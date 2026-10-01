#!/usr/bin/env python3
"""
Headless imagery download into a plain ``{z}/{x}/{y}.jpg`` tree.

Uses the same tile plans, imagery sources (USGS below z16, Google Hybrid z16-20)
and fetch code as ``atak_downloader_finalbuild.py``, but needs no GUI, no phone
and no adb. Output is a standard XYZ ("slippy map") directory that any
offline tile viewer can read, e.g. copy it to ``/tiles`` on a Squatch Mesh SD card.

In a terminal it shows a live curses dashboard (q to stop); piped or with
``--plain`` it prints progress lines instead.

Examples:
    # 10-mile radius, zooms 3-16
    python3 scripts/xyz_tile_download.py --center 47.66 -117.42 --radius 10 --zoom 3-16 -o ~/tiles

    # Whole state(s) at low/mid zoom
    python3 scripts/xyz_tile_download.py --state Idaho --state Washington --zoom 3-13 -o ~/tiles

    # Same area as an existing ATAK imagery package, at the zooms you choose
    python3 scripts/xyz_tile_download.py --like ATAK_SQL_Sandpoint.sqlite --zoom 3-16 -o ~/tiles

    # Bounding box (west south east north), just count tiles
    python3 scripts/xyz_tile_download.py --bbox -117.6 47.5 -117.2 47.8 --zoom 3-17 -o ~/tiles --dry-run

Re-running is safe: tiles already on disk are skipped.
"""
from __future__ import annotations

import argparse
import curses
import sqlite3
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import atak_downloader_finalbuild as dl
from imagery_tile_selection import (
    METERS_PER_MILE,
    build_tiles_for_state_result,
    compute_tiles_for_radius,
    lonlat_to_tile,
    tile_lonlat_bounds,
)

PNG_MAGIC = b"\x89PNG"
STATUSES = ("downloaded", "existing", "missing", "failed")


def parse_zoom_range(text: str) -> List[int]:
    lo, sep, hi = text.partition("-")
    lo_i = int(lo)
    hi_i = int(hi) if sep else lo_i
    if not (0 <= lo_i <= hi_i <= dl.GOOGLE_HYBRID_ZOOM_MAX):
        raise argparse.ArgumentTypeError(f"zoom range must be within 0-{dl.GOOGLE_HYBRID_ZOOM_MAX}, low <= high")
    return list(range(lo_i, hi_i + 1))


def tiles_for_bbox(west: float, south: float, east: float, north: float, z: int) -> List[Tuple[int, int]]:
    x0, y1 = lonlat_to_tile(west, south, z)
    x1, y0 = lonlat_to_tile(east, north, z)
    return [(x, y) for x in range(min(x0, x1), max(x0, x1) + 1) for y in range(min(y0, y1), max(y0, y1) + 1)]


def sqlite_key_to_zxy(key: int) -> Tuple[int, int, int]:
    """Inverse of the osmdroid key ``((z << z) + x << z) + y`` used in ATAK imagery SQLite files."""
    z = 0
    while (z + 1) << (2 * (z + 1)) <= key:
        z += 1
    rest = key - (z << (2 * z))
    return z, rest >> z, rest & ((1 << z) - 1)


def area_from_sqlite(path: Path) -> Tuple:
    """
    Area covered by an ATAK imagery package. Radius packages carry their square footprint as
    ``bounds`` metadata (see ``square_lonlat_footprint_for_radius_miles``), which inverts exactly
    to centre + radius; anything else falls back to the bbox of its deepest zoom's tiles.
    """
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        meta = dict(conn.execute("SELECT key, value FROM ATAK_metadata"))
        if "bounds" in meta:
            w, s, e, n = (float(v) for v in meta["bounds"].split(","))
            lat = (s + n) / 2
            miles = (n - s) / 2 * 111_320.0 / METERS_PER_MILE
            return ("radius", lat, (w + e) / 2, round(miles, 6))
        lo, hi = conn.execute("SELECT MIN(key), MAX(key) FROM tiles").fetchone()
        if lo is None:
            raise ValueError(f"{path} has no tiles")
        zmax = sqlite_key_to_zxy(hi)[0]
        base = zmax << (2 * zmax)
        xs, ys = zip(*(sqlite_key_to_zxy(k)[1:] for (k,) in conn.execute(
            "SELECT key FROM tiles WHERE key >= ?", (base,))))
        w, _, _, n = tile_lonlat_bounds(min(xs), min(ys), zmax)
        _, e, s, _ = tile_lonlat_bounds(max(xs), max(ys), zmax)
        return ("bbox", w, s, e, n)
    finally:
        conn.close()


def area_label(area: Tuple) -> str:
    kind, *v = area
    if kind == "radius":
        return f"{v[2]:g} mi around {v[0]:.5f}, {v[1]:.5f}"
    if kind == "bbox":
        return "bbox " + ", ".join(f"{c:.4f}" for c in v)
    return v[0]


def source_for_zoom(z: int) -> str:
    return "Google Hybrid" if dl.is_google_hybrid_zoom(z) else "USGS"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def human_duration(seconds: float) -> str:
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class Job:
    """State shared between the download thread and whichever UI is drawing it."""

    def __init__(self, args: argparse.Namespace, areas: List[Tuple], out_dir: Path) -> None:
        self.args = args
        self.areas = areas
        self.out_dir = out_dir
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.finished = threading.Event()
        self.phase = "planning"
        self.detail = ""
        self.plan_counts: Dict[int, int] = {}
        self.done_counts: Dict[int, int] = {z: 0 for z in args.zoom}
        self.counts = {s: 0 for s in STATUSES}
        self.bytes = 0
        self.start = 0.0
        self.end = 0.0
        self.recent: deque = deque(maxlen=200)
        self.warnings: List[str] = []
        self.error: Optional[str] = None

    @property
    def total(self) -> int:
        return sum(self.plan_counts.values())

    @property
    def done(self) -> int:
        return sum(self.counts.values())

    def note(self, msg: str) -> None:
        with self.lock:
            self.recent.append(f"{datetime.now():%H:%M:%S} {msg.rstrip()}")

    def area_label(self) -> str:
        return " + ".join(area_label(a) for a in self.areas)


def build_plan(job: Job) -> List[Tuple[int, int, int]]:
    """Unique (z, x, y) tiles covering every requested area, ordered by zoom."""
    wanted = [a[1] for a in job.areas if a[0] == "state"]
    states = dl.load_states(dl.STATE_GEOJSON_PATH) if wanted else {}
    unknown = [s for s in wanted if s not in states]
    if unknown:
        raise ValueError(f"Unknown state(s): {', '.join(unknown)}. Known: {', '.join(sorted(states))}")

    plan: List[Tuple[int, int, int]] = []
    for z in job.args.zoom:
        if job.stop.is_set():
            break
        job.detail = f"z{z}"
        level: Set[Tuple[int, int]] = set()
        for kind, *v in job.areas:
            if kind == "radius":
                level.update(compute_tiles_for_radius(v[0], v[1], v[2], z))
            elif kind == "bbox":
                level.update(tiles_for_bbox(*v, z))
            else:
                result = build_tiles_for_state_result(
                    v[0], states[v[0]], z, geojson_path=dl.STATE_GEOJSON_PATH, tile_plan_dir=dl.TILE_PLAN_DIR
                )
                level.update(result.tiles)
        with job.lock:
            job.plan_counts[z] = len(level)
        plan.extend((z, x, y) for x, y in sorted(level))
    return plan


def tile_path(out_dir: Path, z: int, x: int, y: int, ext: str) -> Path:
    return out_dir / str(z) / str(x) / f"{y}.{ext}"


def fetch(out_dir: Path, z: int, x: int, y: int) -> Tuple[int, str, int]:
    if tile_path(out_dir, z, x, y, "png").exists():
        return z, "existing", 0
    jpg = tile_path(out_dir, z, x, y, "jpg")
    status, nbytes = dl.fetch_tile(z, x, y, jpg, state_label="")
    # Viewers pick the decoder from the extension; keep PNG payloads named .png.
    if status == "downloaded":
        with open(jpg, "rb") as f:
            if f.read(4) == PNG_MAGIC:
                jpg.rename(jpg.with_suffix(".png"))
    return z, status, nbytes


def download(job: Job, plan: List[Tuple[int, int, int]]) -> None:
    workers = max(1, job.args.workers)
    todo = iter(plan)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        while True:
            # Keep a bounded number of tiles in flight so state-sized plans don't queue millions of futures.
            # On stop, submit nothing more and let in-flight tiles finish (no truncated files left behind).
            while not job.stop.is_set() and len(pending) < workers * 4:
                tile = next(todo, None)
                if tile is None:
                    break
                pending.add(pool.submit(fetch, job.out_dir, *tile))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            with job.lock:
                for fut in finished:
                    z, status, nbytes = fut.result()
                    job.counts[status] += 1
                    job.done_counts[z] += 1
                    job.bytes += nbytes


def check_tiles(job: Job) -> None:
    """Flag tiles small offline viewers may refuse: oversized files and progressive JPEGs."""
    max_kb = job.args.check_kb
    big = progressive = 0
    for path in job.out_dir.rglob("*.jpg"):
        if job.stop.is_set():
            return
        data = path.read_bytes()
        if len(data) > max_kb * 1024:
            big += 1
        # SOF2 marker before start-of-scan = progressive DCT (TJpgDec-based firmware can't decode it).
        sos = data.find(b"\xff\xda")
        if b"\xff\xc2" in data[: sos if sos >= 0 else len(data)]:
            progressive += 1
    if big:
        job.warnings.append(f"{big:,} tile(s) larger than {max_kb} KB")
    if progressive:
        job.warnings.append(f"{progressive:,} progressive JPEG tile(s)")


def work(job: Job) -> None:
    try:
        plan = build_plan(job)
        if job.args.dry_run or job.stop.is_set() or not plan:
            return
        dl.load_google_hybrid_tile_url_template()  # once here, not raced by every worker
        job.phase = "downloading"
        job.start = time.monotonic()
        download(job, plan)
        job.end = time.monotonic()
        if job.args.check_kb > 0 and not job.stop.is_set():
            job.phase = "checking"
            check_tiles(job)
    except Exception as e:  # surfaced by the UI, not lost in a thread
        job.error = f"{type(e).__name__}: {e}"
    finally:
        job.end = job.end or time.monotonic()
        job.phase = "stopped" if job.stop.is_set() else "done"
        job.finished.set()


def rate_and_eta(job: Job) -> Tuple[float, float, Optional[float]]:
    elapsed = ((job.end if job.finished.is_set() else time.monotonic()) - job.start) if job.start else 0.0
    rate = job.done / elapsed if elapsed > 0 else 0.0
    eta = (job.total - job.done) / rate if rate > 0 else None
    return elapsed, rate, eta


# -----------------------------
# Plain (non-TTY) output
# -----------------------------


def run_plain(job: Job) -> None:
    last = None
    while not job.finished.wait(2):
        with job.lock:
            lines = list(job.recent)
            job.recent.clear()
        for line in lines:
            print(line, file=sys.stderr)
        if job.phase == "downloading":
            _, rate, eta = rate_and_eta(job)
            c = job.counts
            msg = (
                f"{job.done:,}/{job.total:,}  new {c['downloaded']:,}  skip {c['existing']:,}  "
                f"404 {c['missing']:,}  fail {c['failed']:,}  {rate:.0f}/s  "
                f"eta {human_duration(eta) if eta is not None else '?'}"
            )
            if msg != last:
                print(msg, file=sys.stderr, flush=True)
                last = msg
    for line in job.recent:
        print(line, file=sys.stderr)


# -----------------------------
# Curses dashboard
# -----------------------------

BAR_FULL = "█"
BAR_EMPTY = "░"


class Dashboard:
    def __init__(self, scr, job: Job) -> None:
        self.scr = scr
        self.job = job
        curses.curs_set(0)
        scr.nodelay(True)
        self.c = {}
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            for i, (name, fg) in enumerate(
                [("ok", curses.COLOR_GREEN), ("warn", curses.COLOR_YELLOW), ("bad", curses.COLOR_RED),
                 ("accent", curses.COLOR_CYAN), ("dim", curses.COLOR_BLUE)],
                start=1,
            ):
                curses.init_pair(i, fg, -1)
                self.c[name] = curses.color_pair(i)
            curses.init_pair(10, curses.COLOR_BLACK, curses.COLOR_CYAN)
            self.c["title"] = curses.color_pair(10) | curses.A_BOLD
        self.attr = lambda name: self.c.get(name, curses.A_NORMAL)

    def put(self, y: int, x: int, text: str, attr: int = 0) -> None:
        h, w = self.scr.getmaxyx()
        if 0 <= y < h and x < w:
            try:
                self.scr.addnstr(y, x, text, w - x - (1 if y == h - 1 else 0), attr)
            except curses.error:
                pass

    def bar(self, y: int, x: int, width: int, frac: float, attr: int) -> None:
        width = max(1, width)
        filled = int(round(width * max(0.0, min(1.0, frac))))
        self.put(y, x, BAR_FULL * filled, attr)
        self.put(y, x + filled, BAR_EMPTY * (width - filled), self.attr("dim"))

    def draw(self) -> None:
        job, scr = self.job, self.scr
        scr.erase()
        h, w = scr.getmaxyx()
        with job.lock:
            plan = dict(job.plan_counts)
            done = dict(job.done_counts)
            counts = dict(job.counts)
            nbytes = job.bytes
            recent = list(job.recent)
        elapsed, rate, eta = rate_and_eta(job)
        total, finished = sum(plan.values()), sum(counts.values())

        title = " XYZ Tile Download "
        state = {"planning": f"planning {job.detail}", "downloading": "downloading",
                 "checking": "checking tiles", "done": "done", "stopped": "stopped"}[job.phase]
        if job.stop.is_set() and not job.finished.is_set():
            state = "stopping…"
        self.put(0, 0, (title + " " * w)[:w], self.attr("title"))
        self.put(0, max(len(title) + 1, w - len(state) - 2), state, self.attr("title"))

        y = 2
        self.put(y, 2, "Area   ", curses.A_BOLD); self.put(y, 9, job.area_label()); y += 1
        out = str(job.out_dir)
        if len(out) > w - 10:
            out = "…" + out[-(w - 11):]
        self.put(y, 2, "Output ", curses.A_BOLD); self.put(y, 9, out); y += 2

        # Per-zoom table
        num_w = max(7, len(f"{max(plan.values(), default=0):,}"))
        bar_x = 2 + 4 + 15 + 2 * (num_w + 2)
        bar_w = max(5, min(40, w - bar_x - 8))
        self.put(y, 2, f"{'z':<4}{'source':<15}{'tiles':>{num_w}}  {'done':>{num_w}}", curses.A_BOLD)
        y += 1
        zooms = job.args.zoom
        room = max(1, h - y - 12)
        if len(zooms) > room:  # show the zooms around the one being worked on
            active = next((z for z in zooms if done.get(z, 0) < plan.get(z, 1)), zooms[-1])
            i = max(0, min(zooms.index(active) - room // 2, len(zooms) - room))
            zooms = zooms[i:i + room]
        for z in zooms:
            n, d = plan.get(z), done.get(z, 0)
            src = source_for_zoom(z)
            tiles = f"{n:,}" if n is not None else "…"
            self.put(y, 2, f"{z:<4}", self.attr("accent") | curses.A_BOLD)
            self.put(y, 6, f"{src:<15}{tiles:>{num_w}}  {d:>{num_w},}")
            if n:
                frac = d / n
                self.bar(y, bar_x, bar_w, frac, self.attr("ok") if frac >= 1 else self.attr("accent"))
                self.put(y, bar_x + bar_w + 1, f"{frac * 100:3.0f}%")
            y += 1
        y += 1

        # Overall
        frac = finished / total if total else 0.0
        self.put(y, 2, f"{'Total':<19}{total:>{num_w},}  {finished:>{num_w},}", curses.A_BOLD)
        self.bar(y, bar_x, bar_w, frac, self.attr("ok") | curses.A_BOLD)
        self.put(y, bar_x + bar_w + 1, f"{frac * 100:3.0f}%", curses.A_BOLD)
        y += 2

        x = 2
        for label, key, attr in (("new", "downloaded", "ok"), ("on disk", "existing", "accent"),
                                 ("404", "missing", "warn"), ("failed", "failed", "bad")):
            text = f"{counts[key]:,} {label}"
            self.put(y, x, text, self.attr(attr) if counts[key] else 0)
            x += len(text) + 4
        y += 1
        avg = nbytes / counts["downloaded"] if counts["downloaded"] else 0
        remaining = total - finished
        stats = [
            f"{human_bytes(nbytes)} fetched",
            f"{rate:,.0f} tiles/s",
            f"{human_bytes(nbytes / elapsed if elapsed else 0)}/s",
            f"elapsed {human_duration(elapsed)}",
        ]
        if eta is not None and remaining and not job.finished.is_set():
            stats.append(f"eta {human_duration(eta)}")
        if avg and remaining:
            stats.append(f"~{human_bytes(avg * remaining)} to go")
        self.put(y, 2, "  ·  ".join(stats), self.attr("dim") | curses.A_BOLD)
        y += 2

        # Log / warnings
        msgs = [(m, "warn") for m in job.warnings]
        if job.error:
            msgs.append((job.error, "bad"))
        msgs += [(m, "bad" if "ERROR" in m else "dim") for m in recent]
        log_rows = h - y - 2
        for m, attr in msgs[-log_rows:] if log_rows > 0 else []:
            self.put(y, 2, m, self.attr(attr))
            y += 1

        footer = " q stop · tiles already on disk are skipped on re-run "
        if job.finished.is_set():
            footer = " any key to exit "
        self.put(h - 1, 0, (footer + " " * w)[:w], self.attr("title"))
        scr.refresh()

    def loop(self) -> None:
        while True:
            try:
                self.draw()
                ch = self.scr.getch()
                if job_done := self.job.finished.is_set():
                    if ch != -1:
                        return
                elif ch in (ord("q"), ord("Q"), 27):
                    self.job.stop.set()
                time.sleep(0.1 if not job_done else 0.2)
            except KeyboardInterrupt:
                if self.job.finished.is_set():
                    return
                self.job.stop.set()


def run_curses(job: Job) -> None:
    curses.wrapper(lambda scr: Dashboard(scr, job).loop())


def print_summary(job: Job) -> None:
    c = job.counts
    out = sys.stderr
    for z in job.args.zoom:
        if z in job.plan_counts:
            print(f"  z{z:<3} {source_for_zoom(z):<14} {job.plan_counts[z]:>10,} tiles", file=out)
    print(f"Area: {job.area_label()}", file=out)
    print(f"Total: {job.total:,} tiles -> {job.out_dir}", file=out)
    if job.phase != "planning" and job.start:
        elapsed, _, _ = rate_and_eta(job)
        print(
            f"{'Stopped' if job.stop.is_set() else 'Done'} in {human_duration(elapsed)}: "
            f"{c['downloaded']:,} downloaded ({human_bytes(job.bytes)}), {c['existing']:,} already present, "
            f"{c['missing']:,} not available (404), {c['failed']:,} failed",
            file=out,
        )
        if job.args.check_kb > 0 and job.phase == "done":
            for warning in job.warnings:
                print(f"WARNING: {warning}", file=out)
            if not job.warnings:
                print(f"Check OK: all JPEG tiles are baseline and <= {job.args.check_kb} KB", file=out)
    if job.error:
        print(f"ERROR: {job.error}", file=out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Download imagery tiles into a {z}/{x}/{y}.jpg tree (no GUI or phone needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:", 1)[1],
    )
    area = ap.add_argument_group("area (combine as many as you like)")
    area.add_argument("--center", nargs=2, type=float, metavar=("LAT", "LON"), help="radius centre")
    area.add_argument("--radius", type=float, metavar="MILES", help="radius in miles (with --center)")
    area.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    area.add_argument("--state", action="append", metavar="NAME", help="US state name, repeatable")
    area.add_argument("--like", action="append", type=Path, metavar="SQLITE",
                      help="same area as an existing ATAK imagery .sqlite package, repeatable")
    ap.add_argument("--zoom", type=parse_zoom_range, default=parse_zoom_range("3-16"), metavar="MIN-MAX",
                    help="zoom levels, e.g. 3-16 (default) or 14")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output directory (gets z/x/y.jpg)")
    ap.add_argument("--workers", type=int, default=dl.MAX_DOWNLOAD_WORKERS, help="parallel downloads")
    ap.add_argument("--dry-run", action="store_true", help="only count tiles")
    ap.add_argument("--plain", action="store_true", help="plain progress lines instead of the dashboard")
    ap.add_argument("--check-kb", type=int, default=96, metavar="KB",
                    help="after download, warn about tiles over this size or progressive JPEGs "
                         "(default 96, the Squatch Mesh pager limit; 0 to skip)")
    args = ap.parse_args(argv)

    if bool(args.center) != (args.radius is not None):
        ap.error("--center and --radius go together")
    areas: List[Tuple] = []
    if args.center:
        areas.append(("radius", args.center[0], args.center[1], args.radius))
    if args.bbox:
        areas.append(("bbox", *args.bbox))
    areas += [("state", name) for name in args.state or []]
    for path in args.like or []:
        try:
            areas.append(area_from_sqlite(path.expanduser()))
        except (sqlite3.Error, ValueError) as e:
            ap.error(f"--like {path}: {e}")
    if not areas:
        ap.error("give an area: --center/--radius, --bbox, --state or --like")

    job = Job(args, areas, args.out.expanduser().resolve())
    # The shared downloader logs to stdout; route it into the job so it can't scribble over the dashboard.
    dl.log = lambda msg: (dl.LOGGER._fh.write(f"{msg}\n"), job.note(msg))

    worker = threading.Thread(target=work, args=(job,), daemon=True)
    worker.start()
    use_curses = not (args.plain or args.dry_run) and sys.stdout.isatty() and sys.stdin.isatty()
    try:
        if use_curses:
            run_curses(job)
        else:
            run_plain(job)
    except KeyboardInterrupt:
        job.stop.set()
        print("\nStopping; finishing in-flight tiles…", file=sys.stderr)
    worker.join()
    print_summary(job)
    if job.error:
        return 1
    if job.stop.is_set():
        return 130
    return 1 if job.counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

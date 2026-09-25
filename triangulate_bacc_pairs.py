#!/usr/bin/env python3
"""Triangulate every stereo pair listed in BACC_image_pairs_triangulation.csv.

Uses the WISC field-line geometry from guillaume_triangulation_WISC.ipynb.
Paste the LYR and NYA image folders into LYR_DIR and NYA_DIR below.
Leave PREVIEW_ONLY = True to check the lat/lon box on RGB images, then
set it to False for the full field-line run. REUSE_TRACES traces each
geographic box once, saves those paths under the output folder, and
re-samples brightness on later pairs and later runs.

    python triangulate_bacc_pairs.py
"""

from __future__ import annotations

import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wisc_triangulation import (
    DEFAULT_TRACE,
    Station,
    build_fieldline_grid,
    event_times,
    filter_cached_field_lines,
    make_plots,
    plot_region_box,
)

SCRIPT_DIR = Path(__file__).resolve().parent

LYR_DIR = "/Users/guillaume/Code/random/BACC_frames_direct/LYR/2020/01/03" 
NYA_DIR = "/Users/guillaume/Code/random/BACC_frames_direct/NYA/2020/01/03" 
PREVIEW_ONLY = False
REUSE_TRACES = True
SAVE_SEPARATE_PLOTS = False
SAMPLES = ("greyscale", "RGBproduct")

CAL_DIR = SCRIPT_DIR / "Data"
LYR_CAL = "StarCal_BACC_LYR_2020.h5"
NYA_CAL = "StarCal_BACC_NYA_2020.h5"
PAIRS_CSV = SCRIPT_DIR / "BACC_image_pairs_triangulation - 0901.csv"

_STAMP = re.compile(r"(\d{8})_(\d{6})")


def _require_dir(name: str, value: str | Path) -> Path:
    if not value:
        sys.exit(f"Paste the {name} folder path at the top of this script.")
    path = Path(value).expanduser()
    if not path.is_dir():
        sys.exit(f"{name}={path} is not a directory")
    return path


def event_dt_from_name(filename: str) -> datetime:
    match = _STAMP.search(filename)
    if not match:
        raise ValueError(f"No YYYYMMDD_HHMMSS stamp in {filename!r}")
    return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and (value != value)):
        return ""
    return f"{float(value):.1f}"


def _trace_cache_path(save_dir: Path, cache_key) -> Path:
    lat_range, lon_range, alt_fl, step, upper, lower, interval, dates = cache_key
    name = (
        f"lat{lat_range[0]:.4f}-{lat_range[1]:.4f}"
        f"_lon{lon_range[0]:.4f}-{lon_range[1]:.4f}"
        f"_alt{alt_fl}_step{step}_u{upper}_l{lower}_i{interval}"
        f"_d{dates}.npz"
    )
    return save_dir / "fieldline_cache" / name


def _save_fieldline_paths(path: Path, paths: list, lyr_cal: Path, nya_cal: Path) -> None:
    if not paths:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        lat_fl=np.array([p["lat_fl"] for p in paths], dtype=float),
        lon_fl=np.array([p["lon_fl"] for p in paths], dtype=float),
        alt=np.stack([np.asarray(p["alt"], dtype=float) for p in paths]),
        px_lyr=np.stack([np.asarray(p["px_lyr"], dtype=float) for p in paths]),
        py_lyr=np.stack([np.asarray(p["py_lyr"], dtype=float) for p in paths]),
        px_nya=np.stack([np.asarray(p["px_nya"], dtype=float) for p in paths]),
        py_nya=np.stack([np.asarray(p["py_nya"], dtype=float) for p in paths]),
        lyr_cal=np.array(str(lyr_cal.resolve())),
        nya_cal=np.array(str(nya_cal.resolve())),
    )


def _load_fieldline_paths(path: Path, lyr_cal: Path, nya_cal: Path):
    try:
        with np.load(path, allow_pickle=False) as data:
            stored_lyr = str(data["lyr_cal"].item())
            stored_nya = str(data["nya_cal"].item())
            if stored_lyr != str(lyr_cal.resolve()) or stored_nya != str(nya_cal.resolve()):
                print(f"Ignoring {path}: StarCal paths do not match this run")
                return None
            paths = []
            for i in range(len(data["lat_fl"])):
                paths.append(
                    {
                        "lat_fl": float(data["lat_fl"][i]),
                        "lon_fl": float(data["lon_fl"][i]),
                        "alt": np.array(data["alt"][i], copy=True),
                        "px_lyr": np.array(data["px_lyr"][i], copy=True),
                        "py_lyr": np.array(data["py_lyr"][i], copy=True),
                        "px_nya": np.array(data["px_nya"][i], copy=True),
                        "py_nya": np.array(data["py_nya"][i], copy=True),
                    }
                )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Ignoring unreadable field-line cache {path}: {exc}")
        return None
    if not paths:
        return None
    return paths


def empty_stats() -> dict:
    keys = (
        "n_lines",
        "lyr_mean_km",
        "lyr_std_km",
        "lyr_sem_km",
        "lyr_median_km",
        "nya_mean_km",
        "nya_std_km",
        "nya_sem_km",
        "nya_median_km",
        "both_mean_km",
        "both_std_km",
        "both_sem_km",
        "both_median_km",
        "lyr_mean_profile_peak_km",
        "nya_mean_profile_peak_km",
        "both_mean_profile_peak_km",
    )
    stats = {k: float("nan") for k in keys}
    stats["n_lines"] = 0
    return stats


def main() -> int:
    lyr_dir = _require_dir("LYR_DIR", LYR_DIR)
    nya_dir = _require_dir("NYA_DIR", NYA_DIR)

    if not PAIRS_CSV.is_file():
        sys.exit(f"Pairs CSV not found: {PAIRS_CSV}")
    cal_lyr = CAL_DIR / LYR_CAL
    cal_nya = CAL_DIR / NYA_CAL
    if not cal_lyr.is_file() or not cal_nya.is_file():
        sys.exit(f"StarCal files not found in {CAL_DIR}")

    with PAIRS_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        sys.exit(f"No pairs in {PAIRS_CSV}")

    save_dir = SCRIPT_DIR / "out" / PAIRS_CSV.stem
    save_dir.mkdir(parents=True, exist_ok=True)
    results_csv = save_dir / "WISC_batch_results.csv"
    print(f"Saving plots to {save_dir}")

    lyr = None
    nya = None
    results = []
    trace_cache = {}

    for row in rows:
        lyr_name = row["LYR image"]
        nya_name = row["NYA image"]
        lyr_path = lyr_dir / lyr_name
        nya_path = nya_dir / nya_name
        if not lyr_path.is_file():
            sys.exit(f"LYR image not found: {lyr_path}")
        if not nya_path.is_file():
            sys.exit(f"NYA image not found: {nya_path}")

        event_dt = event_dt_from_name(lyr_name)
        dates, ut_time, _date_time, iso_basic = event_times(event_dt)
        lat_range = (float(row["lat1"]), float(row["lat2"]))
        lon_range = (float(row["lon1"]), float(row["lon2"]))
        region = {
            "lat1": lat_range[0],
            "lat2": lat_range[1],
            "lon1": lon_range[0],
            "lon2": lon_range[1],
            "label": "R1",
        }

        print(f"\n=== {lyr_name} / {nya_name}  {event_dt.isoformat()} ===")
        print(f"LYR {lyr_path}")
        print(f"NYA {nya_path}")
        print(f"box lat={lat_range} lon={lon_range}")

        if lyr is None:
            lyr = Station("LYR", lyr_path, cal_lyr)
            nya = Station("NYA", nya_path, cal_nya)
            print(
                f"LYR {lyr.lat:.6f}N {lyr.lon:.6f}E {lyr.alt_km * 1000:.0f} m, "
                f"NYA {nya.lat:.6f}N {nya.lon:.6f}E {nya.alt_km * 1000:.0f} m"
            )
        else:
            lyr.set_image(lyr_path)
            nya.set_image(nya_path)

        if PREVIEW_ONLY:
            plot_region_box(
                lat_range,
                lon_range,
                lyr=lyr,
                nya=nya,
                save_dir=str(save_dir),
                save_prefix=iso_basic,
                region=region,
                event_dt=event_dt,
            )
            continue

        if REUSE_TRACES:
            cache_key = (
                lat_range,
                lon_range,
                DEFAULT_TRACE["alt_fl"],
                DEFAULT_TRACE["step"],
                DEFAULT_TRACE["upper"],
                DEFAULT_TRACE["lower"],
                DEFAULT_TRACE["interval"],
                dates,
            )
            if cache_key not in trace_cache:
                cache_file = _trace_cache_path(save_dir, cache_key)
                loaded = (
                    _load_fieldline_paths(cache_file, cal_lyr, cal_nya)
                    if cache_file.is_file()
                    else None
                )
                if loaded is not None:
                    print(f"Loaded cached field-line paths from {cache_file}")
                    trace_cache[cache_key] = loaded
                else:
                    print(f"Building field-line cache for box {lat_range}, {lon_range}")
                    paths = build_fieldline_grid(
                        lat_range,
                        lon_range,
                        DEFAULT_TRACE["alt_fl"],
                        DEFAULT_TRACE["step"],
                        DEFAULT_TRACE["upper"],
                        DEFAULT_TRACE["lower"],
                        DEFAULT_TRACE["interval"],
                        lyr,
                        nya,
                        dates,
                        ut_time,
                    )
                    _save_fieldline_paths(cache_file, paths, cal_lyr, cal_nya)
                    if paths:
                        print(f"Saved field-line paths to {cache_file}")
                    trace_cache[cache_key] = paths
            else:
                print("Reusing cached field-line paths for this box")
            paths = trace_cache[cache_key]
        else:
            paths = build_fieldline_grid(
                lat_range,
                lon_range,
                DEFAULT_TRACE["alt_fl"],
                DEFAULT_TRACE["step"],
                DEFAULT_TRACE["upper"],
                DEFAULT_TRACE["lower"],
                DEFAULT_TRACE["interval"],
                lyr,
                nya,
                dates,
                ut_time,
            )

        for sample in SAMPLES:
            print(f"Sampling brightness: {sample}")
            sample_dir = save_dir / sample
            sample_dir.mkdir(parents=True, exist_ok=True)
            all_brightness_values_LYR, all_brightness_values_NYA, alt_fieldline_last, lines_LYR, lines_NYA = (
                filter_cached_field_lines(
                    paths,
                    lyr,
                    nya,
                    DEFAULT_TRACE["alt_fl"],
                    DEFAULT_TRACE["upper"],
                    DEFAULT_TRACE["lower"],
                    DEFAULT_TRACE["peak"],
                    DEFAULT_TRACE["corr_thr"],
                    DEFAULT_TRACE["edge"],
                    DEFAULT_TRACE["frac"],
                    DEFAULT_TRACE["width_limit"],
                    DEFAULT_TRACE["centroid_thr"],
                    lat_range=lat_range,
                    lon_range=lon_range,
                    sample=sample,
                )
            )

            record = {
                "index": row.get("Index", ""),
                "sample": sample,
                "lyr_image": lyr_name,
                "nya_image": nya_name,
                "event_utc": event_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "lat1": lat_range[0],
                "lat2": lat_range[1],
                "lon1": lon_range[0],
                "lon2": lon_range[1],
                "out_dir": str(sample_dir),
            }

            if not all_brightness_values_LYR:
                stats = empty_stats()
            else:
                stats = make_plots(
                    all_brightness_values_LYR,
                    all_brightness_values_NYA,
                    alt_fieldline_last,
                    lines_LYR,
                    lines_NYA,
                    lat_range,
                    lon_range,
                    lyr=lyr,
                    nya=nya,
                    event_dt=event_dt,
                    save_dir=str(sample_dir),
                    save_prefix=iso_basic,
                    region=region,
                    save_separate=SAVE_SEPARATE_PLOTS,
                    sample=sample,
                )

            record.update(stats)
            results.append(record)

    if PREVIEW_ONLY:
        print("\nPREVIEW_ONLY=True: wrote lat/lon box overlays, skipped tracing.")
        print("Set PREVIEW_ONLY = False to run field-line tracing.")
        return 0

    fieldnames = [
        "index",
        "sample",
        "lyr_image",
        "nya_image",
        "event_utc",
        "lat1",
        "lat2",
        "lon1",
        "lon2",
        "n_lines",
        "lyr_mean_km",
        "lyr_median_km",
        "lyr_std_km",
        "lyr_sem_km",
        "nya_mean_km",
        "nya_median_km",
        "nya_std_km",
        "nya_sem_km",
        "both_mean_km",
        "both_median_km",
        "both_std_km",
        "both_sem_km",
        "lyr_mean_profile_peak_km",
        "nya_mean_profile_peak_km",
        "both_mean_profile_peak_km",
        "out_dir",
    ]
    with results_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in results:
            row_out = dict(record)
            for key in fieldnames:
                if key.endswith("_km"):
                    row_out[key] = _fmt(record.get(key))
            writer.writerow(row_out)

    print(f"\nWrote {results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

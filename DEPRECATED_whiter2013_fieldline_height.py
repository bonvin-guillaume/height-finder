#!/usr/bin/env python3
"""Peak auroral emission height from a stereo all-sky pair.

Whiter et al. (2013) Method 2. By default, draw a rectangle on LYR
only. The brightest structure in that box is mapped to lat/lon, field
lines are traced with IGRF, and the same lines are projected onto NYA.

    python3 whiter2013_fieldline_height.py
    python3 whiter2013_fieldline_height.py --every-pixel
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def _ensure_script_cwd() -> None:
    try:
        os.getcwd()
    except OSError:
        os.chdir(str(SCRIPT_DIR))


_ensure_script_cwd()
os.environ.setdefault("GEOPACK_PATH", str(SCRIPT_DIR / ".geopack_data"))
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))
for _var in ("GEOPACK_PATH", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
    Path(os.environ[_var]).mkdir(parents=True, exist_ok=True)

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.interpolate import LinearNDInterpolator  # noqa: E402
from scipy.ndimage import map_coordinates, median_filter  # noqa: E402
from scipy.signal import medfilt  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial import ConvexHull, cKDTree  # noqa: E402

R_EARTH_KM = 6371.2
# DEFAULT_EPOCH = "2019-12-03T11:58:58"
DEFAULT_EPOCH = "2020-01-03T08:57:41"

# Whiter §4.2: random footprints and the two-iteration tests
D_LAT = 0.05
D_LON = 0.075
CORR_1, CENTROID_1_KM = 0.5, 20.0
CORR_2, CENTROID_2_KM = 0.7, 1.0


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class Config:
    data_dir: Path = SCRIPT_DIR / "Data"

    lyr_image: str = "BACC_LYR2020.png"
    nya_image: str = "BACC_NYA2020.png"
    lyr_cal: str = "StarCal_BACC_LYR_2020.h5"
    nya_cal: str = "StarCal_BACC_NYA_2020.h5"
    out_dir: Path = SCRIPT_DIR / "out_BACC_0857"

    # lyr_image: str = "LYR2020.jpg"
    # nya_image: str = "NYA2020.JPG"
    # lyr_cal: str = "StarCal_LYR_2020.h5"
    # nya_cal: str = "StarCal_NYA_2020.h5"
    # out_dir: Path = SCRIPT_DIR / "out_SONY_0857"

    # lyr_image: str = "GREEN_BACC_LYR_20191203_115858.png"
    # nya_image: str = "GREEN_BACC_NYA_20191203_115858.png"
    # lyr_cal: str = "StarCal_BACC_LYR_2020.h5"
    # nya_cal: str = "StarCal_BACC_NYA_2020.h5"
    # out_dir: Path = SCRIPT_DIR / "out_every_pixel"

    time_utc: datetime | None = None  # event time for IGRF
    cluster_station: str = "LYR"  # which image is clustered (--pick forces LYR)
    pick: bool = True  # interactive rectangle on LYR; --no-pick clusters the whole image
    lyr_box: tuple[float, float, float, float] | None = None  # pixel ROI from --pick
    every_pixel: bool = False  # skip clustering; one IGRF seed per ROI pixel
    pixel_step: int = 1  # use every Nth pixel in the ROI when every_pixel is set

    h_min: float = 80.0  # lowest altitude sampled along each field line [km]
    h_max: float = 250.0  # highest altitude sampled along each field line [km]
    h_step: float = 2.0  # altitude spacing of those samples [km]
    footprint_alt: float = 150.0  # map LYR cluster to lat/lon at this height, then IGRF [km]

    el_min: float = 15.0  # ignore pixels below this elevation [deg]
    subsample: int = 4  # every Nth pixel when building az/el → pixel interpolator
    bright_pct: float = 10.0  # cluster the brightest this % of FOV pixels (Whiter)
    star_filter: int = 7  # median-filter size [px] before clustering; 0 disables
    seed: int = 0  # RNG seed for random field-line footprints

    altitudes: np.ndarray = field(default_factory=lambda: np.empty(0))

    def finalise(self) -> None:
        self.altitudes = np.arange(self.h_min, self.h_max + 1e-9, self.h_step)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def date_int(self) -> int:
        return int(self.time_utc.strftime("%Y%m%d"))

    @property
    def ut_hours(self) -> float:
        t = self.time_utc
        return t.hour + t.minute / 60.0 + t.second / 3600.0


# --------------------------------------------------------------------------
# Geometry (WISC / geo.ipynb)
# --------------------------------------------------------------------------


def geo_to_ecef(lat_deg, lon_deg, alt_km):
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    r = R_EARTH_KM + np.asarray(alt_km, dtype=float)
    return np.stack(
        [r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)],
        axis=-1,
    )


def _enu_basis(lat_deg, lon_deg):
    slat, slon = np.radians(lat_deg), np.radians(lon_deg)
    east = np.array([-np.sin(slon), np.cos(slon), 0.0])
    north = np.array(
        [-np.sin(slat) * np.cos(slon), -np.sin(slat) * np.sin(slon), np.cos(slat)]
    )
    up = np.array(
        [np.cos(slat) * np.cos(slon), np.cos(slat) * np.sin(slon), np.sin(slat)]
    )
    return east, north, up


def azel_from_station(site_lat, site_lon, site_alt_km, lat, lon, alt_km):
    origin = geo_to_ecef(site_lat, site_lon, site_alt_km)
    target = geo_to_ecef(lat, lon, alt_km)
    d = target - origin
    east, north, up = _enu_basis(site_lat, site_lon)
    e, n, u = d @ east, d @ north, d @ up
    rng = np.sqrt(e**2 + n**2 + u**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        elev = np.degrees(np.arcsin(np.where(rng > 0, u / rng, np.nan)))
    azim = np.degrees(np.arctan2(e, n)) % 360.0
    return azim, elev


def azel_to_uv(az_deg, el_deg):
    z = 90.0 - np.asarray(el_deg, dtype=float)
    a = np.radians(np.asarray(az_deg, dtype=float))
    return z * np.sin(a), z * np.cos(a)


def look_to_latlon(station: Station, az_deg, el_deg, target_alt_km):
    """Ray from the camera along (az, el) onto the sphere at target_alt_km."""
    origin = geo_to_ecef(station.lat, station.lon, station.alt_km)
    east, north, up = _enu_basis(station.lat, station.lon)
    az = np.radians(np.asarray(az_deg, dtype=float))
    el = np.radians(np.asarray(el_deg, dtype=float))
    direction = (
        np.cos(el) * np.sin(az)
    )[..., None] * east + (
        np.cos(el) * np.cos(az)
    )[..., None] * north + np.sin(el)[..., None] * up
    radius = R_EARTH_KM + target_alt_km
    b = 2.0 * (direction @ origin)
    c = float(origin @ origin) - radius * radius
    disc = b * b - 4.0 * c
    lat = np.full(az.shape, np.nan)
    lon = np.full(az.shape, np.nan)
    ok = disc >= 0.0
    if not np.any(ok):
        return lat, lon
    sqrt_d = np.sqrt(np.maximum(disc, 0.0))
    t1 = (-b - sqrt_d) / 2.0
    t2 = (-b + sqrt_d) / 2.0
    t = np.where((t1 > 1e-6) & (t2 > 1e-6), np.minimum(t1, t2), np.where(t1 > 1e-6, t1, t2))
    hit = ok & (t > 1e-6)
    point = origin + t[hit, None] * direction[hit]
    rad = np.linalg.norm(point, axis=1)
    lat[hit] = np.degrees(np.arcsin(np.clip(point[:, 2] / rad, -1.0, 1.0)))
    lon[hit] = np.degrees(np.arctan2(point[:, 1], point[:, 0]))
    return lat, lon


# --------------------------------------------------------------------------
# Station
# --------------------------------------------------------------------------


class Station:
    def __init__(self, name: str, image_path: Path, cal_path: Path, cfg: Config):
        self.name = name
        self.cfg = cfg
        with h5py.File(cal_path, "r") as handle:
            attrs = dict(handle.attrs)
            az_corner = handle["azimuth_deg"][:].astype(np.float64)
            el_corner = handle["elevation_deg"][:].astype(np.float64)
        self.lat = float(np.ravel(attrs["site_lat_deg"])[0])
        self.lon = float(np.ravel(attrs["site_lon_deg"])[0])
        self.alt_km = float(np.ravel(attrs["site_alt_m"])[0]) / 1000.0
        self.width = int(np.ravel(attrs["image_width"])[0])
        self.height = int(np.ravel(attrs["image_height"])[0])
        for key in ("flip_image_x", "flip_image_y"):
            if int(np.ravel(attrs.get(key, [0]))[0]) != 0:
                raise NotImplementedError(f"{cal_path.name} sets {key}=1")
        expected = (self.height + 1, self.width + 1)
        if az_corner.shape != expected:
            raise ValueError(
                f"{cal_path.name}: azimuth grid {az_corner.shape}, expected {expected}"
            )
        self.image = np.asarray(Image.open(image_path), dtype=np.float64)
        if self.image.ndim == 3:
            self.image = self.image.mean(axis=2)
        if self.image.shape != (self.height, self.width):
            raise ValueError(
                f"{image_path.name} is {self.image.shape[1]}x{self.image.shape[0]} "
                f"but calibration describes {self.width}x{self.height}"
            )
        u_c, v_c = azel_to_uv(az_corner, el_corner)
        self.u = 0.25 * (u_c[:-1, :-1] + u_c[:-1, 1:] + u_c[1:, :-1] + u_c[1:, 1:])
        self.v = 0.25 * (v_c[:-1, :-1] + v_c[:-1, 1:] + v_c[1:, :-1] + v_c[1:, 1:])
        self.el = 0.25 * (
            el_corner[:-1, :-1]
            + el_corner[:-1, 1:]
            + el_corner[1:, :-1]
            + el_corner[1:, 1:]
        )
        self.az = np.degrees(np.arctan2(self.u, self.v)) % 360.0
        self._build_inverse()

    def _build_inverse(self) -> None:
        step = self.cfg.subsample
        py, px = np.mgrid[0 : self.height : step, 0 : self.width : step]
        u, v, el = self.u[::step, ::step], self.v[::step, ::step], self.el[::step, ::step]
        inside = el > self.cfg.el_min
        if inside.sum() < 100:
            raise ValueError(f"{self.name}: almost nothing above el_min")
        self._interp = LinearNDInterpolator(
            np.column_stack([u[inside], v[inside]]),
            np.column_stack([px[inside].astype(float), py[inside].astype(float)]),
        )

    def azel_to_pixel(self, az_deg, el_deg):
        az = np.atleast_1d(np.asarray(az_deg, dtype=float))
        el = np.atleast_1d(np.asarray(el_deg, dtype=float))
        u, v = azel_to_uv(az, el)
        out = np.full((az.size, 2), np.nan)
        ok = np.isfinite(u) & np.isfinite(v) & (el > self.cfg.el_min)
        if ok.any():
            out[ok] = self._interp(np.column_stack([u[ok], v[ok]]))
        px, py = out[:, 0], out[:, 1]
        off = ~(
            np.isfinite(px)
            & np.isfinite(py)
            & (px >= 0)
            & (px <= self.width - 1)
            & (py >= 0)
            & (py <= self.height - 1)
        )
        px[off] = np.nan
        py[off] = np.nan
        return px, py

    def sample(self, px, py):
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        good = np.isfinite(px) & np.isfinite(py)
        out = np.full(px.shape, np.nan)
        if good.any():
            out[good] = map_coordinates(
                self.image, [py[good], px[good]], order=1, mode="nearest"
            )
        return out


# --------------------------------------------------------------------------
# Peak / centroid / P (geo.ipynb, used because Whiter does not specify)
# --------------------------------------------------------------------------


def find_peak(alt, prof):
    valid = np.isfinite(prof)
    if valid.sum() < 5:
        return np.nan, np.nan
    a, p = np.asarray(alt)[valid], np.asarray(prof)[valid]
    smooth = medfilt(p, kernel_size=3)
    i = int(np.argmax(smooth))
    peak_alt, peak_val = float(a[i]), float(smooth[i])
    if 0 < i < len(a) - 1:
        x0, x1, x2 = a[i - 1], a[i], a[i + 1]
        y0, y1, y2 = smooth[i - 1], smooth[i], smooth[i + 1]
        d1, d2, d3 = (x0 - x1) * (x0 - x2), (x1 - x0) * (x1 - x2), (x2 - x0) * (x2 - x1)
        if min(abs(d1), abs(d2), abs(d3)) > 1e-12:
            qa = y0 / d1 + y1 / d2 + y2 / d3
            qb = (
                -y0 * (x1 + x2) / d1
                - y1 * (x0 + x2) / d2
                - y2 * (x0 + x1) / d3
            )
            if qa < 0:
                vertex = -qb / (2 * qa)
                if min(x0, x2) <= vertex <= max(x0, x2):
                    peak_alt = float(vertex)
    return peak_alt, peak_val


def centroid(alt, prof):
    valid = np.isfinite(prof)
    if valid.sum() == 0:
        return np.nan
    a, p = np.asarray(alt)[valid], np.clip(np.asarray(prof)[valid], 0.0, None)
    return float(np.sum(p * a) / np.sum(p)) if np.sum(p) > 0 else np.nan


def brightness_ratio_p(prof):
    v = np.asarray(prof, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    med = np.median(v)
    if med == 0:
        return np.nan
    return float(np.percentile(v, 60) / med)


def mean_above_below(alt, prof, peak_alt):
    valid = np.isfinite(prof)
    a, p = np.asarray(alt)[valid], np.asarray(prof)[valid]
    above = p[a > peak_alt]
    below = p[a < peak_alt]
    ma = float(np.mean(above)) if above.size else np.nan
    mb = float(np.mean(below)) if below.size else np.nan
    return ma, mb


def normalised(prof):
    m = np.nanmax(prof) if np.isfinite(prof).any() else np.nan
    return prof / m if np.isfinite(m) and m > 0 else prof


def _line_result(reason: str = "", **extra):
    res = {
        "peak_lyr": np.nan,
        "peak_nya": np.nan,
        "centroid_lyr": np.nan,
        "centroid_nya": np.nan,
        "corr": np.nan,
        "p_lyr": np.nan,
        "p_nya": np.nan,
        "n_valid": 0,
        "accepted": False,
        "reason": reason,
    }
    res.update(extra)
    return res


def evaluate_line(alt, prof_a, prof_b, corr_thr: float, centroid_thr: float):
    """Whiter's three tests. Peak test uses AND on both sides (paper: both)."""
    both = np.isfinite(prof_a) & np.isfinite(prof_b)
    res = _line_result(n_valid=int(both.sum()))
    if both.sum() < 8:
        res["reason"] = "too few points in both fields of view"
        return res
    pa, va = find_peak(alt, prof_a)
    pb, vb = find_peak(alt, prof_b)
    res["peak_lyr"], res["peak_nya"] = pa, pb
    res["centroid_lyr"] = centroid(alt, prof_a)
    res["centroid_nya"] = centroid(alt, prof_b)
    na, nb = prof_a[both], prof_b[both]
    if np.std(na) > 0 and np.std(nb) > 0:
        res["corr"] = float(np.corrcoef(na, nb)[0, 1])
    p_a, p_b = brightness_ratio_p(prof_a), brightness_ratio_p(prof_b)
    res["p_lyr"], res["p_nya"] = p_a, p_b
    ma_a, mb_a = mean_above_below(alt, prof_a, pa)
    ma_b, mb_b = mean_above_below(alt, prof_b, pb)
    if not (np.isfinite(res["corr"]) and res["corr"] > corr_thr):
        res["reason"] = f"correlation {res['corr']:.2f} <= {corr_thr:g}"
        return res
    lyr_clear = np.isfinite(va) and va > p_a * ma_a and va > p_a * mb_a
    nya_clear = np.isfinite(vb) and vb > p_b * ma_b and vb > p_b * mb_b
    if not lyr_clear:
        res["reason"] = "LYR peak not clear against upper and lower means"
        return res
    if not nya_clear:
        res["reason"] = "NYA peak not clear against upper and lower means"
        return res
    gap = abs(res["centroid_lyr"] - res["centroid_nya"])
    if not (np.isfinite(gap) and gap < centroid_thr):
        res["reason"] = f"centroids differ by {gap:.1f} km"
        return res
    res["accepted"] = True
    return res


def normalise_box(x0, y0, x1, y1, width, height):
    xmin, xmax = sorted((float(x0), float(x1)))
    ymin, ymax = sorted((float(y0), float(y1)))
    xmin = float(np.clip(xmin, 0.0, width - 1))
    xmax = float(np.clip(xmax, 0.0, width - 1))
    ymin = float(np.clip(ymin, 0.0, height - 1))
    ymax = float(np.clip(ymax, 0.0, height - 1))
    if xmax - xmin < 5 or ymax - ymin < 5:
        raise SystemExit("selected rectangle is too small")
    return xmin, ymin, xmax, ymax


def track_hits_box(px, py, box) -> np.ndarray:
    xmin, ymin, xmax, ymax = box
    hit = (
        np.isfinite(px)
        & np.isfinite(py)
        & (px >= xmin)
        & (px <= xmax)
        & (py >= ymin)
        & (py <= ymax)
    )
    return hit.any(axis=1)


def _pick_command() -> str:
    return f"python3 {SCRIPT_DIR / 'whiter2013_fieldline_height.py'}"


def _interactive_backend() -> str:
    for name in ("macosx", "MacOSX", "TkAgg", "QtAgg"):
        try:
            plt.switch_backend(name)
            return plt.get_backend()
        except Exception:
            continue
    raise SystemExit(
        f"--pick needs a display. Run from Terminal.app:\n  {_pick_command()}"
    )


def _show_image(ax, station: Station):
    img = station.image
    step = max(1, img.shape[0] // 900)
    shown = img[::step, ::step]
    vmax = np.percentile(shown, 99.5)
    ax.imshow(
        shown,
        cmap="gray",
        vmin=np.percentile(shown, 5),
        vmax=vmax if vmax > 0 else None,
        extent=(0, station.width, station.height, 0),
    )
    ax.set_xlim(0, station.width)
    ax.set_ylim(station.height, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def pick_rectangle(station: Station, what: str = "the aurora") -> tuple[float, float, float, float]:
    from matplotlib.widgets import RectangleSelector

    fig, ax = plt.subplots(figsize=(9, 9))
    _show_image(ax, station)
    ax.set_title(
        f"{station.name}: drag a rectangle around {what}, "
        "then close the window (or press Enter)",
        fontsize=11,
    )
    chosen = {"box": None}

    def onselect(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        chosen["box"] = (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)

    selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="data",
        interactive=True,
    )

    def on_key(event):
        if event.key in ("enter", "return"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    box = chosen["box"]
    if box is None:
        xmin, xmax, ymin, ymax = selector.extents
        if xmax > xmin and ymax > ymin:
            box = (xmin, ymin, xmax, ymax)
    if box is None:
        raise SystemExit(f"no rectangle selected on {station.name}")
    return normalise_box(*box, station.width, station.height)


def pick_lyr_roi(lyr: Station):
    """Interactive rectangle on LYR only; NYA is not picked."""
    _interactive_backend()
    try:
        return pick_rectangle(lyr, "the aurora")
    finally:
        plt.close("all")
        try:
            plt.switch_backend("Agg")
        except Exception:
            matplotlib.use("Agg", force=True)


# --------------------------------------------------------------------------
# Clustering (Whiter §4.2)
# --------------------------------------------------------------------------


def brightest_cluster(
    station: Station,
    bright_pct: float,
    box=None,
    star_filter: int = 7,
    footprint_alt: float | None = None,
):
    """Largest cluster of the brightest `bright_pct` of FOV pixels.

    Stars are removed first with a 2-D median filter (not used when
    sampling brightness along field lines).  If ``box`` is set, only
    pixels inside it are clustered.  Look directions are mapped to
    lat/lon at ``footprint_alt`` (default: ``station.cfg.footprint_alt``).
    """
    if footprint_alt is None:
        footprint_alt = station.cfg.footprint_alt
    fov = station.el > station.cfg.el_min
    if box is not None:
        xmin, ymin, xmax, ymax = box
        yy, xx = np.mgrid[0 : station.height, 0 : station.width]
        fov = fov & (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)
    if fov.sum() < 50:
        raise SystemExit(
            f"{station.name}: not enough pixels in the FOV"
            + (" / pick box" if box is not None else "")
        )
    work = station.image
    k = int(star_filter)
    if k >= 3:
        if k % 2 == 0:
            k += 1
        if box is not None:
            y0, y1 = int(np.floor(ymin)), int(np.ceil(ymax)) + 1
            x0, x1 = int(np.floor(xmin)), int(np.ceil(xmax)) + 1
            y0, x0 = max(0, y0), max(0, x0)
            y1 = min(station.height, y1)
            x1 = min(station.width, x1)
            work = station.image.copy()
            work[y0:y1, x0:x1] = median_filter(
                station.image[y0:y1, x0:x1], size=k
            )
        else:
            work = median_filter(station.image, size=k)
    threshold = np.percentile(work[fov], 100.0 - bright_pct)
    mask = fov & (work >= threshold)
    ys, xs = np.nonzero(mask)
    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    tree = cKDTree(coords)
    nn = tree.query(coords, k=2)[0][:, 1]
    d_min = float(np.min(nn[nn > 0])) if np.any(nn > 0) else 1.0
    eps = 2.0 * d_min
    n = coords.shape[0]
    pairs = tree.query_pairs(eps, output_type="ndarray")
    if pairs.size == 0:
        labels = np.arange(n)
        n_comp = n
    else:
        graph = coo_matrix(
            (np.ones(pairs.shape[0], dtype=np.uint8), (pairs[:, 0], pairs[:, 1])),
            shape=(n, n),
        )
        graph = graph + graph.T
        n_comp, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels, minlength=n_comp)
    keep = labels == int(np.argmax(counts))
    cluster_mask = np.zeros_like(mask, dtype=bool)
    cluster_mask[ys[keep], xs[keep]] = True
    az, el = station.az[ys[keep], xs[keep]], station.el[ys[keep], xs[keep]]
    lat, lon = look_to_latlon(station, az, el, footprint_alt)
    good = np.isfinite(lat) & np.isfinite(lon)
    if good.sum() < 10:
        raise SystemExit(
            f"cluster did not map to {footprint_alt:g} km; try the other --cluster-station"
        )
    print(
        f"  {station.name}: brightest {bright_pct:g}% = {mask.sum()} px"
        f"{' (inside pick box)' if box is not None else ''}, "
        f"d_min = {d_min:.2f} px, {n_comp} clusters, "
        f"largest {int(counts.max())} px, star median {k if k >= 3 else 0} px"
    )
    return cluster_mask, lat[good], lon[good]


def region_pixels_to_footprints(
    station: Station,
    box,
    footprint_alt: float | None = None,
    step: int = 1,
):
    """Map every (or every Nth) pixel in ``box`` to lat/lon at ``footprint_alt``.

    No brightness clustering.  Pixels below ``el_min`` or that miss the
    altitude sphere are dropped.
    """
    if box is None:
        raise SystemExit("--every-pixel needs a selected region (use --pick)")
    if footprint_alt is None:
        footprint_alt = station.cfg.footprint_alt
    xmin, ymin, xmax, ymax = box
    step = max(1, int(step))
    x0 = max(0, int(np.floor(xmin)))
    y0 = max(0, int(np.floor(ymin)))
    x1 = min(station.width, int(np.ceil(xmax)) + 1)
    y1 = min(station.height, int(np.ceil(ymax)) + 1)
    yy, xx = np.mgrid[y0:y1:step, x0:x1:step]
    keep = (
        (xx >= xmin)
        & (xx <= xmax)
        & (yy >= ymin)
        & (yy <= ymax)
        & (station.el[yy, xx] > station.cfg.el_min)
    )
    ys = yy[keep]
    xs = xx[keep]
    if ys.size == 0:
        raise SystemExit(f"{station.name}: no pixels above el_min in the pick box")
    az = station.az[ys, xs]
    el = station.el[ys, xs]
    lat, lon = look_to_latlon(station, az, el, footprint_alt)
    good = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(good):
        raise SystemExit(
            f"no pixel in the pick box mapped to {footprint_alt:g} km"
        )
    cluster_mask = np.zeros((station.height, station.width), dtype=bool)
    cluster_mask[ys[good], xs[good]] = True
    print(
        f"  {station.name}: {int(keep.sum())} ROI pixels"
        f"{'' if step == 1 else f' (step {step})'}, "
        f"{int(good.sum())} map to {footprint_alt:g} km"
    )
    return cluster_mask, lat[good], lon[good]


def _polygon_area(verts) -> float:
    return float(
        0.5
        * np.abs(
            np.dot(verts[:, 0], np.roll(verts[:, 1], 1))
            - np.dot(verts[:, 1], np.roll(verts[:, 0], 1))
        )
    )


def region_polygon(lat, lon):
    xy = np.column_stack([lon, lat])
    if xy.shape[0] < 3:
        raise SystemExit("cluster region too small for a polygon")
    stride = max(1, xy.shape[0] // 4000)
    sample = xy[::stride]
    if sample.shape[0] < 3:
        sample = xy
    hull = ConvexHull(sample)
    verts = sample[hull.vertices]
    path = MplPath(verts)
    area = _polygon_area(verts)
    return path, area, (float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max()))


def random_footprints(path: MplPath, bbox, d_lat, d_lon, rng: np.random.Generator, area=None):
    lat_lo, lat_hi, lon_lo, lon_hi = bbox
    if area is None:
        area = _polygon_area(path.vertices)
    n_target = max(8, int(round(area / (d_lat * d_lon))))
    lat_out, lon_out = [], []
    tries = 0
    while len(lat_out) < n_target and tries < 40:
        tries += 1
        batch = max(n_target * 4, 64)
        lat = rng.uniform(lat_lo, lat_hi, batch)
        lon = rng.uniform(lon_lo, lon_hi, batch)
        inside = path.contains_points(np.column_stack([lon, lat]))
        lat_out.extend(lat[inside].tolist())
        lon_out.extend(lon[inside].tolist())
    lat_out = np.asarray(lat_out[:n_target], dtype=float)
    lon_out = np.asarray(lon_out[:n_target], dtype=float)
    if lat_out.size == 0:
        raise SystemExit("no random footprints landed inside the cluster region")
    return lat_out, lon_out


def random_around_passers(lat_pass, lon_pass, d_lat, d_lon, rng: np.random.Generator):
    """Iteration 2: spacing / 3 inside a one-spacing box around each passer."""
    if lat_pass.size == 0:
        raise SystemExit("no field lines passed iteration 1")
    d_lat2, d_lon2 = d_lat / 3.0, d_lon / 3.0
    n_each = max(4, int(round((2.0 * d_lat) * (2.0 * d_lon) / (d_lat2 * d_lon2))))
    lat_out = []
    lon_out = []
    for la, lo in zip(lat_pass, lon_pass):
        lat_out.append(rng.uniform(la - d_lat, la + d_lat, n_each))
        lon_out.append(rng.uniform(lo - d_lon, lo + d_lon, n_each))
    return np.concatenate(lat_out), np.concatenate(lon_out)


# --------------------------------------------------------------------------
# IGRF tracing
# --------------------------------------------------------------------------


def _import_geopack():
    _ensure_script_cwd()
    try:
        import PyGeopack as gp
    except Exception as exc:
        raise SystemExit(f"PyGeopack could not be imported ({exc}).") from exc
    return gp


def trace_igrf(lat_seed, lon_seed, cfg: Config):
    """Field-line geographic lat/lon at each altitude.

    Starts at ``cfg.footprint_alt``. PyGeopack's IGRF tracer only fills
    the first start point in a batch, so each field line is traced on
    its own.  MaxLen is kept short so the conjugate hemisphere is never
    reached.
    """
    gp = _import_geopack()
    lat_seed = np.atleast_1d(np.asarray(lat_seed, dtype=float))
    lon_seed = np.atleast_1d(np.asarray(lon_seed, dtype=float))
    n_lines = lat_seed.size
    date, ut = cfg.date_int, cfg.ut_hours
    alts = cfg.altitudes
    lat_out = np.full((n_lines, alts.size), np.nan)
    lon_out = np.full((n_lines, alts.size), np.nan)
    xyz = geo_to_ecef(lat_seed, lon_seed, cfg.footprint_alt) / R_EARTH_KM
    x_gsm, y_gsm, z_gsm = gp.Coords.GEOtoGSM(
        xyz[:, 0].copy(), xyz[:, 1].copy(), xyz[:, 2].copy(), date, ut
    )
    print(f"  tracing {n_lines} field lines (IGRF, {cfg.h_min:g}–{cfg.h_max:g} km)")
    for i in range(n_lines):
        tr = gp.TraceField(
            np.array([x_gsm[i]]),
            np.array([y_gsm[i]]),
            np.array([z_gsm[i]]),
            date,
            ut,
            Model="IGRF",
            CoordIn="GSM",
            alt=float(cfg.h_min),
            TraceDir="both",
            MaxLen=120,
            DSMax=0.001,
            alpha=[],
            Vx=400.0,
            Vy=0.0,
            Vz=0.0,
        )
        nstep = int(np.ravel(tr.nstep)[0])
        if nstep < 3:
            continue
        x, y, z = tr.xgsm[0, :nstep], tr.ygsm[0, :nstep], tr.zgsm[0, :nstep]
        r = np.sqrt(x * x + y * y + z * z)
        alt_km = (r - 1.0) * R_EARTH_KM
        xg, yg, zg = gp.Coords.GSMtoGEO(x.copy(), y.copy(), z.copy(), date, ut)
        lat = np.degrees(np.arcsin(np.clip(zg / r, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(yg, xg))
        dlon = np.abs((lon - lon_seed[i] + 180.0) % 360.0 - 180.0)
        keep = (
            (alt_km >= cfg.h_min - 10.0)
            & (alt_km <= cfg.h_max + 30.0)
            & (np.abs(lat - lat_seed[i]) < 1.5)
            & (dlon < 3.0)
            & np.isfinite(lat)
            & np.isfinite(lon)
        )
        if keep.sum() < 2:
            continue
        order = np.argsort(alt_km[keep])
        alt_s = alt_km[keep][order]
        lat_s = lat[keep][order]
        lon_s = lon[keep][order]
        unique = np.concatenate([[True], np.diff(alt_s) > 1e-3])
        alt_s, lat_s, lon_s = alt_s[unique], lat_s[unique], lon_s[unique]
        if alt_s.size < 2:
            continue
        lon_s = np.degrees(np.unwrap(np.radians(lon_s)))
        lat_out[i] = np.interp(alts, alt_s, lat_s, left=np.nan, right=np.nan)
        lon_out[i] = np.interp(alts, alt_s, lon_s, left=np.nan, right=np.nan)
        if (i + 1) % 500 == 0 or i + 1 == n_lines:
            print(f"    {i + 1}/{n_lines}")
    lon_out = (lon_out + 180.0) % 360.0 - 180.0
    return lat_out, lon_out


def brightness_profiles(station: Station, lat_tr, lon_tr, altitudes):
    lat_tr = np.atleast_2d(lat_tr)
    lon_tr = np.atleast_2d(lon_tr)
    n, m = lat_tr.shape
    alt = np.broadcast_to(np.asarray(altitudes, dtype=float), (n, m))
    az, el = azel_from_station(
        station.lat,
        station.lon,
        station.alt_km,
        lat_tr.ravel(),
        lon_tr.ravel(),
        alt.ravel(),
    )
    px, py = station.azel_to_pixel(az, el)
    dn = station.sample(px, py)
    return dn.reshape((n, m)), px.reshape((n, m)), py.reshape((n, m))


def run_iteration(
    lat_fl,
    lon_fl,
    lyr: Station,
    nya: Station,
    cfg: Config,
    corr_thr: float,
    centroid_thr: float,
):
    lat_tr, lon_tr = trace_igrf(lat_fl, lon_fl, cfg)
    prof_lyr, px_lyr, py_lyr = brightness_profiles(lyr, lat_tr, lon_tr, cfg.altitudes)
    prof_nya, px_nya, py_nya = brightness_profiles(nya, lat_tr, lon_tr, cfg.altitudes)
    in_crop = np.ones(lat_fl.size, dtype=bool)
    if cfg.lyr_box is not None:
        in_crop &= track_hits_box(px_lyr, py_lyr, cfg.lyr_box)
    rows = []
    for i in range(lat_fl.size):
        if not in_crop[i]:
            rows.append(_line_result(reason="outside pick box"))
        else:
            rows.append(
                evaluate_line(
                    cfg.altitudes, prof_lyr[i], prof_nya[i], corr_thr, centroid_thr
                )
            )
    accepted = np.array([r["accepted"] for r in rows])
    tracks = {
        "lyr": [(px_lyr[i], py_lyr[i]) for i in range(lat_fl.size)],
        "nya": [(px_nya[i], py_nya[i]) for i in range(lat_fl.size)],
    }
    return rows, accepted, prof_lyr, prof_nya, tracks


def mean_profile_peak(alt, prof_lyr, prof_nya, accepted):
    idx = np.flatnonzero(accepted)
    if idx.size == 0:
        return np.nan, None
    stack = []
    for i in idx:
        stack.append(normalised(prof_lyr[i]))
        stack.append(normalised(prof_nya[i]))
    mean = np.nanmean(np.vstack(stack), axis=0)
    peak, _ = find_peak(alt, mean)
    return peak, mean


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


def write_csv(path: Path, rows, lat_fl, lon_fl, iteration):
    cols = [
        "iteration",
        "footpoint_lat",
        "footpoint_lon",
        "peak_lyr_km",
        "peak_nya_km",
        "centroid_lyr_km",
        "centroid_nya_km",
        "corr",
        "accepted",
        "reason",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for it, la, lo, r in zip(iteration, lat_fl, lon_fl, rows):
            w.writerow(
                [
                    it,
                    f"{la:.4f}",
                    f"{lo:.4f}",
                    f"{r['peak_lyr']:.2f}",
                    f"{r['peak_nya']:.2f}",
                    f"{r['centroid_lyr']:.2f}",
                    f"{r['centroid_nya']:.2f}",
                    f"{r['corr']:.4f}",
                    int(r["accepted"]),
                    r["reason"],
                ]
            )


def plot_overlay(path, lyr, nya, cluster_mask, cluster_st, tracks, accepted, cfg):
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    idx = np.flatnonzero(accepted)
    show = idx[:: max(1, len(idx) // 80)] if idx.size else []
    for ax, st, key in ((axes[0], lyr, "lyr"), (axes[1], nya, "nya")):
        _show_image(ax, st)
        if st is cluster_st and cluster_mask is not None and cluster_mask.any():
            ys, xs = np.nonzero(cluster_mask)
            step = max(1, ys.size // 12000)
            ax.plot(
                xs[::step],
                ys[::step],
                ".",
                color="cyan",
                ms=1.2,
                alpha=0.45,
                rasterized=True,
            )
        if st is lyr and cfg.lyr_box is not None:
            xmin, ymin, xmax, ymax = cfg.lyr_box
            ax.add_patch(
                Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False,
                    edgecolor="lime",
                    lw=1.6,
                )
            )
        n = len(tracks[key])
        tested = range(0, n, max(1, n // 80)) if n else []
        if not idx.size:
            for i in tested:
                px, py = tracks[key][i]
                ax.plot(px, py, color="0.55", lw=0.45, alpha=0.4)
        for i in show:
            px, py = tracks[key][i]
            ax.plot(px, py, lw=1.0, alpha=0.85)
        title = f"{st.name}  ({idx.size} accepted field lines)"
        if st is cluster_st:
            title += "  [cyan = seed pixels]" if cfg.every_pixel else "  [cyan = cluster]"
        if st is lyr and cfg.lyr_box is not None:
            title += "  [lime = LYR ROI]"
        elif st is nya and cfg.lyr_box is not None:
            title += "  [same lines projected]"
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_profiles(path, alt, prof_lyr, prof_nya, accepted, mean_prof, peak_h):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    idx = np.flatnonzero(accepted)
    for ax, profs, title in (
        (axes[0], prof_lyr, "Longyearbyen"),
        (axes[1], prof_nya, "Ny-Alesund"),
    ):
        for i in idx:
            ax.plot(normalised(profs[i]), alt, lw=0.6, alpha=0.35, color="tab:blue")
        if mean_prof is not None:
            ax.plot(mean_prof, alt, lw=2.5, color="crimson", label="mean accepted")
        if np.isfinite(peak_h):
            ax.axhline(peak_h, ls="--", color="k", lw=1.2, label=f"peak {peak_h:.1f} km")
        ax.set_xlabel("Normalised brightness")
        ax.set_title(f"{title}  ({idx.size} field lines)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("Altitude [km]")
    axes[0].set_ylim(float(alt.min()), float(alt.max()))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_altitude_map(path, lat_all, lon_all, accepted, lyr, nya, footprint_alt):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(lon_all, lat_all, s=8, color="lightgrey", label="tested (iter 2)")
    idx = np.flatnonzero(accepted)
    if idx.size:
        ax.scatter(lon_all[idx], lat_all[idx], s=28, c="tab:red", label="accepted")
    ax.plot(lyr.lon, lyr.lat, "^", color="crimson", ms=10, label="LYR")
    ax.plot(nya.lon, nya.lat, "s", color="tab:orange", ms=8, label="NYA")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"Field-line footprints at {footprint_alt:g} km")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = Config()
    p.add_argument("--time", default=DEFAULT_EPOCH)
    p.add_argument("--data-dir", type=Path, default=d.data_dir)
    p.add_argument("--out", type=Path, default=d.out_dir, dest="out_dir")
    p.add_argument("--lyr-image", default=d.lyr_image)
    p.add_argument("--nya-image", default=d.nya_image)
    p.add_argument("--lyr-cal", default=d.lyr_cal)
    p.add_argument("--nya-cal", default=d.nya_cal)
    p.add_argument(
        "--cluster-station", choices=["LYR", "NYA"], default=d.cluster_station
    )
    p.add_argument(
        "--pick",
        action=argparse.BooleanOptionalAction,
        default=d.pick,
        help="draw a rectangle on LYR (default); --no-pick clusters the whole image",
    )
    p.add_argument(
        "--every-pixel",
        action="store_true",
        default=d.every_pixel,
        help="skip clustering: trace one IGRF field line per pixel in the LYR pick box",
    )
    p.add_argument(
        "--pixel-step",
        type=int,
        default=d.pixel_step,
        help="with --every-pixel, use every Nth pixel (1 = all pixels)",
    )
    p.add_argument("--h-min", type=float, default=d.h_min, help="lowest field-line sample [km]")
    p.add_argument("--h-max", type=float, default=d.h_max, help="highest field-line sample [km]")
    p.add_argument("--h-step", type=float, default=d.h_step, help="altitude sampling step [km]")
    p.add_argument(
        "--footprint-alt",
        type=float,
        default=d.footprint_alt,
        help="map the LYR cluster to lat/lon at this height [km], then trace IGRF",
    )
    p.add_argument("--el-min", type=float, default=d.el_min, help="ignore pixels below this elevation [deg]")
    p.add_argument("--bright-pct", type=float, default=d.bright_pct, help="cluster the brightest this %% of FOV pixels")
    p.add_argument(
        "--star-filter",
        type=int,
        default=d.star_filter,
        help="median-filter size [px] before clustering; 0 disables",
    )
    p.add_argument("--seed", type=int, default=d.seed, help="RNG seed for random footprints")
    a = p.parse_args(argv)
    stamp = a.time.replace("Z", "+00:00")
    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        raise SystemExit(f"--time '{a.time}' is not ISO 8601")
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    cfg = Config(
        data_dir=a.data_dir,
        lyr_image=a.lyr_image,
        nya_image=a.nya_image,
        lyr_cal=a.lyr_cal,
        nya_cal=a.nya_cal,
        out_dir=a.out_dir,
        time_utc=when.astimezone(timezone.utc),
        cluster_station=a.cluster_station,
        pick=a.pick,
        every_pixel=a.every_pixel,
        pixel_step=a.pixel_step,
        h_min=a.h_min,
        h_max=a.h_max,
        h_step=a.h_step,
        footprint_alt=a.footprint_alt,
        el_min=a.el_min,
        bright_pct=a.bright_pct,
        star_filter=a.star_filter,
        seed=a.seed,
    )
    if cfg.h_max <= cfg.h_min:
        raise SystemExit("--h-max must exceed --h-min")
    if cfg.footprint_alt <= 0:
        raise SystemExit("--footprint-alt must be positive")
    if cfg.pixel_step < 1:
        raise SystemExit("--pixel-step must be >= 1")
    if cfg.every_pixel and not cfg.pick:
        raise SystemExit("--every-pixel needs a selected region; omit --no-pick")
    cfg.finalise()
    return cfg


def _print_reasons(rows):
    reasons: dict[str, int] = {}
    for r in rows:
        if r["accepted"]:
            continue
        key = r["reason"]
        if key.startswith("correlation"):
            key = "correlation too low"
        elif key.startswith("centroids"):
            key = "centroids disagree"
        reasons[key] = reasons.get(key, 0) + 1
    for why, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {count:5d}  {why}")


def _select_cluster_station(cfg: Config, lyr: Station, nya: Station) -> Station:
    if not cfg.pick:
        return lyr if cfg.cluster_station == "LYR" else nya
    if cfg.cluster_station != "LYR":
        print("  (--pick uses LYR; ignoring --cluster-station NYA)")
        cfg.cluster_station = "LYR"
    print("\nSelect the aurora on LYR only (drag, then close the window)")
    if cfg.every_pixel:
        print("  Every pixel in that box becomes an IGRF seed (no clustering)")
    else:
        print("  Field lines through that region will be projected onto NYA")
    try:
        cfg.lyr_box = pick_lyr_roi(lyr)
    except SystemExit:
        raise
    except Exception as extra:
        raise SystemExit(
            f"--pick could not open an interactive window ({extra}).\n"
            f"Run from Terminal.app:\n  {_pick_command()}"
        ) from extra
    xmin, ymin, xmax, ymax = cfg.lyr_box
    print(f"  LYR ROI: x {xmin:.0f}–{xmax:.0f}, y {ymin:.0f}–{ymax:.0f}")
    return lyr


def _iter1_fallback(rows1, acc1, lat1, lon1, prof_lyr1, prof_nya1, tracks1):
    idx = np.flatnonzero(acc1)
    return (
        [r for r, ok in zip(rows1, acc1) if ok],
        np.ones(idx.size, dtype=bool),
        lat1[acc1],
        lon1[acc1],
        prof_lyr1[acc1],
        prof_nya1[acc1],
        {
            "lyr": [tracks1["lyr"][i] for i in idx],
            "nya": [tracks1["nya"][i] for i in idx],
        },
    )


def _write_outputs(
    cfg,
    lyr,
    nya,
    cluster_mask,
    cluster_st,
    rows,
    lat,
    lon,
    accepted,
    tracks,
    prof_lyr,
    prof_nya,
    mean_prof,
    peak_h,
    iteration: int,
):
    write_csv(
        cfg.out_dir / "fieldlines.csv",
        rows,
        lat,
        lon,
        np.full(lat.size, iteration, dtype=int),
    )
    plot_profiles(
        cfg.out_dir / "profiles.png",
        cfg.altitudes,
        prof_lyr,
        prof_nya,
        accepted,
        mean_prof,
        peak_h,
    )
    plot_overlay(
        cfg.out_dir / "overlay.png",
        lyr,
        nya,
        cluster_mask,
        cluster_st,
        tracks,
        accepted,
        cfg,
    )
    plot_altitude_map(
        cfg.out_dir / "altitude_map.png",
        lat,
        lon,
        accepted,
        lyr,
        nya,
        cfg.footprint_alt,
    )
    for name in ("fieldlines.csv", "profiles.png", "overlay.png", "altitude_map.png"):
        print(f"  {cfg.out_dir / name}")


def main(argv=None) -> int:
    cfg = parse_args(argv)
    rng = np.random.default_rng(cfg.seed)
    print("Whiter et al. (2013) Method 2")
    print(f"  epoch     : {cfg.time_utc.isoformat()}")
    print(
        f"  window    : {cfg.h_min:g}–{cfg.h_max:g} km in {cfg.h_step:g} km steps "
        f"({cfg.altitudes.size} levels), IGRF"
    )
    if cfg.every_pixel:
        step_txt = "" if cfg.pixel_step == 1 else f", every {cfg.pixel_step}th pixel"
        print(f"  seeds     : every pixel in the LYR pick box{step_txt} (no clustering)")
    else:
        print(f"  cluster   : brightest {cfg.bright_pct:g}% of {cfg.cluster_station}")
    print(f"  seed alt  : {cfg.footprint_alt:g} km (LYR pixels → lat/lon, then IGRF)")

    print("\nLoading images and WISC calibration")
    lyr = Station("LYR", cfg.data_dir / cfg.lyr_image, cfg.data_dir / cfg.lyr_cal, cfg)
    nya = Station("NYA", cfg.data_dir / cfg.nya_image, cfg.data_dir / cfg.nya_cal, cfg)
    for st in (lyr, nya):
        print(
            f"  {st.name}: {st.width}x{st.height} px at "
            f"{st.lat:.3f} N, {st.lon:.3f} E, {st.alt_km * 1000:.0f} m"
        )
    cluster_st = _select_cluster_station(cfg, lyr, nya)

    if cfg.every_pixel:
        print("\nMapping every pixel in the LYR ROI (no clustering)")
        cluster_mask, use_lat, use_lon = region_pixels_to_footprints(
            cluster_st,
            cfg.lyr_box,
            footprint_alt=cfg.footprint_alt,
            step=cfg.pixel_step,
        )
        n_seed = use_lat.size
        if n_seed > 2000:
            print(
                f"  warning: {n_seed} IGRF traces; use --pixel-step N if this is too slow"
            )
        print(
            f"\nEvaluating {n_seed} field lines, corr > {CORR_1:g}, "
            f"centroid < {CENTROID_1_KM:g} km"
        )
        use_rows, use_acc, use_prof_lyr, use_prof_nya, use_tracks = run_iteration(
            use_lat, use_lon, lyr, nya, cfg, CORR_1, CENTROID_1_KM
        )
        print(f"  passed {use_acc.sum()}/{n_seed}")
        _print_reasons(use_rows)
        label = "every pixel"
        iteration = 0
        if not use_acc.any():
            print("  No field line passed the tests; writing overlay and CSV anyway.")
            _write_outputs(
                cfg,
                lyr,
                nya,
                cluster_mask,
                cluster_st,
                use_rows,
                use_lat,
                use_lon,
                use_acc,
                use_tracks,
                use_prof_lyr,
                use_prof_nya,
                None,
                np.nan,
                iteration,
            )
            return 1
    else:
        print("\nClustering the brightest structure")
        cluster_box = cfg.lyr_box if cluster_st is lyr else None
        if cluster_box is None:
            print("  (--no-pick: largest 5% cluster of the whole image, often not the arc)")
        cluster_mask, clat, clon = brightest_cluster(
            cluster_st,
            cfg.bright_pct,
            box=cluster_box,
            star_filter=cfg.star_filter,
            footprint_alt=cfg.footprint_alt,
        )
        path, area, bbox = region_polygon(clat, clon)
        print(
            f"  {cfg.footprint_alt:g} km region: lat {bbox[0]:.2f}–{bbox[1]:.2f}, "
            f"lon {bbox[2]:.2f}–{bbox[3]:.2f}  ({area:.3f} deg²)"
        )

        print(
            f"\nIteration 1: random footprints, corr > {CORR_1:g}, "
            f"centroid < {CENTROID_1_KM:g} km"
        )
        lat1, lon1 = random_footprints(path, bbox, D_LAT, D_LON, rng, area)
        print(f"  {lat1.size} footprints (mean spacing {D_LAT:g}° / {D_LON:g}°)")
        rows1, acc1, prof_lyr1, prof_nya1, tracks1 = run_iteration(
            lat1, lon1, lyr, nya, cfg, CORR_1, CENTROID_1_KM
        )
        print(f"  passed {acc1.sum()}/{lat1.size}")
        _print_reasons(rows1)
        if not acc1.any():
            print("  No field line passed iteration 1.")
            print("\nWriting overlay so you can inspect the selected cluster")
            plot_overlay(
                cfg.out_dir / "overlay.png",
                lyr,
                nya,
                cluster_mask,
                cluster_st,
                tracks1,
                acc1,
                cfg,
            )
            print(f"  {cfg.out_dir / 'overlay.png'}")
            return 1

        print(
            f"\nIteration 2: spacing / 3 around passers, corr > {CORR_2:g}, "
            f"centroid < {CENTROID_2_KM:g} km"
        )
        lat2, lon2 = random_around_passers(lat1[acc1], lon1[acc1], D_LAT, D_LON, rng)
        print(f"  {lat2.size} footprints around {int(acc1.sum())} passers")
        rows2, acc2, prof_lyr, prof_nya, tracks = run_iteration(
            lat2, lon2, lyr, nya, cfg, CORR_2, CENTROID_2_KM
        )
        print(f"  passed {acc2.sum()}/{lat2.size}")
        _print_reasons(rows2)

        use_rows, use_acc, use_lat, use_lon = rows2, acc2, lat2, lon2
        use_prof_lyr, use_prof_nya, use_tracks = prof_lyr, prof_nya, tracks
        label = "iteration 2"
        iteration = 2
        if not acc2.any():
            print("  Iteration 2 accepted none; reporting the iteration-1 mean profile.")
            (
                use_rows,
                use_acc,
                use_lat,
                use_lon,
                use_prof_lyr,
                use_prof_nya,
                use_tracks,
            ) = _iter1_fallback(
                rows1, acc1, lat1, lon1, prof_lyr1, prof_nya1, tracks1
            )
            label = "iteration 1 (iteration 2 empty)"
            iteration = 1

    print("\nResult")
    peak_h, mean_prof = mean_profile_peak(
        cfg.altitudes, use_prof_lyr, use_prof_nya, use_acc
    )
    if not np.isfinite(peak_h):
        print("  No field line passed the tests.")
        return 1
    print(
        f"  peak emission height ({label}): {peak_h:.1f} km"
        f"  n={int(use_acc.sum())}"
    )

    print("\nWriting outputs")
    _write_outputs(
        cfg,
        lyr,
        nya,
        cluster_mask,
        cluster_st,
        use_rows,
        use_lat,
        use_lon,
        use_acc,
        use_tracks,
        use_prof_lyr,
        use_prof_nya,
        mean_prof,
        peak_h,
        iteration,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

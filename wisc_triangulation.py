#!/usr/bin/env python3
"""WISC StarCal field-line triangulation (geometry from guillaume_triangulation_WISC.ipynb).

Functions match the notebook. Stations, event time, and output directory are
passed in so a batch driver can loop over image pairs without changing the
azimuth/elevation mapping, IGRF traces, brightness sampling, or selection tests.
"""

from __future__ import annotations

import os
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
import PyGeopack as gp  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.interpolate import LinearNDInterpolator  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402
from scipy.signal import medfilt  # noqa: E402

R_EARTH_KM = 6371.2

DEFAULT_TRACE = dict(
    alt_fl=165,
    step=0.1,
    upper=85,
    lower=85,
    interval=2,
    peak=10,
    corr_thr=0.7,
    edge=10,
    frac=0.9,
    width_limit=999,
    centroid_thr=20,
)


def event_times(event_dt: datetime) -> tuple[int, float, str, str]:
    """Date integer, decimal UT hours, compact stamp, and ISO 8601 basic UTC."""
    dates = int(event_dt.strftime("%Y%m%d"))
    ut_time = event_dt.hour + event_dt.minute / 60 + event_dt.second / 3600
    date_time = event_dt.strftime("%Y%m%d%H%M%S")
    iso_basic = event_dt.strftime("%Y%m%dT%H%M%SZ")
    return dates, ut_time, date_time, iso_basic


def azel_to_uv(az_deg, el_deg):
    """Map azimuth/elevation to a wrap-safe zenith-angle plane."""
    z = 90.0 - np.asarray(el_deg, dtype=float)
    a = np.radians(np.asarray(az_deg, dtype=float))
    return z * np.sin(a), z * np.cos(a)


def _load_station_images(image_path, height, width):
    """Return greyscale, original RGB, and RGBproduct arrays.

    greyscale is the channel mean. rgb_product is R*G*B / 255**2 (float64, 0–255 scale).
    original is the uint8 RGB frame used for overlays.
    """
    raw = np.asarray(Image.open(image_path))
    if raw.ndim != 3:
        raise ValueError(
            f"{Path(image_path).name} is not RGB; RGBproduct needs three channels"
        )
    if raw.shape[0] != height or raw.shape[1] != width:
        raise ValueError(
            f"{Path(image_path).name} is {raw.shape[1]}x{raw.shape[0]} "
            f"but calibration describes {width}x{height}"
        )
    original = raw[:, :, :3]
    channels = original.astype(np.float64)
    greyscale = channels.mean(axis=2)
    rgb_product = np.prod(channels, axis=2) / (255.0 ** 2)
    return greyscale, original, rgb_product


def sampling_image(station, sample):
    if sample == "greyscale":
        return station.greyscale
    if sample == "RGBproduct":
        return station.rgb_product
    raise ValueError(f"Unknown brightness sample {sample!r}")


class Station:
    """All-sky camera with WISC StarCal az/el grids in raw pixel coordinates."""

    def __init__(self, name, image_path, cal_path, el_min=15.0, subsample=4):
        self.name = name
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
                raise NotImplementedError(f"{Path(cal_path).name} sets {key}=1")

        self.greyscale, self.original, self.rgb_product = _load_station_images(
            image_path, self.height, self.width
        )

        expected = (self.height + 1, self.width + 1)
        if az_corner.shape != expected:
            raise ValueError(
                f"{Path(cal_path).name}: azimuth grid {az_corner.shape}, expected {expected}"
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
        self.el_min = el_min
        self._build_inverse(subsample)

    def set_image(self, image_path):
        """Replace the brightness images without rebuilding the StarCal interpolator."""
        self.greyscale, self.original, self.rgb_product = _load_station_images(
            image_path, self.height, self.width
        )

    def _build_inverse(self, step):
        py, px = np.mgrid[0 : self.height : step, 0 : self.width : step]
        u, v, el = self.u[::step, ::step], self.v[::step, ::step], self.el[::step, ::step]
        inside = el > self.el_min
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
        ok = np.isfinite(u) & np.isfinite(v) & (el > self.el_min)
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


def geo_to_ecef(lat_deg, lon_deg, alt_km):
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    r = R_EARTH_KM + np.asarray(alt_km, dtype=float)
    return np.stack(
        [r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)],
        axis=-1,
    )


def azel_from_station(station, lat, lon, alt_km):
    origin = geo_to_ecef(station.lat, station.lon, station.alt_km)
    target = geo_to_ecef(lat, lon, alt_km)
    d = target - origin
    slat, slon = np.radians(station.lat), np.radians(station.lon)
    east = np.array([-np.sin(slon), np.cos(slon), 0.0])
    north = np.array(
        [-np.sin(slat) * np.cos(slon), -np.sin(slat) * np.sin(slon), np.cos(slat)]
    )
    up = np.array(
        [np.cos(slat) * np.cos(slon), np.cos(slat) * np.sin(slon), np.sin(slat)]
    )
    e, n, u = d @ east, d @ north, d @ up
    rng = np.sqrt(e**2 + n**2 + u**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        elev = np.degrees(np.arcsin(np.where(rng > 0, u / rng, np.nan)))
    azim = np.degrees(np.arctan2(e, n)) % 360.0
    return azim, elev


def field_line_tracing(lat_fl, lon_fl, alt_fl, upper, lower, interval, dates, ut_time):
    """
    A version that uses `alt_fl` as the central altitude,
    where the actual starting point of the trace is at `(alt_fl + upper)`,
    and follows the magnetic field lines only downwards from that point.
    """

    lat_fieldline = []
    lon_fieldline = []
    alt_fieldline = []

    # Altitude range to be tracked (top to bottom)
    alt_top = alt_fl + upper
    alt_bottom = alt_fl - lower
    alt_range = np.arange(alt_top, alt_bottom - 1e-6, -interval)

    # The starting point is alt_top ( = alt_fl + upper)
    r_geo = (R_EARTH_KM + alt_top) / R_EARTH_KM  # given in Re (Earth radii)

    lat_rad = np.deg2rad(lat_fl)
    lon_rad = np.deg2rad(lon_fl)

    # (x, y, z) in GEO at alt_top
    x_geo = r_geo * np.cos(lat_rad) * np.cos(lon_rad)
    y_geo = r_geo * np.cos(lat_rad) * np.sin(lon_rad)
    z_geo = r_geo * np.sin(lat_rad)

    # ECEF/GEO (Earth-centered, Earth-fixed/geocentric) → Geocentric Solar Magnetospheric (GSM)
    # The starting point remains the same throughout!
    x_gsm, y_gsm, z_gsm = gp.Coords.GEOtoGSM(x_geo, y_geo, z_geo, dates, ut_time)

    # From alt_top to alt_bottom, trace only downwards
    for alt in alt_range:
        trace = gp.TraceField(
            x_gsm, y_gsm, z_gsm, dates, ut_time, coord_In="GSM", alt=alt
        )

        lat_fieldline.append(trace.GlatN)
        lon_fieldline.append(trace.GlonN)
        alt_fieldline.append(alt)

    # If we reorder the altitudes here in ascending order (from ‘low’ to ‘high’),
    # the orientation will match that of the existing profile rendering
    lat_fieldline = lat_fieldline[::-1]
    lon_fieldline = lon_fieldline[::-1]
    alt_fieldline = alt_fieldline[::-1]

    return lat_fieldline, lon_fieldline, alt_fieldline


def inverse_mapping(lat_fieldline, lon_fieldline, alt_fieldline, station):
    """Pixel coordinates in the raw StarCal image from latitude, longitude, and altitude."""
    lat = np.array([np.ravel(v)[0] for v in lat_fieldline], dtype=float)
    lon = np.array([np.ravel(v)[0] for v in lon_fieldline], dtype=float)
    alt = np.array([np.ravel(v)[0] for v in alt_fieldline], dtype=float)
    az, el = azel_from_station(station, lat, lon, alt)
    px, py = station.azel_to_pixel(az, el)
    return px, py


def brightness_vs_altitude(px_lyr, py_lyr, px_nya, py_nya, alt_fieldline, lyr, nya, sample="greyscale"):
    def sample_pixels(image, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        out = np.full(x.shape, np.nan)
        if good.any():
            out[good] = map_coordinates(
                image, [y[good], x[good]], order=1, mode="nearest"
            )
        return out

    brightness_LYR = sample_pixels(sampling_image(lyr, sample), px_lyr, py_lyr)
    brightness_NYA = sample_pixels(sampling_image(nya, sample), px_nya, py_nya)
    return brightness_LYR, brightness_NYA, np.asarray(alt_fieldline)


def fit_and_find_peak_med(altitudes, brightness):
    valid_idx = ~np.isnan(brightness)
    altitudes, brightness = altitudes[valid_idx], brightness[valid_idx]

    if len(altitudes) == 0:
        return None, None, None

    brightness_smooth = medfilt(brightness, kernel_size=3)

    peak_idx = np.argmax(brightness_smooth)
    peak_altitude = altitudes[peak_idx]
    peak_brightness = brightness_smooth[peak_idx]

    return altitudes, brightness_smooth, (peak_altitude, peak_brightness)


def fit_and_find_peak(altitudes, brightness):
    valid_idx = ~np.isnan(brightness)  # Remove NaN values
    altitudes, brightness = altitudes[valid_idx], brightness[valid_idx]

    if len(altitudes) < 4:  # Not enough data for cubic fitting
        return None, None, None

    poly_coeffs = np.polyfit(altitudes, brightness, 3)  # Cubic fit
    poly_func = np.poly1d(poly_coeffs)  # Create polynomial function

    # Generate smooth curve for plotting
    alt_smooth = np.linspace(min(altitudes), max(altitudes), 100)
    brightness_smooth = poly_func(alt_smooth)

    # Find peak (maximum value of the fitted curve)
    peak_idx = np.argmax(brightness_smooth)
    peak_altitude = alt_smooth[peak_idx]
    peak_brightness = brightness_smooth[peak_idx]

    return poly_func, peak_altitude, peak_brightness


def calculation_for_conditions(brightness_LYR, brightness_NYA, alt_fieldline):
    """
    Parameters:
        brightness_LYR (list): list of brightness values for LYR.
        brightness_NYA (list): list of brightness values for NYA.
        alt_fieldline (list): list of altitudes for each point.

    Returns:
        dict: Dictionary containing correlation, peak altitude, peak brightness, centre, percentile ratio,
              mean brightness above peak altitude, mean brightness below peak altitude for both graphs.
    """
    # Remove NaN values for correlation computation
    valid_indices = ~np.isnan(brightness_LYR) & ~np.isnan(brightness_NYA)
    correlation = np.corrcoef(brightness_LYR[valid_indices], brightness_NYA[valid_indices])[0, 1]

    poly_func_LYR, peak_alt_LYR, peak_brightness_LYR = fit_and_find_peak(alt_fieldline, brightness_LYR)
    poly_func_NYA, peak_alt_NYA, peak_brightness_NYA = fit_and_find_peak(alt_fieldline, brightness_NYA)

    # Compute Centre (Σ(brightness_values * altitude) / Σbrightness_values)
    def compute_centre(altitudes, brightness_values):
        valid_idx = ~np.isnan(brightness_values)
        altitudes, brightness_values = altitudes[valid_idx], brightness_values[valid_idx]
        return np.sum(brightness_values * altitudes) / np.sum(brightness_values) if np.sum(brightness_values) != 0 else np.nan

    centre_LYR = compute_centre(alt_fieldline, brightness_LYR)
    centre_NYA = compute_centre(alt_fieldline, brightness_NYA)

    # Compute P (60% percentile / median)
    def compute_p(brightness_values):
        valid_values = brightness_values[~np.isnan(brightness_values)]
        if len(valid_values) == 0:
            return np.nan
        percentile_60 = np.percentile(valid_values, 60)
        median_brightness = np.median(valid_values)
        return percentile_60 / median_brightness if median_brightness != 0 else np.nan

    percentile_LYR = compute_p(brightness_LYR)
    percentile_NYA = compute_p(brightness_NYA)

    # Compute mean brightness above and below peak altitude
    def mean_above_below_peak(altitudes, brightness_values, peak_altitude):
        valid_idx = ~np.isnan(brightness_values)
        altitudes, brightness_values = altitudes[valid_idx], brightness_values[valid_idx]

        above = brightness_values[altitudes > peak_altitude]
        below = brightness_values[altitudes < peak_altitude]

        mean_above = np.mean(above) if len(above) > 0 else np.nan
        mean_below = np.mean(below) if len(below) > 0 else np.nan

        return mean_above, mean_below

    mean_above_LYR, mean_below_LYR = mean_above_below_peak(alt_fieldline, brightness_LYR, peak_alt_LYR)
    mean_above_NYA, mean_below_NYA = mean_above_below_peak(alt_fieldline, brightness_NYA, peak_alt_NYA)

    return correlation, peak_alt_LYR, peak_brightness_LYR, centre_LYR, percentile_LYR, mean_above_LYR, mean_below_LYR, peak_alt_NYA, peak_brightness_NYA, centre_NYA, percentile_NYA, mean_above_NYA, mean_below_NYA


def compute_peak_width(altitudes, brightness_values, frac):
    """
    altitudes: altitude array (ascending, km)
    brightness_values: intensity along the field line (same length, NaNs allowed)
    frac: frac=0.9 means the width is measured where brightness is still ≥ 90% of the peak

    Returns:
        width_km : width [km] (high − low)
        low_alt  : altitude where intensity drops below the threshold (below the peak)
        high_alt : altitude where intensity drops below the threshold (above the peak)
    """
    altitudes = np.asarray(altitudes)
    brightness_values = np.asarray(brightness_values)

    valid = np.isfinite(brightness_values)
    if valid.sum() < 3:
        return np.inf, np.nan, np.nan

    alts = altitudes[valid]
    vals = brightness_values[valid]

    # Smooth it out a little (same as in `fit_and_find_peak_med`)
    smooth = medfilt(vals, kernel_size=3)

    # Find the peak
    peak_idx = np.argmax(smooth)
    peak_alt = alts[peak_idx]
    peak_val = smooth[peak_idx]
    if peak_val <= 0:
        return np.inf, np.nan, np.nan

    threshold = peak_val * frac

    # Trace downwards
    i_low = peak_idx
    while i_low > 0 and smooth[i_low] >= threshold:
        i_low -= 1
    low_alt = alts[i_low]

    # Trace upwards
    i_high = peak_idx
    n = len(alts) - 1
    while i_high < n and smooth[i_high] >= threshold:
        i_high += 1
    high_alt = alts[i_high]

    width_km = high_alt - low_alt
    return width_km, low_alt, high_alt


def build_fieldline_grid(
    lat_range,
    lon_range,
    alt_fl,
    step,
    upper,
    lower,
    interval,
    lyr,
    nya,
    dates,
    ut_time,
):
    """Trace every seed in the box and map it to LYR/NYA pixels (no brightness yet)."""
    lat_fl_values = np.arange(*lat_range, step)
    lon_fl_values = np.arange(*lon_range, step)
    paths = []
    n_seed = len(lat_fl_values) * len(lon_fl_values)
    print(f"Tracing {n_seed} field lines for box {lat_range}, {lon_range}")
    for lat_fl in lat_fl_values:
        for lon_fl in lon_fl_values:
            lat_td, lon_td, alt_td = field_line_tracing(
                lat_fl, lon_fl, alt_fl, upper, lower, interval, dates, ut_time
            )
            px_lyr, py_lyr = inverse_mapping(lat_td, lon_td, alt_td, lyr)
            px_nya, py_nya = inverse_mapping(lat_td, lon_td, alt_td, nya)
            paths.append(
                {
                    "lat_fl": lat_fl,
                    "lon_fl": lon_fl,
                    "alt": np.asarray(alt_td),
                    "px_lyr": np.asarray(px_lyr).ravel(),
                    "py_lyr": np.asarray(py_lyr).ravel(),
                    "px_nya": np.asarray(px_nya).ravel(),
                    "py_nya": np.asarray(py_nya).ravel(),
                }
            )
    return paths


def filter_cached_field_lines(
    paths,
    lyr,
    nya,
    alt_fl,
    upper,
    lower,
    peak,
    corr_thr,
    edge,
    frac,
    width_limit,
    centroid_thr,
    lat_range=None,
    lon_range=None,
    sample="greyscale",
):
    """Sample brightness on the current images and keep lines that pass cond."""
    all_brightness_values_LYR = []
    all_brightness_values_NYA = []
    lines_LYR = []
    lines_NYA = []

    alt_fieldline_last = None
    sample_size = 0

    for path in paths:
        px_lyr, py_lyr = path["px_lyr"], path["py_lyr"]
        px_nya, py_nya = path["px_nya"], path["py_nya"]
        alt_td = path["alt"]

        brightness_LYR_td, brightness_NYA_td, alt_td = brightness_vs_altitude(
            px_lyr, py_lyr, px_nya, py_nya, alt_td, lyr, nya, sample=sample
        )

        corr_td, palt1_td, pray1_td, cen1_td, p1_td, ma1_td, mb1_td, \
            palt2_td, pray2_td, cen2_td, p2_td, ma2_td, mb2_td = \
            calculation_for_conditions(brightness_LYR_td, brightness_NYA_td, alt_td)

        width90_augo1_td, low90_augo1_td, high90_augo1_td = \
            compute_peak_width(alt_td, brightness_LYR_td, frac=frac)
        width90_augso_td, low90_augso_td, high90_augso_td = \
            compute_peak_width(alt_td, brightness_NYA_td, frac=frac)

        cond = (
            abs(palt1_td - palt2_td) < peak and
            alt_fl - lower + edge < palt1_td < alt_fl + upper - edge and
            alt_fl - lower + edge < palt2_td < alt_fl + upper - edge and
            abs(cen1_td - cen2_td) < centroid_thr and
            corr_td > corr_thr and
            (pray1_td > ma1_td * p1_td or pray1_td > mb1_td * p1_td) and
            (pray2_td > ma2_td * p2_td or pray2_td > mb2_td * p2_td) and
            width90_augo1_td <= width_limit and
            width90_augso_td <= width_limit
        )

        if cond:
            all_brightness_values_LYR.append(np.array(brightness_LYR_td))
            all_brightness_values_NYA.append(np.array(brightness_NYA_td))
            lines_LYR.append((np.asarray(px_lyr).ravel(), np.asarray(py_lyr).ravel()))
            lines_NYA.append((np.asarray(px_nya).ravel(), np.asarray(py_nya).ravel()))
            alt_fieldline_last = np.array(alt_td)
            sample_size += 1

    if sample_size == 0 or alt_fieldline_last is None:
        box = f"[{lat_range}, {lon_range}] " if lat_range is not None else ""
        print(f"{box}{sample}: no valid field lines.")
        return [], [], np.array([]), [], []

    return all_brightness_values_LYR, all_brightness_values_NYA, alt_fieldline_last, lines_LYR, lines_NYA


def trace_and_filter_field_lines(
    lat_range,
    lon_range,
    alt_fl,
    step,
    upper,
    lower,
    interval,
    peak,
    corr_thr,
    edge,
    frac,
    width_limit,
    centroid_thr,
    lyr,
    nya,
    dates,
    ut_time,
    sample="greyscale",
):
    """
    Traces field lines for a given (lat_range, lon_range) and
    returns information on the field lines that satisfy the conditions.
    """
    paths = build_fieldline_grid(
        lat_range,
        lon_range,
        alt_fl,
        step,
        upper,
        lower,
        interval,
        lyr,
        nya,
        dates,
        ut_time,
    )
    return filter_cached_field_lines(
        paths,
        lyr,
        nya,
        alt_fl,
        upper,
        lower,
        peak,
        corr_thr,
        edge,
        frac,
        width_limit,
        centroid_thr,
        lat_range=lat_range,
        lon_range=lon_range,
        sample=sample,
    )


def _peak_altitudes(profiles, alt):
    peaks = []
    for p in profiles:
        p = np.asarray(p, dtype=float)
        if not np.any(np.isfinite(p)):
            continue
        peaks.append(float(alt[int(np.nanargmax(p))]))
    return np.asarray(peaks)


def _peak_stats(peaks):
    mu = np.nanmean(peaks)
    sig = np.nanstd(peaks, ddof=1)
    sem = sig / np.sqrt(len(peaks))
    med = np.nanmedian(peaks)
    return mu, sig, sem, med


def summarise_peaks(all_brightness_values_LYR, all_brightness_values_NYA, alt):
    peaks_lyr = _peak_altitudes(all_brightness_values_LYR, alt)
    peaks_nya = _peak_altitudes(all_brightness_values_NYA, alt)
    peaks_all = np.concatenate([peaks_lyr, peaks_nya]) if len(peaks_lyr) or len(peaks_nya) else np.array([])
    mu1, sig1, sem1, med1 = _peak_stats(peaks_lyr) if len(peaks_lyr) else (np.nan, np.nan, np.nan, np.nan)
    mu2, sig2, sem2, med2 = _peak_stats(peaks_nya) if len(peaks_nya) else (np.nan, np.nan, np.nan, np.nan)
    mu, sig, sem, med = _peak_stats(peaks_all) if len(peaks_all) else (np.nan, np.nan, np.nan, np.nan)
    return {
        "n_lines": len(all_brightness_values_LYR),
        "lyr_mean_km": mu1,
        "lyr_std_km": sig1,
        "lyr_sem_km": sem1,
        "lyr_median_km": med1,
        "nya_mean_km": mu2,
        "nya_std_km": sig2,
        "nya_sem_km": sem2,
        "nya_median_km": med2,
        "both_mean_km": mu,
        "both_std_km": sig,
        "both_sem_km": sem,
        "both_median_km": med,
    }


def _imshow_station(ax, station, sample=None):
    img = station.original if sample is None else sampling_image(station, sample)
    if np.asarray(img).ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)


def _region_box_pixels(lat_range, lon_range, region, lyr, nya, h=150.0):
    if region is None:
        region = {
            "lat1": lat_range[0],
            "lat2": lat_range[1],
            "lon1": lon_range[0],
            "lon2": lon_range[1],
        }
    lats = np.linspace(*lat_range, 40)
    lons = np.linspace(*lon_range, 40)
    lat_b = np.concatenate([lats, np.full_like(lons, lats[-1]), lats[::-1], np.full_like(lons, lats[0])])
    lon_b = np.concatenate([np.full_like(lats, lons[0]), lons, np.full_like(lats, lons[-1]), lons[::-1]])
    alt_b = np.full_like(lat_b, h)
    x_lyr, y_lyr = inverse_mapping(lat_b, lon_b, alt_b, lyr)
    x_nya, y_nya = inverse_mapping(lat_b, lon_b, alt_b, nya)
    corners = [
        (region["lat1"], region["lon1"], "w"),
        (region["lat1"], region["lon2"], "w"),
        (region["lat2"], region["lon2"], "w"),
        (region["lat2"], region["lon1"], "w"),
    ]
    c_lat = np.array([c[0] for c in corners])
    c_lon = np.array([c[1] for c in corners])
    c_alt = np.full(4, h)
    cx1, cy1 = inverse_mapping(c_lat, c_lon, c_alt, lyr)
    cx2, cy2 = inverse_mapping(c_lat, c_lon, c_alt, nya)
    return region, (x_lyr, y_lyr), (x_nya, y_nya), corners, (cx1, cy1), (cx2, cy2)


def _title_time(event_dt):
    if event_dt is None:
        return ""
    return event_dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _draw_latlon_box(ax_lyr, ax_nya, lat_range, lon_range, lyr, nya, region=None, event_dt=None, sample=None):
    label_str = _title_time(event_dt)
    region, (x_lyr, y_lyr), (x_nya, y_nya), corners, (cx1, cy1), (cx2, cy2) = _region_box_pixels(
        lat_range, lon_range, region, lyr, nya
    )
    _imshow_station(ax_lyr, lyr, sample=sample)
    _imshow_station(ax_nya, nya, sample=sample)
    ax_lyr.plot(x_lyr, y_lyr, "w", lw=1.0)
    ax_nya.plot(x_nya, y_nya, "w", lw=1.0)
    for ax, xs, ys in ((ax_lyr, cx1, cy1), (ax_nya, cx2, cy2)):
        for (lat, lon, colour), x, y in zip(corners, xs, ys):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ax.scatter(x, y, c=colour, s=50, zorder=5)
            ax.annotate(
                f"{lat:.1f}N, {lon:.1f}E",
                (x, y),
                textcoords="offset points",
                xytext=(6, 6),
                color=colour,
                fontsize=8,
            )
    ax_lyr.set_title(f"LYR: {label_str}")
    ax_nya.set_title(f"NYA: {label_str}")
    return label_str, region, (x_lyr, y_lyr), (x_nya, y_nya)


def plot_region_box(lat_range, lon_range, lyr, nya, save_dir, save_prefix=None, region=None, event_dt=None):
    """RGB overlay of the geographic box at 150 km, with no field-line traces."""
    fig_im, (ax_im1, ax_im2) = plt.subplots(1, 2, figsize=(12, 6))
    _draw_latlon_box(ax_im1, ax_im2, lat_range, lon_range, lyr, nya, region=region, event_dt=event_dt)
    plt.tight_layout()
    if save_prefix is not None:
        path_im = os.path.join(save_dir, f"{save_prefix}_fieldlines.png")
        plt.savefig(path_im, bbox_inches="tight")
        print(f"Saved: {path_im}")
    plt.close(fig_im)


_FIELD_LINE_COLORS = [
    "#ff0000", "#ff4000", "#ff8000", "#ffbf00", "#ffff00",
    "#bfff00", "#80ff00", "#40ff00", "#00ff80", "#00ffbf",
    "#00ffff", "#00bfff", "#0080ff", "#0040ff", "#0000ff",
]


def _fieldline_colors(n_lines):
    n_colors = len(_FIELD_LINE_COLORS)
    return [_FIELD_LINE_COLORS[(idx + 5) % n_colors] for idx in range(n_lines)]


def _scatter_fieldlines(ax_lyr, ax_nya, lines_LYR, lines_NYA, line_colors):
    for (x1, y1), (x2, y2), c in zip(lines_LYR, lines_NYA, line_colors):
        m1 = np.isfinite(x1) & np.isfinite(y1)
        m2 = np.isfinite(x2) & np.isfinite(y2)
        ax_lyr.scatter(np.asarray(x1)[m1], np.asarray(y1)[m1], c=c, s=0.1)
        ax_nya.scatter(np.asarray(x2)[m2], np.asarray(y2)[m2], c=c, s=0.1)


def _save_figure(fig, save_dir, save_prefix, suffix):
    if save_prefix is not None:
        path = os.path.join(save_dir, f"{save_prefix}_{suffix}.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def _mark_peak_mean(ax, mu, sem):
    if not np.isfinite(mu):
        return
    label = f"Mean of peaks {mu:.1f} km"
    if np.isfinite(sem):
        ax.axhspan(mu - sem, mu + sem, color="tab:blue", alpha=0.2, zorder=0)
        label = f"Mean of peaks {mu:.1f}±{sem:.1f} km"
    ax.axhline(mu, color="tab:blue", lw=1.5, label=label)


def _mark_peak_median(ax, median):
    if not np.isfinite(median):
        return
    ax.axhline(median, color="tab:orange", ls=":", lw=1.5, label=f"Median of peaks {median:.1f} km")


def _draw_station_profiles(
    ax1,
    ax2,
    all_brightness_values_LYR,
    all_brightness_values_NYA,
    line_colors,
    alt,
    mean1_s,
    mean2_s,
    pk_alt1,
    pk_alt2,
    label_str,
    show_title=True,
    stats=None,
):
    for prof, c in zip(all_brightness_values_LYR, line_colors):
        ax1.plot(prof, alt, color=c, alpha=0.8, lw=1.2)
    ax1.plot(mean1_s, alt, color="black", lw=2.5, label="Mean (smoothed)")
    ax1.axhline(pk_alt1, ls="--", color="k", lw=1, label=f"Argmax mean {pk_alt1:.0f} km")
    if stats is not None:
        _mark_peak_mean(ax1, stats["lyr_mean_km"], stats["lyr_sem_km"])
        _mark_peak_median(ax1, stats["lyr_median_km"])
    ax1.set_xlabel("Brightness (pixel value)")
    ax1.set_ylabel("Altitude (km)")
    if show_title:
        ax1.set_title(f"LYR {label_str}")
    ax1.grid(True)
    ax1.legend(loc="best")

    for prof, c in zip(all_brightness_values_NYA, line_colors):
        ax2.plot(prof, alt, color=c, alpha=0.8, lw=1.2)
    ax2.plot(mean2_s, alt, color="black", lw=2.5, label="Mean (smoothed)")
    ax2.axhline(pk_alt2, ls="--", color="k", lw=1, label=f"Argmax mean {pk_alt2:.0f} km")
    if stats is not None:
        _mark_peak_mean(ax2, stats["nya_mean_km"], stats["nya_sem_km"])
        _mark_peak_median(ax2, stats["nya_median_km"])
    ax2.set_xlabel("Brightness (pixel value)")
    if show_title:
        ax2.set_title(f"NYA {label_str}")
    ax2.grid(True)
    ax2.legend(loc="best")


def _plot_fieldlines_figure(
    lines_LYR,
    lines_NYA,
    line_colors,
    lat_range,
    lon_range,
    lyr,
    nya,
    save_dir,
    save_prefix,
    region,
    event_dt,
    sample,
):
    fig, (ax_lyr, ax_nya) = plt.subplots(1, 2, figsize=(12, 6))
    _draw_latlon_box(
        ax_lyr, ax_nya, lat_range, lon_range, lyr, nya, region=region, event_dt=event_dt, sample=sample
    )
    _scatter_fieldlines(ax_lyr, ax_nya, lines_LYR, lines_NYA, line_colors)
    fig.tight_layout()
    _save_figure(fig, save_dir, save_prefix, "fieldlines")


def _plot_profiles_figure(
    all_brightness_values_LYR,
    all_brightness_values_NYA,
    line_colors,
    alt,
    mean1_s,
    mean2_s,
    pk_alt1,
    pk_alt2,
    label_str,
    save_dir,
    save_prefix,
    stats,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    _draw_station_profiles(
        ax1,
        ax2,
        all_brightness_values_LYR,
        all_brightness_values_NYA,
        line_colors,
        alt,
        mean1_s,
        mean2_s,
        pk_alt1,
        pk_alt2,
        label_str,
        stats=stats,
    )
    fig.tight_layout()
    _save_figure(fig, save_dir, save_prefix, "profiles_single")


_USE_BRIGHTNESS_SAMPLE = object()


def _plot_combined_figure(
    all_brightness_values_LYR,
    all_brightness_values_NYA,
    lines_LYR,
    lines_NYA,
    line_colors,
    alt,
    mean1_s,
    mean2_s,
    pk_alt1,
    pk_alt2,
    label_str,
    lat_range,
    lon_range,
    lyr,
    nya,
    save_dir,
    save_prefix,
    region,
    event_dt,
    sample,
    stats,
    image_sample=_USE_BRIGHTNESS_SAMPLE,
    suffix="combined",
):
    if image_sample is _USE_BRIGHTNESS_SAMPLE:
        image_sample = sample
    fig = plt.figure(figsize=(16, 14))
    grid = fig.add_gridspec(2, 2)
    ax_lyr = fig.add_subplot(grid[0, 0])
    ax_nya = fig.add_subplot(grid[0, 1])
    ax_prof_lyr = fig.add_subplot(grid[1, 0])
    ax_prof_nya = fig.add_subplot(grid[1, 1], sharey=ax_prof_lyr)
    _draw_latlon_box(
        ax_lyr,
        ax_nya,
        lat_range,
        lon_range,
        lyr,
        nya,
        region=region,
        event_dt=event_dt,
        sample=image_sample,
    )
    _scatter_fieldlines(ax_lyr, ax_nya, lines_LYR, lines_NYA, line_colors)
    _draw_station_profiles(
        ax_prof_lyr,
        ax_prof_nya,
        all_brightness_values_LYR,
        all_brightness_values_NYA,
        line_colors,
        alt,
        mean1_s,
        mean2_s,
        pk_alt1,
        pk_alt2,
        label_str,
        show_title=False,
        stats=stats,
    )
    fig.suptitle(
        f"Number of accepted fieldlines: {len(all_brightness_values_LYR)}\n"
        f"Brightness sample: {sample}"
    )
    fig.tight_layout()
    _save_figure(fig, save_dir, save_prefix, suffix)


def make_plots(
    all_brightness_values_LYR,
    all_brightness_values_NYA,
    alt_fieldline_last,
    lines_LYR,
    lines_NYA,
    lat_range,
    lon_range,
    lyr,
    nya,
    event_dt,
    save_dir,
    save_prefix=None,
    region=None,
    save_separate=False,
    sample="greyscale",
):
    label_str = _title_time(event_dt)
    _, (x_lyr, y_lyr), (x_nya, y_nya), _, _, _ = _region_box_pixels(
        lat_range, lon_range, region, lyr, nya
    )
    colors_cycle = _FIELD_LINE_COLORS
    line_colors = _fieldline_colors(len(lines_LYR))

    def fieldlines_shell_ASC(alt, lines_LYR, lines_NYA):
        # Fieldlines piercing shells at 140 and 200 km (Sony 08:57:35 only)
        if event_dt == datetime(2020, 1, 3, 8, 57, 35, tzinfo=timezone.utc):
            h_shell = 140.0
            i = int(np.argmin(np.abs(alt - h_shell)))
            h_shell250 = 200.0
            i250 = int(np.argmin(np.abs(alt - h_shell250)))
            fig, ax = plt.subplots(2, 2, figsize=(24, 22))
            _imshow_station(ax[0, 0], lyr)
            _imshow_station(ax[0, 1], nya)
            ax[0, 0].plot(x_lyr, y_lyr, "w", lw=1.0)
            ax[0, 1].plot(x_nya, y_nya, "w", lw=1.0)
            for idx, ((x1, y1), (x2, y2)) in enumerate(zip(lines_LYR, lines_NYA)):
                c = colors_cycle[idx % len(colors_cycle)]
                ax[0, 0].scatter(x1[i], y1[i], s=10, c=c)
                ax[0, 1].scatter(x2[i], y2[i], s=10, c=c)
            ax[0, 0].set_title(f"LYR at {alt[i]:.0f} km", fontsize=14)
            ax[0, 1].set_title(f"NYA at {alt[i]:.0f} km", fontsize=14)
            _imshow_station(ax[1, 0], lyr)
            _imshow_station(ax[1, 1], nya)
            ax[1, 0].plot(x_lyr, y_lyr, "w", lw=1.0)
            ax[1, 1].plot(x_nya, y_nya, "w", lw=1.0)
            for idx, ((x1, y1), (x2, y2)) in enumerate(zip(lines_LYR, lines_NYA)):
                c = colors_cycle[idx % len(colors_cycle)]
                ax[1, 0].scatter(x1[i250], y1[i250], s=10, c=c)
                ax[1, 1].scatter(x2[i250], y2[i250], s=10, c=c)
            ax[1, 0].set_title(f"LYR at {alt[i250]:.0f} km", fontsize=14)
            ax[1, 1].set_title(f"NYA at {alt[i250]:.0f} km", fontsize=14)
            plt.tight_layout()
            if save_prefix is not None:
                path_shell = os.path.join(save_dir, f"{save_prefix}_fieldlines_shell.png")
                plt.savefig(path_shell, bbox_inches="tight")
                print(f"Saved: {path_shell}")
            plt.close(fig)

    # Brightness-Altitude profiles
    arr1 = np.vstack(all_brightness_values_LYR)  # (n_line, n_alt)
    arr2 = np.vstack(all_brightness_values_NYA)

    mean1 = np.nanmean(arr1, axis=0)
    mean2 = np.nanmean(arr2, axis=0)

    mean1_s = medfilt(mean1, kernel_size=3)
    mean2_s = medfilt(mean2, kernel_size=3)

    alt = alt_fieldline_last

    idx_pk1 = np.nanargmax(mean1)
    pk_ray1 = mean1_s[idx_pk1]
    pk_alt1 = alt[idx_pk1]

    idx_pk2 = np.nanargmax(mean2)
    pk_ray2 = mean2_s[idx_pk2]
    pk_alt2 = alt[idx_pk2]

    print(f"{label_str} LYR:{pk_alt1}, NYA:{pk_alt2}")

    stats = summarise_peaks(all_brightness_values_LYR, all_brightness_values_NYA, alt)
    stats["lyr_mean_profile_peak_km"] = float(pk_alt1)
    stats["nya_mean_profile_peak_km"] = float(pk_alt2)
    _ = pk_ray1, pk_ray2

    print(
        f"LYR  {stats['lyr_mean_km']:.1f} ± {stats['lyr_std_km']:.1f} km (std),  "
        f"SEM {stats['lyr_sem_km']:.1f} km,  median {stats['lyr_median_km']:.1f} km"
    )
    print(
        f"NYA  {stats['nya_mean_km']:.1f} ± {stats['nya_std_km']:.1f} km (std),  "
        f"SEM {stats['nya_sem_km']:.1f} km,  median {stats['nya_median_km']:.1f} km"
    )
    print(
        f"Both {stats['both_mean_km']:.1f} ± {stats['both_std_km']:.1f} km (std),  "
        f"SEM {stats['both_sem_km']:.1f} km,  median {stats['both_median_km']:.1f} km"
    )

    _plot_combined_figure(
        all_brightness_values_LYR,
        all_brightness_values_NYA,
        lines_LYR,
        lines_NYA,
        line_colors,
        alt,
        mean1_s,
        mean2_s,
        pk_alt1,
        pk_alt2,
        label_str,
        lat_range,
        lon_range,
        lyr,
        nya,
        save_dir,
        save_prefix,
        region,
        event_dt,
        sample,
        stats,
    )
    _plot_combined_figure(
        all_brightness_values_LYR,
        all_brightness_values_NYA,
        lines_LYR,
        lines_NYA,
        line_colors,
        alt,
        mean1_s,
        mean2_s,
        pk_alt1,
        pk_alt2,
        label_str,
        lat_range,
        lon_range,
        lyr,
        nya,
        save_dir,
        save_prefix,
        region,
        event_dt,
        sample,
        stats,
        image_sample=None,
        suffix="combined_coloured",
    )
    if save_separate:
        _plot_fieldlines_figure(
            lines_LYR,
            lines_NYA,
            line_colors,
            lat_range,
            lon_range,
            lyr,
            nya,
            save_dir,
            save_prefix,
            region,
            event_dt,
            sample,
        )
        _plot_profiles_figure(
            all_brightness_values_LYR,
            all_brightness_values_NYA,
            line_colors,
            alt,
            mean1_s,
            mean2_s,
            pk_alt1,
            pk_alt2,
            label_str,
            save_dir,
            save_prefix,
            stats,
        )

    def make_normalised_brightness_profiles(
        all_brightness_values_LYR, all_brightness_values_NYA, alt, save_prefix=None
    ):
        arrLYR = np.vstack(all_brightness_values_LYR)
        arrNYA = np.vstack(all_brightness_values_NYA)

        maxLYR = np.nanmax(arrLYR)
        maxNYA = np.nanmax(arrNYA)

        normLYR = arrLYR / maxLYR
        normNYA = arrNYA / maxNYA

        mean_normLYR = np.nanmean(normLYR, axis=0)
        mean_normNYA = np.nanmean(normNYA, axis=0)
        mean_norm_both = np.nanmean(np.vstack([normLYR, normNYA]), axis=0)
        std_norm_both = np.nanstd(np.vstack([normLYR, normNYA]), axis=0)

        pk_lyr_idx = int(np.nanargmax(mean_normLYR))
        pk_nya_idx = int(np.nanargmax(mean_normNYA))
        _ = float(alt[pk_lyr_idx]), float(alt[pk_nya_idx])
        pk_idx = int(np.nanargmax(mean_norm_both))
        pk_alt = float(alt[pk_idx])

        fig, ax = plt.subplots(figsize=(8, 10))

        ax.plot(medfilt(mean_norm_both, kernel_size=3), alt, color="magenta", lw=3.5, label="Mean")
        ax.fill_betweenx(
            alt,
            mean_norm_both - std_norm_both,
            mean_norm_both + std_norm_both,
            color="magenta",
            alpha=0.2,
            label="STD",
        )

        ax.axhline(pk_alt, ls="--", color="k", lw=2.0, label=f"Argmax Mean {pk_alt:.0f} km")
        ax.axhline(
            stats["both_mean_km"],
            color="k",
            lw=2,
            label=f"Mean of all peaks {stats['both_mean_km']:.0f}±{stats['both_sem_km']:.0f} km",
        )

        ax.set_xlabel("Normalised brightness", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylabel("Altitude (km)", fontsize=14)
        ax.set_title(
            r"$\bf{Brightness\ profile\ using\ WISC\ StarCal}$"
            + f"\n{label_str}\nNumber of accepted fieldlines: {len(all_brightness_values_LYR)}",
            fontsize=16,
        )
        ax.grid(True)
        ax.legend(loc="best")

        plt.tight_layout()
        if save_prefix is not None:
            path_prof = os.path.join(save_dir, f"{save_prefix}_normalised_brightness_profile.png")
            plt.savefig(path_prof, bbox_inches="tight")
            print(f"Saved: {path_prof}")
        plt.close(fig)

        stats["both_mean_profile_peak_km"] = pk_alt

    # make_normalised_brightness_profiles(
    #     all_brightness_values_LYR, all_brightness_values_NYA, alt, save_prefix=save_prefix
    # )
    # fieldlines_shell_ASC(alt, lines_LYR, lines_NYA)
    return stats

# Auroral height from stereo all-sky images (WISC StarCal)

`guillaume_triangulation_WISC.ipynb` estimates the **peak emission height of aurora** from a pair of all-sky cameras at Longyearbyen (LYR) and Ny-Ålesund (NYA). It follows the magnetic **field-line method** of Whiter et al. (2010 / 2013): the same field line is projected into both images, brightness is sampled along altitude, and lines that look like the same auroral structure at both sites are kept.

This notebook uses **WISC StarCal** HDF5 files for the camera pointing (azimuth/elevation grids in raw pixel coordinates). The same geometry is in `wisc_triangulation.py`; `triangulate_bacc_pairs.py` runs it on every row of `BACC_image_pairs_triangulation.csv`. A related script in this repo, `whiter2013_fieldline_height.py`, implements a similar Method 2 pipeline as a command-line tool.

## Method

1. Load a stereo image pair and the matching StarCal files.
2. Over a geographic box, start field lines at a reference altitude and trace them with IGRF / Tsyganenko via [PyGeopack](https://github.com/mattkjames7/PyGeopack).
3. Convert each `(lat, lon, alt)` sample to azimuth/elevation from each station, then to image pixels using the StarCal grids.
4. Sample image brightness along the line to get a brightness–altitude profile at LYR and NYA.
5. Keep only field lines that pass the stereo tests (similar peak altitudes, high correlation, similar intensity-weighted centroids, well-defined peaks).
6. Report peak altitudes (mean ± std / SEM) and save overlay and profile figures.

## Requirements

Python 3 with the packages in `requirements.txt`:

```
numpy
scipy
h5py
matplotlib
pillow
PyGeopack
```

Install with:

```bash
pip install -r requirements.txt
```

PyGeopack needs geomagnetic model files. On first use it typically downloads them; if that fails, set `GEOPACK_PATH` to a writable directory.

Run the notebook or the batch script from the repository root so that `Data/` and `out/` resolve correctly.

## Input data

Place calibrations (and notebook images) under `Data/`. The batch script reads LYR/NYA frames from the `LYR_DIR` and `NYA_DIR` paths at the top of `triangulate_bacc_pairs.py`; StarCal files still come from `Data/`. Each `Station` needs:

| Input | Role |
| --- | --- |
| All-sky image (PNG/JPG) | LYR or NYA RGB frame; overlays use the original, brightness is the channel mean or RGBproduct |
| StarCal HDF5 | `azimuth_deg`, `elevation_deg` corner grids of shape `(height+1, width+1)`, plus site attributes |

StarCal attributes used: `site_lat_deg`, `site_lon_deg`, `site_alt_m`, `image_width`, `image_height`. Image size must match the calibration. `flip_image_x` / `flip_image_y` are not supported.

The cell that constructs `lyr` and `nya` is where you select the event. Commented examples include BACC 2020-01-03, Sony 2020-01-03, and Sony 2021-12-15. The active setup is:

- Event time: `2020-01-03 08:57:50 UTC`
- Calibrations: `StarCal_BACC_LYR_2020.h5`, `StarCal_BACC_NYA_2020.h5`
- Images: currently both stations point at `RGB_product_BACC_NYA_20200103_085750.png` — swap in the matching LYR/NYA frames before a real run

Also set `event_dt` to the same instant as the images. That timestamp is used for the magnetic field model and for output file names.

## How to run the batch script

`triangulate_bacc_pairs.py` loads every LYR/NYA pair in `BACC_image_pairs_triangulation.csv`, uses that row’s geographic box, and takes the event time from the filename stamp (`BACC_LYR_20200103_081358.png` → `2020-01-03 08:13:58 UTC`). Geometry, tracing parameters, and acceptance tests are those of the notebook (`alt_fl=165`, `step=0.1`, `upper=lower=85`, `interval=2`, `peak=10`, `corr_thr=0.7`, `edge=0`, `frac=0.9`, `width_limit=999`, `centroid_thr=20`).

Paste the image folders at the top of `triangulate_bacc_pairs.py`, and leave `PREVIEW_ONLY = True` until the geographic box looks right:

```python
LYR_DIR = "/path/to/lyr"
NYA_DIR = "/path/to/nya"
PREVIEW_ONLY = True
REUSE_TRACES = True
```

Then run:

```bash
python triangulate_bacc_pairs.py
```

Preview writes `{YYYYMMDDThhmmssZ}_fieldlines.png` with the lat/lon box on the **original RGB** frames and does not trace field lines. Edit `lat1`…`lon2` in the CSV if the box is wrong, re-run the preview, then set `PREVIEW_ONLY = False` for the full batch (trace, select, profiles, `results.csv`). Overlays stay on the original RGB frame.

Brightness is sampled twice from each pair: the greyscale channel mean, and an RGBproduct `R*G*B / 255**2`. Figures go to `out/<pairs-csv-stem>/greyscale/` and `out/<pairs-csv-stem>/RGBproduct/`. `WISC_batch_results.csv` in the parent folder has a `sample` column for both.

`REUSE_TRACES = True` traces each unique geographic box once (using the first pair’s event time) and reuses those paths and pixel coordinates for later pairs and for both brightness images. Set it to `False` to re-trace every pair with its own timestamp.

Same folder for both stations is fine. If either path is empty or not a directory, the script exits with an error. Calibrations default to `Data/StarCal_BACC_LYR_2020.h5` and `Data/StarCal_BACC_NYA_2020.h5`.

All figures from a run go in `out/<pairs-csv-stem>/greyscale/` and `out/<pairs-csv-stem>/RGBproduct/`, with ISO 8601 basic prefixes (`YYYYMMDDThhmmssZ_fieldlines.png`, profiles, normalised profile). Peak statistics for both samples are written to `WISC_batch_results.csv` in the parent folder.

The batch run keeps every field line that passes the stereo tests. It does not apply the notebook’s optional post-hoc slice of `selected_*` arrays.

## How to run the notebook

Open `guillaume_triangulation_WISC.ipynb` and run all cells in order.

The tracing cell defines a geographic box and the altitude window. Current values:

```python
region = [
    {'lat1': 77.2, 'lat2': 78.4, 'lon1': 11.0, 'lon2': 16.6, 'label': 'R1'},
]

trace_and_filter_field_lines(
    lat_range, lon_range,
    alt_fl=165, step=0.1,
    upper=85, lower=85, interval=2,
    peak=10, corr_thr=0.7, edge=0, frac=0.9, width_limit=999, centroid_thr=20,
)
```

That traces field lines on a `0.1°` lat/lon grid, from **80 to 250 km** (`alt_fl ± 85 km`) every **2 km**.

Optional: after tracing, subset `selected_*` arrays if you only want to plot some of the accepted lines.

## Field-line acceptance tests

A line is kept when all of the following hold (see the markdown cell above the tracing call for the intended “strict” set):

| Parameter | Meaning |
| --- | --- |
| `peak` | Peak altitudes at LYR and NYA differ by less than this (km) |
| `corr_thr` | Pearson correlation of the two brightness profiles |
| `edge` | Reject peaks within this many km of the top or bottom of the altitude window |
| `centroid_thr` | Intensity-weighted altitude centroids differ by less than this (km) |
| `frac`, `width_limit` | Peak width at `frac` of maximum (e.g. 90%) must be ≤ `width_limit` km |
| peak shape | Peak brightness must stand out relative to the mean above/below the peak |

Looser numbers (`peak=10`, `corr_thr=0.7`, `edge=0`, `width_limit=999`) accept more lines; tighter numbers (`peak=5`, `corr_thr=0.8`, `edge=10`, `width_limit=40`) match the notebook’s documented example.

## Outputs

Notebook figures go to `out/greyscale_out_fiedline_WISC_<YYYYMMDDHHMMSS>/`. Batch figures go to `out/<pairs-csv-stem>/greyscale/` and `out/<pairs-csv-stem>/RGBproduct/`, with `WISC_batch_results.csv` in the parent folder.

| File | Content |
| --- | --- |
| `*_fieldlines.png` | Geographic box at 150 km on the RGB pair (preview), plus accepted field lines after a full run |
| `*_profiles_single.png` | Brightness vs altitude for each accepted line (LYR and NYA), with mean profile and peak |
| `*_normalised_brightness_profile.png` | Mean normalised profile of both stations, with std band |

Peak statistics (mean altitude, standard deviation, and standard error of the mean, for LYR, NYA, and both combined) are printed, and the batch script also writes them to `WISC_batch_results.csv` in the run folder.

## Notebook layout

| Section | What it does |
| --- | --- |
| Map `(lat, lon, alt)` to pixels | `Station` class: load StarCal + image, inverse az/el → pixel interpolator |
| Field-line tracing functions | GEO/ECEF geometry, PyGeopack traces, pixel mapping, brightness sampling, selection metrics |
| Trace fieldlines | Geographic box, `trace_and_filter_field_lines(...)` |
| Plot | Image overlays, single-station profiles, combined normalised profile |

## Related files

- `wisc_triangulation.py` — notebook Station class, GEO/ECEF geometry, PyGeopack traces, selection tests, and plots
- `triangulate_bacc_pairs.py` — batch driver over `BACC_image_pairs_triangulation.csv`
- `guillaume_triangulation.ipynb` — earlier triangulation notebook (mapping can be replaced by StarCal)
- `whiter2013_fieldline_height.py` — script version of the Whiter Method 2 pipeline
- `geo.ipynb` / `geo_Guillaume.ipynb` — geographic / mapping helpers

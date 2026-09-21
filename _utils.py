import os, re, glob, json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2, theilslopes 

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

G, Y, R, C, B, W = "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[34m", "\033[0m"

# =============================================================================
# Run-wise file-path resolution
# =============================================================================

def extract_run_number(path: str) -> int | None:
    """Pull the integer run number out of a filename like '...Run01234.0000.h5'."""
    match = re.search(r"Run(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


def resolve_paths_by_run(
    unique_runs, pattern_new: str, pattern_old: str | None, subrun_glob: str, run_glob: str = "Run*"
) -> dict:
    """Glob a NEW and (optionally) OLD data-tier pattern once, then map
    run_number -> resolved path, with NEW taking priority over OLD.

    `subrun_glob` is the run/subrun wildcard fragment to widen into `run_glob`
    before globbing, e.g. "Run?????.????.h5" -> "Run*".
    """
    new_files = glob.glob(pattern_new.replace(subrun_glob, run_glob)) if pattern_new else []
    old_files = glob.glob(pattern_old.replace(subrun_glob, run_glob)) if pattern_old else []

    new_map = {extract_run_number(f): f for f in new_files if extract_run_number(f) is not None}
    old_map = {extract_run_number(f): f for f in old_files if extract_run_number(f) is not None}

    return {run: new_map.get(run, old_map.get(run)) for run in unique_runs}


# =============================================================================
# DL1 datacheck (HDF5) reading
# =============================================================================

BASE_COLS_DCHECK = ["runnumber", "subrun"]

COLS_COSMICS_SRWISE_DCHECKS = [
    "runnumber", "subrun", "time", "elapsed_time", "events", "azimuth", "altitude",
    "num_contained_mu_rings", "mu_effi_mean", "mu_effi_stddev", "mu_width_mean", "mu_width_stddev",
    "mu_radius_mean", "mu_radius_stddev", "mu_intensity_mean", "mu_hg_peak_sample", "mu_hg_peak_sample_stddev",
]
COLS_CIS_SRWISE_DCHECKS = [
    "corrected_elapsed_time", "ra_tel", "dec_tel",
    "picture_thresh", "boundary_thresh", "diffuse_nsb_std",
    "cosmics_rate", "cosmics_cleaned_rate",
    "intensity_at_half_peak_rate", "cosmics_peak_rate",
    "cosmics_rate_at_422_pe", "delta_cosmics_rate_at_422_pe",
    "cosmics_spectral_index", "delta_cosmics_spectral_index",
    "ZD_corrected_intensity_at_half_peak_rate", "ZD_corrected_cosmics_peak_rate",
    "ZD_corrected_cosmics_rate_at_422_pe", "ZD_corrected_delta_cosmics_rate_at_422_pe",
    "ZD_corrected_cosmics_spectral_index",
    "intensity_spectrum_fit_p_value", "intensity_at_reference_rate", "light_yield",
]
COLS_FF_SRWISE_DCHECKS = [
    "events", "charge_mean", "charge_stddev", "rel_time_mean", "rel_time_stddev",
]  # -> prefixed "ff_"
COLS_PED_SRWISE_DCHECKS = [
    "events", "charge_mean", "charge_stddev", "fraction_pulses_above10", "fraction_pulses_above30",
]  # -> prefixed "pedestal_"

# FINAL wanted order of columns
DCHECK_COLUMN_ORDER = [
    "obs_id", "subrun", "time", "elapsed_time", "corrected_elapsed_time", "events",
    "az", "zd", "alt", "ra", "dec", 
    
    "picture_thresh", "boundary_thresh", "diffuse_nsb_std",
    
    "ff_events", "ff_charge_mean", "ff_charge_stddev", "ff_rel_time_mean", "ff_rel_time_stddev",
    
    "pedestal_events", "pedestal_charge_mean", "pedestal_charge_stddev",
    "pedestal_fraction_pulses_above10", "pedestal_fraction_pulses_above30",
    
    "cosmics_rate", "cosmics_cleaned_rate", "intensity_at_half_peak_rate", "cosmics_peak_rate",
    "cosmics_rate_at_422_pe", "delta_cosmics_rate_at_422_pe", "cosmics_spectral_index",
    "delta_cosmics_spectral_index",
    
    "ZD_corrected_intensity_at_half_peak_rate", "ZD_corrected_cosmics_peak_rate",
    "ZD_corrected_cosmics_rate_at_422_pe", "ZD_corrected_delta_cosmics_rate_at_422_pe",
    "ZD_corrected_cosmics_spectral_index",
    
    "intensity_spectrum_fit_p_value", "intensity_at_reference_rate", "light_yield",
    
    
    "num_contained_mu_rings", "mu_effi_mean", "mu_effi_stddev",
    "mu_width_mean", "mu_width_stddev", "mu_radius_mean", "mu_radius_stddev",
    "mu_intensity_mean", "mu_hg_peak_sample", "mu_hg_peak_sample_stddev",
    
    "dcheck_fname", "dl1_srunwise_fname", "dl1_runwise_fname", "dl2_fname", "dl3_fname",
]


def _select(store: pd.HDFStore, key: str, cols: list[str], required: bool) -> pd.DataFrame | None:
    """Read one key from an open HDFStore, keeping BASE_COLS_DCHECK + requested cols that exist."""
    try:
        df = store.get(key)
    except KeyError:
        if required:
            raise
        return None

    wanted = BASE_COLS_DCHECK + [c for c in cols if c not in BASE_COLS_DCHECK]
    present = [c for c in wanted if c in df.columns]
    missing = set(wanted) - set(present)
    if missing:
        logger.warning("%s missing columns: %s", key, sorted(missing))

    if not set(BASE_COLS_DCHECK).issubset(present):
        if required:
            raise KeyError(f"{key} lacks runnumber/subrun")
        return None

    return df[present].copy()


def read_dcheck_file(path: Path) -> pd.DataFrame | None:
    """Read the cosmics/cosmics_intensity_spectrum/flatfield/pedestals tables of one
    DL1 datacheck HDF5 file and merge them into a single subrun-wise DataFrame."""
    try:
        with pd.HDFStore(path, mode="r") as store:
            d_cosmics = _select(store, "/cosmics/table", COLS_COSMICS_SRWISE_DCHECKS, required=True)
            d_cis = _select(store, "/cosmics_intensity_spectrum", COLS_CIS_SRWISE_DCHECKS, required=False)
            d_ff = _select(store, "/flatfield/table", COLS_FF_SRWISE_DCHECKS, required=False)
            d_ped = _select(store, "/pedestals/table", COLS_PED_SRWISE_DCHECKS, required=False)
    except Exception as exc:
        logger.warning("skipping %s: %s", path.name, exc)
        return None

    # Converting angles to degrees
    for col in ("azimuth", "altitude"):
        d_cosmics[col] = np.rad2deg(d_cosmics[col])

    if d_ff is not None:
        d_ff = d_ff.rename(columns={c: f"ff_{c}" for c in d_ff.columns if c not in BASE_COLS_DCHECK})
    if d_ped is not None:
        d_ped = d_ped.rename(columns={c: f"pedestal_{c}" for c in d_ped.columns if c not in BASE_COLS_DCHECK})

    merged = d_cosmics
    for extra in (d_cis, d_ff, d_ped):
        if extra is not None:
            merged = merged.merge(extra, on=BASE_COLS_DCHECK, how="left")

    merged["dcheck_fname"] = path.name
    return merged


def fix_dcheck_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw unix timestamps into datetimes and rename to the project's short column names."""
    df["time"] = df["time"].apply(datetime.fromtimestamp)
    df["zd"] = 90.0 - df["altitude"]
    return df.rename(columns={"azimuth": "az", "altitude": "alt", "ra_tel": "ra", "dec_tel": "dec"})


def order_dcheck_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns per DCHECK_COLUMN_ORDER, appending any unlisted extras at the end."""
    front = [c for c in DCHECK_COLUMN_ORDER if c in df.columns]
    return df[front + [c for c in df.columns if c not in front]]


# =============================================================================
# Weather-station (WS) day-file parsing
# =============================================================================

WS_COL_NAMES = [
    "Time", "Date", "TE", "DR", "WR", "FE", "WG", "WS", "WD", "WV", "TK", "TD", "TS", "RA", "RS", "ST",
]
WS_STRIP_RE = r"[A-Za-z]*([+\-]?\d*\.?\d+)"  # strips starting letter prefixes, e.g. "T12.3" -> "12.3"

WS_COL_MAP = {
    "TE": "temperature",         # °C
    "DR": "pressure",            # mmHg
    "FE": "humidity",            # %
    "WG": "wind_speed",          # km/h
    "WS": "wind_gust",           # km/h
    "WD": "wind_speed_average",  # km/h
    "TD": "tng_dust",            # µg/m³
    "TS": "tng_seeing",          # arcsec
    "ST": "rain",
}


def load_ws_day(day: pd.Timestamp, path_ws: Path) -> pd.DataFrame | None:
    """Parse one WS day file (`<path_ws>/<YYYY_MM>/WSalldata_<YYMMDD>.txt`).
    Returns a DataFrame indexed by Datetime, or None if the file is missing/empty."""
    fpath = path_ws / day.strftime("%Y_%m") / day.strftime("WSalldata_%y%m%d.txt")
    if not fpath.exists():
        return None

    df = pd.read_csv(
        fpath, sep=r",\s*", skiprows=1, header=None, names=WS_COL_NAMES,
        dtype=str, engine="python", on_bad_lines="skip",
    )
    if df.empty:
        return None

    df.index = pd.to_datetime(
        df["Date"].str.strip() + " " + df["Time"].str.strip(), format="%d.%m.%y %H:%M:%S", errors="coerce"
    )
    df.index.name = "Datetime"
    df = df.drop(columns=["Date", "Time"])

    return (
        df.stack(future_stack=True)
        .str.extract(WS_STRIP_RE, expand=False)
        .unstack()
        .astype(float)
    )


# =============================================================================
# SLURM batch submission with on-disk caching
# =============================================================================

def is_job_complete(out_path: Path) -> bool:
    """A batch is done if either a `.done` sentinel exists (inline runs) or the
    Slurm `.out` file exists and contains 'Finished' — both modes share one cache."""
    if out_path.with_suffix(".done").exists():
        return True
    return out_path.exists() and "Finished" in out_path.read_text(errors="replace")

def run_batches(
    items, job_name_fn, cmd_fn, out_dir: Path, *,
    process_inline: bool, resume_from: int = 0, overwrite: bool = False,
    slurm_opts: str = "-p short", part_file_fn=None,
) -> tuple[int, int]:
    """Submit (or run inline) one job per item in `items`, skipping already-completed
    batches. Shared by the WS-run-relation, intensity-cut and PSF-query stages.
    job_name_fn(idx, item) -> str      unique job name, also used for the .out/.done cache
    cmd_fn(item)           -> str      the python/bash command line to execute
    part_file_fn(idx, item) -> Path    optional; if given, a batch is only treated as
                                        complete when this file also exists on disk
                                        (guards against stale .out/.done pointing at
                                        deleted/missing data). Callers that don't pass
                                        this behave exactly as before.
    Returns (n_processed, n_skipped).
    """
    processed = skipped = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(items[resume_from:], start=resume_from):
        job_name = job_name_fn(idx, item)
        out_path = out_dir / f"{job_name}.out"
        part_file = part_file_fn(idx, item) if part_file_fn else None

        if overwrite:
            # Clear stale cache/data for this batch so a rerun can't be falsely
            # certified "complete" by leftover files from a previous run.
            out_path.unlink(missing_ok=True)
            out_path.with_suffix(".done").unlink(missing_ok=True)
            if part_file is not None:
                part_file.unlink(missing_ok=True)
        else:
            complete = is_job_complete(out_path)  # <-- unchanged, untouched
            if complete and part_file is not None and not part_file.exists():
                logger.warning(
                    "[CACHE MISMATCH] batch %d (%s) marked complete by log but "
                    "part-file %s is missing — rerunning.", idx, job_name, part_file
                )
                complete = False
            if complete:
                skipped += 1
                logger.info("[CACHE SKIP] batch %d (%s) already complete.", idx, job_name)
                continue

        processed += 1
        cmd = cmd_fn(item)
        if process_inline:
            logger.info("[inline] batch %d: %s", idx, job_name)
            result = subprocess.run(cmd, shell=True, text=True, check=False)
            if result.returncode == 0:
                out_path.with_suffix(".done").touch()
            else:
                logger.warning("batch %d failed (rc=%d)", idx, result.returncode)
        else:
            slurm_cmd = f"sbatch {slurm_opts} -J {job_name} -o {out_path} --wrap='{cmd}'"
            result = subprocess.run(slurm_cmd, shell=True, text=True, check=False)
            if result.returncode == 0:
                logger.info("Submitted batch %d: %s", idx, job_name)
            else:
                logger.warning("sbatch failed for batch %d", idx)
    return processed, skipped


def chunk_row_batches(n_total: int, batch_size: int) -> list[tuple[int, int]]:
    """Split `range(n_total)` into inclusive (i1, i2) index batches of `batch_size` rows."""
    return [(i, min(i + batch_size - 1, n_total - 1)) for i in range(0, n_total, batch_size)]


def chunk_list(items: list, batch_size: int) -> list[list]:
    """Split a list into consecutive chunks of at most `batch_size` items."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def dedupe_text_file(path: Path) -> tuple[int, int]:
    """Remove duplicate lines from a text file in place. Returns (n_total, n_unique)."""
    lines = path.read_text().splitlines(keepends=True)
    unique_lines = np.unique(lines)
    path.write_text("".join(unique_lines))
    return len(lines), len(unique_lines)


def create_catb(dl1_path, output_path):
    """
    Extracts catB tables from a DL1 file and saves them in the structure
    expected by lstchain_dl1ab.
    """
    with tables.open_file(dl1_path, 'r') as f_in:
        with tables.open_file(output_path, 'w') as f_out:
            # Create the /tel_1 group required by the script
            f_out.create_group("/", "tel_1")
            
            # Map the internal monitoring tables
            for name in ['calibration', 'pedestal', 'flatfield']:
                data = read_table(f_in, f"/dl1/monitoring/telescope/catB/{name}")
                write_table(data, f_out, f"/tel_1/{name}")
    
    print(f"CatB file created at: {output_path}")
    return output_path

# =============================================================================
# JSON helpers
# =============================================================================

def read_json_files(file_paths) -> list:
    """Read multiple JSON files and return a list of their contents."""
    data = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            data.append(json.load(f))
    return data


def merge_json_data(data_list: list[dict]) -> dict:
    """Shallow-merge a list of JSON dicts (later entries overwrite earlier ones)."""
    merged = {}
    for data in data_list:
        merged.update(data)
    return merged


def modify_json_data(target_dict: dict, change_dict: dict) -> dict:
    """Recursively update `target_dict` in place with values from `change_dict`."""
    for key, value in change_dict.items():
        if isinstance(value, dict):
            target_dict.setdefault(key, {})
            modify_json_data(target_dict[key], value)
        else:
            target_dict[key] = value
    return target_dict


def write_json_file(data, output_file) -> None:
    """Write JSON data to a file, pretty-printed."""
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


# =============================================================================
# Stats and fit helpers
# =============================================================================

def calc_light_yield(p0_fit, p1_fit, sigma_p0_fit, sigma_p1_fit, p0_ref):
	"""
	Calculate the light yield based on the reference point, AR, alphaR, A2, and alpha2.

	Parameters:
	refpoint (float): The reference point.
	AR (float): The AR value.
	alphaR (float): The alphaR value.
	A2 (float): The A2 value.
	alpha2 (float): The alpha2 value.

	Returns:
	float: The calculated light yield.
	float: The error in the light yield
	"""

	power_factor = (- 1 / ( 1 + p1_fit))

	ly	= (p0_fit / p0_ref) ** power_factor
    
	dlydA = (p0_fit / p0_ref) ** power_factor * 1 / (1 + p1_fit)	* (-1) / p0_fit
	dlyda = (p0_fit / p0_ref) ** power_factor * 1 / (1 + p1_fit)**2 * np.log(p0_fit / p0_ref)

	delta_ly = np.sqrt((dlydA) ** 2 * sigma_p0_fit ** 2 + (dlyda) ** 2 * sigma_p1_fit ** 2)
    
	return ly, delta_ly
    
def weighted_mean(x, w):
    """NaN-safe weighted mean; falls back to a plain nanmean if weights are unusable."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    good = ~(np.isnan(x) | np.isnan(w))
    if good.sum() == 0 or np.nansum(w[good]) <= 0:
        return np.nanmean(x)
    return np.average(x[good], weights=w[good])
    
def fit_drdi(telapsed, drdi, u_drdi, *, min_value=1e-2, max_value=2.4, trim_edges=False, min_points=3):
    telapsed = np.asarray(telapsed, dtype=float)
    drdi = np.asarray(drdi, dtype=float)
    u_drdi = np.asarray(u_drdi, dtype=float)

    # sanity cut no longer depends on whether the error is known -- a point
    # with a valid drdi but an unknown (NaN) error is still a usable point,
    # just one we can't weight yet
    finite = np.isfinite(drdi)
    in_range = (drdi >= min_value) & (drdi <= max_value) if max_value is not None or min_value is not None else True
    good = finite & in_range

    have_err = np.isfinite(u_drdi) & (u_drdi >= 1e-5)

    fit_mask = good.copy()
    if trim_edges:
        idx_good = np.flatnonzero(fit_mask)
        if len(idx_good) > min_points:
            fit_mask[idx_good[0]] = False
            fit_mask[idx_good[-1]] = False

    x_full = np.cumsum(telapsed)
    x, y, ye = x_full[fit_mask], drdi[fit_mask], u_drdi[fit_mask]
    use_w = have_err[fit_mask]

    if len(x) > 2:
        try:
            if use_w.all():
                # full error info -> proper weighted fit, errors trusted as absolute
                params, pcov, info, _, _ = curve_fit(
                    lambda x, a, b: a + b * x, x, y, sigma=ye,
                    absolute_sigma=True, p0=[0, 1], full_output=True)
            elif use_w.any():
                # mixed: impute a large fallback sigma for the unknown points so
                # they're included but barely count, instead of forcing the fit
                fallback = np.nanmedian(ye[use_w]) * 10
                ye_filled = np.where(use_w, ye, fallback)
                params, pcov, info, _, _ = curve_fit(
                    lambda x, a, b: a + b * x, x, y, sigma=ye_filled,
                    absolute_sigma=True, p0=[0, 1], full_output=True)
            else:
                # no error info at all (placeholder column) -> plain unweighted OLS.
                # absolute_sigma defaults to False here, so curve_fit rescales pcov
                # by the fit's own residual variance -- you still get real, finite
                # u_p0/u_p1, just derived from scatter around the line instead of
                # from errors you don't have yet.
                params, pcov, info, _, _ = curve_fit(
                    lambda x, a, b: a + b * x, x, y, p0=[0, 1], full_output=True)

            p0, p1 = params
            u_p0, u_p1 = np.sqrt(pcov[0, 0]), np.sqrt(pcov[1, 1])
            ndf = len(x) - 2
            chi2_val = np.sum(info["fvec"] ** 2)
            pval = 1 - chi2.cdf(chi2_val, ndf) if use_w.any() else np.nan  # only meaningful if weighted
            return p0, p1, u_p0, u_p1, chi2_val, pval, ndf, False, good
        except Exception:
            pass

    return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, len(x), True, good

def chi2F(params, f, x, y, uy) -> float:
    """Chi-squared of model `f(x, *params)` against data (y, uy)."""
    y_pred = f(x, *params)
    residuals = (y - y_pred) / uy
    return np.sum(residuals ** 2)


def powerlaw(x, norm, pindex):
    return norm * x ** pindex


def straight_line(x, intercept, slope):
    return intercept + slope * x


def pol2(x, a, b, c):
    return a + b * x + c * x * x


def sort_based(x_array, ref_array):
    """Sort `ref_array` ascending and rearrange `x_array` the same way.
    Returns (sorted_ref_array, reordered_x_array)."""
    order = np.argsort(ref_array)
    return np.asarray(ref_array)[order], np.asarray(x_array)[order]
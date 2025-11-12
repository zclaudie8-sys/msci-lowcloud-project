#!/usr/bin/env python3
"""Estimate 6-CCF Ridge sensitivities (seasonal vs deseasonal panels).

The script loads gridded monthly panel data, optionally removes the monthly
climatology, fits a local Ridge regression with non-local (windowed)
predictors at every valid grid point, and then stores/plots the inferred
sensitivities.

See ``python scripts/fig01_sensitivity_season_vs_deseason.py --help`` for the
command-line interface.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from matplotlib import pyplot as plt
from matplotlib import path as mpath
from matplotlib.colors import TwoSlopeNorm

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import yaml  # optional, only needed when regions.yaml is present
except ModuleNotFoundError:  # pragma: no cover - PyYAML may be absent in CI
    yaml = None

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PREDICTOR_ORDER: tuple[str, ...] = (
    "SST",
    "EIS",
    "RH700",
    "OMEGA700",
    "WS10",
    "SSTADV",
)
SEASONALITY_CHOICES = {"SEASON", "DESEASON", "BOTH"}
DEFAULT_REGIONS = {
    "SEP": {"lat": (-30.0, -10.0), "lon": (-100.0, -80.0)},
    "SEA": {"lat": (-25.0, -5.0), "lon": (-15.0, 15.0)},
    "NEP": {"lat": (15.0, 30.0), "lon": (-135.0, -115.0)},
    "NEA": {"lat": (15.0, 30.0), "lon": (-35.0, -15.0)},
    "SEI": {"lat": (-35.0, -15.0), "lon": (80.0, 110.0)},
}

# -----------------------------------------------------------------------------
# Helper data structures
# -----------------------------------------------------------------------------
@dataclass
class PanelData:
    predictors: Mapping[str, xr.DataArray]
    target: xr.DataArray
    mask: np.ndarray


@dataclass
class FitOutputs:
    sensitivities: xr.DataArray
    r2: xr.DataArray
    nsamples: xr.DataArray
    alpha: xr.DataArray


# -----------------------------------------------------------------------------
# Argument parsing helpers
# -----------------------------------------------------------------------------
def parse_alphas(value: str | None) -> np.ndarray:
    if not value:
        return np.logspace(-3.0, 3.0, 25)
    parts = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.append(float(chunk))
    if not parts:
        raise ValueError("No valid alphas parsed from input string.")
    return np.array(parts, dtype=float)


def parse_lon_lat_range(value: str | Sequence[str] | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [v.strip() for v in value.split(",") if v.strip()]
    else:
        parts = list(value)
    if len(parts) != 4:
        raise ValueError("--lon-lat-range expects four comma-separated values: lon_min,lon_max,lat_min,lat_max")
    lon_min, lon_max, lat_min, lat_max = map(float, parts)
    return lon_min, lon_max, lat_min, lat_max


def parse_time_window(value: str | None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if value is None:
        return None
    try:
        start_str, end_str = value.split(":")
        start = pd.to_datetime(start_str.strip(), format="%Y-%m")
        end = pd.to_datetime(end_str.strip(), format="%Y-%m")
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("--window must be formatted as YYYY-MM:YYYY-MM") from exc
    if end < start:
        raise ValueError("--window end must be >= start")
    return start, end


def parse_out_dirs(raw: Sequence[str] | None) -> dict[str, Path]:
    defaults = {
        "figs": Path("figs"),
        "results": Path("results"),
        "tables": Path("tables"),
        "logs": Path("logs"),
    }
    if not raw:
        return defaults
    resolved = dict(defaults)
    ordered_keys = ["figs", "results", "tables", "logs"]
    for item in raw:
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, val = item.split("=", 1)
            key = key.strip().lower()
            if key not in defaults:
                raise ValueError(f"Unknown output alias '{key}'. Choose from {list(defaults)}.")
            resolved[key] = Path(val.strip())
        else:
            # positional override following figs, results, tables, logs order
            if not ordered_keys:
                raise ValueError("Too many positional --out values provided.")
            key = ordered_keys.pop(0)
            resolved[key] = Path(item)
    return resolved


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def standardize_array(da: xr.DataArray, lon_lat_range: tuple[float, float, float, float] | None) -> xr.DataArray:
    # Rename dimensions if necessary
    rename = {}
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if rename:
        da = da.rename(rename)

    if "time" not in da.dims:
        raise ValueError("Expected 'time' dimension in panel data.")
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError("Expected 'lat' and 'lon' dimensions in panel data.")

    # Convert longitude to [-180, 180)
    lon = da["lon"].values
    lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
    if not np.allclose(lon, lon_wrapped):
        da = da.assign_coords(lon=lon_wrapped)
    da = da.sortby("lon")

    # Sort lat ascending for consistent orientation
    da = da.sortby("lat")

    # Clip to |lat| <= 60 by default
    lat = da["lat"].values
    mask_lat = (lat >= -60.0) & (lat <= 60.0)
    if mask_lat.sum() == 0:
        raise ValueError("Latitude range selection removed all data.")
    da = da.sel(lat=da["lat"].values[mask_lat])

    if lon_lat_range is not None:
        lon_min, lon_max, lat_min, lat_max = lon_lat_range
        if lon_min < lon_max:
            da = da.sel(lon=slice(lon_min, lon_max))
        else:  # wrap-around selection
            sel1 = da.sel(lon=slice(lon_min, 180))
            sel2 = da.sel(lon=slice(-180, lon_max))
            da = xr.concat([sel1, sel2], dim="lon")
        da = da.sel(lat=slice(lat_min, lat_max))

    return da


def load_single_var(path: Path, lon_lat_range: tuple[float, float, float, float] | None) -> xr.DataArray:
    if not path.exists():
        raise FileNotFoundError(f"Panel variable not found: {path}")
    with xr.open_dataset(path) as ds:
        data_vars = list(ds.data_vars)
        if not data_vars:
            raise ValueError(f"Dataset {path} has no data variables.")
        name = data_vars[0]
        da = ds[name]
        da.load()
    da = standardize_array(da, lon_lat_range)
    return da


def deseason_array(da: xr.DataArray) -> xr.DataArray:
    grouped = da.groupby("time.month")
    clim = grouped.mean("time")
    return grouped - clim


def maybe_deseason(da: xr.DataArray, seasonality: str) -> xr.DataArray:
    if seasonality == "SEASON":
        return da
    if seasonality == "DESEASON":
        return deseason_array(da)
    raise ValueError(f"Unexpected seasonality flag: {seasonality}")


def load_panel(data_root: Path, seasonality: str, yvar: str,
               lon_lat_range: tuple[float, float, float, float] | None,
               logger: logging.Logger) -> PanelData:
    season_dir = data_root / seasonality.lower()
    has_explicit = season_dir.exists()

    # Load target from preferred seasonality, fallback to season for deseasonalisation
    def _load_var(var: str) -> xr.DataArray:
        if seasonality == "DESEASON" and not has_explicit:
            season_path = data_root / "season" / f"{var}.nc"
            da = load_single_var(season_path, lon_lat_range)
            return deseason_array(da)
        else:
            path = season_dir / f"{var}.nc"
            return load_single_var(path, lon_lat_range)

    predictors = {}
    for pred in PREDICTOR_ORDER:
        try:
            predictors[pred] = _load_var(pred)
        except FileNotFoundError as exc:
            logger.error("Missing predictor %s for %s seasonality", pred, seasonality)
            raise exc

    target = _load_var(yvar)

    # Align all variables on common coords (intersection)
    arrays = list(predictors.values()) + [target]
    arrays = xr.align(*arrays, join="inner")
    predictors = {name: arr for name, arr in zip(PREDICTOR_ORDER, arrays[:-1])}
    target = arrays[-1]

    sst = predictors["SST"]
    mask = sst.isfinite().any(dim="time").values.astype(bool)
    return PanelData(predictors=predictors, target=target, mask=mask)


def apply_time_window(panel: PanelData, window: tuple[pd.Timestamp, pd.Timestamp] | None) -> PanelData:
    if window is None:
        return panel
    start, end = window
    slice_da = {name: arr.sel(time=slice(start, end)) for name, arr in panel.predictors.items()}
    target = panel.target.sel(time=slice(start, end))
    if target.sizes.get("time", 0) == 0:
        raise ValueError("Time window selection removed all samples.")
    mask = panel.mask
    return PanelData(predictors=slice_da, target=target, mask=mask)


# -----------------------------------------------------------------------------
# Region helpers
# -----------------------------------------------------------------------------
def _points_in_polygon(lon: np.ndarray, lat: np.ndarray, vertices: Sequence[Sequence[float]]) -> np.ndarray:
    points = np.column_stack([lon.ravel(), lat.ravel()])
    path = mpath.Path(vertices)
    mask_flat = path.contains_points(points)
    return mask_flat.reshape(lat.shape)


def load_region_masks(lat: np.ndarray, lon: np.ndarray, region_names: Sequence[str],
                      logger: logging.Logger) -> dict[str, np.ndarray]:
    region_defs: dict[str, dict] = {}
    cfg_path = Path("config/regions.yaml")
    if cfg_path.exists() and yaml is not None:
        with cfg_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                region_defs = loaded
            else:
                logger.warning("regions.yaml did not contain a dictionary; ignoring file.")
    masks: dict[str, np.ndarray] = {}

    lon2d, lat2d = np.meshgrid(lon, lat)

    for name in region_names:
        info = None
        if name in region_defs:
            info = region_defs[name]
        if info is None:
            info = DEFAULT_REGIONS.get(name)
            if info is None:
                logger.warning("Region '%s' not found in config; using global mask (all True).", name)
                masks[name] = np.ones_like(lat2d, dtype=bool)
                continue
        if "polygon" in info:
            vertices = info["polygon"]
            mask = _points_in_polygon(lon2d, lat2d, vertices)
        else:
            lon_min, lon_max = info.get("lon", info.get("lon_range", info.get("lon_bounds", (None, None))))
            lat_min, lat_max = info.get("lat", info.get("lat_range", info.get("lat_bounds", (None, None))))
            if lon_min is None or lon_max is None or lat_min is None or lat_max is None:
                raise ValueError(f"Region {name} missing lon/lat bounds in definition.")
            if lon_min <= lon_max:
                mask_lon = (lon2d >= lon_min) & (lon2d <= lon_max)
            else:  # wrap
                mask_lon = (lon2d >= lon_min) | (lon2d <= lon_max)
            mask_lat = (lat2d >= lat_min) & (lat2d <= lat_max)
            mask = mask_lon & mask_lat
        masks[name] = mask.astype(bool)
    return masks


# -----------------------------------------------------------------------------
# Modelling helpers
# -----------------------------------------------------------------------------
def build_feature_matrix(predictors: Mapping[str, xr.DataArray], lat_idx: int, lon_idx: int,
                         half_window: int) -> np.ndarray:
    arrays = []
    lat_slice = slice(lat_idx - half_window, lat_idx + half_window + 1)
    lon_slice = slice(lon_idx - half_window, lon_idx + half_window + 1)
    for pred in PREDICTOR_ORDER:
        window = predictors[pred][:, lat_slice, lon_slice].values
        arrays.append(window.reshape(window.shape[0], -1))
    return np.concatenate(arrays, axis=1)


def fit_ridge(panel: PanelData, alphas: np.ndarray, neigh: int, logger: logging.Logger,
              min_samples: int) -> FitOutputs:
    half = neigh // 2
    target = panel.target
    predictors = panel.predictors
    mask = panel.mask

    time_index = pd.DatetimeIndex(target["time"].values)
    years_all = time_index.year.values

    lat_vals = target["lat"].values
    lon_vals = target["lon"].values
    nlat = len(lat_vals)
    nlon = len(lon_vals)

    sens_arr = np.full((len(PREDICTOR_ORDER), nlat, nlon), np.nan, dtype=float)
    r2_arr = np.full((nlat, nlon), np.nan, dtype=float)
    nsamples_arr = np.zeros((nlat, nlon), dtype=int)
    alpha_arr = np.full((nlat, nlon), np.nan, dtype=float)

    total_cells = 0
    fitted_cells = 0

    for j in range(half, nlat - half):
        for i in range(half, nlon - half):
            total_cells += 1
            if not mask[j, i]:
                continue
            X = build_feature_matrix(predictors, j, i, half)
            y = target[:, j, i].values
            valid = np.isfinite(y)
            if valid.sum() == 0:
                continue
            valid &= np.isfinite(X).all(axis=1)
            if valid.sum() < min_samples:
                continue
            X_valid = X[valid]
            y_valid = y[valid]
            years = years_all[valid]
            unique_years = np.unique(years)
            if unique_years.size < 2:
                continue
            n_splits = min(5, unique_years.size)
            if n_splits < 2:
                continue
            cv = GroupKFold(n_splits=n_splits)
            pipe = Pipeline([
                ("scale", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", RidgeCV(alphas=alphas, cv=cv, scoring="r2")),
            ])
            try:
                pipe.fit(X_valid, y_valid, ridge__groups=years)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Ridge fit failed at lat %.2f lon %.2f: %s", lat_vals[j], lon_vals[i], exc)
                continue
            ridge: RidgeCV = pipe.named_steps["ridge"]
            scaler: StandardScaler = pipe.named_steps["scale"]
            scale = scaler.scale_.copy()
            scale[~np.isfinite(scale)] = 1.0
            scale[scale == 0.0] = 1.0
            coef = ridge.coef_ / scale
            offset = 0
            for idx, pred in enumerate(PREDICTOR_ORDER):
                span = neigh * neigh
                block = coef[offset:offset + span]
                sens_arr[idx, j, i] = np.nansum(block)
                offset += span
            r2_arr[j, i] = pipe.score(X_valid, y_valid)
            nsamples_arr[j, i] = int(valid.sum())
            alpha_arr[j, i] = float(ridge.alpha_)
            fitted_cells += 1

    logger.info("Ridge fit complete: %d/%d grid cells fitted", fitted_cells, total_cells)
    return FitOutputs(
        sensitivities=xr.DataArray(
            sens_arr,
            dims=("var", "lat", "lon"),
            coords={"var": list(PREDICTOR_ORDER), "lat": lat_vals, "lon": lon_vals},
        ),
        r2=xr.DataArray(r2_arr, dims=("lat", "lon"), coords={"lat": lat_vals, "lon": lon_vals}),
        nsamples=xr.DataArray(nsamples_arr, dims=("lat", "lon"), coords={"lat": lat_vals, "lon": lon_vals}),
        alpha=xr.DataArray(alpha_arr, dims=("lat", "lon"), coords={"lat": lat_vals, "lon": lon_vals}),
    )


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def _maybe_import_cartopy(use_cartopy: bool, logger: logging.Logger):
    if not use_cartopy:
        return False, None, None
    try:  # pragma: no cover - optional dependency
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ModuleNotFoundError:
        logger.warning("Cartopy requested but not available; falling back to plain matplotlib.")
        return False, None, None
    return True, ccrs, cfeature


def _compute_vlim(datasets: Sequence[xr.DataArray]) -> float:
    values = np.concatenate([np.abs(ds.values[np.isfinite(ds.values)]) for ds in datasets if ds is not None])
    if values.size == 0:
        return 1.0
    return np.nanpercentile(values, 95)


def _plot_maps(data: xr.DataArray, title: str, outfile: Path, use_cartopy: bool,
               ccrs_mod=None, cfeature_mod=None, vlim: float = 1.0, dry_run: bool = False) -> None:
    n_vars = data.sizes["var"]
    nrows = 2
    ncols = math.ceil(n_vars / nrows)
    figsize = (4 * ncols, 3.5 * nrows)
    if use_cartopy:
        proj = ccrs_mod.PlateCarree()
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={"projection": proj})
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(nrows, ncols)
    lon = data["lon"].values
    lat = data["lat"].values
    for idx, var in enumerate(data["var"].values):
        ax = axes.flat[idx]
        field = data.sel(var=var)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)
        if use_cartopy:
            mesh = ax.pcolormesh(lon, lat, field, transform=ccrs_mod.PlateCarree(), cmap="coolwarm", norm=norm)
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature_mod.BORDERS, linewidth=0.3)
        else:
            mesh = ax.pcolormesh(lon, lat, field, cmap="coolwarm", norm=norm, shading="auto")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        ax.set_title(str(var))
        fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.7)
    for extra_ax in axes.flat[n_vars:]:
        extra_ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if not dry_run:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150)
    plt.close(fig)


def _plot_difference(deseason: xr.DataArray, season: xr.DataArray, outfile: Path,
                     use_cartopy: bool, ccrs_mod=None, cfeature_mod=None,
                     vlim: float = 1.0, dry_run: bool = False) -> None:
    diff = deseason - season
    _plot_maps(diff, "Deseason minus Season", outfile, use_cartopy, ccrs_mod, cfeature_mod, vlim, dry_run)


def compute_area_weighted_mean(field: xr.DataArray, mask: np.ndarray) -> tuple[float, float, int]:
    data = field.values
    lat = field["lat"].values
    weights = np.cos(np.deg2rad(lat))[:, None]
    mask_valid = mask & np.isfinite(data)
    if not np.any(mask_valid):
        return np.nan, np.nan, 0
    combined_weight = weights * mask_valid
    total_weight = float(np.nansum(combined_weight))
    mean = float(np.nansum(data * combined_weight) / total_weight)
    resid = (data - mean) * mask_valid
    var = float(np.nansum((resid ** 2) * weights) / total_weight)
    if var < 0:
        var = 0.0
    count = int(np.sum(mask_valid))
    se = math.sqrt(var) / math.sqrt(max(count, 1))
    return mean, se, count


def _plot_region_bars(deseason: xr.DataArray | None, season: xr.DataArray | None,
                       masks: Mapping[str, np.ndarray], outfile: Path,
                       dry_run: bool = False) -> None:
    predictors = list(deseason["var"].values if deseason is not None else season["var"].values)
    regions = list(masks.keys())
    n_pred = len(predictors)
    x = np.arange(len(regions))
    width = 0.35

    fig, axes = plt.subplots(nrows=math.ceil(n_pred / 2), ncols=2, figsize=(12, 3 * math.ceil(n_pred / 2)))
    axes = np.array(axes).reshape(-1)

    for idx, pred in enumerate(predictors):
        ax = axes[idx]
        bars_d = []
        err_d = []
        bars_s = []
        err_s = []
        for region in regions:
            mask = masks[region]
            if deseason is not None:
                mean_d, se_d, _ = compute_area_weighted_mean(deseason.sel(var=pred), mask)
            else:
                mean_d, se_d = np.nan, np.nan
            if season is not None:
                mean_s, se_s, _ = compute_area_weighted_mean(season.sel(var=pred), mask)
            else:
                mean_s, se_s = np.nan, np.nan
            bars_d.append(mean_d)
            err_d.append(se_d)
            bars_s.append(mean_s)
            err_s.append(se_s)
        offset = width / 2
        if deseason is not None:
            ax.bar(x - offset, bars_d, width, yerr=err_d, label="Deseason", color="#1f77b4", alpha=0.85)
        if season is not None:
            ax.bar(x + offset, bars_s, width, yerr=err_s, label="Season", color="#ff7f0e", alpha=0.6, hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels(regions)
        ax.set_title(pred)
        ax.axhline(0, color="black", linewidth=0.8)
        if idx == 0:
            ax.legend()
    for extra in axes[n_pred:]:
        extra.axis("off")
    fig.tight_layout()
    if not dry_run:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
def setup_logger(logfile: Path) -> logging.Logger:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fig01_sensitivity")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(logfile, mode="w", encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.addHandler(stream)
    logger.info("Logging to %s", logfile)
    return logger


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ridge sensitivities for CCF predictors (season vs deseason).")
    parser.add_argument("--yvar", required=True, choices=["SWCRE", "LWCRE", "LCF"], help="Target variable")
    parser.add_argument("--neigh", type=int, default=5, help="Neighborhood window size (odd integer, default 5)")
    parser.add_argument("--alphas", default=None, help="Comma-separated alpha list for RidgeCV")
    parser.add_argument("--seasonality", default="BOTH", choices=SEASONALITY_CHOICES,
                        help="Which seasonality to evaluate")
    parser.add_argument("--lon-lat-range", dest="lon_lat_range", default=None,
                        help="Optional lon_min,lon_max,lat_min,lat_max selection")
    parser.add_argument("--window", default=None, help="Optional YYYY-MM:YYYY-MM time subset")
    parser.add_argument("--regions", default="SEP,SEA,NEP,NEA,SEI", help="Comma-separated region keys")
    parser.add_argument("--data-root", default="data/panel", help="Root folder containing panel NetCDFs")
    parser.add_argument("--out", nargs="*", default=None,
                        help="Override output directories. Use positional args (figs results tables logs) or key=value pairs.")
    parser.add_argument("--cartopy", action="store_true", help="Use cartopy for map backgrounds if available")
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline without writing outputs")
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    if args.neigh % 2 != 1:
        raise ValueError("--neigh must be an odd integer so that a central grid cell exists.")
    alphas = parse_alphas(args.alphas)
    lon_lat_range = parse_lon_lat_range(args.lon_lat_range)
    window = parse_time_window(args.window)
    out_dirs = parse_out_dirs(args.out)
    for path in out_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    logfile = out_dirs["logs"] / f"fig01_{args.yvar}.log"
    logger = setup_logger(logfile)
    if args.dry_run:
        logger.info("Dry run enabled: outputs will not be saved.")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    seasonality_modes = [args.seasonality] if args.seasonality != "BOTH" else ["DESEASON", "SEASON"]
    panels: dict[str, PanelData] = {}
    for seasonality in seasonality_modes:
        logger.info("Loading data for %s", seasonality)
        panel = load_panel(data_root, seasonality, args.yvar, lon_lat_range, logger)
        panel = apply_time_window(panel, window)
        panels[seasonality] = panel
        logger.info("Loaded panel for %s: %d time steps, %d lat, %d lon",
                    seasonality, panel.target.sizes["time"], panel.target.sizes["lat"], panel.target.sizes["lon"])

    min_samples = 18 if window is not None else 24

    fits: dict[str, FitOutputs] = {}
    for seasonality, panel in panels.items():
        logger.info("Fitting Ridge models for %s", seasonality)
        fit = fit_ridge(panel, alphas, args.neigh, logger, min_samples)
        fits[seasonality] = fit
        valid_vals = fit.sensitivities.values[np.isfinite(fit.sensitivities.values)]
        if valid_vals.size:
            logger.info("%s sensitivity stats: mean %.4f, median %.4f, 95th |value| %.4f",
                        seasonality, float(np.nanmean(valid_vals)), float(np.nanmedian(valid_vals)),
                        float(np.nanpercentile(np.abs(valid_vals), 95)))
        logger.info("%s R^2 median %.3f", seasonality, float(np.nanmedian(fit.r2.values)))
        if np.isnan(fit.nsamples.values).all():
            logger.info("%s sample count median nan", seasonality)
        else:
            samples_median = float(np.nanmedian(fit.nsamples.values))
            logger.info("%s sample count median %.1f", seasonality, samples_median)

    # Save NetCDF outputs
    for seasonality, fit in fits.items():
        out_dir = out_dirs["results"] / "ridge" / seasonality.lower()
        outfile = out_dir / f"sensitivities_{args.yvar}.nc"
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            ds = xr.Dataset({
                "sensitivity": fit.sensitivities,
                "r2": fit.r2,
                "nsamples": fit.nsamples,
                "alpha": fit.alpha,
            })
            time_vals = pd.to_datetime(panels[seasonality].target.time.values)
            ds.attrs.update({
                "yvar": args.yvar,
                "neigh": args.neigh,
                "alphas": ",".join(f"{a:g}" for a in alphas),
                "seasonality": seasonality,
                "time_range": f"{time_vals[0].strftime('%Y-%m-%d')}:{time_vals[-1].strftime('%Y-%m-%d')}",
            })
            ds.to_netcdf(outfile)
            logger.info("Saved sensitivities to %s", outfile)
        else:
            logger.info("[dry-run] Skipping save to %s", outfile)

    use_cartopy, ccrs_mod, cfeature_mod = _maybe_import_cartopy(args.cartopy, logger)

    fit_deseason = fits.get("DESEASON")
    deseason_data = fit_deseason.sensitivities if fit_deseason is not None else None
    fit_season = fits.get("SEASON")
    season_data = fit_season.sensitivities if fit_season is not None else None

    if deseason_data is not None:
        vlim = _compute_vlim([deseason_data, season_data] if season_data is not None else [deseason_data])
        outfile = out_dirs["figs"] / "fig01a_maps_deseason.png"
        _plot_maps(deseason_data, "Deseason sensitivities", outfile, use_cartopy, ccrs_mod, cfeature_mod, vlim, args.dry_run)
        logger.info("Generated deseason map figure at %s", outfile)
    if season_data is not None:
        vlim = _compute_vlim([season_data, deseason_data] if deseason_data is not None else [season_data])
        outfile = out_dirs["figs"] / "fig01b_maps_season.png"
        _plot_maps(season_data, "Season sensitivities", outfile, use_cartopy, ccrs_mod, cfeature_mod, vlim, args.dry_run)
        logger.info("Generated season map figure at %s", outfile)
    if deseason_data is not None and season_data is not None:
        vlim = _compute_vlim([deseason_data - season_data])
        outfile = out_dirs["figs"] / "fig01c_maps_diff.png"
        _plot_difference(deseason_data, season_data, outfile, use_cartopy, ccrs_mod, cfeature_mod, vlim, args.dry_run)
        logger.info("Generated difference map at %s", outfile)

    # Region bars
    if deseason_data is not None or season_data is not None:
        reference = deseason_data if deseason_data is not None else season_data
        lat_vals = reference["lat"].values
        lon_vals = reference["lon"].values
        region_names = [r.strip() for r in args.regions.split(",") if r.strip()]
        masks = load_region_masks(lat_vals, lon_vals, region_names, logger)
        outfile = out_dirs["figs"] / "fig01d_regionbars.png"
        _plot_region_bars(deseason_data, season_data, masks, outfile, args.dry_run)
        logger.info("Generated region bar plot at %s", outfile)

    logger.info("All tasks finished.")
    return 0


if __name__ == "__main__":
    cli_args = parse_args()
    sys.exit(main(cli_args))

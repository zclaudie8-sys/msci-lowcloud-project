#!/usr/bin/env python3
"""Cloud-controlling factor (CCF) regression and sensitivity analysis.

This script implements the core regression engine used throughout the
low-cloud analysis pipeline. It loads gridded panels of monthly data,
constructs non-local neighbourhood predictors, performs ridge regression
with leave-one-year-out cross validation, and computes both coefficient
sensitivities and permutation-based importance metrics.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold


FACTOR_NAMES: Tuple[str, ...] = (
    "SST",
    "EIS",
    "RH700",
    "OMEGA700",
    "WS10",
    "SSTADV",
)

DEFAULT_REGION_BOXES: Mapping[str, Sequence[float]] = {
    # [lon_min, lon_max, lat_min, lat_max]
    "SEP": (-110.0, -70.0, -35.0, -15.0),
    "SEA": (-30.0, 0.0, -30.0, 0.0),
    "NEP": (-155.0, -115.0, 15.0, 30.0),
    "NEA": (-30.0, 15.0, 15.0, 30.0),
    "SEI": (90.0, 150.0, -30.0, 0.0),
}

SEASONALITY_CHOICES = ("SEASON", "DESEASON", "BOTH")


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def parse_float_list(raw: str) -> List[float]:
    if not raw:
        return []
    values: List[float] = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        values.append(float(value))
    return values


def parse_str_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_lon_lat_range(raw: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not raw:
        return None
    parts = [float(p) for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--lon-lat-range expects four comma-separated numbers")
    lon_min, lon_max, lat_min, lat_max = parts
    return lon_min, lon_max, lat_min, lat_max


def parse_window(raw: Optional[str]) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not raw:
        return None
    try:
        start_str, end_str = raw.split(":", 1)
    except ValueError as exc:
        raise ValueError("--window expects START:END (YYYY-MM:YYYY-MM)") from exc
    start = pd.to_datetime(start_str.strip())
    end = pd.to_datetime(end_str.strip())
    return start, end


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cloud-controlling-factor ridge analysis")
    parser.add_argument("--yvar", required=True, help="Target variable: SWCRE, LWCRE, or LCF")
    parser.add_argument("--seasonality", default="SEASON", choices=SEASONALITY_CHOICES)
    parser.add_argument("--regions", default="SEP,SEA,NEP,NEA,SEI", help="Comma-separated region keys")
    parser.add_argument("--neigh", type=int, default=5, help="Neighbourhood width (must be odd)")
    parser.add_argument("--alphas", default="1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100,300,1000")
    parser.add_argument("--block", type=int, default=3, help="Block length (months) for permutation shuffles")
    parser.add_argument("--shuffles", type=int, default=20, help="Number of permutation replicates per factor")
    parser.add_argument("--lon-lat-range", dest="lon_lat_range", help="lon_min,lon_max,lat_min,lat_max")
    parser.add_argument("--window", help="Time window START:END (YYYY-MM:YYYY-MM)")
    parser.add_argument("--data-root", default="data/panel", help="Root directory for panel data")
    parser.add_argument("--cmip-csv", help="Optional CMIP stacked panel CSV (for logging only)")
    parser.add_argument("--out", default="results/ccf", help="Output directory for NetCDF results")
    parser.add_argument("--fig", dest="fig_dir", help="Optional figure output directory")
    parser.add_argument("--cartopy", action="store_true", help="Enable cartopy figure production (not implemented)")
    parser.add_argument("--dry-run", action="store_true", help="Run analysis without writing outputs")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for permutation importance")
    return parser


# ---------------------------------------------------------------------------
# IO utilities
# ---------------------------------------------------------------------------

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_regions(region_keys: Sequence[str]) -> Dict[str, Sequence[float]]:
    config_path = Path("configs/config.yaml")
    boxes: Dict[str, Sequence[float]] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf8") as fh:
            cfg = yaml.safe_load(fh)
        if isinstance(cfg, dict) and "regions" in cfg and isinstance(cfg["regions"], dict):
            for key, value in cfg["regions"].items():
                if isinstance(value, (list, tuple)) and len(value) == 4:
                    boxes[key.upper()] = tuple(float(v) for v in value)
    for key in region_keys:
        key_upper = key.upper()
        if key_upper in boxes:
            continue
        if key_upper in DEFAULT_REGION_BOXES:
            boxes[key_upper] = DEFAULT_REGION_BOXES[key_upper]
        else:
            raise KeyError(f"Region '{key}' not found in configuration or defaults")
    return boxes


def standardise_longitude(da: xr.DataArray) -> xr.DataArray:
    """Return a copy of *da* with longitude in [-180, 180)."""
    lon_name = "lon" if "lon" in da.coords else "longitude"
    lon = da[lon_name]
    lon_std = ((lon + 180.0) % 360.0) - 180.0
    return da.assign_coords({lon_name: lon_std}).sortby(lon_name)


def subset_lon_lat(
    da: xr.DataArray,
    lon_lat_range: Optional[Tuple[float, float, float, float]],
) -> xr.DataArray:
    if lon_lat_range is None:
        return da
    lon_min, lon_max, lat_min, lat_max = lon_lat_range
    lat_name = "lat" if "lat" in da.coords else "latitude"
    lon_name = "lon" if "lon" in da.coords else "longitude"
    da = da.sortby(lat_name)
    da = da.sel({lat_name: slice(lat_min, lat_max)})
    if lon_min <= lon_max:
        da = da.sel({lon_name: slice(lon_min, lon_max)})
    else:
        left = da.sel({lon_name: slice(lon_min, 180)})
        right = da.sel({lon_name: slice(-180, lon_max)})
        da = xr.concat([left, right], dim=lon_name)
    return da


def subset_time(da: xr.DataArray, window: Optional[Tuple[pd.Timestamp, pd.Timestamp]]) -> xr.DataArray:
    if window is None:
        return da
    start, end = window
    return da.sel(time=slice(start, end))


def deseasonalise(da: xr.DataArray) -> xr.DataArray:
    clim = da.groupby("time.month").mean("time")
    return da.groupby("time.month") - clim


def load_panel_variable(
    data_root: Path,
    seasonality: str,
    var: str,
    fallback: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    seasonality_lower = seasonality.lower()
    path = data_root / seasonality_lower / f"{var}.nc"
    if path.exists():
        da = xr.open_dataarray(path)
    else:
        if fallback is None:
            raise FileNotFoundError(f"Panel file not found: {path}")
        da = deseasonalise(fallback)
    return da


def load_panel_group(
    data_root: Path,
    seasonality: str,
    variables: Sequence[str],
    base_cache: Optional[Dict[str, xr.DataArray]] = None,
) -> Dict[str, xr.DataArray]:
    cache: Dict[str, xr.DataArray] = {}
    fallback_group = base_cache if seasonality.lower() == "deseason" else None
    for var in variables:
        fallback = None
        if fallback_group is not None and var in fallback_group:
            fallback = fallback_group[var]
        cache[var] = load_panel_variable(data_root, seasonality, var, fallback=fallback)
    return cache


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_neighbourhood(da: xr.DataArray, size: int) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Neighbourhood size must be odd")
    half = size // 2
    data = da.values
    padded = np.pad(data, ((0, 0), (half, half), (0, 0)), mode="edge")
    padded = np.pad(padded, ((0, 0), (0, 0), (half, half)), mode="wrap")
    neighbourhoods = []
    for dy in range(size):
        for dx in range(size):
            window = padded[:, dy : dy + data.shape[1], dx : dx + data.shape[2]]
            neighbourhoods.append(window)
    stacked = np.stack(neighbourhoods, axis=-1)
    return stacked  # shape: (time, lat, lon, size*size)


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def _prepare_cv_splits(groups: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Need at least two unique years for cross-validation")
    splitter = GroupKFold(n_splits=unique_groups.size)
    indices = np.arange(groups.size)
    return [(train, test) for train, test in splitter.split(indices, groups=groups)]


def _standardise(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    train_s = (train - mean) / std
    test_s = (test - mean) / std
    return train_s, test_s, mean, std


def select_alpha(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, float, np.ndarray]:
    best_alpha = None
    best_r2 = -np.inf
    best_preds = None
    for alpha in alphas:
        preds = np.empty_like(y, dtype=float)
        for train_idx, test_idx in splits:
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            train_s, test_s, _, _ = _standardise(X_train, X_test)
            model = Ridge(alpha=alpha)
            model.fit(train_s, y_train)
            preds[test_idx] = model.predict(test_s)
        score = r2_score(y, preds)
        if score > best_r2:
            best_r2 = score
            best_alpha = alpha
            best_preds = preds.copy()
    if best_alpha is None or best_preds is None:
        raise RuntimeError("Unable to determine best alpha")
    return float(best_alpha), float(best_r2), best_preds


def cross_validated_r2(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    alpha: float,
) -> float:
    preds = np.empty_like(y, dtype=float)
    for train_idx, test_idx in splits:
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        train_s, test_s, _, _ = _standardise(X_train, X_test)
        model = Ridge(alpha=alpha)
        model.fit(train_s, y_train)
        preds[test_idx] = model.predict(test_s)
    return float(r2_score(y, preds))


def fit_final_model(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_s = (X - mean) / std
    model = Ridge(alpha=alpha)
    model.fit(X_s, y)
    coef = model.coef_ / std
    intercept = float(model.intercept_ - np.dot(coef, mean))
    return coef.astype(float), intercept, mean, std


# ---------------------------------------------------------------------------
# Permutation utilities
# ---------------------------------------------------------------------------

def block_permutation_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if block <= 1 or n <= block:
        return rng.permutation(n)
    blocks = [np.arange(i, min(i + block, n)) for i in range(0, n, block)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)


# ---------------------------------------------------------------------------
# Region utilities
# ---------------------------------------------------------------------------

def region_subset(da: xr.DataArray, box: Sequence[float]) -> xr.DataArray:
    lon_min, lon_max, lat_min, lat_max = box
    lat_name = "lat" if "lat" in da.coords else "latitude"
    lon_name = "lon" if "lon" in da.coords else "longitude"
    da = da.sortby(lat_name)
    da = standardise_longitude(da)
    da = da.sel({lat_name: slice(lat_min, lat_max)})
    if lon_min <= lon_max:
        return da.sel({lon_name: slice(lon_min, lon_max)})
    left = da.sel({lon_name: slice(lon_min, 180)})
    right = da.sel({lon_name: slice(-180, lon_max)})
    return xr.concat([left, right], dim=lon_name).sortby(lon_name)


def area_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    lat_name = "lat" if "lat" in da.coords else "latitude"
    lon_name = "lon" if "lon" in da.coords else "longitude"
    lat_vals = da[lat_name].values
    lon_vals = da[lon_name].values
    weights = xr.DataArray(
        np.cos(np.deg2rad(lat_vals))[:, None] * np.ones((lat_vals.size, lon_vals.size)),
        coords={lat_name: lat_vals, lon_name: lon_vals},
        dims=(lat_name, lon_name),
    )
    return da.weighted(weights).mean(dim=(lat_name, lon_name))


# ---------------------------------------------------------------------------
# Main analysis routine
# ---------------------------------------------------------------------------

def process_seasonality(
    args: argparse.Namespace,
    seasonality: str,
    regions: Mapping[str, Sequence[float]],
    logger: logging.Logger,
) -> None:
    data_root = Path(args.data_root)
    yvar = args.yvar.upper()
    variables = list(FACTOR_NAMES) + [yvar]

    base_season_cache: Optional[Dict[str, xr.DataArray]] = None
    if seasonality.lower() == "deseason":
        base_season_cache = load_panel_group(data_root, "season", variables)

    panels = load_panel_group(data_root, seasonality, variables, base_cache=base_season_cache)

    lon_lat_range = parse_lon_lat_range(args.lon_lat_range)
    time_window = parse_window(args.window)

    for key, da in panels.items():
        da = standardise_longitude(da)
        da = subset_lon_lat(da, lon_lat_range)
        da = subset_time(da, time_window)
        panels[key] = da

    time_coord = panels[yvar].coords["time"]
    years = pd.to_datetime(time_coord.values).year.astype(int)

    lat_name = "lat" if "lat" in panels[yvar].coords else "latitude"
    lon_name = "lon" if "lon" in panels[yvar].coords else "longitude"
    lat = panels[yvar][lat_name].values
    lon = panels[yvar][lon_name].values

    sst = panels["SST"]
    ocean_mask = np.isfinite(sst.mean("time").values)
    lat_mask = (np.abs(lat) <= 60.0)[:, None]
    mask = ocean_mask & lat_mask

    size = args.neigh
    factor_neigh: Dict[str, np.ndarray] = {}
    for factor in FACTOR_NAMES:
        factor_neigh[factor] = build_neighbourhood(panels[factor], size)

    all_features = np.concatenate([factor_neigh[f] for f in FACTOR_NAMES], axis=-1)
    target = panels[yvar].values

    n_time, n_lat, n_lon, n_features = all_features.shape
    logger.info(
        "Seasonality %s: loaded data with %d months, %d lat × %d lon, %d features",
        seasonality,
        n_time,
        n_lat,
        n_lon,
        n_features,
    )

    sensitivities = np.full((len(FACTOR_NAMES), n_lat, n_lon), np.nan, dtype=float)
    delta_r2 = np.full((len(FACTOR_NAMES), n_lat, n_lon), np.nan, dtype=float)
    base_r2 = np.full((n_lat, n_lon), np.nan, dtype=float)
    best_alpha = np.full((n_lat, n_lon), np.nan, dtype=float)
    sample_count = np.zeros((n_lat, n_lon), dtype=int)

    rng = np.random.default_rng(args.seed)
    alphas = sorted(parse_float_list(args.alphas))
    if not alphas:
        raise ValueError("At least one alpha value must be provided")

    lon_lat_mask_indices = np.argwhere(mask)
    logger.info("Evaluating ridge models on %d ocean grid cells", lon_lat_mask_indices.shape[0])

    for lat_idx, lon_idx in lon_lat_mask_indices:
        X_point = all_features[:, lat_idx, lon_idx, :]
        y_point = target[:, lat_idx, lon_idx]
        valid = np.isfinite(y_point) & np.all(np.isfinite(X_point), axis=1)
        if valid.sum() < 24:  # require at least two years of monthly data
            continue
        X_valid = X_point[valid]
        y_valid = y_point[valid]
        groups = years[valid]
        try:
            splits = _prepare_cv_splits(groups)
        except ValueError:
            continue
        alpha, r2_base, _ = select_alpha(X_valid, y_valid, groups, alphas, splits)
        coef, _, _, _ = fit_final_model(X_valid, y_valid, alpha)
        best_alpha[lat_idx, lon_idx] = alpha
        base_r2[lat_idx, lon_idx] = r2_base
        sample_count[lat_idx, lon_idx] = int(valid.sum())
        for factor_idx, factor in enumerate(FACTOR_NAMES):
            start = factor_idx * (size * size)
            end = (factor_idx + 1) * (size * size)
            sensitivities[factor_idx, lat_idx, lon_idx] = coef[start:end].sum()
        if args.shuffles > 0:
            for factor_idx in range(len(FACTOR_NAMES)):
                start = factor_idx * (size * size)
                end = (factor_idx + 1) * (size * size)
                permuted_deltas: List[float] = []
                for _ in range(args.shuffles):
                    perm_idx = block_permutation_indices(X_valid.shape[0], args.block, rng)
                    permuted = X_valid.copy()
                    permuted[:, start:end] = X_valid[perm_idx, start:end]
                    r2_perm = cross_validated_r2(permuted, y_valid, splits, alpha)
                    permuted_deltas.append(r2_base - r2_perm)
                if permuted_deltas:
                    delta_r2[factor_idx, lat_idx, lon_idx] = float(np.mean(permuted_deltas))
        else:
            delta_r2[:, lat_idx, lon_idx] = np.nan

    # Build xarray objects for output
    coords = {"factor": list(FACTOR_NAMES), lat_name: lat, lon_name: lon}
    sens_da = xr.DataArray(
        sensitivities,
        coords=coords,
        dims=("factor", lat_name, lon_name),
        name="sensitivity",
    )
    delta_da = xr.DataArray(
        delta_r2,
        coords=coords,
        dims=("factor", lat_name, lon_name),
        name="delta_r2",
    )
    r2_da = xr.DataArray(
        base_r2,
        coords={lat_name: lat, lon_name: lon},
        dims=(lat_name, lon_name),
        name="r2_base",
    )
    alpha_da = xr.DataArray(
        best_alpha,
        coords={lat_name: lat, lon_name: lon},
        dims=(lat_name, lon_name),
        name="alpha",
    )
    n_da = xr.DataArray(
        sample_count,
        coords={lat_name: lat, lon_name: lon},
        dims=(lat_name, lon_name),
        name="n_samples",
    )

    finite_r2 = base_r2[np.isfinite(base_r2)]
    if finite_r2.size:
        median_r2 = float(np.median(finite_r2))
        p25 = float(np.percentile(finite_r2, 25))
        p75 = float(np.percentile(finite_r2, 75))
    else:
        median_r2 = float("nan")
        p25 = float("nan")
        p75 = float("nan")

    valid_samples = sample_count[sample_count > 0]
    if valid_samples.size:
        sample_median = float(np.median(valid_samples))
    else:
        sample_median = float("nan")

    logger.info(
        "Seasonality %s: base R² median %.3f (25th=%.3f, 75th=%.3f)",
        seasonality,
        median_r2,
        p25,
        p75,
    )
    if not np.isnan(sample_median):
        logger.info("Seasonality %s: median valid samples per grid %.1f", seasonality, sample_median)

    out_dir = Path(args.out)
    sens_path = out_dir / f"sensitivities_{seasonality.lower()}_{yvar}.nc"
    delta_path = out_dir / f"deltaR2_{seasonality.lower()}_{yvar}.nc"
    if not args.dry_run:
        ensure_parent(sens_path)
        ensure_parent(delta_path)
        xr.Dataset(
            {
                "sensitivity": sens_da,
                "alpha": alpha_da,
                "r2_base": r2_da,
                "n_samples": n_da,
            }
        ).to_netcdf(sens_path)
        xr.Dataset({"delta_r2": delta_da}).to_netcdf(delta_path)

    # Regional summaries
    region_rows: List[Dict[str, object]] = []
    delta_rows: List[Dict[str, object]] = []
    r2_rows: List[Dict[str, object]] = []
    for region, box in regions.items():
        sens_region = area_weighted_mean(region_subset(sens_da, box))
        delta_region = area_weighted_mean(region_subset(delta_da, box))
        r2_region = area_weighted_mean(region_subset(r2_da, box))
        for factor in FACTOR_NAMES:
            region_rows.append(
                {
                    "region": region,
                    "factor": factor,
                    "seasonality": seasonality.upper(),
                    "yvar": yvar,
                    "sensitivity": float(sens_region.sel(factor=factor)),
                    "base_r2": float(r2_region),
                }
            )
            delta_rows.append(
                {
                    "region": region,
                    "factor": factor,
                    "seasonality": seasonality.upper(),
                    "yvar": yvar,
                    "delta_r2": float(delta_region.sel(factor=factor)),
                }
            )
        r2_rows.append(
            {
                "region": region,
                "seasonality": seasonality.upper(),
                "yvar": yvar,
                "base_r2": float(r2_region),
            }
        )

    if not args.dry_run:
        tables_dir = Path("tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        summary_path = tables_dir / f"ccf_region_summary_{seasonality.lower()}_{yvar}.csv"
        delta_csv_path = tables_dir / f"ccf_deltaR2_{seasonality.lower()}_{yvar}.csv"
        base_csv_path = tables_dir / f"ccf_baseR2_{seasonality.lower()}_{yvar}.csv"
        pd.DataFrame(region_rows).to_csv(summary_path, index=False)
        pd.DataFrame(delta_rows).to_csv(delta_csv_path, index=False)
        pd.DataFrame(r2_rows).drop_duplicates().to_csv(base_csv_path, index=False)

    if not args.dry_run:
        log_dir = Path("logs")
        log_path = log_dir / f"ccf_analysis_{seasonality.lower()}_{yvar}.log"
        ensure_parent(log_path)
        with log_path.open("w", encoding="utf8") as log_file:
            log_file.write(
                f"Seasonality: {seasonality}\nTarget: {yvar}\n"
                f"Grid cells evaluated: {int(np.isfinite(base_r2).sum())}\n"
                f"Median R2: {median_r2:.3f}\n"
                f"R2 IQR: {p25:.3f}–{p75:.3f}\n"
                f"Sample count median: {sample_median:.1f}\n"
            )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    seasonality_options = [args.seasonality.upper()]
    if args.seasonality.upper() == "BOTH":
        seasonality_options = ["SEASON", "DESEASON"]

    regions = load_regions(parse_str_list(args.regions))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("ccf_analysis")

    if args.cmip_csv:
        cmip_path = Path(args.cmip_csv)
        if cmip_path.exists():
            logger.info("CMIP CSV provided: %s (%.1f MB)", cmip_path, cmip_path.stat().st_size / 1e6)
        else:
            logger.warning("CMIP CSV path does not exist: %s", cmip_path)

    for seasonality in seasonality_options:
        logger.info("Starting analysis for seasonality: %s", seasonality)
        process_seasonality(args, seasonality, regions, logger)
        logger.info("Completed analysis for seasonality: %s", seasonality)


if __name__ == "__main__":
    main()

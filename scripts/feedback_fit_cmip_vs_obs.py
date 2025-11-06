#!/usr/bin/env python3
"""Compute OBS vs CMIP6 AMIP feedback fits over 2003–2014 with manifest lookup."""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scripts.io_manifest import load_manifest, resolve_from_manifest
from scripts.path_constants import (
    DEFAULT_FIG_DIRS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TABLES_DIR,
    ensure_dir,
    first_existing,
    make_parent,
)

DEFAULT_REGIONS = ["SEP", "SEA", "NEP", "NEA", "SEI"]
REGION_COLORS = {
    "SEP": "#1f77b4",
    "SEA": "#ff7f0e",
    "NEP": "#2ca02c",
    "NEA": "#d62728",
    "SEI": "#9467bd",
}


@dataclass(frozen=True)
class MetricConfig:
    key: str
    obs_y_vars: Sequence[str]
    obs_x_vars: Sequence[str]
    cmip_y_vars: Sequence[str]
    cmip_x_vars: Sequence[str]
    title: str
    unit_label: str
    scale_factor: float


METRIC_CONFIG: Dict[str, MetricConfig] = {
    "lambda_cld_sw": MetricConfig(
        key="lambda_cld_sw",
        obs_y_vars=("swcre", "clswlow"),
        obs_x_vars=("ts", "tas"),
        cmip_y_vars=("swcre", "clswlow"),
        cmip_x_vars=("ts", "tas"),
        title="λ_cld,SW",
        unit_label="W m$^{-2}$ K$^{-1}$",
        scale_factor=1.0,
    ),
    "dlcf_deis": MetricConfig(
        key="dLCF_dEIS",
        obs_y_vars=("cllmodis", "lcf"),
        obs_x_vars=("eislts", "eis", "lts"),
        cmip_y_vars=("lcf", "cllmodis"),
        cmip_x_vars=("eis", "lts", "eislts"),
        title="dLCF/dEIS",
        unit_label="% K$^{-1}$",
        scale_factor=100.0,
    ),
}

METRIC_ALIASES = {
    "lambda_sw": "lambda_cld_sw",
    "dlcf_deis": "dLCF_dEIS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OBS vs CMIP6 AMIP feedback fits (2003–2014)."
    )
    parser.add_argument(
        "--metrics",
        help="Comma separated list of metrics (e.g., lambda_cld_sw,dLCF_dEIS).",
    )
    parser.add_argument(
        "--metric",
        help="Single metric (legacy flag; equivalent to providing --metrics).",
    )
    parser.add_argument(
        "--regions",
        help="Comma separated list of regions (default: configured regions).",
    )
    parser.add_argument(
        "--cmip-csv",
        help="Explicit path to CMIP6 AMIP stacked monthly table (optional).",
    )
    parser.add_argument(
        "--hac-lag",
        type=int,
        default=12,
        help="Maximum lag for Newey–West HAC estimator (default: 12).",
    )
    parser.add_argument(
        "--hac-lags",
        type=int,
        help="Alias for --hac-lag (legacy flag).",
    )
    parser.add_argument(
        "--deseasonalize",
        dest="deseasonalize",
        action="store_true",
        default=True,
        help="Remove monthly climatology before regression (default: enabled).",
    )
    parser.add_argument(
        "--no-deseasonalize",
        dest="deseasonalize",
        action="store_false",
        help="Disable deseasonalization before regression.",
    )
    parser.add_argument(
        "--no-intercept",
        action="store_true",
        help="Fit regressions without an intercept term.",
    )
    parser.add_argument(
        "--manifest",
        help="Optional path to output_manifest.json for artifact resolution.",
    )
    parser.add_argument(
        "--tables-dir",
        help="Directory where summary tables are written (default: tables).",
    )
    parser.add_argument(
        "--fig-dir",
        help="Directory where figures are written (default: fig/figures).",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory where logs are written (default: logs).",
    )
    return parser.parse_args()


def parse_metric_keys(metrics_arg: Optional[str], metric_arg: Optional[str]) -> List[str]:
    requested: List[str] = []
    for raw in (metrics_arg, metric_arg):
        if not raw:
            continue
        for piece in raw.split(","):
            piece = piece.strip()
            if not piece:
                continue
            requested.append(piece)

    if not requested:
        raise ValueError("At least one metric must be provided via --metrics/--metric.")

    normalized: List[str] = []
    seen = set()
    for key in requested:
        lookup = METRIC_ALIASES.get(key.lower(), key)
        canonical = lookup if lookup in METRIC_CONFIG else lookup.lower()
        if canonical not in METRIC_CONFIG:
            raise ValueError(f"Unknown metric '{key}'.")
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def load_regions(regions_arg: Optional[str]) -> List[str]:
    if regions_arg:
        return [r.strip().upper() for r in regions_arg.split(",") if r.strip()]

    cfg_path = Path("configs/regions.yaml")
    if cfg_path.exists():
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(cfg_path.read_text())
            if isinstance(data, dict):
                return [str(key).upper() for key in data.keys()]
        except Exception:
            pass

    cfg_yaml = Path("configs/config.yaml")
    if cfg_yaml.exists():
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(cfg_yaml.read_text())
            if isinstance(data, dict):
                if "regions" in data and isinstance(data["regions"], dict):
                    return [str(k).upper() for k in data["regions"].keys()]
                project = data.get("project")
                if isinstance(project, dict) and isinstance(project.get("regions"), list):
                    return [str(r).upper() for r in project["regions"]]
        except Exception:
            pass

    return DEFAULT_REGIONS.copy()


def setup_logger(logs_dir: Path) -> Tuple[logging.Logger, Path]:
    ensure_dir(logs_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"feedback_fit_{timestamp}.log"
    logger = logging.getLogger("feedback_fit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_path


def case_insensitive_path(path: Path) -> Optional[Path]:
    if path.exists():
        return path
    parent = path.parent
    if not parent.exists():
        return None
    target = path.name.lower()
    for child in parent.iterdir():
        if child.name.lower() == target:
            return child
    return None


def resolve_obs_path(
    manifest: Optional[Dict[str, object]],
    var: str,
    region: str,
    logger: logging.Logger,
) -> Tuple[Optional[Path], List[str]]:
    placeholders = {"REGION": region, "var": var}
    manifest_paths = resolve_from_manifest(
        manifest,
        {
            "artifact_type": "table",
            "path_like_contains": ["regional_monthly", "<REGION>", "<var>"],
            "placeholders": placeholders,
        },
    )

    candidates: List[Path] = [Path(path) for path in manifest_paths]
    fallback = Path(DEFAULT_OUTPUT_DIR) / "regional_monthly" / f"{var}_mean_2003-2014_{region}.csv"
    candidates.append(fallback)
    candidates.append(fallback.with_name(fallback.name.lower()))

    tried: List[str] = []
    for candidate in candidates:
        candidate_path = candidate if candidate.is_absolute() else Path(candidate)
        tried.append(str(candidate_path))
        resolved = case_insensitive_path(candidate_path)
        if resolved and resolved.exists():
            logger.info(
                "Resolved OBS path for var '%s' region '%s': %s", var, region, resolved
            )
            return resolved, tried

    return None, tried


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen: set[str] = set()
    ordered: List[Path] = []
    for path in paths:
        normalized = str(Path(path))
        if normalized not in seen:
            ordered.append(Path(path))
            seen.add(normalized)
    return ordered


def gather_cmip_candidates(
    metric_cfg: MetricConfig,
    manifest: Optional[Dict[str, object]],
    override: Optional[str],
) -> List[Path]:
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override))

    manifest_paths = resolve_from_manifest(
        manifest,
        {
            "artifact_type": "table",
            "path_like_contains": ["cmip", "amip"],
        },
    )
    metric_tokens = [metric_cfg.key.lower()] + [v.lower() for v in metric_cfg.cmip_y_vars]
    metric_tokens += [v.lower() for v in metric_cfg.cmip_x_vars]
    for path_str in manifest_paths:
        lower_path = path_str.lower()
        if any(token in lower_path for token in metric_tokens):
            candidates.append(Path(path_str))

    for path_str in manifest_paths:
        candidates.append(Path(path_str))

    output_dir = Path(DEFAULT_OUTPUT_DIR)
    candidates.append(output_dir / "cmip_amip_monthly_2003-2014.csv")
    candidates.append(output_dir / "cmip_amip_vs_obs_2003-2014.csv")
    candidates.append(output_dir / "cmip_amip_monthly.csv")
    candidates.append(output_dir / "cmip_amip_vs_obs.csv")
    for var in metric_cfg.cmip_y_vars:
        candidates.append(output_dir / f"cmip_amip_{var}_vs_obs_2003-2014.csv")
    for var in metric_cfg.cmip_x_vars:
        candidates.append(output_dir / f"cmip_amip_{var}_vs_obs_2003-2014.csv")

    for pattern in [
        "cmip_amip*_2003-2014.csv",
        "cmip_amip*monthly*.csv",
        "cmip_amip*vs_obs*.csv",
    ]:
        candidates.extend(output_dir.glob(pattern))
    return unique_paths(candidates)


def try_load_cmip_table(path: Path, metric_cfg: MetricConfig) -> Tuple[Optional[pd.DataFrame], str]:
    if not path.exists():
        return None, "file not found"

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - logging detail
        return None, f"failed to read CSV ({exc})"

    if df.empty:
        return None, "table is empty"

    columns_lower = {c.lower(): c for c in df.columns}

    def get_column(candidates: Sequence[str], required: bool = True) -> Optional[str]:
        for candidate in candidates:
            key = candidate.lower()
            if key in columns_lower:
                return columns_lower[key]
        if required:
            raise KeyError(f"Missing columns {candidates}")
        return None

    try:
        rename_map = {
            get_column(["model"]): "model",
            get_column(["region"]): "region",
        }
    except KeyError as exc:
        return None, str(exc)

    time_col = None
    for candidate in ("time", "date"):
        if candidate in columns_lower:
            time_col = columns_lower[candidate]
            break
    if time_col:
        rename_map[time_col] = "time"
    else:
        if "month" in columns_lower:
            rename_map[columns_lower["month"]] = "month"
        if "year" in columns_lower:
            rename_map[columns_lower["year"]] = "year"

    try:
        y_col = get_column(metric_cfg.cmip_y_vars)
        x_col = get_column(metric_cfg.cmip_x_vars)
    except KeyError as exc:
        return None, str(exc)

    rename_map[y_col] = "y"
    rename_map[x_col] = "x"

    family_col = get_column(["family"], required=False)
    if family_col:
        rename_map[family_col] = "family"

    df = df.rename(columns=rename_map)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    elif {"year", "month"}.issubset(df.columns):
        df["time"] = pd.to_datetime(
            {
                "year": pd.to_numeric(df["year"], errors="coerce").astype("Int64"),
                "month": pd.to_numeric(df["month"], errors="coerce").astype("Int64"),
                "day": 1,
            }
        )
    else:
        return None, "no parseable time information"

    df = df.dropna(subset=["time"])
    df["model"] = df["model"].astype(str)
    df["region"] = df["region"].astype(str)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df[(df["time"].dt.year >= 2003) & (df["time"].dt.year <= 2014)]

    keep_cols = ["model", "region", "time", "y", "x"]
    if "family" in df.columns:
        keep_cols.append("family")
    df = df[keep_cols]
    df["region"] = df["region"].str.upper()
    df = df.sort_values(["model", "region", "time"]).reset_index(drop=True)
    return df, ""


def load_cmip_for_metric(
    metric_cfg: MetricConfig,
    manifest: Optional[Dict[str, object]],
    override: Optional[str],
    logger: logging.Logger,
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    candidates = gather_cmip_candidates(metric_cfg, manifest, override)
    errors: List[str] = []
    for candidate in candidates:
        df, err = try_load_cmip_table(candidate, metric_cfg)
        if df is not None:
            logger.info(
                "Loaded CMIP data for metric %s from %s", metric_cfg.key, candidate
            )
            return df, [str(candidate)]
        errors.append(f"{candidate}: {err}")
    logger.error(
        "Unable to locate CMIP data for %s. Tried paths: %s",
        metric_cfg.key,
        "; ".join(str(p) for p in candidates) if candidates else "<none>",
    )
    for message in errors:
        logger.error("  %s", message)
    return None, [str(p) for p in candidates]


def read_monthly_csv(path: Path, logger: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        logger.warning("%s is empty", path)
        return pd.DataFrame(columns=["time", "value"])

    columns_lower = {c.lower(): c for c in df.columns}

    time_col = None
    for candidate in ("time", "date", "month"):
        if candidate in columns_lower:
            time_col = columns_lower[candidate]
            break
    if time_col is None:
        time_col = df.columns[0]

    value_col = None
    for candidate in ("value", "val", "mean", "data", "series", "climate"):
        if candidate in columns_lower:
            value_col = columns_lower[candidate]
            break
    if value_col is None:
        non_time = [c for c in df.columns if c != time_col]
        if not non_time:
            raise ValueError(f"Unable to identify value column in {path}.")
        value_col = non_time[-1]

    df = df[[time_col, value_col]].copy()
    df.columns = ["time", "value"]

    time_parsed = pd.to_datetime(df["time"], errors="coerce")
    if time_parsed.isna().all():
        columns_lower = {c.lower(): c for c in df.columns}
        if {"year", "month"}.issubset(columns_lower):
            year_col = columns_lower["year"]
            month_col = columns_lower["month"]
            df["time"] = pd.to_datetime(
                {
                    "year": pd.to_numeric(df[year_col], errors="coerce").astype("Int64"),
                    "month": pd.to_numeric(df[month_col], errors="coerce").astype("Int64"),
                    "day": 1,
                }
            )
        else:
            raise ValueError(
                f"Time column in {path} could not be parsed to datetime; provide YYYY-MM format."
            )
    else:
        df["time"] = time_parsed

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time", "value"]).sort_values("time")
    df = df[(df["time"].dt.year >= 2003) & (df["time"].dt.year <= 2014)]
    df = df.groupby("time", as_index=False)["value"].mean()

    logger.info("Read %s: %d monthly records (2003–2014 filter).", path.name, len(df))
    return df


def prepare_series(df: pd.DataFrame, deseasonalize: bool) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    series = df.set_index("time")["value"].sort_index()
    if deseasonalize:
        climatology = series.groupby(series.index.month).transform("mean")
        series = series - climatology
    return series


def build_anomaly_pair(
    y_df: pd.DataFrame,
    x_df: pd.DataFrame,
    dataset_label: str,
    deseasonalize: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    y_series = prepare_series(y_df, deseasonalize)
    x_series = prepare_series(x_df, deseasonalize)
    paired = pd.concat([y_series.rename("y"), x_series.rename("x")], axis=1).dropna()
    logger.info(
        "Built monthly anomalies over 2003–2014 for %s: n=%d", dataset_label, len(paired)
    )
    return paired


def run_hac_regression(
    paired: pd.DataFrame,
    dataset_label: str,
    hac_lag: int,
    scale_factor: float,
    include_intercept: bool,
    logger: logging.Logger,
) -> Optional[Dict[str, float]]:
    n_samples = len(paired)
    if n_samples < 24:
        logger.warning(
            "%s: insufficient paired samples after anomaly alignment (n=%d).",
            dataset_label,
            n_samples,
        )
        return None

    if include_intercept:
        X = sm.add_constant(paired["x"].values)
    else:
        X = paired[["x"]].values
    model = sm.OLS(paired["y"].values, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lag}
    )

    slope_index = 1 if include_intercept else 0
    slope = float(model.params[slope_index]) * scale_factor
    se_hac = float(model.bse[slope_index]) * scale_factor
    t_val = float(model.tvalues[slope_index])
    p_val = float(model.pvalues[slope_index])
    r_sq = float(model.rsquared)

    logger.info(
        "%s OLS(HAC lags=%d): b=%.4f, SE=%.4f, t=%.3f, p=%.3f, R2=%.3f, n=%d",
        dataset_label,
        hac_lag,
        slope,
        se_hac,
        t_val,
        p_val,
        r_sq,
        n_samples,
    )

    return {
        "b": slope,
        "SE_HAC": se_hac,
        "t": t_val,
        "p": p_val,
        "R2": r_sq,
        "n": n_samples,
        "sign": bool(p_val < 0.05),
    }


def process_region(
    region: str,
    metric_cfg: MetricConfig,
    manifest: Optional[Dict[str, object]],
    cmip_df: pd.DataFrame,
    tables_dir: Path,
    deseasonalize: bool,
    include_intercept: bool,
    hac_lag: int,
    logger: logging.Logger,
) -> Optional[Tuple[Dict[str, float], List[Dict[str, float]], Dict[str, Optional[str]]]]:
    logger.info("Processing region %s for metric %s.", region, metric_cfg.key)

    obs_y_path = None
    y_tried: List[str] = []
    for var in metric_cfg.obs_y_vars:
        obs_y_path, tried = resolve_obs_path(manifest, var, region, logger)
        y_tried.extend(tried)
        if obs_y_path:
            break
    if obs_y_path is None:
        logger.error(
            "Unable to locate OBS Y data for %s (vars=%s). Tried paths: %s",
            region,
            ",".join(metric_cfg.obs_y_vars),
            "; ".join(y_tried) if y_tried else "<none>",
        )
        return None

    obs_x_path = None
    x_tried: List[str] = []
    for var in metric_cfg.obs_x_vars:
        obs_x_path, tried = resolve_obs_path(manifest, var, region, logger)
        x_tried.extend(tried)
        if obs_x_path:
            break
    if obs_x_path is None:
        logger.error(
            "Unable to locate OBS X data for %s (vars=%s). Tried paths: %s",
            region,
            ",".join(metric_cfg.obs_x_vars),
            "; ".join(x_tried) if x_tried else "<none>",
        )
        return None

    obs_y_df = read_monthly_csv(obs_y_path, logger)
    obs_x_df = read_monthly_csv(obs_x_path, logger)

    obs_pair = build_anomaly_pair(obs_y_df, obs_x_df, "OBS", deseasonalize, logger)
    obs_result = run_hac_regression(
        obs_pair,
        "OBS",
        hac_lag,
        metric_cfg.scale_factor,
        include_intercept,
        logger,
    )
    if obs_result is None:
        logger.error("Failed to compute OBS regression for %s due to insufficient data.", region)
        return None

    rows: List[Dict[str, object]] = [
        {
            "dataset": "obs",
            "model": "OBS",
            "region": region,
            "metric": metric_cfg.key,
            **obs_result,
        }
    ]

    region_df = cmip_df[cmip_df["region"] == region]
    model_rows: List[Dict[str, float]] = []
    plot_info: Dict[str, Optional[str]] = {}

    if region_df.empty:
        logger.warning("No CMIP entries found for region %s.", region)
    else:
        for model in sorted(region_df["model"].unique()):
            model_data = region_df[region_df["model"] == model]
            y_df = model_data[["time", "y"]].rename(columns={"y": "value"})
            x_df = model_data[["time", "x"]].rename(columns={"x": "value"})
            pair = build_anomaly_pair(y_df, x_df, model, deseasonalize, logger)
            result = run_hac_regression(
                pair,
                model,
                hac_lag,
                metric_cfg.scale_factor,
                include_intercept,
                logger,
            )
            if result is None:
                continue
            rows.append(
                {
                    "dataset": "cmip",
                    "model": model,
                    "region": region,
                    "metric": metric_cfg.key,
                    **result,
                }
            )
            model_rows.append({"model": model, **result})
            if "family" in model_data.columns:
                fam_series = model_data["family"].dropna().astype(str)
                plot_info[model] = fam_series.mode().iloc[0] if not fam_series.empty else None
            else:
                plot_info[model] = None

    table_path = tables_dir / f"feedback_fit_{metric_cfg.key}_{region}.csv"
    ensure_dir(table_path.parent)
    table_df = pd.DataFrame(rows)[
        ["dataset", "model", "region", "metric", "b", "SE_HAC", "t", "p", "R2", "n", "sign"]
    ]
    table_df["dataset"] = pd.Categorical(
        table_df["dataset"], categories=["obs", "cmip"], ordered=True
    )
    table_df = table_df.sort_values(["dataset", "model"]).reset_index(drop=True)
    table_df.to_csv(table_path, index=False)
    logger.info("Wrote table: %s", table_path)

    return obs_result, model_rows, plot_info


def make_scatter(
    metric_cfg: MetricConfig,
    regions: List[str],
    obs_results: Dict[str, Dict[str, float]],
    model_results: Dict[str, List[Dict[str, float]]],
    plot_infos: Dict[str, Dict[str, Optional[str]]],
    fig_path: Path,
    logger: logging.Logger,
) -> None:
    if not regions:
        return

    make_parent(fig_path)
    fig, ax = plt.subplots(figsize=(6, 6))

    all_values: List[float] = []

    if len(regions) == 1:
        region = regions[0]
        obs_row = obs_results[region]
        model_rows = model_results.get(region, [])
        obs_b = obs_row["b"]
        obs_se = obs_row["SE_HAC"]
        families = [plot_infos.get(region, {}).get(row["model"]) for row in model_rows]
        unique_families = [f for f in sorted({fam for fam in families if fam})]

        if model_rows:
            if unique_families:
                cmap = plt.get_cmap("tab10")
                color_map = {fam: cmap(i % cmap.N) for i, fam in enumerate(unique_families)}
                labels_added: set[str] = set()
                for row, fam in zip(model_rows, families):
                    color = color_map.get(fam, REGION_COLORS.get(region, "#1f77b4"))
                    label = None
                    if fam and fam not in labels_added:
                        label = fam
                        labels_added.add(fam)
                    ax.errorbar(
                        obs_b,
                        row["b"],
                        xerr=obs_se,
                        yerr=row["SE_HAC"],
                        fmt="o",
                        color=color,
                        ecolor=color,
                        elinewidth=1.0,
                        capsize=3,
                        label=label,
                    )
                    ax.annotate(
                        row["model"],
                        (obs_b, row["b"]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                    )
                if labels_added:
                    ax.legend(title="Model family", loc="best")
            else:
                color = REGION_COLORS.get(region, "#1f77b4")
                for row in model_rows:
                    ax.errorbar(
                        obs_b,
                        row["b"],
                        xerr=obs_se,
                        yerr=row["SE_HAC"],
                        fmt="o",
                        color=color,
                        ecolor=color,
                        elinewidth=1.0,
                        capsize=3,
                    )
                    ax.annotate(
                        row["model"],
                        (obs_b, row["b"]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                    )
        else:
            color = REGION_COLORS.get(region, "#1f77b4")
            ax.errorbar(
                obs_b,
                obs_b,
                xerr=obs_se,
                yerr=obs_se,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=3,
                label="OBS",
            )
            ax.annotate(
                "OBS",
                (obs_b, obs_b),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        all_values.extend([obs_b] + [row["b"] for row in model_rows])
    else:
        legend_handles = {}
        for region in regions:
            obs_row = obs_results.get(region)
            if not obs_row:
                continue
            obs_b = obs_row["b"]
            obs_se = obs_row["SE_HAC"]
            models = model_results.get(region, [])
            color = REGION_COLORS.get(region, "#1f77b4")
            label = None
            if region not in legend_handles:
                label = region
                legend_handles[region] = color
            if models:
                for row in models:
                    ax.errorbar(
                        obs_b,
                        row["b"],
                        xerr=obs_se,
                        yerr=row["SE_HAC"],
                        fmt="o",
                        color=color,
                        ecolor=color,
                        elinewidth=1.0,
                        capsize=3,
                        label=label,
                    )
                    ax.annotate(
                        f"{row['model']} ({region})",
                        (obs_b, row["b"]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                    )
                    label = None
            else:
                ax.errorbar(
                    obs_b,
                    obs_b,
                    xerr=obs_se,
                    yerr=obs_se,
                    fmt="s",
                    color=color,
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=3,
                    label=label,
                )
            all_values.append(obs_b)
            all_values.extend([row["b"] for row in models])

        if legend_handles:
            ax.legend(loc="best", title="Region")

    if not all_values:
        logger.warning("No data available for plotting %s; skipping figure.", metric_cfg.key)
        plt.close(fig)
        return

    vmin = float(np.nanmin(all_values))
    vmax = float(np.nanmax(all_values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = -1.0, 1.0
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    padding = (vmax - vmin) * 0.05
    vmin -= padding
    vmax += padding
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", color="gray", linewidth=1.0)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel(f"OBS {metric_cfg.title} ({metric_cfg.unit_label})")
    ax.set_ylabel(f"CMIP {metric_cfg.title} ({metric_cfg.unit_label})")
    ax.set_title(
        f"{metric_cfg.title} comparison ({', '.join(regions)}, 2003–2014)"
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info("Saved figure: %s", fig_path)


def main() -> None:
    args = parse_args()
    try:
        metric_keys = parse_metric_keys(args.metrics, args.metric)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    regions = load_regions(args.regions)

    hac_lag = args.hac_lags if args.hac_lags is not None else args.hac_lag
    include_intercept = not args.no_intercept

    tables_dir = Path(args.tables_dir) if args.tables_dir else Path(DEFAULT_TABLES_DIR)
    ensure_dir(tables_dir)

    if args.fig_dir:
        fig_dir = Path(args.fig_dir)
    else:
        default_candidates = [Path(name) for name in DEFAULT_FIG_DIRS]
        fig_dir = first_existing(default_candidates) or default_candidates[0]
    ensure_dir(fig_dir)

    logs_dir = Path(args.logs_dir)
    logger, log_path = setup_logger(logs_dir)
    logger.info("Logging to %s", log_path)
    logger.info("Metrics: %s", ", ".join(metric_keys))
    logger.info("Regions: %s", ", ".join(regions))
    logger.info("Tables dir: %s", tables_dir.resolve())
    logger.info("Figures dir: %s", fig_dir.resolve())

    manifest = load_manifest(args.manifest)
    if args.manifest:
        if manifest:
            logger.info("Loaded manifest: %s", args.manifest)
        else:
            logger.warning("Manifest not found or unreadable: %s", args.manifest)

    processed_any = False

    for metric_key in metric_keys:
        metric_cfg = METRIC_CONFIG[metric_key]
        logger.info("Starting metric %s", metric_cfg.key)
        cmip_df, tried_paths = load_cmip_for_metric(metric_cfg, manifest, args.cmip_csv, logger)
        if cmip_df is None:
            logger.error(
                "Skipping metric %s due to missing CMIP data. Tried: %s",
                metric_cfg.key,
                "; ".join(tried_paths) if tried_paths else "<none>",
            )
            continue

        obs_results: Dict[str, Dict[str, float]] = {}
        model_results: Dict[str, List[Dict[str, float]]] = {}
        plot_infos: Dict[str, Dict[str, Optional[str]]] = {}
        processed_regions: List[str] = []

        for region in regions:
            result = process_region(
                region,
                metric_cfg,
                manifest,
                cmip_df,
                tables_dir,
                args.deseasonalize,
                include_intercept,
                hac_lag,
                logger,
            )
            if result is None:
                continue
            obs_result, model_rows, plot_info = result
            obs_results[region] = obs_result
            model_results[region] = model_rows
            plot_infos[region] = plot_info
            processed_regions.append(region)

        if not processed_regions:
            logger.warning("No regions processed successfully for metric %s.", metric_cfg.key)
            continue

        processed_any = True
        regions_token = ",".join(processed_regions)
        fig_path = fig_dir / f"feedback_fit_{metric_cfg.key}_{regions_token}.png"
        make_scatter(
            metric_cfg,
            processed_regions,
            obs_results,
            model_results,
            plot_infos,
            fig_path,
            logger,
        )

    if not processed_any:
        logger.error("No metrics were processed successfully.")
        sys.exit(1)

    logger.info("Completed feedback fit processing.")


if __name__ == "__main__":
    main()

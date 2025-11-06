#!/usr/bin/env python3
"""Compute OBS vs CMIP6 AMIP feedback fits over 2003–2014.

This script estimates two metrics on a regional basis using monthly anomalies:

* ``lambda_sw``: shortwave low-cloud feedback (dSWCRE / dTs).
* ``dLCF_dEIS``: stability–low-cloud sensitivity (dLCF / dEIS).

Monthly anomalies are computed by removing the calendar-month climatology over
2003–2014 for each dataset (OBS, individual CMIP models). Ordinary least
squares regressions are fitted with Newey–West (HAC) standard errors. Outputs
include per-region tables, diagnostic scatter plots, and logs documenting the
processing steps.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


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
    obs_y: str
    obs_x: str
    cmip_y: str
    cmip_x: str
    title: str
    unit_label: str
    scale_factor: float


METRIC_CONFIG: Dict[str, MetricConfig] = {
    "lambda_sw": MetricConfig(
        key="lambda_sw",
        obs_y="swcre",
        obs_x="ts",
        cmip_y="swcre",
        cmip_x="ts",
        title="λ_cld,SW",
        unit_label="W m$^{-2}$ K$^{-1}$",
        scale_factor=1.0,
    ),
    "dLCF_dEIS": MetricConfig(
        key="dLCF_dEIS",
        obs_y="cllmodis",
        obs_x="eislts",
        cmip_y="lcf",
        cmip_x="eis",
        title="dLCF/dEIS",
        unit_label="% K$^{-1}$",
        scale_factor=100.0,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OBS vs CMIP6 AMIP feedback fits (2003–2014)."
    )
    parser.add_argument(
        "--metric",
        required=True,
        choices=list(METRIC_CONFIG.keys()),
        help="Metric to evaluate (lambda_sw or dLCF_dEIS).",
    )
    parser.add_argument(
        "--regions",
        help="Comma separated list of regions (default: all available).",
    )
    parser.add_argument(
        "--cmip-csv",
        default="output/cmip_amip_monthly_2003-2014.csv",
        help="Path to CMIP6 AMIP monthly stacked table (2003–2014).",
    )
    parser.add_argument(
        "--hac-lags",
        type=int,
        default=12,
        help="Maximum lag for Newey–West HAC estimator (default: 12).",
    )
    return parser.parse_args()


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


def setup_logger(metric: str, region: str, logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger_name = f"feedback_fit_{metric}_{region}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(logs_dir / f"feedback_fit_{metric}_{region}.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def find_case_insensitive(directory: Path, filename: str) -> Path:
    target = filename.lower()
    for path in directory.iterdir():
        if path.name.lower() == target:
            return path
    raise FileNotFoundError(f"File not found (case-insensitive match): {filename}")


def read_monthly_csv(path: Path, logger: logging.Logger) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty:
        logger.warning(f"{path} is empty.")
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

    logger.info(f"Read {path.name}: {len(df)} monthly records (2003–2014 filter).")
    return df


def compute_anomalies(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    series = df.set_index("time")["value"].sort_index()
    climatology = series.groupby(series.index.month).transform("mean")
    anomalies = series - climatology
    return anomalies


def build_anomaly_pair(
    y_df: pd.DataFrame,
    x_df: pd.DataFrame,
    dataset_label: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    y_anom = compute_anomalies(y_df)
    x_anom = compute_anomalies(x_df)
    paired = pd.concat([y_anom.rename("y"), x_anom.rename("x")], axis=1).dropna()
    logger.info(
        f"Built monthly anomalies over 2003–2014 for {dataset_label}: n={len(paired)}"
    )
    return paired


def run_hac_regression(
    paired: pd.DataFrame,
    dataset_label: str,
    hac_lags: int,
    scale_factor: float,
    logger: logging.Logger,
) -> Optional[Dict[str, float]]:
    n_samples = len(paired)
    if n_samples < 24:
        logger.warning(
            f"{dataset_label}: insufficient paired samples after anomaly alignment (n={n_samples})."
        )
        return None

    X = sm.add_constant(paired["x"].values)
    model = sm.OLS(paired["y"].values, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )

    slope = float(model.params[1]) * scale_factor
    se_hac = float(model.bse[1]) * scale_factor
    t_val = float(model.tvalues[1])
    p_val = float(model.pvalues[1])
    r_sq = float(model.rsquared)

    logger.info(
        f"{dataset_label} OLS(HAC lags={hac_lags}): "
        f"b={slope:.4f}, SE={se_hac:.4f}, t={t_val:.3f}, p={p_val:.3f}, R2={r_sq:.3f}, n={n_samples}"
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


def load_cmip_table(path: Path, metric_cfg: MetricConfig) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    columns_lower = {c.lower(): c for c in df.columns}

    def get_column(name: str) -> str:
        key = name.lower()
        if key in columns_lower:
            return columns_lower[key]
        raise KeyError(f"Column '{name}' not found in CMIP table.")

    rename_map = {
        get_column("model"): "model",
        get_column("region"): "region",
    }

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

    rename_map[get_column(metric_cfg.cmip_y)] = "y"
    rename_map[get_column(metric_cfg.cmip_x)] = "x"

    if "family" in columns_lower:
        rename_map[columns_lower["family"]] = "family"

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
        raise ValueError(
            "CMIP table must contain a parseable 'time' column or both 'year' and 'month' columns."
        )

    df["model"] = df["model"].astype(str)
    df["region"] = df["region"].astype(str)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.dropna(subset=["time"])  # drop rows with invalid dates
    df = df[(df["time"].dt.year >= 2003) & (df["time"].dt.year <= 2014)]

    keep_cols = ["model", "region", "time", "y", "x"]
    if "family" in df.columns:
        keep_cols.append("family")
    df = df[keep_cols]

    df["region"] = df["region"].str.upper()
    df = df.sort_values(["model", "region", "time"])
    return df


def make_scatter(
    region: str,
    metric_cfg: MetricConfig,
    obs_row: Dict[str, float],
    model_rows: List[Dict[str, float]],
    plot_info: Dict[str, Optional[str]],
    fig_path: Path,
    single_region: bool,
    logger: logging.Logger,
) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))

    obs_b = obs_row["b"]
    obs_se = obs_row["SE_HAC"]

    if model_rows:
        y_vals = np.array([row["b"] for row in model_rows], dtype=float)
        y_err = np.array([row["SE_HAC"] for row in model_rows], dtype=float)
        x_vals = np.full_like(y_vals, obs_b)
        x_err = np.full_like(y_vals, obs_se)

        families = [plot_info.get(row["model"], None) for row in model_rows]
        unique_families = [f for f in sorted(set(families)) if f]

        if single_region and unique_families:
            cmap = plt.get_cmap("tab10")
            color_map = {
                fam: cmap(i % cmap.N) for i, fam in enumerate(unique_families)
            }
            colors = [color_map.get(fam, REGION_COLORS.get(region, "#1f77b4")) for fam in families]
            handles = []
            labels_added = set()
            for x, y, xe, ye, color, model_name, fam in zip(
                x_vals, y_vals, x_err, y_err, colors, [row["model"] for row in model_rows], families
            ):
                ax.errorbar(
                    x,
                    y,
                    xerr=xe,
                    yerr=ye,
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=3,
                    label=fam if fam and fam not in labels_added else None,
                )
                if fam and fam not in labels_added:
                    handles.append(ax.plot([], [], "o", color=color, label=fam)[0])
                    labels_added.add(fam)
                ax.annotate(model_name, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
            if handles:
                ax.legend(handles=handles, title="Model family", loc="best")
        else:
            color = REGION_COLORS.get(region, "#1f77b4")
            ax.errorbar(
                x_vals,
                y_vals,
                xerr=x_err,
                yerr=y_err,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=3,
                linestyle="none",
            )
            for x, y, model_name in zip(x_vals, y_vals, [row["model"] for row in model_rows]):
                ax.annotate(model_name, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    else:
        ax.scatter([obs_b], [obs_b], color=REGION_COLORS.get(region, "#1f77b4"), label="OBS")
        ax.annotate("OBS", (obs_b, obs_b), textcoords="offset points", xytext=(5, 5), fontsize=8)

    all_values = np.array([obs_b] + [row["b"] for row in model_rows]) if model_rows else np.array([obs_b])
    vmin = float(np.nanmin(all_values))
    vmax = float(np.nanmax(all_values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = -1.0, 1.0
    if np.isclose(vmin, vmax):
        pad = max(1.0, abs(vmin) * 0.1 if vmin != 0 else 0.5)
        vmin -= pad
        vmax += pad
    else:
        pad = 0.1 * (vmax - vmin)
        vmin -= pad
        vmax += pad

    ax.plot([vmin, vmax], [vmin, vmax], color="k", linestyle="--", linewidth=1.0)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    ax.set_xlabel(f"OBS {metric_cfg.title} ({metric_cfg.unit_label})")
    ax.set_ylabel(f"CMIP {metric_cfg.title} ({metric_cfg.unit_label})")
    ax.set_title(f"{metric_cfg.title} comparison ({region}, 2003–2014)")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved figure: {fig_path}")


def process_region(
    region: str,
    metric_cfg: MetricConfig,
    hac_lags: int,
    cmip_df: pd.DataFrame,
    single_region: bool,
    logger: logging.Logger,
) -> None:
    logger.info(f"Processing region {region} for metric {metric_cfg.key}.")

    obs_dir = Path("output/regional_monthly")
    try:
        obs_y_path = find_case_insensitive(
            obs_dir, f"{metric_cfg.obs_y}_mean_2003-2014_{region}.csv"
        )
        obs_x_path = find_case_insensitive(
            obs_dir, f"{metric_cfg.obs_x}_mean_2003-2014_{region}.csv"
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return

    obs_y_df = read_monthly_csv(obs_y_path, logger)
    obs_x_df = read_monthly_csv(obs_x_path, logger)

    obs_pair = build_anomaly_pair(obs_y_df, obs_x_df, "OBS", logger)
    obs_result = run_hac_regression(obs_pair, "OBS", hac_lags, metric_cfg.scale_factor, logger)
    if obs_result is None:
        logger.error("Failed to compute OBS regression due to insufficient data.")
        return

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
    if region_df.empty:
        logger.warning(f"No CMIP entries found for region {region}.")
        model_rows: List[Dict[str, float]] = []
        plot_info: Dict[str, Optional[str]] = {}
    else:
        model_rows = []
        plot_info = {}
        for model in sorted(region_df["model"].unique()):
            model_data = region_df[region_df["model"] == model]
            y_df = model_data[["time", "y"]].rename(columns={"y": "value"})
            x_df = model_data[["time", "x"]].rename(columns={"x": "value"})
            pair = build_anomaly_pair(y_df, x_df, model, logger)
            result = run_hac_regression(pair, model, hac_lags, metric_cfg.scale_factor, logger)
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
                fam = (
                    model_data["family"].dropna().astype(str).mode().iloc[0]
                    if not model_data["family"].dropna().empty
                    else None
                )
                plot_info[model] = fam
            else:
                plot_info[model] = None

    table_path = Path("tables") / f"feedback_fit_{metric_cfg.key}_{region}.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_df = pd.DataFrame(rows)[
        ["dataset", "model", "region", "metric", "b", "SE_HAC", "t", "p", "R2", "n", "sign"]
    ]
    # Ensure deterministic order: OBS row first, then models alphabetical
    table_df["dataset"] = pd.Categorical(
        table_df["dataset"], categories=["obs", "cmip"], ordered=True
    )
    table_df = table_df.sort_values(["dataset", "model"]).reset_index(drop=True)
    table_df.to_csv(table_path, index=False)
    logger.info(f"Wrote table: {table_path}")

    fig_path = Path("fig") / f"feedback_fit_{metric_cfg.key}_{region}.png"
    make_scatter(region, metric_cfg, rows[0], model_rows, plot_info, fig_path, single_region, logger)


def main() -> None:
    args = parse_args()
    metric_cfg = METRIC_CONFIG[args.metric]
    regions = load_regions(args.regions)
    single_region = len(regions) == 1

    cmip_path = Path(args.cmip_csv)
    cmip_df = load_cmip_table(cmip_path, metric_cfg)

    for region in regions:
        logger = setup_logger(metric_cfg.key, region, Path("logs"))
        try:
            process_region(region, metric_cfg, args.hac_lags, cmip_df, single_region, logger)
        finally:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)


if __name__ == "__main__":
    main()

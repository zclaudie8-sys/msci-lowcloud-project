#!/usr/bin/env python3
"""Estimate longwave low-cloud feedback (lambda_lw = dLWCRE / dTs) for 2003–2014.

This script computes the longwave component of the low-cloud feedback using
monthly anomalies of LWCRE and surface temperature (Ts). Observational products
are compared against CMIP6 AMIP models on a per-region basis. Each dataset's
calendar-month climatology over 2003–2014 is removed separately for the
predictor and predictand prior to fitting ordinary least squares regressions
with Newey–West (HAC) standard errors.

Outputs
-------
* ``tables/feedback_fit_lambda_lw_<REGION>.csv`` — regression summary table.
* ``figures/feedback_fit_lambda_lw_<REGION>.png`` — diagnostic scatter plot of
  model vs observational slopes including HAC error bars.
* ``logs/feedback_fit_lambda_lw_<REGION>.log`` — processing log.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


DEFAULT_REGIONS = ["SEP", "SEA", "NEP", "NEA", "SEI"]
OBS_PREFIX = Path("output/regional_monthly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate longwave low-cloud feedback (λ_cld,LW = dLWCRE/dTs)."
    )
    parser.add_argument(
        "--regions",
        help="Comma separated list of regions (default: from config or preset).",
    )
    parser.add_argument(
        "--cmip-csv",
        default="output/cmip_amip_monthly_2003-2014.csv",
        help="Path to CMIP6 AMIP stacked monthly table (default: output/cmip_amip_monthly_2003-2014.csv).",
    )
    parser.add_argument(
        "--hac-lags",
        type=int,
        default=12,
        help="Maximum lag for the Newey–West HAC estimator (default: 12).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load inputs and preview anomalies without writing outputs.",
    )
    return parser.parse_args()


def load_regions(regions_arg: Optional[str]) -> List[str]:
    if regions_arg:
        return [r.strip().upper() for r in regions_arg.split(",") if r.strip()]

    config_paths = [Path("configs/regions.yaml"), Path("configs/config.yaml")]
    for cfg_path in config_paths:
        if not cfg_path.exists():
            continue
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(cfg_path.read_text())
        except Exception:  # pragma: no cover - config parsing is best effort
            continue

        if isinstance(data, dict):
            if cfg_path.name == "regions.yaml":
                return [str(key).upper() for key in data.keys()]
            if "regions" in data and isinstance(data["regions"], dict):
                return [str(k).upper() for k in data["regions"].keys()]
            project = data.get("project")
            if isinstance(project, dict) and isinstance(project.get("regions"), list):
                return [str(r).upper() for r in project["regions"]]
    return DEFAULT_REGIONS.copy()


def setup_logger(region: str, dry_run: bool) -> logging.Logger:
    logger_name = f"feedback_fit_lambda_lw_{region}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if not dry_run:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            logs_dir / f"feedback_fit_lambda_lw_{region}.log"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _find_column(columns_lower: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        key = candidate.lower()
        if key in columns_lower:
            return columns_lower[key]
    return None


def read_obs_series(path: Path, logger: logging.Logger) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty:
        logger.warning(f"{path} is empty.")
        return pd.DataFrame(columns=["time", "value"])

    columns_lower = {c.lower(): c for c in df.columns}
    time_col = _find_column(columns_lower, ["time", "date", "month"])
    if time_col is None:
        time_col = df.columns[0]
    value_col = _find_column(columns_lower, ["value", "val", "mean", "data"])
    if value_col is None:
        non_time = [c for c in df.columns if c != time_col]
        if not non_time:
            raise ValueError(f"Unable to identify value column in {path}.")
        value_col = non_time[-1]

    df = df[[time_col, value_col]].copy()
    df.columns = ["time", "value"]

    parsed_time = pd.to_datetime(df["time"], errors="coerce")
    if parsed_time.isna().all():
        if {"year", "month"}.issubset(columns_lower):
            df["time"] = pd.to_datetime(
                {
                    "year": pd.to_numeric(df[columns_lower["year"]], errors="coerce").astype("Int64"),
                    "month": pd.to_numeric(df[columns_lower["month"]], errors="coerce").astype("Int64"),
                    "day": 1,
                }
            )
        else:
            raise ValueError(
                f"Time column in {path} could not be parsed to datetime; provide YYYY-MM format."
            )
    else:
        df["time"] = parsed_time

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time", "value"]).sort_values("time")
    df = df[(df["time"].dt.year >= 2003) & (df["time"].dt.year <= 2014)]
    df = df.groupby("time", as_index=False)["value"].mean()
    logger.info(f"Read {path.name}: {len(df)} monthly records (2003–2014 filter).")
    return df


def compute_anomaly_pairs(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame(columns=["y", "x"])

    sorted_df = paired.dropna(subset=["time", "y", "x"]).sort_values("time")
    sorted_df = sorted_df.set_index("time")

    y = sorted_df["y"]
    x = sorted_df["x"]

    y_clim = y.groupby(y.index.month).transform("mean")
    x_clim = x.groupby(x.index.month).transform("mean")

    anomalies = pd.DataFrame(
        {
            "y": y - y_clim,
            "x": x - x_clim,
        }
    )
    anomalies = anomalies.dropna()
    return anomalies


def run_hac_regression(
    anomalies: pd.DataFrame,
    label: str,
    region: str,
    hac_lags: int,
    logger: logging.Logger,
) -> Optional[Dict[str, float]]:
    n_samples = len(anomalies)
    if n_samples < 24:
        logger.warning(
            f"{label}: insufficient paired samples after anomaly alignment (REGION={region}, n={n_samples})."
        )
        return None

    X = sm.add_constant(anomalies["x"].values)
    model = sm.OLS(anomalies["y"].values, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )

    slope = float(model.params[1])
    se_hac = float(model.bse[1])
    t_val = float(model.tvalues[1])
    p_val = float(model.pvalues[1])
    r_sq = float(model.rsquared)

    logger.info(
        "OLS(HAC lags=%d): b=%.4f, SE=%.4f, t=%.3f, p=%.3f, R2=%.3f",
        hac_lags,
        slope,
        se_hac,
        t_val,
        p_val,
        r_sq,
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


def load_cmip_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["model", "region", "time", "y", "x"])

    columns_lower = {c.lower(): c for c in df.columns}

    def require_column(*names: str) -> str:
        column = _find_column(columns_lower, names)
        if column is None:
            raise KeyError(f"Missing column in CMIP table; expected one of {names}.")
        return column

    rename_map = {
        require_column("model"): "model",
        require_column("region"): "region",
        require_column("lwcre"): "y",
        require_column("ts"): "x",
    }

    time_col = _find_column(columns_lower, ["time", "date"])
    if time_col:
        rename_map[time_col] = "time"
    else:
        year_col = _find_column(columns_lower, ["year"])
        month_col = _find_column(columns_lower, ["month"])
        if year_col and month_col:
            rename_map[year_col] = "year"
            rename_map[month_col] = "month"
        else:
            raise KeyError(
                "CMIP table must include a parseable 'time' column or both 'year' and 'month'."
            )

    if "family" in columns_lower:
        rename_map[columns_lower["family"]] = "family"

    df = df.rename(columns=rename_map)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        df["time"] = pd.to_datetime(
            {
                "year": pd.to_numeric(df["year"], errors="coerce").astype("Int64"),
                "month": pd.to_numeric(df["month"], errors="coerce").astype("Int64"),
                "day": 1,
            }
        )

    df = df.dropna(subset=["time"])  # drop invalid dates
    df["model"] = df["model"].astype(str)
    df["region"] = df["region"].astype(str).str.upper()
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")

    keep_cols = ["model", "region", "time", "y", "x"]
    if "family" in df.columns:
        keep_cols.append("family")

    df = df[keep_cols]
    df = df[(df["time"].dt.year >= 2003) & (df["time"].dt.year <= 2014)]
    df = df.sort_values(["model", "region", "time"])
    return df


def combine_obs_pairs(region: str, logger: logging.Logger) -> pd.DataFrame:
    lwcre_path = OBS_PREFIX / f"lwcre_mean_2003-2014_{region}.csv"
    ts_path = OBS_PREFIX / f"ts_mean_2003-2014_{region}.csv"

    lwcre_df = read_obs_series(lwcre_path, logger)
    ts_df = read_obs_series(ts_path, logger)

    merged = pd.merge(lwcre_df, ts_df, on="time", how="outer", suffixes=("_y", "_x"))
    merged = merged.rename(columns={"value_y": "y", "value_x": "x"})
    return merged[["time", "y", "x"]]


def make_scatter(
    region: str,
    obs_row: Dict[str, float],
    model_rows: List[Dict[str, float]],
    fig_path: Path,
    logger: logging.Logger,
) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))

    obs_b = obs_row["b"]
    obs_se = obs_row["SE_HAC"]

    if model_rows:
        for row in model_rows:
            label = row["model"]
            ax.errorbar(
                obs_b,
                row["b"],
                xerr=obs_se,
                yerr=row["SE_HAC"],
                fmt="o",
                label=label,
                capsize=3,
            )
    else:
        ax.scatter([obs_b], [obs_b], color="black", marker="x", label="OBS")

    all_vals = [obs_b]
    if np.isfinite(obs_se):
        all_vals.extend([obs_b - obs_se, obs_b + obs_se])
    for row in model_rows:
        all_vals.append(row["b"])
        if np.isfinite(row["SE_HAC"]):
            all_vals.extend([row["b"] - row["SE_HAC"], row["b"] + row["SE_HAC"]])

    if not all_vals:
        all_vals = [0.0]

    min_val = min(all_vals)
    max_val = max(all_vals)
    if min_val == max_val:
        pad = 1.0
    else:
        pad = 0.1 * (max_val - min_val)
    lims = (min_val - pad, max_val + pad)

    ax.plot(lims, lims, "k--", linewidth=1, label="1:1 line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("OBS λ_cld,LW (W m$^{-2}$ K$^{-1}$)")
    ax.set_ylabel("CMIP λ_cld,LW (W m$^{-2}$ K$^{-1}$)")
    ax.set_title(f"λ_cld,LW – {region} (2003–2014)")
    ax.grid(True, linestyle=":", alpha=0.5)
    if model_rows:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved scatter plot to {fig_path}.")


def process_region(
    region: str,
    cmip_df: pd.DataFrame,
    hac_lags: int,
    dry_run: bool,
) -> None:
    logger = setup_logger(region, dry_run=dry_run)

    # Observational anomalies
    obs_pairs = combine_obs_pairs(region, logger)
    obs_anoms = compute_anomaly_pairs(obs_pairs)
    logger.info(
        "Built monthly anomalies over 2003–2014 for OBS (REGION=%s, n=%d)",
        region,
        len(obs_anoms),
    )

    if dry_run:
        print(f"[DRY-RUN] OBS {region}: shape={obs_pairs.shape}")
        print(obs_pairs.head())
        print(f"[DRY-RUN] OBS {region} anomalies head:\n{obs_anoms.head()}")

    obs_result = None
    if not dry_run:
        obs_result = run_hac_regression(obs_anoms, "OBS", region, hac_lags, logger)
        if obs_result is None:
            logger.warning("Skipping region %s due to insufficient OBS samples.", region)
            return

    # CMIP models
    region_df = cmip_df[cmip_df["region"] == region]
    model_results: List[Dict[str, float]] = []
    model_rows_for_plot: List[Dict[str, float]] = []

    for model_name, group in region_df.groupby("model"):
        group_pairs = group[["time", "y", "x"]].copy()
        model_anoms = compute_anomaly_pairs(group_pairs)
        logger.info(
            "Built monthly anomalies over 2003–2014 for %s (REGION=%s, n=%d)",
            model_name,
            region,
            len(model_anoms),
        )

        if dry_run:
            print(f"[DRY-RUN] {model_name} {region}: shape={group_pairs.shape}")
            print(group_pairs.head())
            print(f"[DRY-RUN] {model_name} {region} anomalies head:\n{model_anoms.head()}")
            continue

        stats = run_hac_regression(model_anoms, model_name, region, hac_lags, logger)
        if stats is None:
            continue

        row = {
            "dataset": "cmip",
            "model": model_name,
            "region": region,
            "metric": "lambda_lw",
        }
        row.update(stats)
        model_results.append(row)

        plot_row = {"model": model_name}
        plot_row.update(stats)
        model_rows_for_plot.append(plot_row)

    if dry_run:
        print(f"[DRY-RUN] Completed dry run for region {region}.")
        return

    if obs_result is None:
        # Already handled earlier.
        return

    obs_row = {
        "dataset": "obs",
        "model": "OBS",
        "region": region,
        "metric": "lambda_lw",
    }
    obs_row.update(obs_result)

    table_rows = [obs_row] + model_results
    if not table_rows:
        logger.warning("No valid regression results for region %s.", region)
        return

    table_df = pd.DataFrame(table_rows)
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / f"feedback_fit_lambda_lw_{region}.csv"
    table_df = table_df[
        [
            "dataset",
            "model",
            "region",
            "metric",
            "b",
            "SE_HAC",
            "t",
            "p",
            "R2",
            "n",
            "sign",
        ]
    ]
    table_df.to_csv(table_path, index=False)
    logger.info(f"Wrote table to {table_path}.")

    fig_path = Path("figures") / f"feedback_fit_lambda_lw_{region}.png"
    make_scatter(region, obs_row, model_rows_for_plot, fig_path, logger)


def main() -> None:
    args = parse_args()
    regions = load_regions(args.regions)

    cmip_df = load_cmip_table(Path(args.cmip_csv))

    if args.dry_run:
        print("[DRY-RUN] Loaded CMIP table with shape:", cmip_df.shape)
        print(cmip_df.head())

    for region in regions:
        try:
            process_region(region, cmip_df, args.hac_lags, args.dry_run)
        except FileNotFoundError as exc:
            if args.dry_run:
                print(f"[DRY-RUN] Missing file for region {region}: {exc}")
            else:
                raise

    if args.dry_run:
        print("[DRY-RUN] Completed preview. Exiting without writing outputs.")
        return


if __name__ == "__main__":
    main()

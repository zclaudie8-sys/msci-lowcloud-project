#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feedback fit: λ_cld,LW = d(LWCRE_low)/d(Ts), 2003–2014 (journal-grade)

Features:
- Unified outputs via --out-dir (tables / figures / logs)
- Methods: OLS or Deming (EIV) + HAC robust SE
- HAC lag selection: fixed / andrews / grid  (no external bandwidth import)
- Season fixed effects (--season-fe) or classic de-seasoning
- Optional controls (--controls eis,oni,...) auto-load by region from output/regional_monthly
- Moving-block bootstrap (MBB) CIs (--mbb N, --block L)
- Per-region figures + combined ALL-REGIONS figure
- Metadata in tables: git_sha, run_datetime, method, hac_select, hac_lags_used, spec, controls, ci_method
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ----------------------------
# Defaults & IO layout
# ----------------------------
DEFAULT_REGIONS = ["SEP", "SEA", "NEP", "NEA", "SEI"]
OBS_PREFIX = Path("output/regional_monthly")

REGION_COLORS = {
    "SEP": "#1f77b4",
    "SEA": "#ff7f0e",
    "NEP": "#2ca02c",
    "NEA": "#d62728",
    "SEI": "#9467bd",
}

# ----------------------------
# Utilities
# ----------------------------
def _find_column(columns_lower: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for cand in candidates:
        key = cand.lower()
        if key in columns_lower:
            return columns_lower[key]
    return None


def _normalize_month_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align monthly stamps to Month-Start (00:00), aggregate duplicates,
    and clip to 2003–2014.
    """
    if df.empty:
        return df
    if "time" not in df.columns:
        raise ValueError("normalize_month_index: missing 'time' column.")

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time")
    out["time"] = out["time"].dt.to_period("M").dt.to_timestamp()

    lower = {c.lower(): c for c in out.columns}
    vcol = None
    for k in ("value", "val", "mean", "data"):
        if k in lower:
            vcol = lower[k]
            break
    if vcol is None:
        non_time = [c for c in out.columns if c != "time"]
        if not non_time:
            raise ValueError("normalize_month_index: cannot infer value column.")
        vcol = non_time[-1]

    out = (
        out.groupby("time", as_index=False)[vcol]
        .mean()
        .rename(columns={vcol: "value"})
        .sort_values("time")
    )
    out = out[(out["time"].dt.year >= 2003) & (out["time"].dt.year <= 2014)]
    return out


def month_dummies(time_like) -> pd.DataFrame:
    """
    Return 12 month dummies aligned to the row index of the input.
    Accepts a pandas Series (preferred) or a DatetimeIndex. Ensures float dtype.
    """
    if isinstance(time_like, pd.Series):
        months = pd.to_datetime(time_like, errors="coerce").dt.month.astype("Int64")
        d = pd.get_dummies(months, prefix="M", drop_first=False)
        d.index = time_like.index
    elif isinstance(time_like, pd.DatetimeIndex):
        months = pd.Series(time_like.month, index=pd.RangeIndex(len(time_like)))
        d = pd.get_dummies(months.astype(int), prefix="M", drop_first=False)
    else:
        s = pd.Series(pd.to_datetime(time_like, errors="coerce"))
        months = s.dt.month.astype("Int64")
        d = pd.get_dummies(months, prefix="M", drop_first=False)
        d.index = s.index
    return d.astype(float).fillna(0.0)


def git_sha_short() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return sha
    except Exception:
        return "NA"

# ----------------------------
# Data loading
# ----------------------------
def load_regions(regions_arg: Optional[str]) -> List[str]:
    if regions_arg:
        return [r.strip().upper() for r in regions_arg.split(",") if r.strip()]

    # Config fallback
    for cfg in (Path("configs/regions.yaml"), Path("configs/config.yaml")):
        if not cfg.exists():
            continue
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(cfg.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            if cfg.name == "regions.yaml":
                return [str(k).upper() for k in data.keys()]
            if "regions" in data and isinstance(data["regions"], dict):
                return [str(k).upper() for k in data["regions"].keys()]
            proj = data.get("project")
            if isinstance(proj, dict) and isinstance(proj.get("regions"), list):
                return [str(r).upper() for r in proj["regions"]]
    return DEFAULT_REGIONS.copy()


def read_obs_series(path: Path, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}

    # time
    tcol = _find_column(lower, ("time", "date", "month"))
    if tcol is None:
        if "year" in lower and "month" in lower:
            df["time"] = pd.to_datetime(
                {
                    "year": pd.to_numeric(df[lower["year"]], errors="coerce").astype("Int64"),
                    "month": pd.to_numeric(df[lower["month"]], errors="coerce").astype("Int64"),
                    "day": 1,
                }
            )
        else:
            raise ValueError(f"{path} has no parseable time")
    else:
        df["time"] = pd.to_datetime(df[tcol], errors="coerce")

    # value
    vcol = _find_column(lower, ("value", "val", "mean", "data"))
    if vcol is None:
        non_time = [c for c in df.columns if c != "time" and c != tcol]
        if not non_time:
            raise ValueError(f"{path}: cannot infer value column")
        vcol = non_time[-1]

    df = df[["time", vcol]].rename(columns={vcol: "value"})
    df = df.dropna(subset=["time", "value"])
    df = _normalize_month_index(df)
    if logger:
        logger.info("OBS %s rows=%d (2003–2014, normalized)", path.name, len(df))
    return df


def _first_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def combine_obs_pairs(region: str, logger: logging.Logger, controls: List[str]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    # LWCRE_low
    lw_candidates = [
        OBS_PREFIX / f"lwcre_mean_2003-2014_{region}.csv",
        OBS_PREFIX / f"cllwlow_mean_2003-2014_{region}.csv",
    ]
    ts_candidates = [
        OBS_PREFIX / f"ts_mean_2003-2014_{region}.csv",
        OBS_PREFIX / f"tas_mean_2003-2014_{region}.csv",
    ]
    lw_path = _first_existing(lw_candidates)
    ts_path = _first_existing(ts_candidates)
    if not lw_path.exists():
        raise FileNotFoundError(f"Missing OBS LWCRE_low for {region}. Tried: {', '.join(str(p) for p in lw_candidates)}")
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing OBS Ts for {region}. Tried: {', '.join(str(p) for p in ts_candidates)}")

    lw_df = read_obs_series(lw_path, logger)
    ts_df = read_obs_series(ts_path, logger)
    merged = pd.merge(lw_df, ts_df, on="time", how="inner", suffixes=("_y", "_x")).rename(
        columns={"value_y": "y", "value_x": "x"}
    )  # y=lwcre, x=ts

    # Controls (optional)
    control_frames: Dict[str, pd.DataFrame] = {}
    for cname in controls:
        cname_lower = cname.lower()
        ccands = [
            OBS_PREFIX / f"{cname_lower}_mean_2003-2014_{region}.csv",
            OBS_PREFIX / f"{cname_lower}lts_mean_2003-2014_{region}.csv",  # eislts → eislts_mean_...
            OBS_PREFIX / f"{cname_lower}_2003-2014_{region}.csv",
        ]
        cpath = _first_existing(ccands)
        if cpath.exists():
            cdf = read_obs_series(cpath, logger)
            control_frames[cname_lower] = cdf.rename(columns={"value": cname_lower})
            merged = pd.merge(merged, control_frames[cname_lower], on="time", how="inner")
            logger.info("OBS control added: %s (%d rows)", cname_lower, len(control_frames[cname_lower]))
        else:
            logger.warning("OBS control skipped, not found: %s (tried %s)", cname_lower, [str(p) for p in ccands])

    return merged[["time"] + [c for c in merged.columns if c != "time"]], control_frames


def load_cmip_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["model", "region", "time", "y", "x"])

    lower = {c.lower(): c for c in df.columns}
    def req(*names: str) -> str:
        col = _find_column(lower, names)
        if col is None:
            raise KeyError(f"Missing column; expected one of {names}")
        return col

    rename = {
        req("model"): "model",
        req("region"): "region",
        req("lwcre", "cllwlow", "y"): "y",
        req("ts", "tas", "x"): "x",
    }
    tcol = _find_column(lower, ("time", "date"))
    if tcol:
        rename[tcol] = "time"
    else:
        ycol = req("year")
        mcol = req("month")
        rename[ycol] = "year"
        rename[mcol] = "month"

    if "family" in lower:
        rename[lower["family"]] = "family"

    df = df.rename(columns=rename)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        df["time"] = pd.to_datetime({"year": df["year"], "month": df["month"], "day": 1}, errors="coerce")
    df = df.dropna(subset=["time"])
    df["time"] = df["time"].dt.to_period("M").dt.to_timestamp()
    df = df[(df["time"].dt.year >= 2003) & (df["time"].dt.year <= 2014)]
    df["model"] = df["model"].astype(str)
    df["region"] = df["region"].astype(str).str.upper()
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    keep = ["model", "region", "time", "y", "x"] + (["family"] if "family" in df.columns else [])
    df = df[keep].sort_values(["model", "region", "time"])
    return df

# ----------------------------
# Regression cores
# ----------------------------
@dataclass
class FitConfig:
    method: str = "OLS"            # "OLS" or "Deming"
    hac_select: str = "fixed"      # "fixed", "andrews", "grid"
    hac_lags: int = 12             # used when fixed or as cap
    season_fe: bool = False        # month fixed effects
    mbb: int = 0                   # bootstrap reps; 0 means disabled
    block: int = 9                 # block length for MBB


def select_hac_lags(y: np.ndarray, x: np.ndarray, cfg: FitConfig) -> int:
    """
    Choose HAC maxlags:
      - fixed   : cfg.hac_lags
      - andrews : L = floor( 4 * (n/100)^(2/9) ), capped by cfg.hac_lags (if provided)
      - grid    : choose from {6,9,12,18} by minimal HAC SE
    """
    n = len(y)
    if n <= 2:
        return max(1, min(cfg.hac_lags, 3))

    if cfg.hac_select == "fixed":
        return int(cfg.hac_lags)

    if cfg.hac_select == "andrews":
        L = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        L = max(1, L)
        if cfg.hac_lags is not None:
            L = min(L, int(cfg.hac_lags))
        return L

    if cfg.hac_select == "grid":
        candidates = [6, 9, 12, 18]
        X = sm.add_constant(x)
        best_L, best_se = candidates[0], np.inf
        for L in candidates:
            res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
            se = float(res.bse[1])
            if se < best_se:
                best_se, best_L = se, L
        return best_L

    return int(cfg.hac_lags)


def deming_fit(y: np.ndarray, x: np.ndarray, ratio: float = 1.0) -> Tuple[float, float]:
    """
    Closed-form Deming regression for y ~ a + b x, error ratio λ=Var(ε_y)/Var(ε_x).
    Returns (a, b). SE will be estimated via HAC on residuals by re-fitting OLS.
    """
    xbar, ybar = np.nanmean(x), np.nanmean(y)
    Sxx = np.nanvar(x, ddof=1)
    Syy = np.nanvar(y, ddof=1)
    Sxy = np.nanmean((x - xbar) * (y - ybar))
    num = Syy - ratio * Sxx + np.sqrt((Syy - ratio * Sxx) ** 2 + 4 * ratio * Sxy ** 2)
    den = 2 * Sxy
    b = num / den
    a = ybar - b * xbar
    return float(a), float(b)


def mbb_ci(y: np.ndarray, x: np.ndarray, cfg: FitConfig, design_extras: Optional[pd.DataFrame],
           slope_func) -> Tuple[float, float]:
    """
    Moving Block Bootstrap CI for slope (2.5%, 97.5%).
    slope_func: function (y, x, extras) -> slope
    """
    n, L, B = len(y), cfg.block, cfg.mbb
    if B <= 0 or n < 2 or L < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(12345)
    k = int(np.ceil(n / L))
    slopes = []
    extra_arr = design_extras.values if design_extras is not None else None

    for _ in range(B):
        starts = rng.integers(0, n - L + 1, size=k)
        sel = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
        yb = y[sel]
        xb = x[sel]
        extras_b = None
        if extra_arr is not None:
            extras_b = pd.DataFrame(extra_arr[sel, :], columns=design_extras.columns)

        try:
            slopes.append(slope_func(yb, xb, extras_b))
        except Exception:
            continue

    if not slopes:
        return np.nan, np.nan
    lo, hi = np.nanpercentile(slopes, [2.5, 97.5])
    return float(lo), float(hi)


def run_fit(y: np.ndarray, x: np.ndarray, cfg: FitConfig, extras: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    Core estimator with:
    - OLS or Deming slope
    - HAC robust SE (with selected lags)
    - optional MBB CI
    extras: extra regressors (controls + FE dummies), WITHOUT constant
    """
    # force float dtype
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if extras is not None and not extras.empty:
        extras = extras.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        Xcore = np.column_stack([x, np.asarray(extras.values, dtype=float)])
    else:
        Xcore = x.reshape(-1, 1)

    X_for_se = sm.add_constant(Xcore.astype(float), has_constant="add")

    def hac_fit_stats() -> Tuple[float, float, float, float, float]:
        res = sm.OLS(y, X_for_se).fit(cov_type="HAC", cov_kwds={"maxlags": select_hac_lags(y, x, cfg)})
        b = float(res.params[1])
        se = float(res.bse[1])
        tval = float(res.tvalues[1])
        pval = float(res.pvalues[1])
        r2 = float(res.rsquared)
        return b, se, tval, pval, r2

    if cfg.method.upper() in ("DEMMING", "DEMING"):
        _a, _b = deming_fit(y, x, ratio=1.0)  # slope anchor (for reporting only)
        b, se, tval, pval, r2 = hac_fit_stats()  # SE via HAC with full design
    else:
        b, se, tval, pval, r2 = hac_fit_stats()

    result = {
        "b": b,
        "SE_HAC": se,
        "t": tval,
        "p": pval,
        "R2": r2,
        "n": len(y),
        "hac_lags_used": select_hac_lags(y, x, cfg),
    }

    # MBB CI
    if cfg.mbb > 0:
        def slope_func(yb, xb, extras_b):
            if extras_b is not None and not extras_b.empty:
                Xb = sm.add_constant(np.column_stack([xb, extras_b.values]))
            else:
                Xb = sm.add_constant(xb.reshape(-1, 1))
            rb = sm.OLS(yb, Xb).fit()
            return float(rb.params[1])

        lo, hi = mbb_ci(y, x, cfg, extras if extras is not None else None, slope_func)
        result["ci_low"] = lo
        result["ci_high"] = hi
        result["ci_method"] = "MBB"
    else:
        result["ci_low"] = np.nan
        result["ci_high"] = np.nan
        result["ci_method"] = "HAC"

    return result

# ----------------------------
# Construction of design
# ----------------------------
def compute_design(series_df: pd.DataFrame, season_fe: bool, extra_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """
    Build (y,x,extras) with either:
      - season_fe=True: add month dummies; no pre-demean
      - season_fe=False: pre-demean by month (classic anomalies); controls also demeaned
    series_df columns: ['time','y','x', <controls...>]
    """
    df = series_df.dropna().sort_values("time").copy()

    if season_fe:
        mdum = month_dummies(df["time"])

        extras = pd.DataFrame(index=df.index)
        for c in extra_cols:
            if c in df.columns:
                extras[c] = df[c].values

        extras = pd.concat([extras, mdum], axis=1)
        extras = extras.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

        y = pd.to_numeric(df["y"], errors="coerce").astype(float).values
        x = pd.to_numeric(df["x"], errors="coerce").astype(float).values
        return y, x, extras

    # Pre-demean by calendar month
    df = df.set_index("time")

    def demean_by_month(s: pd.Series) -> pd.Series:
        return s - s.groupby(s.index.month).transform("mean")

    y_anom = demean_by_month(df["y"])
    x_anom = demean_by_month(df["x"])

    extras = pd.DataFrame(index=df.index)
    for c in extra_cols:
        if c in df.columns:
            extras[c] = demean_by_month(pd.to_numeric(df[c], errors="coerce"))

    align = pd.DataFrame({"y": y_anom, "x": x_anom}).dropna()
    if not extras.empty:
        extras = extras.loc[align.index]
        extras = extras.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    else:
        extras = None

    return align["y"].values.astype(float), align["x"].values.astype(float), extras

# ----------------------------
# Plotting
# ----------------------------
def make_scatter(region: str, obs_row: Dict[str, float], model_rows: List[Dict[str, float]], fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    obs_b = obs_row["b"]; obs_se = obs_row["SE_HAC"]

    for row in model_rows:
        # 优先画 CI（若有），否则 HAC SE
        yerr = (row.get("ci_high") - row.get("b")) if np.isfinite(row.get("ci_high", np.nan)) else row["SE_HAC"]
        ax.errorbar(obs_b, row["b"], xerr=obs_se, yerr=yerr, fmt="o", capsize=3, alpha=0.9, label=row["model"])

    vals = [obs_b] + [r["b"] for r in model_rows]
    if np.isfinite(obs_se):
        vals += [obs_b - obs_se, obs_b + obs_se]
    for r in model_rows:
        se = r["SE_HAC"]
        vals += [r["b"] - se, r["b"] + se]
    vmin, vmax = min(vals), max(vals)
    pad = 0.1 * (vmax - vmin) if vmax > vmin else 1.0
    lims = (vmin - pad, vmax + pad)
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    ax.set_xlim(lims); ax.set_ylim(lims)

    ax.set_xlabel("OBS λ_cld,LW (W m$^{-2}$ K$^{-1}$)")
    ax.set_ylabel("CMIP λ_cld,LW (W m$^{-2}$ K$^{-1}$)")
    ax.set_title(f"λ_cld,LW – {region} (2003–2014)")
    ax.grid(True, ls=":", alpha=0.5)
    if len(model_rows) <= 18:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def make_combined_all_regions_figure(all_results: Dict[str, Tuple[Dict[str, float], List[Dict[str, float]]]], out_fig: Path) -> None:
    if not all_results:
        return
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    xs, ys = [], []
    for region, (obs_row, model_rows) in all_results.items():
        if not obs_row or not model_rows:
            continue
        color = REGION_COLORS.get(region, "gray")
        xb = obs_row["b"]; xse = obs_row["SE_HAC"]
        for mr in model_rows:
            yb = mr["b"]
            yerr = (mr.get("ci_high") - mr.get("b")) if np.isfinite(mr.get("ci_high", np.nan)) else mr["SE_HAC"]
            ax.errorbar(xb, yb, xerr=xse, yerr=yerr, fmt="o", capsize=2.5, alpha=0.85, color=color, label=region)
            xs.extend([xb - xse, xb + xse]); ys.extend([yb - yerr, yb + yerr])

    if xs and ys:
        vmin = min(xs + ys); vmax = max(xs + ys)
    else:
        vmin, vmax = -1.0, 1.0
    pad = 0.1 * (vmax - vmin) if vmax > vmin else 1.0
    lims = (vmin - pad, vmax + pad)
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("OBS λ_cld,LW (W m$^{-2}$ K$^{-1}$)")
    ax.set_ylabel("CMIP λ_cld,LW (W m$^{-2}$ K$^{-1}$)")
    ax.set_title("λ_cld,LW – All Regions (2003–2014)")

    # dedupe region legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set(); h2, l2 = [], []
    for h, lab in zip(handles, labels):
        if lab in REGION_COLORS and lab not in seen:
            seen.add(lab); h2.append(h); l2.append(lab)
    ax.legend(h2, l2, title="Region", loc="best", fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)

# ----------------------------
# Logging
# ----------------------------
def setup_logger(region: str, out_dir: Path, to_file: bool) -> logging.Logger:
    logger = logging.getLogger(f"fit_{region}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    if to_file:
        logp = out_dir / "logs"
        logp.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logp / f"feedback_fit_lambda_lw_{region}.log")
        fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

# ----------------------------
# Region processing
# ----------------------------
def process_region(region: str, cmip_df: pd.DataFrame, out_dir: Path, cfg: FitConfig,
                   controls: List[str], dry_run: bool, meta: Dict[str, str]) -> Tuple[Optional[Dict[str, float]], List[Dict[str, float]]]:
    logger = setup_logger(region, out_dir, to_file=not dry_run)

    obs_pairs, _control_frames = combine_obs_pairs(region, logger, controls)

    # Build design
    y_obs, x_obs, extras_obs = compute_design(obs_pairs, cfg.season_fe, extra_cols=controls)
    logger.info("OBS design ready: n=%d (season_fe=%s)", len(y_obs), cfg.season_fe)

    obs_row = None
    if not dry_run:
        obs_stats = run_fit(y_obs, x_obs, cfg, extras_obs)
        obs_row = {
            "dataset": "obs", "model": "OBS", "region": region, "metric": "lambda_lw",
            **obs_stats,
            **meta,
            "spec": "FE" if cfg.season_fe else "demean",
            "method": cfg.method.upper(),
            "hac_select": cfg.hac_select,
            "controls": ",".join(controls) if controls else "none",
        }
        if obs_row["n"] < 24:
            logger.warning("OBS insufficient samples after alignment: n=%d", obs_row["n"])

    # CMIP models
    region_df = cmip_df[cmip_df["region"] == region]
    model_rows: List[Dict[str, float]] = []
    for mname, g in region_df.groupby("model"):
        dfm = g[["time", "y", "x"]].sort_values("time").copy()
        dfm["time"] = pd.to_datetime(dfm["time"]).dt.to_period("M").dt.to_timestamp()
        # bring OBS controls onto model time (large-scale indices)
        for cname in controls:
            if cname in obs_pairs.columns:
                dfm = pd.merge(dfm, obs_pairs[["time", cname]], on="time", how="left")

        y_mod, x_mod, extras_mod = compute_design(dfm, cfg.season_fe, extra_cols=controls)
        if dry_run:
            logger.info("[DRY] %s n=%d", mname, len(y_mod))
            continue
        stats = run_fit(y_mod, x_mod, cfg, extras_mod)
        row = {
            "dataset": "cmip", "model": mname, "region": region, "metric": "lambda_lw",
            **stats,
            **meta,
            "spec": "FE" if cfg.season_fe else "demean",
            "method": cfg.method.upper(),
            "hac_select": cfg.hac_select,
            "controls": ",".join(controls) if controls else "none",
        }
        model_rows.append(row)

    # Write table & figure
    if not dry_run and obs_row and model_rows:
        tdir = out_dir / "tables"; fdir = out_dir / "figures"
        tdir.mkdir(parents=True, exist_ok=True); fdir.mkdir(parents=True, exist_ok=True)
        tpath = tdir / f"feedback_fit_lambda_lw_{region}.csv"
        cols = ["dataset","model","region","metric","b","SE_HAC","t","p","R2","n","hac_lags_used",
                "ci_low","ci_high","ci_method","method","spec","hac_select","controls","git_sha","run_datetime"]
        pd.DataFrame([obs_row] + model_rows)[cols].to_csv(tpath, index=False)
        logger.info("Wrote table: %s", tpath)
        make_scatter(region, obs_row, model_rows, fdir / f"feedback_fit_lambda_lw_{region}.png")

    return obs_row, model_rows

# ----------------------------
# CLI & main
# ----------------------------
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Feedback fit: λ_cld,LW = dLWCRE_low / dTs")
    p.add_argument("--regions", help="Comma-separated regions (default from config/preset).")
    p.add_argument("--cmip-csv", required=True, help="CMIP6 AMIP monthly stack (model×region×time×(lwcre,ts)).")
    p.add_argument("--out-dir", default="output/feedback_fit/lwcre_vs_ts", help="Base output dir.")
    p.add_argument("--method", choices=["OLS","Deming"], default="OLS", help="Regression core.")
    p.add_argument("--hac-select", choices=["fixed","andrews","grid"], default="fixed", help="HAC lag selection.")
    p.add_argument("--hac-lags", type=int, default=12, help="Max lag for HAC (if fixed) or cap for andrews.")
    p.add_argument("--season-fe", action="store_true", help="Use month fixed effects instead of pre-demean.")
    p.add_argument("--controls", default="", help="Comma list of control names (e.g., 'eis,oni').")
    p.add_argument("--mbb", type=int, default=0, help="Moving-block bootstrap reps (0=off).")
    p.add_argument("--block", type=int, default=9, help="Block length for MBB.")
    p.add_argument("--dry-run", action="store_true", help="Preview without writing outputs.")
    return p.parse_args()


def main() -> None:
    args = parse_cli()
    out_dir = Path(args.out_dir)
    regions = load_regions(args.regions)
    cmip_df = load_cmip_table(Path(args.cmip_csv))
    controls = [c.strip().lower() for c in args.controls.split(",") if c.strip()]

    cfg = FitConfig(
        method=args.method,
        hac_select=args.hac_select,
        hac_lags=args.hac_lags,
        season_fe=args.season_fe,
        mbb=args.mbb,
        block=args.block,
    )

    meta = {
        "git_sha": git_sha_short(),
        "run_datetime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if args.dry_run:
        print("[DRY] regions:", regions)
        print("[DRY] cmip_df:", cmip_df.shape, cmip_df.head())
        print("[DRY] cfg:", cfg)
        print("[DRY] controls:", controls)

    combined: Dict[str, Tuple[Dict[str, float], List[Dict[str, float]]]] = {}
    for r in regions:
        try:
            obs_row, model_rows = process_region(r, cmip_df, out_dir, cfg, controls, args.dry_run, meta)
            if (not args.dry_run) and obs_row and model_rows:
                combined[r] = (obs_row, model_rows)
        except FileNotFoundError as e:
            if args.dry_run:
                print(f"[DRY] Missing for {r}: {e}")
            else:
                raise

    if args.dry_run:
        print("[DRY] Completed.")
        return

    # Combined figure
    if combined:
        make_combined_all_regions_figure(combined, (out_dir / "figures") / "feedback_fit_lambda_lw_ALL_REGIONS.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBS vs CMIP6 AMIP feedback fits over 2003–2014 (publication-grade).

主图（推荐）: 分面图（每区一格），横轴=模型斜率 b，竖虚线=OBS 斜率，灰带=OBS 95% CI，模型为水平误差棒。
附录（可选）: 1:1 散点（x=OBS b, y=CMIP b），并显示 Region 图例。

统计: 月异常（默认去季节化）+ OLS with Newey–West(HAC)；可无截距。
dLCF/dEIS: 自动把 LCF 序列统一到“百分数(%)”再回归（避免 0–1 / 0–100 混用）。
输出: output/feedback_fit/{tables,fig,logs} (可用参数覆盖)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm

# ---------- 可选工程依赖（缺失时用兜底） ----------
try:
    from scripts.io_manifest import load_manifest, resolve_from_manifest  # type: ignore
except Exception:
    def load_manifest(_): return None
    def resolve_from_manifest(_m, _q): return []

try:
    from scripts.path_constants import (  # type: ignore
        DEFAULT_OUTPUT_DIR, DEFAULT_TABLES_DIR, DEFAULT_FIG_DIRS,
        ensure_dir, first_existing, make_parent
    )
except Exception:
    DEFAULT_OUTPUT_DIR = "output"
    DEFAULT_TABLES_DIR = "tables"
    DEFAULT_FIG_DIRS = ["fig", "figures"]
    def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
    def first_existing(paths):
        for p in paths:
            if Path(p).exists(): return Path(p)
        return None
    def make_parent(p: Path): ensure_dir(p.parent)

# ---------- 固定输出根 ----------
OUTPUT_ROOT = Path("output/feedback_fit")
TABLES_DEFAULT = OUTPUT_ROOT / "tables"
FIG_DEFAULT    = OUTPUT_ROOT / "fig"
LOGS_DEFAULT   = OUTPUT_ROOT / "logs"

DEFAULT_REGIONS = ["SEP", "SEA", "NEP", "NEA", "SEI"]
REGION_COLORS = {"SEP":"#1f77b4","SEA":"#ff7f0e","NEP":"#2ca02c","NEA":"#d62728","SEI":"#9467bd"}
# 可选：区域对应的 marker（便于论文中黑白打印也能区分）
REGION_MARKERS = {"SEP":"o","SEA":"s","NEP":"^","NEA":"D","SEI":"P"}

@dataclass(frozen=True)
class MetricConfig:
    key: str
    obs_y_vars: Sequence[str]
    obs_x_vars: Sequence[str]
    cmip_y_vars: Sequence[str]
    cmip_x_vars: Sequence[str]
    title: str
    unit_label: str
    scale_factor: float  # 仅线性缩放；LCF 的 % 统一在读取阶段做

METRIC_CONFIG: Dict[str, MetricConfig] = {
    # λ_cld,SW = d(SWCRE)/d(Ts)
    "lambda_cld_sw": MetricConfig(
        key="lambda_cld_sw",
        obs_y_vars=("swcre","clswlow"),
        obs_x_vars=("ts","tas"),
        cmip_y_vars=("swcre","clswlow"),
        cmip_x_vars=("ts","tas"),
        title="λ_cld,SW",
        unit_label="W m$^{-2}$ K$^{-1}$",
        scale_factor=1.0,
    ),
    # dLCF/dEIS：注意 LCF 会在读入时统一为“百分数(%)”
    "dLCF_dEIS": MetricConfig(
        key="dLCF_dEIS",
        obs_y_vars=("cllmodis","lcf"),
        obs_x_vars=("eislts","eis","lts"),
        cmip_y_vars=("lcf","cllmodis"),
        cmip_x_vars=("eis","lts","eislts"),
        title="dLCF/dEIS",
        unit_label="% K$^{-1}$",
        scale_factor=1.0,
    ),
}
METRIC_ALIASES = {"lambda_sw":"lambda_cld_sw", "dlcf_deis":"dLCF_dEIS"}

# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compare OBS vs CMIP6 AMIP feedback fits (2003–2014).")
    p.add_argument("--metrics", help="Comma-separated metrics, e.g., lambda_cld_sw,dLCF_dEIS")
    p.add_argument("--metric",  help="Single metric (alias for --metrics)")
    p.add_argument("--regions", help="Comma-separated regions")
    p.add_argument("--cmip-csv", help="Explicit CMIP stacked monthly CSV (optional)")
    p.add_argument("--hac-lag", type=int, default=12, help="Newey–West max lags (default 12)")
    p.add_argument("--hac-lags", type=int, help="Alias of --hac-lag")
    p.add_argument("--deseasonalize", dest="deseasonalize", action="store_true", default=True,
                   help="Remove monthly climatology before regression (default ON)")
    p.add_argument("--no-deseasonalize", dest="deseasonalize", action="store_false",
                   help="Disable deseasonalization")
    p.add_argument("--no-intercept", action="store_true", help="Fit without intercept")
    p.add_argument("--manifest", help="output_manifest.json (optional)")
    p.add_argument("--tables-dir", help=f"Tables dir (default {TABLES_DEFAULT})")
    p.add_argument("--fig-dir",    help=f"Figures dir (default {FIG_DEFAULT})")
    p.add_argument("--logs-dir",   help=f"Logs dir (default {LOGS_DEFAULT})")
    p.add_argument("--label-topn", type=int, default=4, help="Facet: annotate farthest Top-N models (per region)")
    p.add_argument("--style", choices=["facet","xy"], default="facet",
                   help="facet=publication main figure; xy=legacy 1:1 scatter")
    p.add_argument("--no-text", action="store_true",
                   help="XY scatter: do not annotate model labels")
    return p.parse_args()

def parse_metric_keys(metrics: Optional[str], metric: Optional[str]) -> List[str]:
    req: List[str] = []
    for raw in (metrics, metric):
        if raw: req += [s.strip() for s in raw.split(",") if s.strip()]
    if not req: raise ValueError("At least one metric via --metrics/--metric")
    out, seen = [], set()
    for key in req:
        can = METRIC_ALIASES.get(key.lower(), key.lower())
        if can not in METRIC_CONFIG: raise ValueError(f"Unknown metric '{key}'")
        if can not in seen: out.append(can); seen.add(can)
    return out

def load_regions(regions_arg: Optional[str]) -> List[str]:
    if regions_arg: return [r.strip().upper() for r in regions_arg.split(",") if r.strip()]
    for cfg in ["configs/regions.yaml","configs/config.yaml"]:
        p = Path(cfg)
        if p.exists():
            try:
                import yaml  # type: ignore
                data = yaml.safe_load(p.read_text())
                if isinstance(data, dict):
                    if "regions" in data and isinstance(data["regions"], dict):
                        return [str(k).upper() for k in data["regions"].keys()]
                    proj = data.get("project")
                    if isinstance(proj, dict) and isinstance(proj.get("regions"), list):
                        return [str(r).upper() for r in proj["regions"]]
            except Exception:
                pass
    return DEFAULT_REGIONS.copy()

def setup_logger(logs_dir: Path) -> Tuple[logging.Logger, Path]:
    ensure_dir(logs_dir)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"feedback_fit_{ts}.log"
    lg = logging.getLogger("feedback_fit"); lg.setLevel(logging.INFO); lg.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(sh)
    return lg, log_path

# -------------- IO helpers --------------
def case_insensitive_path(path: Path) -> Optional[Path]:
    if path.exists(): return path
    if not path.parent.exists(): return None
    tgt = path.name.lower()
    for c in path.parent.iterdir():
        if c.name.lower() == tgt: return c
    return None

def resolve_obs_path(manifest: Optional[Dict[str,object]], var: str, region: str, logger: logging.Logger
                    ) -> Tuple[Optional[Path], List[str]]:
    placeholders = {"REGION":region, "var":var}
    mp = resolve_from_manifest(manifest, {
        "artifact_type":"table",
        "path_like_contains":["regional_monthly","<REGION>","<var>"],
        "placeholders": placeholders
    })
    candidates = [Path(s) for s in mp]
    fb = Path(DEFAULT_OUTPUT_DIR) / "regional_monthly" / f"{var}_mean_2003-2014_{region}.csv"
    candidates += [fb, fb.with_name(fb.name.lower())]
    tried: List[str] = []
    for c in candidates:
        tried.append(str(c))
        r = case_insensitive_path(c)
        if r and r.exists():
            logger.info("Resolved OBS path for var '%s' region '%s': %s", var, region, r)
            return r, tried
    return None, tried

def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen=set(); out=[]
    for p in paths:
        s=str(Path(p))
        if s not in seen: out.append(Path(p)); seen.add(s)
    return out

def gather_cmip_candidates(metric_cfg: MetricConfig, manifest, override) -> List[Path]:
    cands: List[Path] = []
    if override: cands.append(Path(override))
    mpaths = resolve_from_manifest(manifest, {"artifact_type":"table","path_like_contains":["cmip","amip"]})
    tokens = [metric_cfg.key.lower(), *[v.lower() for v in metric_cfg.cmip_y_vars+metric_cfg.cmip_x_vars]]
    for s in mpaths:
        low=s.lower()
        if any(t in low for t in tokens): cands.append(Path(s))
    cands.append(Path(DEFAULT_OUTPUT_DIR) / "cmip_amip_monthly_2003-2014.csv")
    for v in metric_cfg.cmip_y_vars+metric_cfg.cmip_x_vars:
        cands.append(Path(DEFAULT_OUTPUT_DIR) / f"cmip_amip_{v}_vs_obs_2003-2014.csv")
    return unique_paths(cands)

def try_load_cmip_table(path: Path, metric_cfg: MetricConfig) -> Tuple[Optional[pd.DataFrame], str]:
    if not path.exists(): return None, "file not found"
    try: df = pd.read_csv(path)
    except Exception as e: return None, f"read fail({e})"
    if df.empty: return None, "empty table"
    cols = {c.lower(): c for c in df.columns}
    def get_col(cands: Sequence[str], required=True):
        for c in cands:
            if c.lower() in cols: return cols[c.lower()]
        if required: raise KeyError(f"Missing {cands}")
        return None
    try:
        ren = {get_col(["model"]):"model", get_col(["region"]):"region"}
    except KeyError as e: return None, str(e)
    # time
    if "time" in cols: ren[cols["time"]] = "time"
    elif "date" in cols: ren[cols["date"]] = "time"
    else:
        if "year" in cols: ren[cols["year"]] = "year"
        if "month" in cols: ren[cols["month"]] = "month"
    try:
        ren[get_col(metric_cfg.cmip_y_vars)] = "y"
        ren[get_col(metric_cfg.cmip_x_vars)] = "x"
    except KeyError as e: return None, str(e)
    fam = get_col(["family"], required=False)
    if fam: ren[fam] = "family"
    df = df.rename(columns=ren)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    elif {"year","month"}.issubset(df.columns):
        df["time"] = pd.to_datetime({"year":pd.to_numeric(df["year"],errors="coerce").astype("Int64"),
                                     "month":pd.to_numeric(df["month"],errors="coerce").astype("Int64"),
                                     "day":1})
    else:
        return None, "no time"
    df = df.dropna(subset=["time"])
    df["model"]  = df["model"].astype(str)
    df["region"] = df["region"].astype(str).str.upper()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df[(df["time"].dt.year>=2003)&(df["time"].dt.year<=2014)]
    keep=["model","region","time","y","x"] + (["family"] if "family" in df.columns else [])
    return df[keep].sort_values(["model","region","time"]).reset_index(drop=True), ""

# ----------- 读 CSV + 单位统一 -----------
def read_monthly_csv(path: Path, logger: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        logger.warning("%s is empty", path)
        return pd.DataFrame(columns=["time","value"])
    cols = {c.lower(): c for c in df.columns}
    # time列
    time_col = None
    for c in ("time","date","month"):
        if c in cols: time_col = cols[c]; break
    if time_col is None: time_col = df.columns[0]
    # value列
    value_col = None
    for c in ("value","val","mean","data","series","climate"):
        if c in cols: value_col = cols[c]; break
    if value_col is None:
        non_time = [c for c in df.columns if c != time_col]
        if not non_time: raise ValueError(f"No value column in {path}")
        value_col = non_time[-1]
    df = df[[time_col, value_col]].copy()
    df.columns = ["time","value"]
    t = pd.to_datetime(df["time"], errors="coerce")
    if t.isna().all():
        if {"year","month"}.issubset(cols):
            y, m = cols["year"], cols["month"]
            df["time"] = pd.to_datetime({"year":pd.to_numeric(df[y],errors="coerce").astype("Int64"),
                                         "month":pd.to_numeric(df[m],errors="coerce").astype("Int64"),
                                         "day":1})
        else:
            raise ValueError(f"Unparseable time in {path}")
    else:
        df["time"] = t
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["time","value"]).sort_values("time")
    df = df[(df["time"].dt.year>=2003)&(df["time"].dt.year<=2014)]
    df = df.groupby("time", as_index=False)["value"].mean()
    logger.info("Read %s: %d monthly records (2003–2014).", path.name, len(df))
    return df

def maybe_to_percent(df: pd.DataFrame) -> pd.DataFrame:
    """若 LCF 在 0–1（99分位 < 1.5），乘 100 统一到百分数；否则原样返回。"""
    if df.empty: return df
    ser = pd.to_numeric(df["value"], errors="coerce")
    if ser.quantile(0.99) < 1.5:
        df = df.copy()
        df["value"] = ser * 100.0
    return df

def prepare_series(df: pd.DataFrame, deseasonalize: bool) -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    s = df.set_index("time")["value"].sort_index()
    if deseasonalize:
        clim = s.groupby(s.index.month).transform("mean")
        s = s - clim
    return s

def build_anomaly_pair(y_df: pd.DataFrame, x_df: pd.DataFrame, dataset_label: str,
                       metric_key: str, deseasonalize: bool, logger: logging.Logger) -> pd.DataFrame:
    # dLCF/dEIS 统一 LCF 到百分数
    if metric_key == "dLCF_dEIS":
        y_df = maybe_to_percent(y_df)
    y = prepare_series(y_df, deseasonalize)
    x = prepare_series(x_df, deseasonalize)
    paired = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    logger.info("Built anomalies for %s: n=%d", dataset_label, len(paired))
    return paired

# ----------- 回归 -----------
def run_hac_regression(paired: pd.DataFrame, dataset_label: str, hac_lag: int,
                       scale_factor: float, include_intercept: bool, logger: logging.Logger) -> Optional[Dict[str,float]]:
    n = len(paired)
    if n < 24:
        logger.warning("%s: insufficient samples (n=%d).", dataset_label, n)
        return None
    X = sm.add_constant(paired["x"].values) if include_intercept else paired[["x"]].values
    m = sm.OLS(paired["y"].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag})
    idx = 1 if include_intercept else 0
    b    = float(m.params[idx])  * scale_factor
    se   = float(m.bse[idx])     * scale_factor
    tval = float(m.tvalues[idx]); pval = float(m.pvalues[idx]); r2 = float(m.rsquared)
    logger.info("%s OLS(HAC=%d): b=%.4f, SE=%.4f, t=%.2f, p=%.3f, R2=%.3f, n=%d",
                dataset_label, hac_lag, b, se, tval, pval, r2, n)
    return {"b":b, "SE_HAC":se, "t":tval, "p":pval, "R2":r2, "n":n, "sign": bool(pval<0.05)}

# ----------- 单区域处理 -----------
def process_region(region: str, metric_cfg: MetricConfig, manifest,
                   cmip_df: pd.DataFrame, tables_dir: Path, deseasonalize: bool,
                   include_intercept: bool, hac_lag: int, logger: logging.Logger
                   ) -> Optional[Tuple[Dict[str,float], List[Dict[str,float]], Dict[str, Optional[str]]]]:
    logger.info("Region %s / metric %s", region, metric_cfg.key)

    # OBS Y
    y_path = None; tried=[]
    for v in metric_cfg.obs_y_vars:
        y_path, t = resolve_obs_path(manifest, v, region, logger); tried += t
        if y_path: break
    if y_path is None:
        logger.error("OBS Y missing for %s. Tried: %s", region, "; ".join(tried)); return None

    # OBS X
    x_path = None; tried=[]
    for v in metric_cfg.obs_x_vars:
        x_path, t = resolve_obs_path(manifest, v, region, logger); tried += t
        if x_path: break
    if x_path is None:
        logger.error("OBS X missing for %s. Tried: %s", region, "; ".join(tried)); return None

    y_obs = read_monthly_csv(y_path, logger)
    x_obs = read_monthly_csv(x_path, logger)
    pair_obs = build_anomaly_pair(y_obs, x_obs, "OBS", metric_cfg.key, deseasonalize, logger)
    res_obs  = run_hac_regression(pair_obs, "OBS", hac_lag, metric_cfg.scale_factor, include_intercept, logger)
    if res_obs is None: return None

    rows: List[Dict[str,object]] = [{"dataset":"obs","model":"OBS","region":region,"metric":metric_cfg.key, **res_obs}]

    # 模型
    region_df = cmip_df[cmip_df["region"]==region]
    model_rows: List[Dict[str,float]] = []
    plot_info: Dict[str, Optional[str]] = {}

    if region_df.empty:
        logger.warning("No CMIP entries for %s", region)
    else:
        for model in sorted(region_df["model"].unique()):
            md = region_df[region_df["model"]==model]
            y_df = md[["time","y"]].rename(columns={"y":"value"})
            x_df = md[["time","x"]].rename(columns={"x":"value"})
            pair = build_anomaly_pair(y_df, x_df, model, metric_cfg.key, deseasonalize, logger)
            res  = run_hac_regression(pair, model, hac_lag, metric_cfg.scale_factor, include_intercept, logger)
            if res is None: continue
            rows.append({"dataset":"cmip","model":model,"region":region,"metric":metric_cfg.key, **res})
            model_rows.append({"model":model, **res})
            if "family" in md.columns:
                s = md["family"].dropna().astype(str)
                plot_info[model] = s.mode().iloc[0] if not s.empty else None
            else:
                plot_info[model] = None

    # 表
    table_path = tables_dir / f"feedback_fit_{metric_cfg.key}_{region}.csv"
    ensure_dir(table_path.parent)
    tb = pd.DataFrame(rows)[["dataset","model","region","metric","b","SE_HAC","t","p","R2","n","sign"]]
    tb["dataset"] = pd.Categorical(tb["dataset"], categories=["obs","cmip"], ordered=True)
    tb.sort_values(["dataset","model"]).reset_index(drop=True).to_csv(table_path, index=False)
    logger.info("Wrote table: %s", table_path)
    return res_obs, model_rows, plot_info

# ----------- 分面主图 -----------
def plot_facet(metric_cfg: MetricConfig, regions: List[str],
               obs: Dict[str,Dict[str,float]],
               models: Dict[str,List[Dict[str,float]]],
               fig_path: Path, logger: logging.Logger, label_topn: int=4,
               title_suffix: str = "") -> None:
    make_parent(fig_path)
    n=len(regions); ncols = 5 if n>=5 else n; nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8*ncols, 3.6*nrows), sharex=False, sharey=False)
    if nrows*ncols==1: axes=np.array([axes])
    axes=axes.ravel()

    vals=[]
    for r in regions:
        vals.append(obs[r]["b"])
        vals += [m["b"] for m in models.get(r,[])]
    vmin=np.nanmin(vals) if len(vals) else -1.0
    vmax=np.nanmax(vals) if len(vals) else  1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax): vmin, vmax=-1.0,1.0
    if vmin==vmax: vmin-=0.5; vmax+=0.5
    pad=0.1*(vmax-vmin); xlim=(vmin-pad, vmax+pad)

    for ax, region in zip(axes, regions):
        b=obs[region]["b"]; se=obs[region]["SE_HAC"]
        ax.axvspan(b-1.96*se, b+1.96*se, color="0.85", alpha=0.6, zorder=0)
        ax.axvline(b, color="0.3", ls="--", lw=1.2)

        ms = models.get(region, [])
        if ms:
            names=[m["model"] for m in ms]
            xs   = np.array([m["b"] for m in ms], float)
            xerr = np.array([m["SE_HAC"] for m in ms], float)
            yind = np.arange(len(ms))[::-1]
            ax.errorbar(xs, yind, xerr=xerr, fmt=REGION_MARKERS.get(region,"o"),
                        ms=4, lw=1.0, capsize=2, mfc="white", mec="k", ecolor="0.2", alpha=0.95)
            if label_topn > 0:
                idx=np.argsort(np.abs(xs-b))[::-1][:label_topn]
                for i in idx:
                    ax.annotate(names[i], xy=(xs[i], yind[i]), xytext=(4,0),
                                textcoords="offset points", va="center", fontsize=7)
        ax.set_yticks([])
        ax.set_xlim(xlim); ax.set_title(region, fontsize=11, pad=4)
        ax.grid(ls=":", alpha=0.4)

    for k in range(len(regions), len(axes)): axes[k].axis("off")
    fig.supylabel("Models (sorted)", x=0.005)
    fig.supxlabel(f"{metric_cfg.title} ({metric_cfg.unit_label})")
    fig.suptitle(f"{metric_cfg.title} comparison ({', '.join(regions)}, 2003–2014){title_suffix}", y=0.98)
    fig.tight_layout(rect=[0,0.02,1,0.95]); fig.savefig(fig_path, dpi=300); plt.close(fig)
    logger.info("Saved figure: %s", fig_path)

# ----------- 1:1 散点（附录，带 Region 图例） -----------
def plot_xy(metric_cfg: MetricConfig, regions: List[str],
            obs: Dict[str,Dict[str,float]],
            models: Dict[str,List[Dict[str,float]]],
            fig_path: Path, logger: logging.Logger,
            no_text: bool = False, title_suffix: str = "") -> None:
    make_parent(fig_path)
    fig, ax = plt.subplots(figsize=(6,6))
    vals=[]
    legend_handles: Dict[str, Line2D] = {}

    for r in regions:
        ob=obs[r]["b"]; se=obs[r]["SE_HAC"]; ms=models.get(r,[])
        color=REGION_COLORS.get(r,"k")
        marker=REGION_MARKERS.get(r,"o")

        # 为图例准备代理句柄（只加一次）
        if r not in legend_handles:
            legend_handles[r] = Line2D([], [], linestyle="none", marker=marker,
                                       markersize=6, color=color, label=r)

        if ms:
            for row in ms:
                ax.errorbar(ob, row["b"], xerr=se, yerr=row["SE_HAC"],
                            fmt=marker, color=color, ecolor=color,
                            elinewidth=1.0, capsize=3)
                if not no_text:
                    ax.annotate(f"{row['model']} ({r})", (ob, row["b"]),
                                xytext=(5,5), textcoords="offset points", fontsize=8)
            vals.append(ob); vals += [row["b"] for row in ms]

    if not vals:
        logger.warning("No points to plot"); plt.close(fig); return

    vmin=float(np.nanmin(vals)); vmax=float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax): vmin, vmax=-1.0,1.0
    if vmin==vmax: vmin-=0.5; vmax+=0.5
    pad=0.05*(vmax-vmin); vmin-=pad; vmax+=pad

    ax.plot([vmin,vmax],[vmin,vmax], ls="--", c="gray")
    ax.set_xlim(vmin,vmax); ax.set_ylim(vmin,vmax)
    ax.set_xlabel(f"OBS {metric_cfg.title} ({metric_cfg.unit_label})")
    ax.set_ylabel(f"CMIP {metric_cfg.title} ({metric_cfg.unit_label})")
    ax.set_title(f"{metric_cfg.title} comparison ({', '.join(regions)}, 2003–2014){title_suffix}")
    if legend_handles:
        ax.legend(handles=list(legend_handles.values()),
                  title="Region", loc="best", frameon=False)
    ax.grid(ls=":", alpha=0.5); fig.tight_layout(); fig.savefig(fig_path, dpi=300); plt.close(fig)
    logger.info("Saved figure: %s", fig_path)

# ---------------- main ----------------
def main() -> None:
    args = parse_args()
    try:
        metric_keys = parse_metric_keys(args.metrics, args.metric)
    except ValueError as e:
        print(str(e), file=sys.stderr); sys.exit(2)

    regions = load_regions(args.regions)
    hac_lag = args.hac_lags if args.hac_lags is not None else args.hac_lag
    include_intercept = not args.no_intercept

    tables_dir = Path(args.tables_dir) if args.tables_dir else TABLES_DEFAULT
    fig_dir    = Path(args.fig_dir)    if args.fig_dir    else FIG_DEFAULT
    logs_dir   = Path(args.logs_dir)   if args.logs_dir   else LOGS_DEFAULT
    ensure_dir(tables_dir); ensure_dir(fig_dir); ensure_dir(logs_dir)

    logger, log_path = setup_logger(logs_dir)
    logger.info("Log -> %s", log_path)
    logger.info("Metrics: %s", ", ".join(metric_keys))
    logger.info("Regions: %s", ", ".join(regions))
    logger.info("Tables: %s", tables_dir.resolve())
    logger.info("Figures: %s", fig_dir.resolve())

    manifest = load_manifest(args.manifest) if args.manifest else None
    if args.manifest and not manifest:
        logger.warning("Manifest not found or unreadable: %s", args.manifest)

    title_suffix = " (deseasoned)" if args.deseasonalize else " (no deseason)"

    processed_any=False
    for mkey in metric_keys:
        mc = METRIC_CONFIG[mkey]
        # 找 CMIP 堆叠表
        cmip_df, tried = try_find_cmip(mc, manifest, args.cmip_csv, logger)
        if cmip_df is None:
            logger.error("Skip %s: CMIP data not found. Tried: %s", mc.key, "; ".join(tried)); continue

        obs_results: Dict[str,Dict[str,float]] = {}
        model_results: Dict[str,List[Dict[str,float]]] = {}
        ok_regions=[]
        for r in regions:
            out = process_region(r, mc, manifest, cmip_df, tables_dir,
                                 args.deseasonalize, include_intercept, hac_lag, logger)
            if out is None: continue
            obs_result, model_rows, _ = out
            obs_results[r] = obs_result
            model_results[r] = model_rows
            ok_regions.append(r)

        if not ok_regions:
            logger.warning("No region processed for %s", mc.key); continue

        processed_any=True
        token=",".join(ok_regions)
        fpath = fig_dir / f"feedback_fit_{mc.key}_{token}_{args.style}.png"
        if args.style == "facet":
            plot_facet(mc, ok_regions, obs_results, model_results, fpath,
                       logger, args.label_topn, title_suffix)
        else:
            plot_xy(mc, ok_regions, obs_results, model_results, fpath,
                    logger, args.no_text, title_suffix)

    if not processed_any:
        logger.error("No metrics processed successfully."); sys.exit(1)
    logger.info("Done.")

def try_find_cmip(metric_cfg: MetricConfig, manifest, override, logger):
    cands = gather_cmip_candidates(metric_cfg, manifest, override)
    errs=[]; tried=[]
    for c in cands:
        tried.append(str(c))
        df, err = try_load_cmip_table(c, metric_cfg)
        if df is not None:
            logger.info("Loaded CMIP data for %s from %s", metric_cfg.key, c)
            return df, tried
        errs.append(f"{c}: {err}")
    for e in errs: logger.error("  %s", e)
    return None, tried

if __name__ == "__main__":
    main()

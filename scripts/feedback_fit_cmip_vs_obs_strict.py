#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feedback_fit_cmip_vs_obs_strict.py  (STRICT, config-aware, NO GUESSING)

比较（默认 2003–2014，可用 --span 覆盖）：
  (1) λ_cld,SW = d(SWCRE)/d(Ts)
  (2) d(LCF)/d(EIS)

读取优先级（严格按你的目录）：
  1) CERES_<REGION>_monthly.csv   （SWCRE）
     ERA5_<REGION>_monthly.csv    （Ts / EIS）
     MODIS_<REGION>_monthly.csv   （LCF）
  2) 回退：{prefix}_region_mean_{span}_{REGION}.csv
          {prefix}_climatology_{span}_{REGION}.csv

统计：
  * y ~ β x + const
  * HAC(Newey–West) 标准误（--hac-lags；--no-hac 关闭）
  * AR(1) 报告：phi_y, phi_x, Ljung–Box(1) p 值（lb_p_y/x），Durbin–Watson（dw）

输出：
  * tables/feedback_fit_<y>_vs_<x>_<REGION>.csv
  * figures/feedback_fit_<y>_vs_<x>_<SCOPE>.png
  * logs/feedback_fit_*.log （由 setup_logger 写入）

  python scripts/feedback_fit_cmip_vs_obs_strict.py \
  --pair swcre_ts \
  --config configs/config.yaml \
  --regions ALL \
  --span 2003-2022 \
  --deseasonalize --hac-lags 12 \
  --obs-map SWCRE=swcre,Ts=ts \
  --cmip-y output/tables/cmip_panel_clswlow_2003-2014.csv \
  --cmip-x output/tables/cmip_panel_ts_2003-2014.csv
# 如果上一步生成的是 tas 面板，就把最后一行改成：
# --cmip-x output/tables/cmip_panel_tas_2003-2014.csv



"""

from __future__ import annotations
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

try:
    from scripts.utils_config import load_config, get_regions, make_output_path, setup_logger
except ModuleNotFoundError:
    from utils_config import load_config, get_regions, make_output_path, setup_logger

# 预置配对
PAIR_DEF = {
    "swcre_ts": ("SWCRE", "Ts"),
    "lcf_eis":  ("LCF",   "EIS"),
}
REGION_COLORS = {"NEP":"C0","NEA":"C1","SEP":"C2","SEA":"C3","SEI":"C4","GLOBAL":"C5"}

# 严格映射（可 --obs-map 覆盖）
DEFAULT_OBS_MAP = {
    "SWCRE": "swcre",
    "LCF":   "cllmodis",
    "Ts":    "ts",
    "EIS":   "eislts",
}

# 变量 → 文件家族（按你目录）
VAR_FAMILY = {
    "SWCRE": "CERES",
    "LCF":   "MODIS",
    "Ts":    "ERA5",
    "EIS":   "ERA5",
}

# 每个变量在月表中的优先列名
VAR_COLUMNS = {
    "SWCRE": ["swcre", "SWCRE", "value"],
    "LCF":   ["cllmodis", "LCF", "value"],
    "Ts":    ["ts", "TS", "value"],
    "EIS":   ["eislts", "EIS", "value"],
}

# ----------------------------
# CLI & helpers
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="CMIP vs OBS feedback fit (STRICT, config-aware, HAC+AR(1)).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pair", choices=PAIR_DEF.keys(), help="预置配对：swcre_ts 或 lcf_eis")
    g.add_argument("--yvar", help="自定义 y 变量（如 SWCRE/LCF）")
    p.add_argument("--xvar", help="自定义 x 变量（如 Ts/EIS）")

    p.add_argument("--regions", default="ALL", help="逗号分隔或 ALL（使用 config.project.regions）")
    p.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"), help="配置文件路径")
    p.add_argument("--span", default="2003-2014", help="年份段，如 2003-2014 或 2003-2022（用于文件名与截取）")
    p.add_argument("--obs-map", default="", help="覆盖观测前缀映射：SWCRE=clswlow,LCF=cllisccp")

    # CMIP 对照表：若不提供，则按变量名 + span 组装默认路径
    p.add_argument("--cmip-y", default=None, help="CMIP y 表（默认 output/cmip_amip_<y>_vs_obs_<span>.csv）")
    p.add_argument("--cmip-x", default=None, help="CMIP x 表（默认 output/cmip_amip_<x>_vs_obs_<span>.csv）")

    p.add_argument("--deseasonalize", action="store_true", help="对 y 去季节；若 --deseason-both 则 x 也去季节")
    p.add_argument("--deseason-both", action="store_true", help="同时对 x 去季节（默认只对 y）")
    p.add_argument("--no-hac", action="store_true", help="关闭 HAC（回归仍做）")
    p.add_argument("--hac-lags", type=int, default=None, help="HAC maxlags，默认经验公式")
    p.add_argument("--alpha", type=float, default=0.05, help="显著性阈值")
    p.add_argument("--out-prefix", default="feedback_fit", help="输出文件前缀")
    return p.parse_args()

def _parse_obs_map(s: str) -> dict:
    m = {}
    if not s:
        return m
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise SystemExit(f"--obs-map 项格式不对：{tok}；应形如 SWCRE=clswlow")
        k, v = tok.split("=", 1)
        m[k.strip()] = v.strip()
    return m

def _regions_to_run(cfg: dict, args) -> list[str]:
    regdict = get_regions(cfg)  # {'NEP': [...], ...}
    if args.regions.strip().upper() == "ALL":
        return list(regdict.keys())
    rs = [r.strip().upper() for r in args.regions.split(",") if r.strip()]
    missing = [r for r in rs if r not in regdict]
    if missing:
        raise KeyError(f"未知区域键：{missing}；可用={list(regdict.keys())}")
    return rs

def _parse_span(span: str) -> tuple[int,int]:
    s = span.strip()
    if "-" not in s:
        raise ValueError("--span 需形如 2003-2014 或 2003-2022")
    a, b = s.split("-", 1)
    y0, y1 = int(a), int(b)
    if y0 > y1:
        y0, y1 = y1, y0
    return y0, y1

# ----------------------------
# 时间处理 & 预处理
# ----------------------------
def _coerce_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        return df.set_index("time").sort_index()
    if {"year","month"}.issubset(df.columns):
        dt = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
        return df.assign(time=dt).set_index("time").sort_index()
    if "month" in df.columns and "value" in df.columns:
        start = pd.Timestamp("2003-01-01")
        df = df.sort_values("month").reset_index(drop=True)
        df["time"] = [start + pd.DateOffset(months=i) for i in range(len(df))]
        return df.set_index("time")
    raise ValueError("需要 'time' 或 ['year','month']（或至少 'month' + 'value'）")

def deseasonalize(series: pd.Series) -> pd.Series:
    clim = series.groupby(series.index.month).transform("mean")
    anom = series - clim
    return anom - anom.mean()

def center(series: pd.Series) -> pd.Series:
    return series - series.mean()

def nw_maxlags(n: int) -> int:
    return int(np.floor(4 * (n/100) ** (2/9))) if n > 1 else 0

def lag1_phi(series: pd.Series) -> float:
    s = pd.Series(series).dropna().values
    if len(s) < 3:
        return np.nan
    return float(np.corrcoef(s[1:], s[:-1])[0, 1])

def ljungbox_p1(series: pd.Series) -> float:
    s = pd.Series(series).dropna().values
    if len(s) < 8:
        return np.nan
    try:
        lb = acorr_ljungbox(s, lags=[1], return_df=True)
        return float(lb["lb_pvalue"].iloc[0])
    except Exception:
        return np.nan

# ----------------------------
# 回归（HAC + AR(1)）
# ----------------------------
def fit_slope_hac(y_in: pd.Series, x_in: pd.Series, use_hac: bool, hac_lags: int | None) -> dict:
    df = pd.concat([y_in.rename("y"), x_in.rename("x")], axis=1).dropna()
    n = len(df)
    if n < 6:
        return {
            "b": np.nan, "se_hac": np.nan, "t": np.nan, "p": np.nan, "R2": np.nan, "n": n,
            "phi_y": np.nan, "phi_x": np.nan, "lb_p_y": np.nan, "lb_p_x": np.nan, "dw": np.nan,
        }

    phi_y = lag1_phi(df["y"]); phi_x = lag1_phi(df["x"])
    lb_p_y = ljungbox_p1(df["y"]); lb_p_x = ljungbox_p1(df["x"])

    X = sm.add_constant(df["x"].values)
    mod = sm.OLS(df["y"].values, X)
    if use_hac:
        L = hac_lags if hac_lags is not None else nw_maxlags(n)
        res = mod.fit(cov_type="HAC", cov_kwds={"maxlags": L})
    else:
        res = mod.fit()

    b = res.params[1]
    se = res.bse[1]
    t = b / se if se > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(t))) if np.isfinite(t) else np.nan
    R2 = res.rsquared
    dw = sm.stats.stattools.durbin_watson(res.resid) if hasattr(sm.stats, "stattools") else np.nan

    return {
        "b": b, "se_hac": se, "t": t, "p": p, "R2": R2, "n": n,
        "phi_y": phi_y, "phi_x": phi_x, "lb_p_y": lb_p_y, "lb_p_x": lb_p_x,
        "dw": float(dw) if np.isscalar(dw) else np.nan,
    }

# ----------------------------
# 读取观测（严格：先月表家族；后回退）
# ----------------------------
def obs_series(cfg: dict, region: str, var_logic: str, obs_map: dict, span: str, logger) -> pd.Series:
    """
    1) 优先读取家族月表：
         CERES_<REGION>_monthly.csv  (SWCRE)
         ERA5_<REGION>_monthly.csv   (Ts/EIS)
         MODIS_<REGION>_monthly.csv  (LCF)
       -> 从其中选出该变量对应的列名（见 VAR_COLUMNS）
       -> 用 --span 截取年份段
    2) 若 1) 不存在，再回退：
         {prefix}_region_mean_{span}_{REGION}.csv
         {prefix}_climatology_{span}_{REGION}.csv
    """
    family = VAR_FAMILY.get(var_logic)
    prefix = obs_map.get(var_logic)
    if not family or not prefix:
        raise KeyError(f"[OBS-MAP 缺失] var={var_logic} 需要 family 与 prefix；请检查 --obs-map/默认映射")

    out_root = Path(cfg["output"]["root"])
    subdir = cfg["output"]["subdirs"]["regional_monthly"]
    pdir = out_root / subdir

    y0, y1 = _parse_span(span)

    # --- (1) 家族月表优先 ---
    monthly_path = pdir / f"{family}_{region}_monthly.csv"
    if monthly_path.exists():
        logger.info(f"[OBS] {var_logic} → using monthly family file: {monthly_path.name}")
        df = pd.read_csv(monthly_path)
        df = _coerce_time_index(df)
        # 变量列优先列表
        for cname in VAR_COLUMNS.get(var_logic, []):
            if cname in df.columns:
                ser = df[cname].astype(float).rename(var_logic)
                ser = ser[(ser.index.year >= y0) & (ser.index.year <= y1)]
                if ser.empty:
                    raise ValueError(f"{monthly_path.name} 在 {y0}-{y1} 无数据")
                return ser
        # 容错：找最后一个数值列
        non_time = [c for c in df.columns if c not in ("time","year","month")]
        num_cols = [c for c in non_time if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError(f"{monthly_path.name} 无可用数值列，列={list(df.columns)}")
        ser = df[num_cols[-1]].astype(float).rename(var_logic)
        ser = ser[(ser.index.year >= y0) & (ser.index.year <= y1)]
        if ser.empty:
            raise ValueError(f"{monthly_path.name} 在 {y0}-{y1} 无数据")
        return ser

    # --- (2) 回退到 *_region_mean_* / *_climatology_* ---
    candidates = [
        pdir / f"{prefix}_region_mean_{span}_{region}.csv",
        pdir / f"{prefix}_climatology_{span}_{region}.csv",
    ]
    p = next((pp for pp in candidates if pp.exists()), None)
    if p is None:
        tried = "\n  - ".join(x.name for x in [monthly_path] + candidates)
        raise FileNotFoundError(
            "[OBS 缺失] 未找到任何观测文件，已尝试：\n  - " + tried +
            f"\n目录：{pdir}\n→ 请检查 --span 是否与文件一致（如 2003-2022），或 --obs-map 前缀是否正确（{prefix}）"
        )

    logger.info(f"[OBS] {var_logic} → using fallback: {p.name}")
    df = pd.read_csv(p)
    df = _coerce_time_index(df)
    # 列选择：优先 'value'，其次是逻辑名/前缀
    candidates_cols = ["value", var_logic, var_logic.upper(), var_logic.lower(), prefix]
    col = next((c for c in candidates_cols if c in df.columns), None)
    if col is None:
        non_time = [c for c in df.columns if c not in ("time","year","month")]
        num_cols = [c for c in non_time if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError(f"{p.name} 无可用数值列，列={list(df.columns)}")
        col = num_cols[-1]
    ser = df[col].astype(float).rename(var_logic)
    ser = ser[(ser.index.year >= y0) & (ser.index.year <= y1)]
    if ser.empty:
        raise ValueError(f"{p.name} 在 {y0}-{y1} 无数据")
    return ser

# ----------------------------
# CMIP panel
# ----------------------------
def cmip_panel(csv_path: Path, region: str, logger) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"[CMIP 缺失] {csv_path}")
    logger.info(f"[CMIP] {csv_path}")
    df = pd.read_csv(csv_path)

    if "time" not in df.columns:
        if {"year","month"}.issubset(df.columns):
            dt = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
            df["time"] = dt
        elif "month" in df.columns:
            start = pd.Timestamp("2003-01-01")
            df = df.sort_values(["region","model","month"])
            df["time"] = [start + pd.DateOffset(months=i % 144) for i in range(len(df))]
        else:
            raise ValueError("CMIP 表缺少时间列")

    if {"model","value","region"}.issubset(df.columns):
        df["time"] = pd.to_datetime(df["time"])
        out = df.loc[df["region"].str.upper()==region.upper(), ["time","model","value"]]
        return out.sort_values(["model","time"]).reset_index(drop=True)

    df["region"] = df["region"].str.upper()
    df = df[df["region"]==region.upper()].copy()
    key = {"region","time","year","month"} & set(df.columns)
    model_cols = [c for c in df.columns if c not in key]
    long = df.melt(id_vars=[c for c in ["time","region"] if c in df.columns],
                   value_vars=model_cols, var_name="model", value_name="value")
    return long.sort_values(["model","time"]).reset_index(drop=True)

# ----------------------------
# 主区域流程
# ----------------------------
def run_region(cfg: dict, region: str, yvar: str, xvar: str,
               obs_map: dict, span: str, deseason_y: bool, deseason_both: bool,
               use_hac: bool, hac_lags: int | None, alpha: float,
               cmip_y_path: Path, cmip_x_path: Path, logger):
    # OBS
    y_obs = obs_series(cfg, region, yvar, obs_map, span, logger)
    x_obs = obs_series(cfg, region, xvar, obs_map, span, logger)
    y_obs, x_obs = y_obs.align(x_obs, join="inner")

    if deseason_y:
        y_in = deseasonalize(y_obs)
        x_in = deseasonalize(x_obs) if deseason_both else center(x_obs)
    else:
        y_in = center(y_obs)
        x_in = center(x_obs)

    stat_obs = fit_slope_hac(y_in, x_in, use_hac=use_hac, hac_lags=hac_lags)
    row_obs = {"dataset":"obs","model":"OBS","region":region,"pair":f"{yvar}~{xvar}", **stat_obs}

    # CMIP
    pan_y = cmip_panel(cmip_y_path, region, logger).copy()
    pan_x = cmip_panel(cmip_x_path, region, logger).copy()
    
    # 统一为 pandas datetime，并派生 年-月 键
    pan_y["time"]  = pd.to_datetime(pan_y["time"])
    pan_x["time"]  = pd.to_datetime(pan_x["time"])
    pan_y["year"]  = pan_y["time"].dt.year
    pan_y["month"] = pan_y["time"].dt.month
    pan_x["year"]  = pan_x["time"].dt.year
    pan_x["month"] = pan_x["time"].dt.month
    
    # 仅用 (model, year, month) 合并，避免日历/日不同造成错失交集
    merged = (
        pan_y.merge(
            pan_x,
            on=["model", "year", "month"],
            suffixes=("_y", "_x"),
            how="inner",
        )
        .sort_values(["model", "year", "month"])
    )
    
    # 记录一下规模，便于排错
    logger.info(f"[MERGE] {region}: y rows={len(pan_y)} x rows={len(pan_x)} -> merged={len(merged)}")
    
    # 生成统一的 time（每月第一天）
    merged["time"] = pd.to_datetime(dict(year=merged["year"], month=merged["month"], day=1))


    rows = [row_obs]
    for mdl, dd in merged.groupby("model"):
        y = pd.Series(merged["value_y"].values, index=merged["time"])
        x = pd.Series(merged["value_x"].values, index=merged["time"])

        if deseason_y:
            y_in = deseasonalize(y)
            x_in = deseasonalize(x) if deseason_both else center(x)
        else:
            y_in = center(y)
            x_in = center(x)
        stat = fit_slope_hac(y_in, x_in, use_hac=use_hac, hac_lags=hac_lags)
        rows.append({"dataset":"cmip","model":str(mdl),"region":region,"pair":f"{yvar}~{xvar}", **stat})

    out = pd.DataFrame(rows)
    out["sign"] = np.where(out["p"]<=alpha, np.where(out["b"]>0, "+","-"), "0")
    return out

# ----------------------------
# 绘图
# ----------------------------
def plot_model_vs_obs(df_all: pd.DataFrame, pair: tuple[str,str], scope: str, fig_path: Path):
    yvar, xvar = pair
    fig, ax = plt.subplots(figsize=(6,6), dpi=160)
    obs_b = df_all[df_all["dataset"]=="obs"][["region","b","se_hac"]].set_index("region")
    for region, sub in df_all[df_all["dataset"]=="cmip"].groupby("region"):
        if region not in obs_b.index:
            continue
        x0 = float(obs_b.loc[region,"b"])
        xerr = float(obs_b.loc[region,"se_hac"]) if np.isfinite(obs_b.loc[region,"se_hac"]) else None
        ax.errorbar([x0]*len(sub), sub["b"].values, xerr=None if xerr is None else xerr,
                    yerr=sub["se_hac"].values, fmt='o', ms=4, alpha=0.85,
                    color=REGION_COLORS.get(region, None), label=region)
    lim = np.nanmax(np.abs(df_all["b"].values))
    lim = 1.05*lim if (np.isfinite(lim) and lim>0) else 1
    ax.plot([-lim, lim], [-lim, lim], lw=1, ls='--')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"b_obs: d({yvar})/d({xvar})"); ax.set_ylabel(f"b_model: d({yvar})/d({xvar})")
    ax.set_title(f"Model vs OBS slope ({yvar}~{xvar}) [{scope}]"); ax.grid(True, ls=":", alpha=0.5)
    handles = [plt.Line2D([0],[0], marker='o', linestyle='', color=c, label=r) for r,c in REGION_COLORS.items()]
    ax.legend(handles=handles, title="Region", ncol=2, frameon=True)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(fig_path); plt.close(fig)

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger(cfg, name="feedback_fit")
    logger.info(f"[CONFIG] loaded from {args.config}")

    if args.pair:
        yvar, xvar = PAIR_DEF[args.pair]
    else:
        if not (args.yvar and args.xvar):
            raise SystemExit("必须提供 --pair 或同时提供 --yvar 与 --xvar")
        yvar, xvar = args.yvar, args.xvar
    logger.info(f"[PAIR] {yvar} ~ {xvar}")

    regions = _regions_to_run(cfg, args)
    logger.info(f"[REGIONS] {regions}")

    obs_map = DEFAULT_OBS_MAP.copy()
    obs_map.update(_parse_obs_map(args.obs_map))
    logger.info(f"[OBS-MAP] {obs_map}")

    # CMIP 默认对照表随 span 变化（仍可通过 --cmip-y/--cmip-x 覆盖）
    cmip_y_path = Path(args.cmip_y or f"output/cmip_amip_{yvar.lower()}_vs_obs_{args.span}.csv")
    cmip_x_path = Path(args.cmip_x or f"output/cmip_amip_{xvar.lower()}_vs_obs_{args.span}.csv")

    # 跑
    all_rows = []
    for r in regions:
        try:
            df = run_region(
                cfg=cfg, region=r, yvar=yvar, xvar=xvar,
                obs_map=obs_map, span=args.span,
                deseason_y=args.deseasonalize, deseason_both=args.deseason_both,
                use_hac=(not args.no_hac), hac_lags=args.hac_lags, alpha=args.alpha,
                cmip_y_path=cmip_y_path, cmip_x_path=cmip_x_path, logger=logger
            )
        except FileNotFoundError as e:
            logger.error(f"[SKIP] {r}: {e}")
            continue
        except Exception as e:
            logger.exception(f"[FAIL] {r}: {e}")
            continue

        out_csv = make_output_path(cfg, "tables", f"{args.out_prefix}_{yvar.lower()}_vs_{xvar.lower()}_{r}.csv")
        df.to_csv(out_csv, index=False)
        logger.info(f"[OUT] {out_csv}")
        all_rows.append(df)

    if not all_rows:
        logger.error("No region finished successfully. Abort plotting.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    scope = "ALL" if len({r for r in df_all['region']}) > 1 else df_all['region'].iloc[0]

    fig_path = make_output_path(cfg, "figures", f"{args.out_prefix}_{yvar.lower()}_vs_{xvar.lower()}_{scope}.png")
    try:
        plot_model_vs_obs(df_all, (yvar,xvar), scope, fig_path)
        logger.info(f"[FIG] {fig_path}")
    except Exception as e:
        logger.exception(f"[PLOT FAIL] {e}")

    out_csv_all = make_output_path(cfg, "tables", f"{args.out_prefix}_{yvar.lower()}_vs_{xvar.lower()}_{scope}.csv")
    df_all.to_csv(out_csv_all, index=False)
    logger.info(f"[OUT] {out_csv_all}")

if __name__ == "__main__":
    main()

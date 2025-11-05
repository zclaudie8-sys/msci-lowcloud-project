#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feedback_fit_cmip_vs_obs.py

比较观测与 CMIP6 AMIP 模型在 2003–2014 年的两类反馈/敏感度：
1) λ_cld,SW = d(SWCRE)/d(Ts)            （短波低云辐射反馈）
2) d(LCF)/d(EIS)                         （稳定度—低云量敏感度）

输入（默认文件模板，可用 --indir_* 参数覆盖）：
- 观测 OBS（需为月度序列 CSV）：
  output/regional_monthly/{var}_mean_2003-2014_<REGION>.csv
  以及对应的预测变量：
    * var in [swcre, clswlow]  -> ts_mean_2003-2014_<REGION>.csv
    * var == cllmodis          -> eislts_mean_2003-2014_<REGION>.csv
- 模式 CMIP：
  * 模型名列表来源：output/cmip_amip_{var}_vs_obs_2003-2014.csv
  * 每个模型的月度序列放在：
    output/regional_monthly/cmip_amip/{MODEL}_{var}_mean_2003-2014_<REGION>.csv
    和
    output/regional_monthly/cmip_amip/{MODEL}_{predictor}_mean_2003-2014_<REGION>.csv

输出：
- 表：   output/tables/feedback_fit_<var>_<region>.csv
- 图：   output/figures/feedback_fit_<var>_<region>.png
- 日志： output/logs/feedback_fit_*.log

回归方法：
- OLS + Newey–West(HAC) 标准误，maxlags=--lags（默认 1）
- 结果包含：b, SE_HAC, t, p, R², n, sign, dataset, model

依赖：pandas, numpy, matplotlib, statsmodels
"""

from __future__ import annotations
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from logging.handlers import RotatingFileHandler
import statsmodels.api as sm


# -------------------------- 日志 --------------------------
def setup_logger(name: str, outdir_log: Path) -> logging.Logger:
    outdir_log.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(outdir_log / f"{name}.log", maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(); sh.setFormatter(fmt); sh.setLevel(logging.INFO)
    logger.addHandler(fh); logger.addHandler(sh)
    logger.info(f"Logger started. Writing to: {outdir_log}")
    return logger


# -------------------------- 参数 --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compare feedback/sensitivity fits: obs vs CMIP AMIP (2003–2014).")
    p.add_argument("--region", required=True, help="NEP/NEA/SEP/SEA/SEI")
    p.add_argument("--var", required=True, choices=["swcre", "clswlow", "cllmodis"],
                   help="目标：swcre/clswlow -> d(SWCRE)/d(Ts); cllmodis -> d(LCF)/d(EIS)")
    p.add_argument("--lags", type=int, default=1, help="Newey–West(HAC) maxlags (default=1)")
    # I/O 覆盖（可不改）
    p.add_argument("--indir_obs", default="output/regional_monthly", help="观测月表目录")
    p.add_argument("--indir_modellist", default="output", help="CMIP 模型列表文件所在的根目录")
    p.add_argument("--indir_cmip_series", default="output/regional_monthly/cmip_amip",
                   help="CMIP 模型的月表目录")
    p.add_argument("--out_tables", default="output/tables", help="输出表目录")
    p.add_argument("--out_figs", default="output/figures", help="输出图目录")
    p.add_argument("--out_logs", default="output/logs", help="日志目录")
    return p.parse_args()


# -------------------------- 工具函数 --------------------------
def log_info(logger: logging.Logger, msg: str):
    logger.info(msg); print(msg)

def read_two_col_csv(path: Path, logger: logging.Logger) -> pd.Series:
    """读取两列或可识别列的 CSV（时间/月份列 + 值列），返回按月份排序的 pd.Series。"""
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # 尝试识别列名
    tcol = cols.get("time") or cols.get("month") or cols.get("m") or list(df.columns)[0]
    vcol = None
    for key in ["value", "val", "y", "series", "climate", "mean", "data"]:
        if key in cols:
            vcol = cols[key]; break
    if vcol is None:
        # 退化：取非时间列里最后一列
        non_time = [c for c in df.columns if c != tcol]
        vcol = non_time[-1]
    s = df[[tcol, vcol]].dropna()
    s.columns = ["t", "v"]
    # 若 t 是 yyyy-mm 列，取月份序（2003-01 → 1 ... 2003-12 → 12 ...）
    try:
        t = pd.to_datetime(s["t"])
        mseq = (t.dt.year - t.dt.year.min()) * 12 + t.dt.month  # 连续月序
        s = pd.Series(s["v"].values, index=mseq.values, name="value")
    except Exception:
        # 如果本就是 1..N 的月份序/索引
        s = pd.Series(s["v"].values, index=pd.to_numeric(s["t"], errors="coerce"), name="value")
    s = s.sort_index()
    logger.info(f"Read {path.name}: {len(s)} points")
    return s

def hac_fit(y: pd.Series, x: pd.Series, lags: int):
    """OLS + HAC 标准误；返回 dict：b, se, t, p, rsq, n。"""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    n = len(df)
    if n < 8:
        return None
    X = sm.add_constant(df["x"].values)
    mod = sm.OLS(df["y"].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    b = float(mod.params[1]); se = float(mod.bse[1]); t = float(mod.tvalues[1]); p = float(mod.pvalues[1])
    rsq = float(mod.rsquared)
    return {"b": b, "se": se, "t": t, "p": p, "rsq": rsq, "n": n, "sign": (p < 0.05)}


# -------------------------- 主逻辑 --------------------------
def main():
    args = parse_args()
    region = args.region.upper()
    var = args.var.lower()
    lags = int(args.lags)

    outdir_tab = Path(args.out_tables); outdir_tab.mkdir(parents=True, exist_ok=True)
    outdir_fig = Path(args.out_figs);   outdir_fig.mkdir(parents=True, exist_ok=True)
    outdir_log = Path(args.out_logs)

    logger = setup_logger(f"feedback_fit_{var}_{region}", outdir_log)
    log_info(logger, f"[CONFIG] var={var}, region={region}, lags={lags}")

    # 变量映射：被解释变量 y 及其预测量 predictor
    # swcre/clswlow -> ts ; cllmodis -> eislts
    if var in ["swcre", "clswlow"]:
        y_name_obs = "swcre" if var == "swcre" else "clswlow"
        x_name_obs = "ts"
        ylabel = "SWCRE (W m⁻²)"
        xlabel = "Ts (K)"
        var_label = "lambda_cld_SW"
    elif var == "cllmodis":
        y_name_obs = "cllmodis"
        x_name_obs = "eislts"
        ylabel = "LCF (%)"
        xlabel = "EIS (K)"
        var_label = "dLCF_dEIS"
    else:
        raise SystemExit("Unknown var")

    # ---------- 读取观测 ----------
    obs_dir = Path(args.indir_obs)

    # y (被解释变量)
    obs_y_file = obs_dir / f"{y_name_obs}_mean_2003-2014_{region}.csv"
    obs_y = read_two_col_csv(obs_y_file, logger)

    # x (预测变量)
    obs_x_file = obs_dir / f"{x_name_obs}_mean_2003-2014_{region}.csv"
    obs_x = read_two_col_csv(obs_x_file, logger)

    # HAC 拟合（观测）
    obs_fit = hac_fit(obs_y, obs_x, lags)
    if obs_fit is None:
        raise SystemExit("Not enough OBS samples for regression.")

    obs_row = {
        "dataset": "OBS", "model": "OBS", "region": region, "var": var_label,
        **obs_fit
    }
    logger.info(f"[OBS] b={obs_fit['b']:.3f} ± {obs_fit['se']:.3f}, R²={obs_fit['rsq']:.3f}, n={obs_fit['n']}")

    # ---------- 读取模型名列表 ----------
    # 用 cmip_amip_{var}_vs_obs_2003-2014.csv 仅提取 model 列
    model_list_file = Path(args.indir_modellist) / f"cmip_amip_{var}_vs_obs_2003-2014.csv"
    if not model_list_file.exists():
        logger.warning(f"Model list file not found: {model_list_file}. Will try to discover series from directory.")
        models = sorted({p.name.split("_")[0] for p in Path(args.indir_cmip_series).glob(f"*_{y_name_obs}_mean_2003-2014_{region}.csv")})
    else:
        df_ml = pd.read_csv(model_list_file)
        if "model" in df_ml.columns:
            models = sorted(set(df_ml["model"].astype(str)))
        else:
            models = sorted({p.name.split("_")[0] for p in Path(args.indir_cmip_series).glob(f"*_{y_name_obs}_mean_2003-2014_{region}.csv")})
    logger.info(f"[MODELS] found {len(models)} candidate models")

    # ---------- 遍历模型 ----------
    rows = [obs_row]
    skipped = 0
    for m in models:
        y_path = Path(args.indir_cmip_series) / f"{m}_{y_name_obs}_mean_2003-2014_{region}.csv"
        x_path = Path(args.indir_cmip_series) / f"{m}_{x_name_obs}_mean_2003-2014_{region}.csv"
        if not y_path.exists() or not x_path.exists():
            logger.warning(f"skip {m}: series missing -> {y_path.exists()=}, {x_path.exists()=}")
            skipped += 1
            continue
        y = read_two_col_csv(y_path, logger)
        x = read_two_col_csv(x_path, logger)
        fit = hac_fit(y, x, lags)
        if fit is None:
            logger.warning(f"skip {m}: insufficient samples after dropna")
            skipped += 1
            continue
        row = {"dataset": "MODEL", "model": m, "region": region, "var": var_label, **fit}
        rows.append(row)

    logger.info(f"[SUMMARY] models processed: {len(rows)-1}, skipped: {skipped}")

    # ---------- 输出表 ----------
    out_df = pd.DataFrame(rows)
    out_csv = outdir_tab / f"feedback_fit_{var}_{region}.csv"
    out_df.to_csv(out_csv, index=False)
    logger.info(f"Saved table -> {out_csv}")

    # ---------- 作图（b_model vs b_obs，含误差条） ----------
    df_model = out_df[out_df["dataset"] == "MODEL"].copy()
    if not df_model.empty:
        b_obs = obs_fit["b"]; se_obs = obs_fit["se"]
        xvals = np.full(len(df_model), b_obs)
        yvals = df_model["b"].values
        yerr  = df_model["se"].values
        # 1:1 参考线（斜率=1，过原点）——在 x 轴上以 b 为单位
        # 我们将把 x、y 都以“斜率 b”为轴，绘 (b_obs, b_model)
        lim_min = min(np.min(yvals - yerr), b_obs - se_obs) - 0.1*abs(b_obs)
        lim_max = max(np.max(yvals + yerr), b_obs + se_obs) + 0.1*abs(b_obs)
        if lim_min == lim_max:
            lim_min -= 1.0; lim_max += 1.0

        plt.figure(figsize=(7.2, 6.2))
        # 观测（竖线+误差区间）
        plt.axvline(b_obs, color="k", lw=1.2, ls="--", alpha=0.8, label="OBS b")
        plt.fill_betweenx([lim_min, lim_max], b_obs - se_obs, b_obs + se_obs, color="gray", alpha=0.15)

        # 模型点（颜色按 region，可选，这里同一 region 一个颜色）
        plt.errorbar(xvals, yvals, yerr=yerr, fmt="o", ms=5, mfc="#1f77b4", mec="none",
                     ecolor="#1f77b4", elinewidth=1.0, capsize=2.5, alpha=0.9, label="MODEL b ± SE")

        # 1:1 线（把 y=x 映射到当前坐标）
        xs = np.linspace(lim_min, lim_max, 200)
        plt.plot(xs, xs, color="orange", lw=1.2, alpha=0.8, label="1:1 line")

        # 美化
        plt.xlim(lim_min, lim_max); plt.ylim(lim_min, lim_max)
        plt.xlabel(f"b_obs  ({var_label})")
        plt.ylabel(f"b_model ({var_label})")
        plt.title(f"Feedback fit ({var_label}) | {region} | 2003–2014  (HAC lags={lags})")
        plt.grid(alpha=0.25, ls=":")
        plt.legend(frameon=False, fontsize=9, loc="upper left")

        # 模型名标注（轻度避免重叠）
        for xi, yi, name in zip(xvals, yvals, df_model["model"].values):
            plt.text(xi + 0.01*(lim_max-lim_min), yi, name, fontsize=7.5, va="center", alpha=0.85)

        fig_path = outdir_fig / f"feedback_fit_{var}_{region}.png"
        plt.tight_layout(); plt.savefig(fig_path, dpi=300, bbox_inches="tight"); plt.close()
        logger.info(f"Saved figure -> {fig_path}")
    else:
        logger.warning("No MODEL rows -> skip figure.")

    logger.info("DONE.")


if __name__ == "__main__":
    main()

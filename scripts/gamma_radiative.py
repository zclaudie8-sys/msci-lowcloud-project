#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monthly radiative sensitivity:
  SWCRE'_m = gamma^(m) * LCF'_m + eta_m
Optionally binned by optical depth (TAU) or effective radius (RE):
  gamma(m | bin)

Inputs (per region):
  - output/regional_monthly/MODIS_<REGION>_monthly.csv   (MODIS_LCF; optional MODIS_TAU / MODIS_RE)
  - output/regional_monthly/CERES_<REGION>_monthly.csv   (CERES_SWCRE)

Preprocessing:
  - Align to month-start timestamps
  - Per-month Z-score for both SWCRE and LCF (and bin variable if needed)
  - OLS with Newey–West (HAC) SE (lag = --lags)

Outputs:
  - output/gamma/GAMMA_<REGION>.csv
  - output/gamma/GAMMA_HARM_<REGION>.csv
  - output/tables/Table3_gamma.csv   (append/update for this region; incl. mean±CI and A,phi)
  - output/figures/Fig3_swcre_vs_lcf_<REGION>.png

Usage:
  python scripts/gamma_radiative.py --region NEP --lags 1
  python scripts/gamma_radiative.py --region NEP --bin_by TAU --bins 3 --lags 1
  # 单区，不分箱
python scripts/gamma_radiative.py --region NEP --lags 1

# 单区，分箱（τ 三分位）
python scripts/gamma_radiative.py --region NEP --bin_by TAU --bins 3 --lags 1

# 一次跑五区（不分箱）
python scripts/gamma_radiative.py --regions ALL --lags 1

# 一次跑五区（分箱）
python scripts/gamma_radiative.py --regions ALL --bin_by RE --bins 4 --lags 1

"""

from __future__ import annotations
from pathlib import Path
import sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

# robust imports
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
try:
    from scripts.utils_config import load_config, make_output_path, setup_logger
    import scripts.utils_climatology as climu
except ModuleNotFoundError:
    from utils_config import load_config, make_output_path, setup_logger
    import utils_climatology as climu

# statsmodels
import statsmodels.api as sm


# -------------------- ARGS --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Monthly radiative sensitivity γ: SWCRE' ~ γ * LCF' (+ optional binning).")
    p.add_argument("--region", help="NEP/NEA/SEP/SEA/SEI")
    p.add_argument("--regions", default="", help="Comma list or ALL (uses project.regions)")
    p.add_argument("--lags", type=int, default=1, help="Newey–West (HAC) max lag for SEs (default=1)")
    p.add_argument("--bin_by", choices=["TAU", "RE"], default="", help="Optional binning variable from MODIS")
    p.add_argument("--bins", type=int, default=3, help="Number of quantile bins if binning (default=3)")
    return p.parse_args()


# -------------------- HELPERS --------------------
def _regions_to_run(cfg: dict, args) -> list[str]:
    """根据 --regions / --region 返回要运行的 region 列表。"""
    if args.regions:
        rs = [r.strip() for r in args.regions.split(",") if r.strip()]
        if len(rs) == 1 and rs[0].upper() == "ALL":
            rs = cfg["project"]["regions"]
        return rs
    if args.region:
        return [args.region]
    raise SystemExit("请提供 --region 或 --regions（可用 ALL）")

def _to_month_start_index(obj):
    x = obj.copy()
    idx = pd.to_datetime(x.index, errors="coerce")
    idx = idx.to_period("M").start_time  # month start
    x.index = idx
    return x.sort_index()

def _zscore_by_month(series: pd.Series) -> pd.Series:
    def _z(g):
        mu = g.mean()
        sd = g.std(ddof=1)
        return (g - mu) / sd if (sd is not None and np.isfinite(sd) and sd > 0) else (g - mu)
    return series.groupby(series.index.month, group_keys=False).apply(_z)

def _read_modis(cfg: dict, region: str, logger):
    """
    优先读 output/regional_monthly/MODIS_<REGION>_monthly.csv（需包含 MODIS_LCF；可选 MODIS_TAU / MODIS_RE）。
    若不存在，则回退到 cllmodis_region_mean_*_<REGION>.csv（两列或无表头），构造最小月表（仅 MODIS_LCF）。
    """
    regdir = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["regional_monthly"]
    f = regdir / f"MODIS_{region}_monthly.csv"
    if f.exists():
        df = pd.read_csv(f)
        if df.shape[0] > 0 and str(df.iloc[0,0]).strip().lower() == "time":
            df = pd.read_csv(f, header=0)
        if "time" not in [c.lower() for c in df.columns]:
            df = df.rename(columns={df.columns[0]: "time"})
        df["time"] = pd.to_datetime(df["time"], errors="coerce", format="%Y-%m-%d")
        df = df.dropna(subset=["time"]).set_index("time").sort_index()
        # 规范 LCF 列名
        if "MODIS_LCF" not in df.columns:
            cand = None
            for c in df.columns:
                if c.lower() in {"lcf","cllmodis","modis_lcf"}:
                    cand = c; break
            if cand:
                df = df.rename(columns={cand: "MODIS_LCF"})
        if "MODIS_LCF" not in df.columns:
            raise KeyError(f"{f.name} 中找不到 MODIS_LCF 列。现有列：{list(df.columns)}")
        return df

    # ---- 回退：cllmodis_region_mean_*_<REGION>.csv → 最小月表（只含 MODIS_LCF）----
    pat = f"cllmodis_region_mean_*_{region}.csv"
    cands = sorted(regdir.glob(pat))
    if not cands:
        raise FileNotFoundError(
            f"MODIS monthly not found: {f}\n且未找到回退文件：{pat}\n"
            f"请先生成 MODIS 月表，或至少有 cllmodis_region_mean_*_{region}.csv。"
        )
    g = cands[-1]
    # 兼容两种写法：无表头两列；或第一行是 'time'
    df = pd.read_csv(g)
    if df.shape[0] > 0 and str(df.iloc[0,0]).strip().lower() == "time":
        df = pd.read_csv(g, header=0)
    if len(df.columns) == 2 and df.columns[0].lower() != "time":
        df.columns = ["time", "MODIS_LCF"]
    if "time" not in [c.lower() for c in df.columns]:
        df.columns = ["time", "MODIS_LCF"]
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    if "MODIS_LCF" not in df.columns:
        # 找除 time 外的第一列
        valcol = [c for c in df.columns if c.lower() != "time"]
        if not valcol:
            raise KeyError(f"{g.name} 解析失败：没有数值列。")
        df = df.rename(columns={valcol[0]: "MODIS_LCF"})
    logger.info(f"[MODIS] 回退使用 {g.name} 组装最小月表（仅 MODIS_LCF）")
    return df


def _read_ceres(cfg: dict, region: str):
    regdir = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["regional_monthly"]
    f = regdir / f"CERES_{region}_monthly.csv"
    if not f.exists():
        raise FileNotFoundError(f"CERES monthly not found: {f}")
    df = pd.read_csv(f)
    if df.shape[0] > 0 and str(df.iloc[0,0]).strip().lower() == "time":
        df = pd.read_csv(f, header=0)
    if "time" not in [c.lower() for c in df.columns]:
        df = df.rename(columns={df.columns[0]: "time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce", format="%Y-%m-%d")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    if "CERES_SWCRE" not in df.columns:
        # 一些版本变量名可能不同，这里给个友好提示
        raise KeyError(f"CERES monthly missing CERES_SWCRE column: {f}. "
                       f"Make sure you mapped swcld to CERES_SWCRE in CERES monthly.")
    return df

def _hac_ols_through_origin(y: pd.Series, x: pd.Series, lags: int):
    """
    OLS with intercept constrained to 0 (standardized anomalies => theory suggests no intercept).
    HAC (Newey–West) SE with given maxlags.
    Returns: dict {gamma, se, t, p, ci_low, ci_high, n, rsq}
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    n = len(df)
    if n < 8:
        return None
    # No constant, slope only
    model = sm.OLS(df["y"].values, df["x"].values)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    gamma = float(res.params[0])
    se = float(res.bse[0])
    t = float(res.tvalues[0])
    p = float(res.pvalues[0])
    z = 1.96
    ci_low = gamma - z * se
    ci_high = gamma + z * se
    # rsq relative to zero-intercept fit
    yhat = df["x"] * gamma
    ss_res = float(((df["y"] - yhat)**2).sum())
    ss_tot = float(((df["y"])**2).sum())  # since mean≈0 after z-score-by-month
    rsq = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return {"gamma": gamma, "se": se, "t": t, "p": p, "ci_low": ci_low, "ci_high": ci_high, "n": n, "rsq": rsq}

def _harmonics_gamma(gm_series: pd.Series):
    """gm_series indexed by month 1..12 -> first harmonic using utils_climatology."""
    months = np.arange(1, 13)
    y = np.array([gm_series.get(m, np.nan) for m in months], dtype=float)
    da = xr.DataArray(y, coords={"month": months}, dims=("month",), name="gamma_m")
    fit = climu.first_harmonic_fit(da, min_valid=8, soft=True)
    return {
        "A": float(fit["amplitude"]),
        "phase_deg": float(fit["phase_deg"]),
        "peak_month": int(fit["peak_month"]),
        "r2_gamma": float(fit["r2"]),
        "n_valid": float(fit.get("n_valid", np.sum(np.isfinite(y)))),
    }

def _quantile_bins(series: pd.Series, K: int) -> pd.Series:
    """Global-quantile bin labels 1..K; ties handled; drop rows with NaN."""
    s = series.dropna()
    if s.empty or K < 2:
        return pd.Series(index=series.index, dtype=float)
    q = np.linspace(0, 1, K+1)
    edges = s.quantile(q).values
    # ensure strictly increasing edges (avoid duplicates due to ties)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-12
    labels = pd.cut(series, bins=edges, include_lowest=True, labels=range(1, K+1))
    return labels.astype("float")


# -------------------- MAIN --------------------
def main():
    args = parse_args()
    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    logger = setup_logger(cfg, name="gamma_radiative")

    # 运行的区域列表
    regions = _regions_to_run(cfg, args)

    # 统一本次运行的配置
    lags = max(0, int(args.lags))
    do_bin = bool(args.bin_by)
    K = int(args.bins) if do_bin else 0
    bin_var_name = f"MODIS_{args.bin_by}" if do_bin else ""
    tag = "unbinned" if not do_bin else f"binned_{args.bin_by}_{K}"
    logger.info(f"Run tag: {tag} | lags={lags} | bin_by={args.bin_by or 'NONE'} | bins={K if do_bin else 0}")

    for region in regions:
        logger.info(f"[RUN] region={region}")

        # ---- 读取 & 对齐 ----
        mod = _read_modis(cfg, region, logger)
        cer = _read_ceres(cfg, region)
        mod = _to_month_start_index(mod)
        cer = _to_month_start_index(cer)
        df = pd.concat([mod, cer], axis=1)

        # 分箱（若启用）
        # 先假定按全局请求分箱
        use_bin = bool(args.bin_by)
        tag_local = "unbinned"  # 每个区域单独的 tag（可能与全局不同）
        
        if use_bin:
            if bin_var_name not in df.columns:
                logger.warning(
                    f"[{region}] 请求分箱 {args.bin_by}，但缺少列 {bin_var_name}；本区降级为 unbinned。"
                )
                use_bin = False
                tag_local = "unbinned"
            else:
                df["_bin"] = _quantile_bins(df[bin_var_name], K)
                if df["_bin"].isna().all():
                    logger.warning(f"[{region}] 分箱标签全 NaN；本区降级为 unbinned。")
                    df.drop(columns=["_bin"], inplace=True)
                    use_bin = False
                    tag_local = "unbinned"
                else:
                    tag_local = f"binned_{args.bin_by}_{K}"
        else:
            tag_local = "unbinned"


        # 月内 Z-score
        df["_LCFz"]   = _zscore_by_month(df["MODIS_LCF"])
        df["_SWCREz"] = _zscore_by_month(df["CERES_SWCRE"])

        # ---- 逐月回归 ----
        rows = []
        gamma_by_month = pd.Series(index=pd.Index(range(1,13), name="month"), dtype=float)

        for m in range(1, 13):
            sub = df[df.index.month == m].copy().dropna(subset=["_LCFz","_SWCREz"])
            if len(sub) == 0:
                logger.warning(f"[{region}] month={m}: no samples → skip")
                continue

            if not use_bin:
                res = _hac_ols_through_origin(sub["_SWCREz"], sub["_LCFz"], lags)
                if res is None:
                    logger.warning(f"[{region}] month={m}: too few samples → skip")
                    continue
                rows.append({"region": region, "month": m, "bin": np.nan, **res})
                gamma_by_month.loc[m] = res["gamma"]
            else:
                # 按 bin 回归
                for b in range(1, K+1):
                    subb = sub[sub["_bin"] == b]
                    if len(subb) < 8:
                        logger.warning(f"[{region}] month={m}, bin={b}: n={len(subb)} < 8 → skip")
                        continue
                    res = _hac_ols_through_origin(subb["_SWCREz"], subb["_LCFz"], lags)
                    if res is None:
                        logger.warning(f"[{region}] month={m}, bin={b}: regression failed → skip")
                        continue
                    rows.append({"region": region, "month": m, "bin": b, **res})
                # 同时计算“本月整体 γ(m)”（用于谐波和表3）
                res_all = _hac_ols_through_origin(sub["_SWCREz"], sub["_LCFz"], lags)
                if res_all is not None:
                    gamma_by_month.loc[m] = res_all["gamma"]

        if not rows:
            logger.warning(f"[{region}] No monthly gamma estimated, skip outputs for this region.")
            continue

        out_df = pd.DataFrame(rows)

        # ---- 保存主表（带 tag）----
        out_csv = make_output_path(cfg, "gamma", f"GAMMA_{region}__{tag_local}.csv")
        cols = ["region","month","bin","gamma","se","t","p","ci_low","ci_high","n","rsq"]
        out_df[cols].to_csv(out_csv, index=False)
        logger.info(f"[{region}] Saved gamma table → {out_csv}")

        # ---- 谐波（带 tag）----
        harm = _harmonics_gamma(gamma_by_month)
        harm_df = pd.DataFrame([harm])
        out_harm = make_output_path(cfg, "gamma", f"GAMMA_HARM_{region}__{tag_local}.csv")
        harm_df.to_csv(out_harm, index=False)
        logger.info(f"[{region}] Saved gamma harmonics → {out_harm}")

        # ---- 表3：均值±CI + (A,φ)（按 region+method 去重追加）----
        tcrit = 2.201
        s = gamma_by_month
        if s.notna().sum() >= 3:
            mu = float(np.nanmean(s.values))
            sd = float(np.nanstd(s.values, ddof=1))
            se = sd / np.sqrt(np.isfinite(s.values).sum())
            lo, hi = mu - tcrit*se, mu + tcrit*se
        else:
            mu = lo = hi = np.nan

        table3_row = pd.DataFrame([{
            "region": region,
            "method": tag_local,
            "bin_by": args.bin_by if use_bin else "",
            "bins": K if use_bin else 0,
            "gamma_mean": mu, "gamma_ci_low": lo, "gamma_ci_high": hi,
            "A": harm["A"], "phase_deg": harm["phase_deg"]
        }])
        table3_path = make_output_path(cfg, "tables", "Table3_gamma.csv")
        if Path(table3_path).exists():
            old = pd.read_csv(table3_path)
            mask = ~((old.get("region","") == region) & (old.get("method","") == tag_local))
            new = pd.concat([old[mask], table3_row], ignore_index=True)
            new.to_csv(table3_path, index=False)
        else:
            table3_row.to_csv(table3_path, index=False)
        logger.info(f"[{region}] Updated Table 3 → {table3_path}")

        # ---- 图（带 tag）----
        # （这里放你刚才的“增强版 Figure3”那段；只把 fig_path 与 title 用上 tag）
        cmap = plt.get_cmap("tab20")
        plt.figure(figsize=(8.4, 6.6))
        all_x = df["_LCFz"].dropna().values
        all_y = df["_SWCREz"].dropna().values
        def _robust_lim(a):
            if a.size == 0: return (-3, 3)
            q1, q3 = np.nanpercentile(a, [25, 75])
            iqr = q3 - q1 if np.isfinite(q3-q1) and (q3-q1) > 0 else 1.0
            lo = q1 - 1.2*iqr; hi = q3 + 1.2*iqr
            lo, hi = float(np.nan_to_num(lo)), float(np.nan_to_num(hi))
            if lo == hi: lo, hi = lo-1.0, hi+1.0
            return (max(lo, -4), min(hi, 4))
        xlim = _robust_lim(all_x); ylim = _robust_lim(all_y)

        # 每月 se/n（整体版本）
        per_m = out_df[out_df["bin"].isna()] if "bin" in out_df.columns and out_df["bin"].notna().any() else out_df
        se_map = {int(k): float(v) for k, v in per_m.groupby("month")["se"].last().items()}
        n_map  = {int(k): int(v)   for k, v in per_m.groupby("month")["n"].last().items()}

        for m in range(1, 13):
            sub = df[df.index.month == m].dropna(subset=["_LCFz","_SWCREz"])
            if len(sub) == 0: continue
            color = cmap((m-1) % 20)
            plt.scatter(sub["_LCFz"], sub["_SWCREz"], s=16, alpha=0.55, label=f"{m}", color=color)
            g = gamma_by_month.get(m, np.nan)
            if np.isfinite(g):
                xs = np.linspace(xlim[0], xlim[1], 50)
                ys = g * xs
                plt.plot(xs, ys, lw=1.4, alpha=0.95, color=color)
                se = se_map.get(m, np.nan); n = n_map.get(m, np.nan)
                txt = rf"{m}: $\gamma$={g:+.2f}" + (rf" ± {1.96*se:.2f}" if np.isfinite(se) else "") + (f" (n={n})" if np.isfinite(n) else "")
                import matplotlib.patheffects as pe  # 顶部只需 import 一次

                # --- 文本摆放（更清晰）---
                xr = xlim[1] - xlim[0]
                yr = ylim[1] - ylim[0]
                right_side = (m % 2 == 1)              # 奇数月放右侧，偶数月放左侧
                x_txt = (xlim[1] - 0.22*xr) if right_side else (xlim[0] + 0.22*xr)
                ha    = "left" if right_side else "right"
                
                # 竖向错位并限制在绘图区内
                y_txt = ys[-1] + (m - 6) * 0.08
                y_txt = min(max(y_txt, ylim[0] + 0.05*yr), ylim[1] - 0.05*yr)
                
                plt.text(
                    x_txt, y_txt, txt,
                    fontsize=9, color=color, ha=ha, va="center", alpha=0.95,
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", boxstyle="round,pad=0.15"),
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="white")]
                )


        plt.axhline(0, lw=0.8, color="k"); plt.axvline(0, lw=0.8, color="k")
        plt.xlim(*xlim); plt.ylim(*ylim)
        plt.xlabel("LCF' (z by month)"); plt.ylabel("SWCRE' (z by month)")
        plt.title(f"SWCRE' vs LCF' with monthly slopes γ(m) | {region} | {tag_local}")
        plt.legend(title="Month", frameon=False, ncol=4, fontsize=8, bbox_to_anchor=(1.02, 1.02), loc="upper left")
        plt.grid(True, alpha=0.25)
        fig_path = make_output_path(cfg, "figures", f"Fig3_swcre_vs_lcf_{region}__{tag_local}.png")
        plt.tight_layout(); plt.savefig(fig_path, dpi=220, bbox_inches="tight"); plt.close()
        logger.info(f"[{region}] Saved Fig3 → {fig_path}")

    logger.info("DONE.")



if __name__ == "__main__":
    main()

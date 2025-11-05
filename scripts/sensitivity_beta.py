#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seasonal (month-by-month) sensitivity:
  LCF'_m = sum_j beta_j^(m) * CCF'_{j,m} + eps_m

Inputs (per region):
  - output/regional_monthly/MODIS_<REGION>_monthly.csv   (needs column MODIS_LCF; fallback to cllmodis_region_mean_*.csv)
  - output/regional_monthly/ERA5_<REGION>_monthly.csv    (columns like ERA5_EIS, ERA5_W500, ERA5_SST, ERA5_TS, ERA5_U10, ERA5_Q)

Preprocessing:
  - For each calendar month m (1..12), z-score LCF and every CCF using that month's (across years) mean/std
  - Check multicollinearity via VIF (warn if > 5)
  - OLS with Newey–West (HAC) standard errors (lag = --lags)

Outputs:
  - output/sensitivity_beta/BETA_<REGION>.csv
      columns: month,var,beta,se,t,p,ci_low,ci_high,n,rsq
  - output/sensitivity_beta/BETA_HARM_<REGION>.csv
      columns: var,A,phase_deg,peak_month,r2_beta,n_valid
  - output/tables/Table2_beta.csv  (append/update one row per var per region)
      columns: region,var,beta_mean,beta_ci_low,beta_ci_high,A,phase_deg
  - output/figures/Fig2_beta_stripes_<REGION>.png  (variables x 12 months heatmap-like stripes)

Usage:
  python scripts/sensitivity_beta.py --region NEP --ccf EIS,W500,SST,U10,Q --lags 1
  # ΔSST 可写成 DSST 或 ΔSST
"""

from __future__ import annotations
from pathlib import Path
import sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------- args ----------
def parse_args():
    p = argparse.ArgumentParser(description="Monthly sensitivities LCF' ~ CCF' (per-month Z-score, HAC SE).")
    p.add_argument("--region", help="Single region key (NEP/NEA/SEP/SEA/SEI)")
    p.add_argument("--regions", default="", help="Comma list or ALL (uses project.regions)")
    p.add_argument("--ccf", required=True, help="Comma list of CCFs, e.g., EIS,W500,SST,U10,Q or DSST/ΔSST")
    p.add_argument("--lags", type=int, default=1, help="Newey–West (HAC) max lag, default 1")
    return p.parse_args()


# ---------- helpers ----------
def _to_month_start_index(obj):
    x = obj.copy()
    idx = pd.to_datetime(x.index, errors="coerce")
    idx = idx.to_period("M").to_timestamp(how="start")  # 月期 -> 月首
    x.index = idx
    return x.sort_index()

def _read_modis_lcf(cfg: dict, region: str, logger):
    """
    Prefer:  output/regional_monthly/MODIS_<REGION>_monthly.csv  (应含列 MODIS_LCF)
    Fallback:cllmodis_region_mean_*_<REGION>.csv   (两列: time, value 或无表头)
    返回: pd.Series(index=datetime, name='LCF')
    """
    reg_dir = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["regional_monthly"]

    # ---- 优先文件 ----
    f1 = reg_dir / f"MODIS_{region}_monthly.csv"
    if f1.exists():
        # 第一次尝试：常规读法
        df = pd.read_csv(f1)
        # 如果第一行是重复表头(比如第一格就是"time")，改用 header=0 再读一次
        if df.shape[0] > 0 and str(df.iloc[0, 0]).strip().lower() == "time":
            df = pd.read_csv(f1, header=0)

        # 处理“时间列名称混乱”的情况
        cols_lower = [c.lower() for c in df.columns]
        if "time" not in cols_lower:
            # 常见：第一列是索引名，或 'Unnamed: 0'
            first = df.columns[0]
            df = df.rename(columns={first: "time"})
        # 统一时间
        df["time"] = pd.to_datetime(df["time"], errors="coerce", format="%Y-%m-%d")
        df = df.dropna(subset=["time"]).set_index("time").sort_index()

        # 找 LCF 列名
        cand = None
        for c in df.columns:
            lc = c.lower()
            if lc in {"modis_lcf", "lcf", "cllmodis"}:
                cand = c; break
        if cand is not None:
            return df[cand].rename("LCF")

        # 如果列里是多变量月表，尝试唯一的 MODIS_* 列
        modis_cols = [c for c in df.columns if c.upper().startswith("MODIS_")]
        if len(modis_cols) == 1:
            return df[modis_cols[0]].rename("LCF")

        logger.warning(f"{f1.name} 找不到 MODIS_LCF 列，尝试 fallback。列={list(df.columns)}")

    # ---- 退路：cllmodis_region_mean_*_<REGION>.csv（两列或无表头）----
    cand_files = sorted(reg_dir.glob(f"cllmodis_region_mean_*_{region}.csv"))
    if not cand_files:
        raise FileNotFoundError(
            f"找不到 {region} 的 MODIS LCF：既没有 {f1.name}，也没有 cllmodis_region_mean_*_{region}.csv"
        )

    # 先尝试“有表头”的常规读
    df = pd.read_csv(cand_files[-1])
    # 如果第一行是 'time'，说明表头被当数据，再用 header=None 指定列名
    if df.shape[0] > 0 and str(df.iloc[0, 0]).strip().lower() == "time":
        df = pd.read_csv(cand_files[-1], header=None, names=["time", "LCF"])
    # 若没有列名，兜底改名
    if "time" not in [c.lower() for c in df.columns]:
        df.columns = ["time", "LCF"]

    df["time"] = pd.to_datetime(df["time"], errors="coerce", format="%Y-%m-%d")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    # 如果第二列不是 LCF 名，就取第一列之外的那一列
    if "LCF" not in df.columns:
        valcol = [c for c in df.columns if c.lower() != "time"][0]
        return df[valcol].rename("LCF")
    return df["LCF"]

def _read_era5_ccf(cfg: dict, region: str) -> pd.DataFrame:
    reg_dir = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["regional_monthly"]
    f = reg_dir / f"ERA5_{region}_monthly.csv"
    if not f.exists():
        raise FileNotFoundError(f"ERA5 monthly file not found: {f}")
    df = pd.read_csv(f)
    # 处理第一行是 'time' 的奇怪情况
    if df.shape[0] > 0 and str(df.iloc[0, 0]).strip().lower() == "time":
        df = pd.read_csv(f, header=0)
    # 统一时间列名
    cols_lower = [c.lower() for c in df.columns]
    if "time" not in cols_lower:
        df = df.rename(columns={df.columns[0]: "time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce", format="%Y-%m-%d")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df


def _aliases(ccf_list_raw: list[str]) -> list[str]:
    """Map input names to ERA5 column suffix names. DSST/ΔSST handled as virtual predictor."""
    out = []
    for s in ccf_list_raw:
        s = s.strip()
        if not s: 
            continue
        if s.upper() in {"ΔSST","DSST"}:
            out.append("DSST")
        else:
            out.append(s.upper())
    return out

def _build_design(df_era5: pd.DataFrame, ccf_list: list[str], logger) -> pd.DataFrame:
    """Return DataFrame with columns exactly the chosen predictors (renamed to their short names)."""
    X = pd.DataFrame(index=df_era5.index)
    # virtual DSST from ERA5_SST
    if "DSST" in ccf_list:
        if "ERA5_SST" not in df_era5.columns:
            raise KeyError("Requested DSST but ERA5_SST not found in ERA5 monthly table.")
        d = df_era5["ERA5_SST"].diff()  # simple month-to-month difference
        X["DSST"] = d
        logger.info("Constructed DSST = month-to-month diff of ERA5_SST")
    # direct mappings
    for name in ccf_list:
        if name == "DSST":
            continue
        col = f"ERA5_{name}"
        if col not in df_era5.columns:
            raise KeyError(f"Predictor column missing: {col}")
        X[name] = df_era5[col]
    return X

def _zscore_by_month(series: pd.Series) -> pd.Series:
    """Per-calendar-month z-score (remove monthly mean, divide by monthly std across years)."""
    def _z(g):
        mu = g.mean()
        sd = g.std(ddof=1)
        return (g - mu) / sd if sd not in (0, None, np.nan) and np.isfinite(sd) and sd > 0 else (g - mu)
    return series.groupby(series.index.month, group_keys=False).apply(_z)

def _vif_report(Xz: pd.DataFrame, logger, month: int):
    try:
        # drop all-NaN columns
        X = Xz.dropna()
        if len(X) < 5 or X.shape[1] < 2:
            return
        X_const = sm.add_constant(X)
        vifs = []
        for i, col in enumerate(X_const.columns[1:]):  # skip const
            v = variance_inflation_factor(X_const.values, i+1)
            vifs.append((col, float(v)))
        bad = [(c, v) for c, v in vifs if v > 5.0]
        if bad:
            logger.warning(f"[VIF] month={month}: high VIF>5 -> {bad}")
    except Exception as e:
        logger.warning(f"[VIF] month={month}: VIF check failed: {e}")

def _hac_ols(y: pd.Series, X: pd.DataFrame, lags: int):
    """Return dict with beta,se,t,p,ci_low,ci_high,rsq and params index equals X columns."""
    # align, drop NaNs
    df = pd.concat([y, X], axis=1).dropna()
    if df.shape[0] < (len(X.columns) + 3):  # minimal samples
        return None, None
    y_ = df.iloc[:, 0]
    X_ = df.iloc[:, 1:]
    Xc = sm.add_constant(X_)
    model = sm.OLS(y_, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    params = model.params.drop("const", errors="ignore")
    se = model.bse.drop("const", errors="ignore")
    t = model.tvalues.drop("const", errors="ignore")
    p = model.pvalues.drop("const", errors="ignore")
    # 95% CI
    z = 1.96
    ci_low = params - z * se
    ci_high = params + z * se
    out = pd.DataFrame({
        "beta": params, "se": se, "t": t, "p": p,
        "ci_low": ci_low, "ci_high": ci_high,
    })
    return out, float(model.rsquared)

def _harmonics_from_monthly(betas_by_var: dict[str, pd.Series], logger, region: str, cfg: dict) -> pd.DataFrame:
    """
    对每个变量的 β(m), m=1..12 做一谐波，返回 DataFrame: var,A,phase_deg,peak_month,r2_beta,n_valid
    注意：climu.first_harmonic_fit 期望 xarray DataArray，带 'month' 维。
    """
    import xarray as xr
    rows = []
    for var, s in betas_by_var.items():
        try:
            months = np.arange(1, 13)
            # 组装 12 个月序列（可能有 NaN）
            y = np.array([s.get(m, np.nan) for m in months], dtype=float)

            # 包成 DataArray，维度名必须叫 'month'
            da = xr.DataArray(
                y,
                coords={"month": months},
                dims=("month",),
                name=f"beta_{var}"
            )

            fit = climu.first_harmonic_fit(da, min_valid=8, soft=True)
            rows.append({
                "var": var,
                "A": float(fit["amplitude"]),
                "phase_deg": float(fit["phase_deg"]),
                "peak_month": int(fit["peak_month"]),
                "r2_beta": float(fit["r2"]),
                "n_valid": float(fit.get("n_valid", np.sum(np.isfinite(y)))),
            })
        except Exception as e:
            logger.warning(f"[HARM] {var}: harmonic fit failed: {e}")
            rows.append({
                "var": var, "A": np.nan, "phase_deg": np.nan,
                "peak_month": np.nan, "r2_beta": np.nan, "n_valid": 0
            })
    return pd.DataFrame(rows)


# ---------- main ----------
def main():
    args = parse_args()
    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    logger = setup_logger(cfg, name="sensitivity_beta")

    # === 组织要跑的区域列表 ===
    def _regions_to_run(cfg, args):
        if args.regions:
            rs = [r.strip() for r in args.regions.split(",") if r.strip()]
            if len(rs) == 1 and rs[0].upper() == "ALL":
                rs = cfg["project"]["regions"]
        elif args.region:
            rs = [args.region]
        else:
            raise SystemExit("请提供 --region 或 --regions（可用 ALL）")
        return rs

    regions = _regions_to_run(cfg, args)
    ccf_list = _aliases(args.ccf.split(","))
    lags = max(0, int(args.lags))

    # === 区域循环 ===
    for region in regions:
        logger.info(f"[RUN] region={region} | predictors={ccf_list}")

        # ---- 读取数据 ----
        lcf  = _read_modis_lcf(cfg, region, logger)
        era5 = _read_era5_ccf(cfg, region)

        # 统一时间到月首，确保索引对齐
        lcf  = _to_month_start_index(lcf)
        era5 = _to_month_start_index(era5)

        # 合并后去掉全 NaN 时间
        df = pd.concat([lcf.rename("LCF"), era5], axis=1).dropna(how="all")
        X_raw = _build_design(df, ccf_list, logger)

        # ---- 逐月回归 ----
        records = []
        betas_by_var = {v: pd.Series(index=pd.Index(range(1,13), name="month"), dtype=float)
                        for v in X_raw.columns}
        for m in range(1, 13):
            sub = pd.concat([df["LCF"], X_raw], axis=1)
            sub = sub[sub.index.month == m].dropna(how="any")
            if len(sub) < (len(X_raw.columns) + 4):
                logger.warning(f"[{region}] month={m}: too few samples ({len(sub)}) → skip")
                continue

            y_z = _zscore_by_month(sub["LCF"])
            X_z = sub[X_raw.columns].copy()
            for col in X_z.columns:
                X_z[col] = _zscore_by_month(X_z[col])

            _vif_report(X_z, logger, m)

            out, rsq = _hac_ols(y_z, X_z, lags)
            if out is None:
                logger.warning(f"[{region}] month={m}: regression failed (insufficient)")
                continue
            out["month"] = m
            out["var"] = out.index
            out["n"] = len(sub)
            out["rsq"] = rsq
            records.append(out.reset_index(drop=True))

            # 保存 β(m)
            for v in X_z.columns:
                if v in out.set_index("var").index:
                    betas_by_var[v].loc[m] = out.set_index("var").loc[v, "beta"]

        if not records:
            logger.warning(f"[{region}] 没有任何月份成功回归，跳过。")
            continue

        beta_df = pd.concat(records, ignore_index=True)

        # ---- 保存逐月 β ----
        out_beta = make_output_path(cfg, "sensitivity_beta", f"BETA_{region}.csv")
        beta_df[["month","var","beta","se","t","p","ci_low","ci_high","n","rsq"]].to_csv(out_beta, index=False)
        logger.info(f"[{region}] Saved betas → {out_beta}")

        # ---- 一谐波拟合 β(m) ----
        harm_df = _harmonics_from_monthly(betas_by_var, logger, region, cfg)
        out_harm = make_output_path(cfg, "sensitivity_beta", f"BETA_HARM_{region}.csv")
        harm_df.to_csv(out_harm, index=False)
        logger.info(f"[{region}] Saved beta harmonics → {out_harm}")

        # ---- Table2: 平均±CI + 谐波 ----
        tcrit = 2.201
        rows = []
        for v in betas_by_var.keys():
            s = betas_by_var[v]
            if s.notna().sum() < 3:
                mu = np.nan; lo = np.nan; hi = np.nan
            else:
                mu = float(np.nanmean(s.values))
                sd = float(np.nanstd(s.values, ddof=1))
                se = sd / np.sqrt(np.sum(np.isfinite(s.values)))
                lo, hi = mu - tcrit*se, mu + tcrit*se
            h = harm_df.set_index("var").loc[v] if v in harm_df.set_index("var").index else pd.Series({"A":np.nan,"phase_deg":np.nan})
            rows.append({
                "region": region, "var": v,
                "beta_mean": mu, "beta_ci_low": lo, "beta_ci_high": hi,
                "A": float(h.get("A", np.nan)), "phase_deg": float(h.get("phase_deg", np.nan))
            })
        table2_row = pd.DataFrame(rows)
        table2_path = make_output_path(cfg, "tables", "Table2_beta.csv")
        if Path(table2_path).exists():
            old = pd.read_csv(table2_path)
            old = old[old["region"] != region]
            new = pd.concat([old, table2_row], ignore_index=True)
            new.to_csv(table2_path, index=False)
        else:
            table2_row.to_csv(table2_path, index=False)
        logger.info(f"[{region}] Updated Table 2 → {table2_path}")

        # ---- 条带图 ----
        vars_order = list(betas_by_var.keys())
        mat = np.full((len(vars_order), 12), np.nan)
        for i, v in enumerate(vars_order):
            s = betas_by_var[v]
            for m in range(1,13):
                mat[i, m-1] = s.get(m, np.nan)
        plt.figure(figsize=(12, 1.6 + 0.35*len(vars_order)))
        im = plt.imshow(mat, aspect="auto", interpolation="nearest")
        plt.colorbar(im, fraction=0.025, pad=0.02, label=r"$\beta$ (z-z)")
        plt.yticks(np.arange(len(vars_order)), vars_order)
        plt.xticks(np.arange(12), [str(m) for m in range(1,13)])
        plt.xlabel("Month"); plt.title(f"β (LCF' ~ CCF') | {region}")
        fig_path = make_output_path(cfg, "figures", f"Fig2_beta_stripes_{region}.png")
        plt.tight_layout(); plt.savefig(fig_path, dpi=200, bbox_inches="tight"); plt.close()
        logger.info(f"[{region}] Saved Fig2 stripes → {fig_path}")

    logger.info("DONE.")


if __name__ == "__main__":
    main()

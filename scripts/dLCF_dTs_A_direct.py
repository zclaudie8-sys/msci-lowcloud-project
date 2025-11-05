#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A-line (direct) estimate of dLCF/dTs with meteorology controlled.

Model (per calendar month m):
    LCF'_m = theta_T^(m) * Ts'_m + sum_j theta_j^(m) * CCF'_{j,m} + eps_m

Inputs (per region):
  - output/regional_monthly/MODIS_<REGION>_monthly.csv   (need column MODIS_LCF; fallback to cllmodis_region_mean_*.csv)
  - output/regional_monthly/ERA5_<REGION>_monthly.csv    (ERA5_TS / ERA5_SST / ERA5_EIS / ERA5_W500 / ...)

Arguments:
  --region NEP --ccf EIS,W500,SST,U10,Q --lags 1

Outputs:
  - output/dLCF_dTs/ALINE_<REGION>.csv        (month, dLCF_dTs, se, t, p, ci_low, ci_high, n, rsq, Ts_name)
  - output/dLCF_dTs/ALINE_HARM_<REGION>.csv   (A, phase_deg, peak_month, r2, n_valid)
  - (append/update) output/tables/Table4_dLCF_dTs_A.csv (region, mean±CI, A, phase_deg)

Notes
- Per-month Z-score (by calendar month) for LCF, Ts and all CCFs.
- Ts predictor is chosen automatically:
    * If user lists SST in --ccf, Ts = ERA5_SST and SST is removed from CCFs to avoid duplication.
    * Else if ERA5_TS exists, Ts = ERA5_TS; else if ERA5_SST exists, Ts = ERA5_SST.
- Newey–West (HAC) SE with lag = --lags (default 1).
- VIF checked per month (info-level warning only).
"""

from __future__ import annotations
from pathlib import Path
import sys, argparse
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm

# --- robust imports so the script runs from project root OR scripts/ ---
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
try:
    from scripts.utils_config import load_config, make_output_path, setup_logger
    import scripts.utils_climatology as climu
except ModuleNotFoundError:
    from utils_config import load_config, make_output_path, setup_logger
    import utils_climatology as climu

# ---------------- args ----------------
def parse_args():
    p = argparse.ArgumentParser(description="A-line direct regression for dLCF/dTs (monthly, with controls).")
    p.add_argument("--region", help="Region key (NEP/NEA/SEP/SEA/SEI)")               # ← 不再 required
    p.add_argument("--regions", default="", help="Comma list or ALL (uses project.regions)")
    p.add_argument("--ccf", required=True,
                   help="Comma list of CCFs, e.g., EIS,W500,SST,U10,Q (SST here means Ts predictor=ERA5_SST)")
    p.add_argument("--lags", type=int, default=1, help="Newey–West (HAC) max lag (default=1)")
    return p.parse_args()


# ---------------- helpers ----------------
def _regions_to_run(cfg: dict, args) -> list[str]:
    """根据 --regions / --region 返回要运行的区域列表。"""
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
    idx = pd.to_datetime(x.index, errors="coerce").to_period("M").start_time
    x.index = idx
    return x.sort_index()

def _zscore_by_month(s: pd.Series) -> pd.Series:
    def _z(g):
        mu = g.mean(); sd = g.std(ddof=1)
        return (g - mu) / sd if (sd is not None and np.isfinite(sd) and sd > 0) else (g - mu)
    return s.groupby(s.index.month, group_keys=False).apply(_z)

def _read_modis_lcf(cfg: dict, region: str, logger):
    regdir = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["regional_monthly"]
    f = regdir / f"MODIS_{region}_monthly.csv"
    if f.exists():
        df = pd.read_csv(f)
        if df.shape[0] > 0 and str(df.iloc[0,0]).strip().lower() == "time":
            df = pd.read_csv(f, header=0)
        if "time" not in [c.lower() for c in df.columns]:
            df = df.rename(columns={df.columns[0]:"time"})
        df["time"] = pd.to_datetime(df["time"], errors="coerce"); df = df.dropna(subset=["time"]).set_index("time")
        # normalize LCF name
        if "MODIS_LCF" not in df.columns:
            for c in df.columns:
                if c.lower() in {"modis_lcf","lcf","cllmodis"}:
                    df = df.rename(columns={c:"MODIS_LCF"}); break
        if "MODIS_LCF" not in df.columns:
            logger.warning(f"{f.name} has no MODIS_LCF; trying fallback.")
    else:
        df = None

    if df is not None and "MODIS_LCF" in df.columns:
        return df["MODIS_LCF"].rename("LCF")

    # fallback: cllmodis_region_mean_* (two-column file)
    cand = sorted(regdir.glob(f"cllmodis_region_mean_*_{region}.csv"))
    if not cand:
        raise FileNotFoundError(f"No MODIS LCF found for {region}.")
    tmp = pd.read_csv(cand[-1])
    if tmp.shape[0] > 0 and str(tmp.iloc[0,0]).strip().lower() == "time":
        tmp = pd.read_csv(cand[-1], header=0)
    if "time" not in [c.lower() for c in tmp.columns]:
        tmp.columns = ["time", "LCF"]
    tmp["time"] = pd.to_datetime(tmp["time"], errors="coerce")
    return tmp.dropna(subset=["time"]).set_index("time")["LCF"].rename("LCF")

def _read_era5_monthly(cfg: dict, region: str) -> pd.DataFrame:
    regdir = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["regional_monthly"]
    f = regdir / f"ERA5_{region}_monthly.csv"
    if not f.exists():
        raise FileNotFoundError(f"ERA5 monthly not found: {f}")
    df = pd.read_csv(f)
    if df.shape[0] > 0 and str(df.iloc[0,0]).strip().lower() == "time":
        df = pd.read_csv(f, header=0)
    if "time" not in [c.lower() for c in df.columns]:
        df = df.rename(columns={df.columns[0]:"time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time"]).set_index("time").sort_index()

def _aliases(ccf_list_raw: list[str]) -> list[str]:
    out = []
    for s in ccf_list_raw:
        s = s.strip().upper()
        if not s: continue
        out.append(s)
    return out
def _regions_to_run(cfg: dict, args) -> list[str]:
    """根据 --regions / --region 返回要运行的区域列表。"""
    if args.regions:
        rs = [r.strip() for r in args.regions.split(",") if r.strip()]
        if len(rs) == 1 and rs[0].upper() == "ALL":
            rs = cfg["project"]["regions"]
        return rs
    if args.region:
        return [args.region]
    raise SystemExit("请提供 --region 或 --regions（可用 ALL）")


def _choose_Ts_and_build_X(df_all: pd.DataFrame, ccf_list: list[str], logger):
    """
    Decide Ts predictor and build CCF matrix.
    - If 'SST' explicitly listed, Ts_col='ERA5_SST' and remove SST from CCFs.
    - Else prefer ERA5_TS; fallback ERA5_SST.
    """
    Ts_col = None
    ccf = [c for c in ccf_list]
    if "SST" in ccf:
        if "ERA5_SST" not in df_all.columns:
            raise KeyError("Requested Ts=SST but ERA5_SST not in ERA5 monthly table.")
        Ts_col = "ERA5_SST"
        ccf = [x for x in ccf if x != "SST"]
        logger.info("Ts predictor chosen: ERA5_SST (because SST is in --ccf)")
    else:
        if "ERA5_TS" in df_all.columns:
            Ts_col = "ERA5_TS"; logger.info("Ts predictor chosen: ERA5_TS")
        elif "ERA5_SST" in df_all.columns:
            Ts_col = "ERA5_SST"; logger.info("Ts predictor chosen: ERA5_SST")
        else:
            raise KeyError("Neither ERA5_TS nor ERA5_SST found in ERA5 monthly table.")

    # Build X with remaining CCFs
    X = pd.DataFrame(index=df_all.index)
    for name in ccf:
        col = f"ERA5_{name}"
        if col not in df_all.columns:
            raise KeyError(f"Predictor column missing: {col}")
        X[name] = df_all[col]
    return Ts_col, X

def _vif_report(Xz: pd.DataFrame, logger, month: int):
    try:
        if Xz.empty or Xz.shape[1] < 2: return
        Xc = sm.add_constant(Xz)
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vifs = []
        for i, col in enumerate(Xc.columns[1:]):
            vifs.append((col, float(variance_inflation_factor(Xc.values, i+1))))
        bad = [(c, v) for c, v in vifs if v > 5.0]
        if bad:
            logger.warning(f"[VIF] month={month}: high VIF>5 -> {bad}")
    except Exception as e:
        logger.warning(f"[VIF] month={month}: VIF check failed: {e}")

def _hac_ols_multi(y: pd.Series, X: pd.DataFrame, lags: int):
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if df.shape[0] < (X.shape[1] + 4):
        return None, None
    y_, X_ = df.iloc[:,0], df.iloc[:,1:]
    Xc = sm.add_constant(X_)
    res = sm.OLS(y_, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    params = res.params.drop("const", errors="ignore")
    se = res.bse.drop("const", errors="ignore")
    t = res.tvalues.drop("const", errors="ignore")
    p = res.pvalues.drop("const", errors="ignore")
    z = 1.96
    ci_low = params - z*se; ci_high = params + z*se
    out = pd.DataFrame({"beta": params, "se": se, "t": t, "p": p,
                        "ci_low": ci_low, "ci_high": ci_high})
    return out, float(res.rsquared)

# ---------------- main ----------------
def main():
    args = parse_args()
    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    logger = setup_logger(cfg, name="dLCF_dTs_A_direct")
  
    ccf_list = _aliases(args.ccf.split(","))
    lags = max(0, int(args.lags))
    regions = _regions_to_run(cfg, args)
    
        
    for region in regions:
        logger.info(f"[RUN] region={region} | ccf={ccf_list} | lags={lags}")
    
        # ---- 以下保留你原来的单区逻辑（从“# 1) load”开始到写 Table4/日志结束）----
        # 1) load
        lcf = _read_modis_lcf(cfg, region, logger)
        era = _read_era5_monthly(cfg, region)
        lcf = _to_month_start_index(lcf)
        era = _to_month_start_index(era)
        df_all = pd.concat([lcf.rename("LCF"), era], axis=1)
    
        Ts_col, X_raw = _choose_Ts_and_build_X(df_all, ccf_list, logger)
    
        # ...（你的逐月回归、保存 ALINE_<REGION>.csv、ALINE_HARM_<REGION>.csv、
        #      更新 Table4_dLCF_dTs_A.csv 的代码原样放在这里，不必改文件名逻辑）...
    
    logger.info("DONE.")

    # unify to month start
    lcf = _to_month_start_index(lcf)
    era = _to_month_start_index(era)
    df_all = pd.concat([lcf.rename("LCF"), era], axis=1)

    # choose Ts predictor & build CCF design
    Ts_col, X_raw = _choose_Ts_and_build_X(df_all, ccf_list, logger)

    # 2) per-month regressions (Z-score by month)
    records = []
    dTs_by_month = pd.Series(index=pd.Index(range(1,13), name="month"), dtype=float)

    for m in range(1, 13):
        sub = pd.concat([df_all["LCF"], df_all[Ts_col], X_raw], axis=1)
        sub = sub[sub.index.month == m].dropna(how="any")
        if len(sub) < (X_raw.shape[1] + 5):  # y + Ts + controls + margin
            logger.warning(f"[{region}] month={m}: too few samples (n={len(sub)}) → skip")
            continue

        # Z-score within month
        y_z  = _zscore_by_month(sub["LCF"])
        Ts_z = _zscore_by_month(sub[Ts_col])
        X_z  = sub[X_raw.columns].copy()
        for c in X_z.columns:
            X_z[c] = _zscore_by_month(X_z[c])

        # Assemble X: [Ts, controls]
        Xz_all = pd.concat([Ts_z.rename("TS"), X_z], axis=1)

        _vif_report(Xz_all, logger, m)

        out, rsq = _hac_ols_multi(y_z, Xz_all, lags)
        if out is None:
            logger.warning(f"[{region}] month={m}: regression failed → skip")
            continue

        if "TS" not in out.index:
            logger.warning(f"[{region}] month={m}: TS coeff missing → skip")
            continue

        row = out.loc["TS"]
        dTs_by_month.loc[m] = float(row["beta"])
        records.append({
            "region": region, "month": m, "Ts_name": Ts_col,
            "dLCF_dTs": float(row["beta"]),
            "se": float(row["se"]), "t": float(row["t"]), "p": float(row["p"]),
            "ci_low": float(row["ci_low"]), "ci_high": float(row["ci_high"]),
            "n": int(len(sub)), "rsq": float(rsq)
        })

    if not records:
        raise SystemExit(f"No monthly estimates produced for region={region}.")

    # 3) save A-line table
    out_df = pd.DataFrame(records)
    out_csv = make_output_path(cfg, "dLCF_dTs", f"ALINE_{region}.csv")
    out_df.to_csv(out_csv, index=False)
    logger.info(f"Saved A-line dLCF/dTs → {out_csv}")

    # 4) harmonics of dLCF/dTs(m)
    months = np.arange(1, 13)
    da = xr.DataArray(
        np.array([dTs_by_month.get(m, np.nan) for m in months], dtype=float),
        coords={"month": months}, dims=("month",), name="dLCF_dTs"
    )
    fit = climu.first_harmonic_fit(da, min_valid=8, soft=True)
    harm_df = pd.DataFrame([{
        "A": float(fit["amplitude"]),
        "phase_deg": float(fit["phase_deg"]),
        "peak_month": int(fit["peak_month"]),
        "r2": float(fit["r2"]),
        "n_valid": float(fit.get("n_valid", np.isfinite(da.values).sum()))
    }])
    out_harm = make_output_path(cfg, "dLCF_dTs", f"ALINE_HARM_{region}.csv")
    harm_df.to_csv(out_harm, index=False)
    logger.info(f"Saved A-line harmonics → {out_harm}")

    # 5) Table 4 (append/update by region)
    tcrit = 2.201  # df≈11
    s = dTs_by_month
    if s.notna().sum() >= 3:
        mu = float(np.nanmean(s.values))
        sd = float(np.nanstd(s.values, ddof=1))
        se = sd / np.sqrt(np.isfinite(s.values).sum())
        lo, hi = mu - tcrit*se, mu + tcrit*se
    else:
        mu = lo = hi = np.nan

    table4_row = pd.DataFrame([{
        "region": region, "method": "A_direct", "Ts_name": Ts_col,
        "dLCF_dTs_mean": mu, "dLCF_dTs_ci_low": lo, "dLCF_dTs_ci_high": hi,
        "A": harm_df["A"].iloc[0], "phase_deg": harm_df["phase_deg"].iloc[0]
    }])
    table4_path = make_output_path(cfg, "tables", "Table4_dLCF_dTs_A.csv")
    if Path(table4_path).exists():
        old = pd.read_csv(table4_path)
        mask = ~((old.get("region","")==region) & (old.get("method","")=="A_direct"))
        pd.concat([old[mask], table4_row], ignore_index=True).to_csv(table4_path, index=False)
    else:
        table4_row.to_csv(table4_path, index=False)
    logger.info(f"Updated Table 4 → {table4_path}")

    logger.info("DONE.")

if __name__ == "__main__":
    main()

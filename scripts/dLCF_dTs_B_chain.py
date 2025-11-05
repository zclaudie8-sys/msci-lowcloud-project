#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B-line (chain decomposition) of dLCF/dTs:

1) For each month m and each CCF j:
      CCF'_{j,m} = kappa_j^(m) * Ts'_m + eps   (no intercept; HAC SE)
2) Read beta_j(m) from Table 2 product: output/sensitivity_beta/BETA_<REGION>.csv
3) Chain estimate:
      dLCF/dTs (m) ~= sum_j beta_j(m) * kappa_j(m)

Inputs (per region):
  - output/sensitivity_beta/BETA_<REGION>.csv     (from sensitivity_beta.py)
  - output/regional_monthly/ERA5_<REGION>.csv     (ERA5_*)
  - output/regional_monthly/MODIS_<REGION>.csv    (only used for robust time index if present)

Args:
  --region NEP --ccf EIS,W500,SST,U10,Q --lags 1

Outputs:
  - output/dLCF_dTs/BCHAIN_<REGION>.csv
     columns: month, dLCF_dTs_chain, Ts_name,  (then per-var: kappa_<var>, beta_<var>, contrib_<var>, n_<var>, rsq_<var>)
  - output/dLCF_dTs/BCHAIN_HARM_<REGION>.csv
     columns: A, phase_deg, peak_month, r2, n_valid
  - Append/merge table:
     output/tables/Table4_dLCF_dTs.csv   (both A-line and B-chain columns if available)

Notes:
- Per-month Z-score for Ts and each CCF.
- Ts choice: if 'SST' appears in --ccf → Ts=ERA5_SST; else prefer ERA5_TS then ERA5_SST.
- HAC(Newey–West) SE with --lags (default 1).
"""

from __future__ import annotations
from pathlib import Path
import sys, argparse
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
try:
    from scripts.utils_config import load_config, make_output_path, setup_logger
except ModuleNotFoundError:
    from utils_config import load_config, make_output_path, setup_logger

# --------- args ----------
def parse_args():
    p = argparse.ArgumentParser(description="B-line chain estimate of dLCF/dTs via beta × kappa.")
    p.add_argument("--region", required=True, help="NEP/NEA/SEP/SEA/SEI")
    p.add_argument("--ccf", required=True, help="Comma list of CCFs, e.g., EIS,W500,SST,U10,Q")
    p.add_argument("--lags", type=int, default=1, help="HAC max lag (default 1)")
    return p.parse_args()

# --------- helpers ----------
def _aliases(lst):
    return [s.strip().upper() for s in lst if s.strip()]

def _to_ms_index(obj):
    x = obj.copy()
    idx = pd.to_datetime(x.index, errors="coerce").to_period("M").start_time
    x.index = idx
    return x.sort_index()

def _z_by_month(s: pd.Series) -> pd.Series:
    def _z(g):
        mu = g.mean(); sd = g.std(ddof=1)
        return (g - mu)/sd if (sd is not None and np.isfinite(sd) and sd > 0) else (g - mu)
    return s.groupby(s.index.month, group_keys=False).apply(_z)

def _read_era5(cfg, region):
    reg = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["regional_monthly"]
    f = reg / f"ERA5_{region}_monthly.csv"
    if not f.exists():
        raise FileNotFoundError(f"ERA5 monthly not found: {f}")
    df = pd.read_csv(f)
    if df.shape[0] > 0 and str(df.iloc[0,0]).strip().lower() == "time":
        df = pd.read_csv(f, header=0)
    if "time" not in [c.lower() for c in df.columns]:
        df = df.rename(columns={df.columns[0]:"time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time"]).set_index("time").sort_index()

def _read_modis_monthly(cfg, region):
    reg = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["regional_monthly"]
    f = reg / f"MODIS_{region}_monthly.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if df.shape[0] > 0 and str(df.iloc[0,0]).strip().lower() == "time":
        df = pd.read_csv(f, header=0)
    if "time" not in [c.lower() for c in df.columns]:
        df = df.rename(columns={df.columns[0]:"time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df

def _choose_Ts(df_all: pd.DataFrame, ccf_list, logger):
    # If user listed SST explicitly -> Ts=ERA5_SST; else prefer ERA5_TS then ERA5_SST.
    if "SST" in ccf_list:
        if "ERA5_SST" not in df_all.columns:
            raise KeyError("Requested Ts=SST but ERA5_SST not found.")
        logger.info("Ts predictor chosen: ERA5_SST (because SST in --ccf)")
        return "ERA5_SST"
    if "ERA5_TS" in df_all.columns:
        logger.info("Ts predictor chosen: ERA5_TS")
        return "ERA5_TS"
    if "ERA5_SST" in df_all.columns:
        logger.info("Ts predictor chosen: ERA5_SST")
        return "ERA5_SST"
    raise KeyError("Neither ERA5_TS nor ERA5_SST found.")

def _hac_slope(y: pd.Series, x: pd.Series, lags: int):
    """No-intercept OLS slope with HAC SE."""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 8:
        return None
    res = sm.OLS(df["y"].values, df["x"].values).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    g = float(res.params[0]); se = float(res.bse[0]); t = float(res.tvalues[0]); p = float(res.pvalues[0])
    z = 1.96; ci_low = g - z*se; ci_high = g + z*se
    # R^2 relative to zero-intercept
    yhat = df["x"] * g
    ss_res = float(((df["y"] - yhat)**2).sum())
    ss_tot = float(((df["y"])**2).sum())
    rsq = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return {"slope": g, "se": se, "t": t, "p": p, "ci_low": ci_low, "ci_high": ci_high, "n": len(df), "rsq": rsq}

# --------- main ----------
def main():
    args = parse_args()
    cfg  = load_config(str(ROOT / "configs" / "config.yaml"))
    logger = setup_logger(cfg, name="dLCF_dTs_B_chain")

    region   = args.region
    ccf_list = _aliases(args.ccf.split(","))
    lags     = max(0, int(args.lags))

    # --- load monthly data ---
    era  = _read_era5(cfg, region)
    mod  = _read_modis_monthly(cfg, region)  # optional
    if mod is not None:
        era = _to_ms_index(era)
        mod = _to_ms_index(mod)
        df_all = era.join(mod, how="outer")
    else:
        df_all = _to_ms_index(era)

    # --- pick Ts predictor ---
    Ts_col = _choose_Ts(df_all, ccf_list, logger)

    # --- read beta table (Table 2 monthly betas) ---
    beta_file = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["sensitivity_beta"] / f"BETA_{region}.csv"
    if not beta_file.exists():
        raise FileNotFoundError(f"BETA file not found: {beta_file}. Run sensitivity_beta.py first.")
    beta_df = pd.read_csv(beta_file)
    # keep only needed CCFs; standardize var names to upper
    beta_df["var"] = beta_df["var"].str.upper()
    beta_df = beta_df[beta_df["var"].isin(ccf_list)]
    if beta_df.empty:
        raise SystemExit(f"No beta rows for requested CCFs in {beta_file}.")

    # --- compute kappa_j(m) by monthly regressions of CCF'_j on Ts' ---
    months = range(1, 13)
    kappa_map = {var: pd.Series(index=pd.Index(months, name="month"), dtype=float) for var in ccf_list}
    kappainfo = {var: {} for var in ccf_list}  # store n, rsq per month
    for var in ccf_list:
        col = f"ERA5_{var}"
        if col not in df_all.columns:
            logger.warning(f"[{region}] missing ERA5 column for {var} -> skip this var in kappa.")
            continue
        for m in months:
            sub = df_all[[col, Ts_col]].copy()
            sub = sub[sub.index.month == m].dropna(how="any")
            if len(sub) < 8:
                continue
            y = _z_by_month(sub[col])
            x = _z_by_month(sub[Ts_col])
            est = _hac_slope(y, x, lags)
            if est is None:
                continue
            kappa_map[var].loc[m] = est["slope"]
            kappainfo[var][m] = {"n": est["n"], "rsq": est["rsq"]}

    # --- chain per month: sum_j beta_j(m) * kappa_j(m) ---
    rows = []
    chain_series = pd.Series(index=pd.Index(months, name="month"), dtype=float)
    for m in months:
        # collect beta and kappa for this month
        betam = beta_df[beta_df["month"] == m].set_index("var")
        tot = 0.0
        contribs = {}
        for var in ccf_list:
            b = float(betam.loc[var, "beta"]) if var in betam.index and pd.notna(betam.loc[var, "beta"]) else np.nan
            k = float(kappa_map[var].loc[m]) if m in kappa_map[var].index and pd.notna(kappa_map[var].loc[m]) else np.nan
            if np.isfinite(b) and np.isfinite(k):
                c = b * k
                tot += c
                contribs[var] = (b, k, c)
        if len(contribs) == 0:
            continue
        chain_series.loc[m] = tot
        # compose row
        row = {"region": region, "month": m, "Ts_name": Ts_col, "dLCF_dTs_chain": tot}
        for var, (b, k, c) in contribs.items():
            row[f"beta_{var}"]   = b
            row[f"kappa_{var}"]  = k
            row[f"contrib_{var}"]= c
            info = kappainfo.get(var, {}).get(m, {})
            if "n" in info:   row[f"n_{var}"]   = info["n"]
            if "rsq" in info: row[f"rsq_{var}"] = info["rsq"]
        rows.append(row)

    if chain_series.dropna().empty:
        raise SystemExit(f"No monthly chain estimates for region={region} (check data overlap).")

    # --- write BCHAIN_<REGION>.csv ---
    out_df = pd.DataFrame(rows).sort_values("month")
    out_csv = make_output_path(cfg, "dLCF_dTs", f"BCHAIN_{region}.csv")
    out_df.to_csv(out_csv, index=False)

    # --- harmonics of chain(m) ---
    da = xr.DataArray(chain_series.reindex(range(1,13)).values,
                      coords={"month": list(range(1,13))}, dims=("month",), name="dLCF_dTs_chain")
    from scripts.utils_climatology import first_harmonic_fit if "scripts.utils_climatology" in sys.modules else None  # safe import
    import importlib
    try:
        climu = importlib.import_module("scripts.utils_climatology")
    except Exception:
        climu = importlib.import_module("utils_climatology")
    fit = climu.first_harmonic_fit(da, min_valid=8, soft=True)
    harm_df = pd.DataFrame([{
        "A": float(fit["amplitude"]),
        "phase_deg": float(fit["phase_deg"]),
        "peak_month": int(fit["peak_month"]),
        "r2": float(fit["r2"]),
        "n_valid": float(fit.get("n_valid", np.isfinite(da.values).sum()))
    }])
    out_harm = make_output_path(cfg, "dLCF_dTs", f"BCHAIN_HARM_{region}.csv")
    harm_df.to_csv(out_harm, index=False)

    # --- update combined Table 4 (merge A & B) ---
    tcrit = 2.201
    s = chain_series
    if s.notna().sum() >= 3:
        mu = float(np.nanmean(s.values))
        sd = float(np.nanstd(s.values, ddof=1))
        se = sd / np.sqrt(np.isfinite(s.values).sum())
        lo, hi = mu - tcrit*se, mu + tcrit*se
    else:
        mu = lo = hi = np.nan

    # prepare this region's B row (and attach harmonics)
    brow = pd.DataFrame([{
        "region": region, "method": "B_chain", "Ts_name": Ts_col,
        "dLCF_dTs_mean_B": mu, "dLCF_dTs_ci_low_B": lo, "dLCF_dTs_ci_high_B": hi,
        "A_B": harm_df["A"].iloc[0], "phase_deg_B": harm_df["phase_deg"].iloc[0]
    }])

    # merge into output/tables/Table4_dLCF_dTs.csv
    t4_path = make_output_path(cfg, "tables", "Table4_dLCF_dTs.csv")
    if Path(t4_path).exists():
        old = pd.read_csv(t4_path)
        # drop existing B row for this region, then append
        mask = ~((old.get("region","")==region) & (old.get("method","")=="B_chain"))
        new = pd.concat([old[mask], brow], ignore_index=True)
        new.to_csv(t4_path, index=False)
    else:
        # if separate A table exists, try to merge
        a_path = make_output_path(cfg, "tables", "Table4_dLCF_dTs_A.csv")
        if Path(a_path).exists():
            aold = pd.read_csv(a_path)
            # rename A columns to A_* and add method="A_direct"
            aold = aold.rename(columns={
                "dLCF_dTs_mean": "dLCF_dTs_mean_A",
                "dLCF_dTs_ci_low": "dLCF_dTs_ci_low_A",
                "dLCF_dTs_ci_high": "dLCF_dTs_ci_high_A",
                "A": "A_A", "phase_deg": "phase_deg_A"
            })
            aold["method"] = "A_direct"
            new = pd.concat([aold, brow], ignore_index=True)
            new.to_csv(t4_path, index=False)
        else:
            brow.to_csv(t4_path, index=False)

    # log
    logger.info(f"Saved B-chain monthly → {out_csv}")
    logger.info(f"Saved B-chain harmonics → {out_harm}")
    logger.info(f"Updated Table 4 (combined A/B) → {t4_path}")
    logger.info("DONE.")

if __name__ == "__main__":
    main()

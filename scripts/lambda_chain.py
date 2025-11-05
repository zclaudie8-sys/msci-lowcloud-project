#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lambda_chain.py

Compute monthly cloud feedback via product:
    lambda_cld^(m) = gamma^(m) * (dLCF/dTs)^(m)
and decompose month-to-month departures into gamma / dLCF_dTs / synergy.

Reads:
  - output/gamma/GAMMA_<REGION>__*.csv  (auto-pick binned_* or unbinned; use overall rows bin=NaN)
  - output/dLCF_dTs/ALINE_<REGION>.csv  (A-line; column dLCF_dTs)
  - output/dLCF_dTs/BCHAIN_<REGION>.csv (B-line; column dLCF_dTs_chain)

Args:
  --region NEP            # single region
  --regions ALL|SEP,SEA   # run multiple
  --method A|B|BOTH       # default BOTH

Outputs per region:
  - output/lambda/LAMBDA_<REGION>_A.csv, ..._B.csv
  - output/lambda/LAMBDA_HARM_<REGION>_A.csv, ..._B.csv
  - output/lambda/LAMBDA_ATTR_<REGION>_A.csv, ..._B.csv
Table 5 (combined):
  - output/tables/Table5_lambda.csv
Figure 4:
  - single region:  output/figures/Fig4_lambda_polar_<REGION>.png
  - multi-region :  output/figures/Fig4_lambda_polar_obs_vs_model.png
"""

from __future__ import annotations
from pathlib import Path
import sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
try:
    from scripts.utils_config import load_config, make_output_path, setup_logger
    import scripts.utils_climatology as climu
except ModuleNotFoundError:
    from utils_config import load_config, make_output_path, setup_logger
    import utils_climatology as climu


# ------------------ ARGS ------------------
def parse_args():
    p = argparse.ArgumentParser(description="λ_cld = γ × (dLCF/dTs) with attribution and harmonics.")
    p.add_argument("--region", help="Single region (NEP/NEA/SEP/SEA/SEI)")
    p.add_argument("--regions", default="", help="Comma list or ALL to use project.regions")
    p.add_argument("--method", choices=["A","B","BOTH"], default="BOTH", help="Use A-line, B-line, or both (default).")
    return p.parse_args()


# ------------------ HELPERS ------------------
def _regions_to_run(cfg, args):
    if args.regions:
        rs = [r.strip() for r in args.regions.split(",") if r.strip()]
        if len(rs)==1 and rs[0].upper()=="ALL":
            rs = cfg["project"]["regions"]
        return rs
    if args.region:
        return [args.region]
    raise SystemExit("请提供 --region 或 --regions（可用 ALL）")

def _pick_gamma_file(gdir: Path, region: str) -> Path:
    """
    选择该 region 的 GAMMA 文件，优先 binned_*，否则 unbinned，再否则无 tag。
    """
    # 1) 先找 binned_ 前缀
    cands = list(gdir.glob(f"GAMMA_{region}__binned_*.csv"))
    # 2) 再找 unbinned
    if not cands:
        cands = list(gdir.glob(f"GAMMA_{region}__unbinned.csv"))
    # 3) 最后找 legacy 无 tag
    if not cands:
        cands = list(gdir.glob(f"GAMMA_{region}.csv"))

    if not cands:
        raise FileNotFoundError(f"No GAMMA file found for {region} under {gdir}")

    # 取最后一个（通常是时间上/命名上最新的）
    return sorted(cands)[-1]


def _read_gamma(cfg, region):
    gdir = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["gamma"]
    fp = _pick_gamma_file(gdir, region)
    df = pd.read_csv(fp)
    # prefer overall rows: bin NaN if column exists, else use as is
    if "bin" in df.columns:
        df_overall = df[df["bin"].isna()] if df["bin"].isna().any() else df
    else:
        df_overall = df
    if "gamma" not in df_overall.columns:
        raise KeyError(f"'gamma' column not found in {fp.name}")
    gm = df_overall[["month","gamma"]].dropna()
    return gm.set_index("month")["gamma"].astype(float), fp.name

def _read_aline(cfg, region):
    f = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["dLCF_dTs"] / f"ALINE_{region}.csv"
    if not f.exists():
        raise FileNotFoundError(f"A-line file not found: {f}")
    df = pd.read_csv(f)
    return df.set_index("month")["dLCF_dTs"].astype(float), f.name

def _read_bchain(cfg, region):
    f = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["dLCF_dTs"] / f"BCHAIN_{region}.csv"
    if not f.exists():
        raise FileNotFoundError(f"B-line file not found: {f}")
    df = pd.read_csv(f)
    return df.set_index("month")["dLCF_dTs_chain"].astype(float), f.name

def _harmonics(series_by_month: pd.Series):
    months = np.arange(1,13)
    da = xr.DataArray(
        np.array([series_by_month.get(m, np.nan) for m in months], float),
        coords={"month": months}, dims=("month",), name="lambda"
    )
    fit = climu.first_harmonic_fit(da, min_valid=8, soft=True)
    return {
        "A": float(fit["amplitude"]),
        "phase_deg": float(fit["phase_deg"]),
        "peak_month": int(fit["peak_month"]),
        "r2": float(fit["r2"]),
        "n_valid": float(fit.get("n_valid", np.isfinite(da.values).sum()))
    }

def _attribution(g: pd.Series, d: pd.Series):
    # Centered decomposition around monthly means
    # Δλ = (g-ḡ) d̄ + ḡ (d-d̄) + (g-ḡ)(d-d̄)
    idx = sorted(set(g.dropna().index) & set(d.dropna().index))
    g = g.reindex(idx); d = d.reindex(idx)
    gbar = float(np.nanmean(g.values)); dbar = float(np.nanmean(d.values))
    cg = (g - gbar) * dbar
    cd = (d - dbar) * gbar
    syn = (g - gbar) * (d - dbar)
    return pd.DataFrame({"month": idx,
                         "contrib_gamma": cg.values,
                         "contrib_dLCF_dTs": cd.values,
                         "synergy": syn.values})


def _mean_ci_12(s: pd.Series):
    tcrit = 2.201
    v = s.dropna().values
    if v.size >= 3:
        mu = float(np.nanmean(v))
        sd = float(np.nanstd(v, ddof=1))
        se = sd / np.sqrt(np.isfinite(v).sum())
        lo, hi = mu - tcrit*se, mu + tcrit*se
    else:
        mu = lo = hi = np.nan
    return mu, lo, hi


# ------------------ MAIN ------------------
def main():
    args = parse_args()
    cfg  = load_config(str(ROOT / "configs" / "config.yaml"))
    logger = setup_logger(cfg, name="lambda_chain")

    regions = _regions_to_run(cfg, args)
    methods = ["A","B"] if args.method=="BOTH" else [args.method]

    # For multi-region figure
    polar_points = []  # (region, method, A, phase_deg)

    for region in regions:
        logger.info(f"[RUN] region={region} | methods={methods}")

        gamma_m, gfile = _read_gamma(cfg, region)  # monthly gamma
        logger.info(f"[{region}] read gamma from {gfile}")

        for method in methods:
            # choose dLCF/dTs
            if method=="A":
                d_m, dfile = _read_aline(cfg, region)
            else:
                d_m, dfile = _read_bchain(cfg, region)
            logger.info(f"[{region}] method {method}: read dLCF/dTs from {dfile}")

            # monthly product on intersection months
            idx = sorted(set(gamma_m.dropna().index) & set(d_m.dropna().index))
            if not idx:
                logger.warning(f"[{region}] method {method}: no overlapping months, skip")
                continue
            g = gamma_m.reindex(idx); d = d_m.reindex(idx)
            lam = g * d

            # save LAMBDA_<REGION>_<M>.csv
            out = pd.DataFrame({"month": idx,
                                "lambda": lam.values,
                                "gamma": g.values,
                                "dLCF_dTs": d.values})
            out_csv = make_output_path(cfg, "lambda", f"LAMBDA_{region}_{method}.csv")
            out.to_csv(out_csv, index=False)
            logger.info(f"[{region}] method {method}: saved → {out_csv}")

            # harmonics
            h = _harmonics(lam)
            harm_df = pd.DataFrame([h])
            out_h = make_output_path(cfg, "lambda", f"LAMBDA_HARM_{region}_{method}.csv")
            harm_df.to_csv(out_h, index=False)
            logger.info(f"[{region}] method {method}: saved harmonics → {out_h}")
            polar_points.append((region, method, h["A"], h["phase_deg"]))

            # attribution
            attr = _attribution(g, d)
            out_attr = make_output_path(cfg, "lambda", f"LAMBDA_ATTR_{region}_{method}.csv")
            attr.to_csv(out_attr, index=False)
            logger.info(f"[{region}] method {method}: saved attribution → {out_attr}")

            # update Table 5
            mu, lo, hi = _mean_ci_12(lam)
            row = pd.DataFrame([{
                "region": region, "method": method,
                "lambda_mean": mu, "lambda_ci_low": lo, "lambda_ci_high": hi,
                "A": h["A"], "phase_deg": h["phase_deg"]
            }])
            t5 = make_output_path(cfg, "tables", "Table5_lambda.csv")
            if Path(t5).exists():
                old = pd.read_csv(t5)
                mask = ~((old.get("region","")==region) & (old.get("method","")==method))
                pd.concat([old[mask], row], ignore_index=True).to_csv(t5, index=False)
            else:
                row.to_csv(t5, index=False)
            logger.info(f"[{region}] method {method}: updated Table 5 → {t5}")

    # Figure 4 — polar comparison
    if len(regions) == 1:
        region = regions[0]
        pts = [(m,A,phi) for r,m,A,phi in [(r, m, A, ph) for (r,m,A,ph) in [(p[0],p[1],p[2],p[3]) for p in polar_points]] if r==region]
        # Ensure both A and B exist
        fig_name = make_output_path(cfg, "figures", f"Fig4_lambda_polar_{region}.png")
    else:
        fig_name = make_output_path(cfg, "figures", "Fig4_lambda_polar_obs_vs_model.png")

    # Plot polar if we have at least one point
    if polar_points:
        plt.figure(figsize=(7.0, 7.0))
        ax = plt.subplot(111, projection="polar")
        # helper to plot
        colors = {"A":"tab:blue","B":"tab:orange"}
        for (r, m, A, ph) in polar_points:
            theta = np.deg2rad(ph % 360.0)
            lbl = f"{r}-{m}"
            ax.scatter([theta],[A], s=60, label=lbl,
                       color=colors.get(m,"tab:gray"), alpha=0.85, edgecolors="none")
            ax.plot([0, theta],[0, A], color=colors.get(m,"tab:gray"), lw=1.4, alpha=0.7)
        # aesthetics
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_title("λ_cld amplitude & phase (A, φ)", va="bottom")
        # de-duplicate legend
        handles, labels = ax.get_legend_handles_labels()
        # combine same label once
        seen=set(); h2=[]; l2=[]
        for h,l in zip(handles, labels):
            if l not in seen:
                seen.add(l); h2.append(h); l2.append(l)
        ax.legend(h2, l2, bbox_to_anchor=(1.05,1.0), loc="upper left", frameon=False, fontsize=8)
        plt.tight_layout(); plt.savefig(fig_name, dpi=220, bbox_inches="tight"); plt.close()

    logger = setup_logger(cfg, name="lambda_chain")  # refresh for last line path echo
    logger.info(f"Saved Fig4 polar → {fig_name}")
    logger.info("DONE.")


if __name__ == "__main__":
    main()

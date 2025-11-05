#!/usr/bin/env python3
# CMIP6 AMIP vs OBS: regional monthly climatology + 1st harmonic (calendar-safe)

from __future__ import annotations
from pathlib import Path
import sys, argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# --- import fallbacks so it runs from ~ or from project root ---
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
try:
    from scripts.utils_config import load_config, get_regions, setup_logger, make_output_path
    from scripts.utils_region import select_region_mean
    from scripts.utils_climatology import first_harmonic_fit
except ModuleNotFoundError:
    from utils_config import load_config, get_regions, setup_logger, make_output_path
    from utils_region import select_region_mean
    from utils_climatology import first_harmonic_fit

VAR_OBS_MAP = {"cllisccp":"cllmodis","clswlow":"clswlow","cllwlow":"cllwlow"}

def wrap_phase_diff(dm: float, do: float) -> float:
    return float((dm - do + 180.0) % 360.0 - 180.0)

def model_name_from_fname(fname: str) -> str:
    base = Path(fname).name
    toks = base[:-3].split("_")
    for i,t in enumerate(toks):
        if t.lower()=="amip" and i>=1: return toks[i-1]
    return toks[2] if len(toks)>=3 else base

def pearsonr12(a: xr.DataArray, b: xr.DataArray) -> float:
    av, bv = a.values.astype(float), b.values.astype(float)
    m = np.isfinite(av) & np.isfinite(bv)
    if m.sum()<3 or av[m].std()==0 or bv[m].std()==0: return np.nan
    return float(np.corrcoef(av[m], bv[m])[0,1])

# ---- calendar-aware slice on 'time' ----
def calendar_aware_slice(da: xr.DataArray, y0: int, y1: int) -> xr.DataArray:
    idx = da.indexes.get("time", None)
    if idx is None:
        return da
    cal = getattr(idx, "calendar", None)
    try:
        import cftime
    except Exception:
        cal = None
    if cal == "360_day":
        start = cftime.Datetime360Day(y0,1,1); end = cftime.Datetime360Day(y1,12,30)
    elif cal in ("365_day","noleap"):
        start = cftime.DatetimeNoLeap(y0,1,1); end = cftime.DatetimeNoLeap(y1,12,31)
    elif cal in ("366_day","all_leap"):
        start = cftime.DatetimeAllLeap(y0,1,1); end = cftime.DatetimeAllLeap(y1,12,31)
    else:
        start = f"{y0}-01-01"; end = f"{y1}-12-31"
    return da.sel(time=slice(start,end))

def main():
    ap = argparse.ArgumentParser(description="CMIP6 AMIP harmonic summary vs OBS (calendar-safe)")
    ap.add_argument("--var", required=True, choices=list(VAR_OBS_MAP.keys()))
    ap.add_argument("--regions", default="", help="Comma-separated regions (e.g., SEP,SEA,NEP)")
    ap.add_argument("--cmip_path", default="/gws/nopw/j04/csgap/ceppi/data/cmip6/5x5/")
    ap.add_argument("--start", type=int, default=2003)
    ap.add_argument("--end", type=int, default=2014)
    args = ap.parse_args()

    cfg = load_config("configs/config.yaml")
    logger = setup_logger(cfg, name="harmonic_summary_cmip_amip")

    y0_cfg, y1_cfg = cfg["project"]["time_span"]
    y_start, y_end = max(y0_cfg, args.start), min(y1_cfg, args.end)
    span = f"{y_start}-{y_end}"
    regions = [r.strip() for r in args.regions.split(",") if r.strip()] if args.regions else cfg["project"]["regions"]
    logger.info(f"Window: {span} | Regions: {regions}")

    # OBS
    obs_var = VAR_OBS_MAP[args.var]
    obs_file = {"cllmodis":"cllmodis_200207-202406.nc",
                "clswlow":"clswlow_200207-202406.nc",
                "cllwlow":"cllwlow_200207-202406.nc"}[obs_var]
    obs_path = Path(cfg["data"]["modis_path"]) / obs_file
    da_obs_full = xr.open_dataset(obs_path)[obs_var]
    logger.info(f"Loaded OBS {obs_var} from {obs_path}")

    # MODELS
    var_dir = Path(args.cmip_path) / args.var
    files = sorted(var_dir.glob("*amip*.nc"))
    if not files: raise FileNotFoundError(f"No AMIP files under {var_dir}")
    logger.info(f"Found {len(files)} AMIP files")

    recs = []
    for f in files:
        model = model_name_from_fname(f.name)
        logger.info(f"[MODEL] {model}")
        try:
            da_mod_full = xr.open_dataset(f)[args.var]
        except Exception as e:
            logger.warning(f"Skip {f.name}: {e}"); continue

        for rg in regions:
            box = get_regions(cfg, [rg])[rg]
            ts_mod = select_region_mean(da_mod_full, box)
            ts_obs = select_region_mean(da_obs_full, box)
            # ★ use calendar-aware slicing here (fixes 360_day issue)
            ts_mod = calendar_aware_slice(ts_mod, y_start, y_end)
            ts_obs = calendar_aware_slice(ts_obs, y_start, y_end)
            if ts_mod.size==0 or ts_obs.size==0: continue

            clim_mod = ts_mod.groupby("time.month").mean("time")
            clim_obs = ts_obs.groupby("time.month").mean("time")

            fit_m = first_harmonic_fit(clim_mod)
            fit_o = first_harmonic_fit(clim_obs)

            dphi  = wrap_phase_diff(fit_m["phase_deg"], fit_o["phase_deg"])
            Arat  = fit_m["amplitude"]/fit_o["amplitude"] if fit_o["amplitude"]!=0 else np.nan
            r12   = pearsonr12(fit_m["climatology"], fit_o["climatology"])

            recs.append({
                "model": model, "region": rg, "var": args.var, "span": span,
                "A_model": fit_m["amplitude"], "phi_model": fit_m["phase_deg"], "R2_model": fit_m["r2"],
                "A_obs":   fit_o["amplitude"], "phi_obs":   fit_o["phase_deg"], "R2_obs":   fit_o["r2"],
                "delta_phase": dphi, "A_ratio": Arat, "corr12": r12,
            })

    df = pd.DataFrame(recs)
    out_csv = make_output_path(cfg, stem=f"cmip_amip_{args.var}_vs_obs", region=None, subdir="results", ext="csv")
    df.to_csv(out_csv, index=False); logger.info(f"Saved summary → {out_csv}")

    # quick plots per region
    for rg in df["region"].unique():
        sub = df[df["region"]==rg]
        figA = make_output_path(cfg, stem=f"cmip_amip_{args.var}_A_vs_obs", region=rg, subdir="figures", ext="png")
        plt.figure(figsize=(7,4)); plt.scatter(sub["A_obs"], sub["A_model"])
        for _,row in sub.iterrows(): plt.annotate(row["model"], (row["A_obs"], row["A_model"]), fontsize=7)
        lim = max(sub["A_obs"].max(), sub["A_model"].max())*1.1 if len(sub)>0 else 1
        plt.plot([0,lim],[0,lim],"k--"); plt.xlabel("Amplitude OBS"); plt.ylabel("Amplitude MODEL")
        plt.title(f"{args.var} amplitude | {rg} | {span}"); plt.tight_layout(); plt.savefig(figA,dpi=200); plt.close()

        figP = make_output_path(cfg, stem=f"cmip_amip_{args.var}_dPhase", region=rg, subdir="figures", ext="png")
        plt.figure(figsize=(8,4)); sub_sorted = sub.sort_values("delta_phase")
        plt.bar(sub_sorted["model"], sub_sorted["delta_phase"]); plt.axhline(0,color="k")
        plt.ylabel("Δphase (deg) model - obs"); plt.xticks(rotation=60,ha="right")
        plt.title(f"{args.var} phase diff | {rg} | {span}"); plt.tight_layout(); plt.savefig(figP,dpi=200); plt.close()

if __name__ == "__main__":
    main()

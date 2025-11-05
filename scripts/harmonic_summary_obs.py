#!/usr/bin/env python3
"""
Summarize first-harmonic fitting for MODIS observations across regions.

Outputs (under output/ and figures/):
  - output/<var>_harmonics_summary_<span>.csv
  - figures/<var>_amplitude_summary_<span>.png
  - figures/<var>_peakmonth_summary_<span>.png

Usage:
  python scripts/harmonic_summary_obs.py --var cllmodis
  # 可选变量: cllmodis, clswlow, cllwlow
"""

from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from utils_config import load_config, get_regions, setup_logger, make_output_path
from utils_region import select_region_mean


# 文件名映射（Paulo 提供的 5x5 月平均）
VAR_TO_FILE = {
    "cllmodis": "cllmodis_200207-202406.nc",
    "clswlow":  "clswlow_200207-202406.nc",
    "cllwlow":  "cllwlow_200207-202406.nc",
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Harmonic summary for MODIS observations.")
    p.add_argument("--var", default="cllmodis", choices=list(VAR_TO_FILE.keys()),
                   help="Variable to process.")
    p.add_argument("--regions", default="", help="Comma-separated region keys (optional).")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) 配置与日志
    cfg = load_config("configs/config.yaml")
    logger = setup_logger(cfg, name="harmonic_summary_obs")

    y0, y1 = cfg["project"]["time_span"]
    all_regions = cfg["project"]["regions"]
    regions = (
        [r.strip() for r in args.regions.split(",") if r.strip()]
        if args.regions else all_regions
    )
    logger.info(f"Time span: {y0}-{y1}")
    logger.info(f"Regions: {regions}")

    # 2) 数据文件
    modis_dir = Path(cfg["data"]["modis_path"])
    fname = VAR_TO_FILE[args.var]
    fpath = modis_dir / fname
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {fpath}")
    logger.info(f"Loading {args.var} from: {fpath}")

    ds = xr.open_dataset(fpath)
    if args.var not in ds.data_vars:
        raise KeyError(f"Variable '{args.var}' not found. Vars in file: {list(ds.data_vars)}")
    da_full = ds[args.var]

    # 3) 遍历区域
    records = []
    for r in regions:
        box = get_regions(cfg, [r])[r]
        logger.info(f"Processing region {r}: {box}")

        ts = select_region_mean(da_full, box)
        ts = ts.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
        clim = ts.groupby("time.month").mean("time")
        fit = first_harmonic_fit(clim)

        records.append({
            "region": r,
            "amplitude": float(fit["amplitude"]),
            "phase_deg": float(fit["phase_deg"]),
            "peak_month": int(fit["peak_month"]),
            "r2": float(fit["r2"]),
        })

    df = pd.DataFrame(records).set_index("region")
    span = f"{y0}-{y1}"

    # 4) 保存 CSV
    span = f"{y0}-{y1}"
    out_csv = make_output_path(cfg, "harmonics", f"{args.var}_harmonics_summary_{span}.csv")

    df.reset_index().to_csv(out_csv, index=False)
    logger.info(f"Saved summary CSV → {out_csv}")

    # 5) 幅度 & 峰月图
    amp_fig  = make_output_path(cfg, "figures", f"{args.var}_amplitude_summary_{span}.png")
    peak_fig = make_output_path(cfg, "figures", f"{args.var}_peakmonth_summary_{span}.png")

    plt.figure(figsize=(7.2,4.2))
    df["amplitude"].plot(kind="bar")
    plt.ylabel("Amplitude (% cloud)")
    plt.title(f"{args.var} amplitude | {span}")
    plt.tight_layout()
    plt.savefig(amp_fig, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved amplitude figure → {amp_fig}")

    plt.figure(figsize=(7.2,4.2))
    df["peak_month"].plot(kind="bar")
    plt.ylabel("Peak month (1–12)")
    plt.yticks(range(1,13))
    plt.title(f"{args.var} peak month | {span}")
    plt.tight_layout()
    plt.savefig(peak_fig, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved peak-month figure → {peak_fig}")

    logger.info("DONE.")

if __name__ == "__main__":
    main()

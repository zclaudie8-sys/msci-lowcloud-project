#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute regional mean time series, monthly climatology, and first-harmonic fit
for MODIS/LCRE on Paulo's 5x5 monthly grids.

Outputs (under output/):
  - regional_monthly/<var>_region_mean_<span>_<REGION>.nc
  - regional_monthly/<var>_region_mean_<span>_<REGION>.csv
  - regional_monthly/<var>_climatology_<span>_<REGION>.csv
  - harmonics/<var>_harmonics_<span>_<REGION>.csv

Usage:
  python scripts/region_mean_modis.py --region SEP --var cllmodis
  # var: cllmodis, clswlow, cllwlow
"""

from __future__ import annotations
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import xarray as xr

# -----------------------------------------------------------------------------
# Robust imports: run from project root OR from scripts/
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))   # scripts/
sys.path.insert(0, str(ROOT))   # project root

try:
    from scripts.utils_config import load_config, get_regions, make_output_path, setup_logger
    from scripts.utils_region  import select_region_mean
    import scripts.utils_climatology as climu
except ModuleNotFoundError:
    from utils_config import load_config, get_regions, make_output_path, setup_logger
    from utils_region  import select_region_mean
    import utils_climatology as climu

# 文件名映射（Paulo 的 5x5 月平均）
VAR_TO_FILE = {
    "cllmodis": "cllmodis_200207-202406.nc",
    "clswlow" : "clswlow_200207-202406.nc",
    "cllwlow" : "cllwlow_200207-202406.nc",
}

# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Regional mean + climatology + harmonic fit for MODIS/LCRE (5x5).")
    p.add_argument("--region", default="SEP", help="Region key in config (SEP/SEA/NEP/NEA/SEI)")
    p.add_argument("--regions", default="", help="Comma list or ALL (uses project.regions)")
    p.add_argument("--var", default="cllmodis", choices=list(VAR_TO_FILE.keys()),
                   help="Variable to process: cllmodis / clswlow / cllwlow")
    p.add_argument("--bbox", default="", help="Custom box 'W,E,S,N' to override --region")
    p.add_argument("--list", action="store_true", help="List files under data.modis_path and exit.")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _regions_to_run(cfg: dict, args) -> list[tuple[str, list[float]]]:
    """
    由 --regions/--region 生成 [(region_key, W,E,S,N), ...]
    若给了 --bbox，则只跑一个 custom。
    """
    if args.bbox:
        parts = [float(x) for x in args.bbox.split(",")]
        if len(parts) != 4:
            raise ValueError("--bbox 需要四个数字：'W,E,S,N'")
        return [("custom", _reorder_box_to_WESN(parts))]

    if args.regions:
        rs = [r.strip() for r in args.regions.split(",") if r.strip()]
        if len(rs) == 1 and rs[0].upper() == "ALL":
            rs = cfg["project"]["regions"]
    elif args.region:
        rs = [args.region]
    else:
        raise SystemExit("请提供 --region 或 --regions（可用 ALL），或 --bbox。")

    regdict = get_regions(cfg)
    out = []
    for r in rs:
        if r not in regdict:
            raise KeyError(f"未知区域键：{r}；可用={list(regdict.keys())}")
        out.append((r, _reorder_box_to_WESN(regdict[r])))
    return out


def _reorder_box_to_WESN(box):
    """
    接受 list/tuple/dict，返回 [W, E, S, N]。
    兼容：
      - [W, E, S, N] / [lon_min, lon_max, lat_min, lat_max]
      - [lat_min, lat_max, lon_min, lon_max]
      - {'lon_min':..,'lon_max':..,'lat_min':..,'lat_max':..}
    """
    if isinstance(box, dict):
        return [float(box["lon_min"]), float(box["lon_max"]),
                float(box["lat_min"]), float(box["lat_max"])]

    if isinstance(box, (list, tuple)) and len(box) == 4:
        b0, b1, b2, b3 = map(float, box)

        def _is_lat(x): return -90.0 <= x <= 90.0
        def _is_lon(x): return -360.0 <= x <= 360.0

        # [W,E,S,N]
        if _is_lon(b0) and _is_lon(b1) and _is_lat(b2) and _is_lat(b3):
            return [b0, b1, b2, b3]
        # [lat_min,lat_max,lon_min,lon_max] → 重排
        if _is_lat(b0) and _is_lat(b1) and _is_lon(b2) and _is_lon(b3):
            return [b2, b3, b0, b1]

    raise ValueError(f"无法识别的 region box 顺序: {box}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    # 1) 配置与日志
    cfg_path = ROOT / "configs" / "config.yaml"
    cfg = load_config(str(cfg_path))
    logger = setup_logger(cfg, name="region_mean_modis")

    # 2) 列表模式（诊断）
    modis_dir = Path(cfg["data"]["modis_path"])
    if args.list:
        print("config file =", (ROOT / "configs" / "config.yaml").resolve())
        print("modis_path  =", modis_dir.resolve())
        files = sorted(modis_dir.glob("*.nc"))
        if not files:
            print("(空) 该目录下没有 .nc 文件"); return
        for f in files:
            try:
                ds0 = xr.open_dataset(f, decode_times=False)
                vars0 = list(ds0.data_vars.keys())
            except Exception as e:
                vars0 = [f"<open failed: {e}>"]
            print(f"- {f.name}: vars={vars0}")
        return

    # 3) 组装需要跑的区域列表
    regions_to_run = _regions_to_run(cfg, args)

    # 4) 数据文件
    fname = VAR_TO_FILE[args.var]
    fpath = modis_dir / fname
    if not fpath.exists():
        candidates = [p.name for p in modis_dir.glob(f"{args.var.split('_')[0]}*.nc")]
        raise FileNotFoundError(
            f"找不到文件：{fpath}\n"
            f"→ 请检查 configs/config.yaml 的 data.modis_path，或更新脚本里的 VAR_TO_FILE。\n"
            f"目录候选：{candidates}"
        )
    logger.info(f"Loading {args.var} from: {fpath}")
    ds = xr.open_dataset(fpath)
    if args.var not in ds.data_vars:
        raise KeyError(f"Variable '{args.var}' not found. Vars in file: {list(ds.data_vars)}")
    da = ds[args.var]

    y0, y1 = cfg["project"]["time_span"]
    span_label = f"{y0}-{y1}"

    # 5) 区域循环
    for region_key, box in regions_to_run:
        logger.info(f"[RUN] region={region_key} | box={box} | span={span_label}")

        # 区域均值 & 截取
        ts = select_region_mean(da, box)
        ts = ts.where((ts["time"].dt.year >= y0) & (ts["time"].dt.year <= y1), drop=True)
        if ts.size == 0:
            logger.warning(f"[{region_key}] 无有效时间点，跳过。"); continue

        # 月气候态（12个月）
        counts = ts.groupby("time.month").count().sel(month=np.arange(1, 13)).values
        clim = ts.groupby("time.month").mean("time", skipna=True).sel(month=np.arange(1, 13))
        months = clim["month"].values
        if not (months == np.arange(1, 13)).all():
            logger.warning(f"[{region_key}] month 顺序异常：{months}"); continue

        # 一阶谐波
        fit = climu.first_harmonic_fit(clim, min_valid=8, soft=True)
        logger.info(
            f"[{region_key}] FIT amp={fit['amplitude']:.4f}, phase={fit['phase_deg']:.1f}°, "
            f"peak={fit['peak_month']}, R2={fit['r2']:.3f}, n_valid={fit.get('n_valid')}"
        )

        # 保存
        # (a) 区域月均
        out_nc  = make_output_path(cfg, "regional_monthly", f"{args.var}_region_mean_{span_label}_{region_key}.nc")
        out_csv = make_output_path(cfg, "regional_monthly", f"{args.var}_region_mean_{span_label}_{region_key}.csv")
        ts.to_netcdf(out_nc); ts.to_series().to_csv(out_csv)
        logger.info(f"[{region_key}] Saved regional mean → {out_nc}")

        # (b) 月气候态
        out_csv_clim = make_output_path(cfg, "regional_monthly", f"{args.var}_climatology_{span_label}_{region_key}.csv")
        pd.Series(clim.to_series()).to_csv(out_csv_clim)

        # (c) 一谐波
        out_csv_fit = make_output_path(cfg, "harmonics", f"{args.var}_harmonics_{span_label}_{region_key}.csv")
        pd.DataFrame([{
            "region": region_key, "span": span_label, "var": args.var,
            "a": fit["a"], "b": fit["b"], "c": fit["c"],
            "amplitude": fit["amplitude"], "phase_deg": fit["phase_deg"],
            "peak_month": fit["peak_month"], "trough_month": fit["trough_month"],
            "r2": fit["r2"], "n_valid": fit.get("n_valid", np.nan),
            "count_month_1":  int(counts[0])  if len(counts) >= 1 else np.nan,
            "count_month_2":  int(counts[1])  if len(counts) >= 2 else np.nan,
            "count_month_3":  int(counts[2])  if len(counts) >= 3 else np.nan,
            "count_month_4":  int(counts[3])  if len(counts) >= 4 else np.nan,
            "count_month_5":  int(counts[4])  if len(counts) >= 5 else np.nan,
            "count_month_6":  int(counts[5])  if len(counts) >= 6 else np.nan,
            "count_month_7":  int(counts[6])  if len(counts) >= 7 else np.nan,
            "count_month_8":  int(counts[7])  if len(counts) >= 8 else np.nan,
            "count_month_9":  int(counts[8])  if len(counts) >= 9 else np.nan,
            "count_month_10": int(counts[9])  if len(counts) >=10 else np.nan,
            "count_month_11": int(counts[10]) if len(counts) >=11 else np.nan,
            "count_month_12": int(counts[11]) if len(counts) >=12 else np.nan,
        }]).to_csv(out_csv_fit, index=False)
        logger.info(f"[{region_key}] Saved harmonic params → {out_csv_fit}")

    logger.info("DONE.")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()

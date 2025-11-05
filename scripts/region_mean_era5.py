#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5 regional monthly means + harmonics (STRICT, NO GUESSING)

使用方式：
  # 仅列出 data/era5 目录下每个 nc 的变量清单（不计算）
  python scripts/region_mean_era5.py --list

  # 仅检查 config.yaml 中 era5_vars 的映射是否正确（不计算）
  python scripts/region_mean_era5.py --check

  # 按映射计算（严格：不做任何猜测）
  python scripts/region_mean_era5.py --region NEP --vars EIS,W500,TS,SST,U10,Q
"""

from __future__ import annotations
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

try:
    from scripts.utils_config import load_config, get_regions, make_output_path, setup_logger
    from scripts.utils_region  import select_region_mean
    import scripts.utils_climatology as climu
except ModuleNotFoundError:
    from utils_config import load_config, get_regions, make_output_path, setup_logger
    from utils_region  import select_region_mean
    import utils_climatology as climu


def parse_args():
    p = argparse.ArgumentParser(description="ERA5 regional monthly means (STRICT mapping).")
    p.add_argument("--region", help="Region key in config (NEP/NEA/SEP/SEA/SEI)")
    p.add_argument("--regions", default="", help="Comma list or ALL (use project.regions)")
    p.add_argument("--vars", help="Comma-separated logical names matching data.era5_vars (e.g., EIS,W500,SST)")
    p.add_argument("--list", action="store_true", help="List all nc files and their variables; no computation.")
    p.add_argument("--check", action="store_true", help="Validate config mappings; no computation.")
    return p.parse_args()

def _regions_to_run(cfg: dict, args) -> list[tuple[str, list[float]]]:
    """按 --regions/--region 生成 [(region_key, WESN), ...] 列表。"""
    # 组装区域名列表
    if getattr(args, "regions", ""):
        rs = [r.strip() for r in args.regions.split(",") if r.strip()]
        if len(rs) == 1 and rs[0].upper() == "ALL":
            rs = cfg["project"]["regions"]
    elif getattr(args, "region", ""):
        rs = [args.region]
    else:
        raise SystemExit("请提供 --region 或 --regions（可用 ALL）")

    regdict = get_regions(cfg)
    out = []
    for r in rs:
        if r not in regdict:
            raise KeyError(f"未知区域键：{r}；可用={list(regdict.keys())}")
        raw_box = regdict[r]                     # [lat_min, lat_max, lon_min, lon_max]
        box = _cfg_latlat_lonlon_to_WESN(raw_box)  # → [W, E, S, N]
        out.append((r, box))
    return out


def _cfg_latlat_lonlon_to_WESN(raw_box):
    """
    配置里的约定： [lat_min, lat_max, lon_min, lon_max]
    这里不做任何猜测，直接转成 [W, E, S, N]
    """
    if not (isinstance(raw_box, (list, tuple)) and len(raw_box) == 4):
        raise ValueError(f"区域必须是 [lat_min, lat_max, lon_min, lon_max]，给到的是：{raw_box}")
    lat_min, lat_max, lon_min, lon_max = map(float, raw_box)
    return [lon_min, lon_max, lat_min, lat_max]  # [W, E, S, N]



def _list_nc_vars(base: Path) -> list[dict]:
    rows = []
    for f in sorted(base.glob("*.nc")):
        try:
            with xr.open_dataset(f) as ds:
                rows.append({
                    "file": f.name,
                    "vars": list(ds.data_vars.keys()),
                    "dims": list(ds.dims.keys()),
                })
        except Exception as e:
            rows.append({"file": f.name, "vars": [f"<open failed: {e}>"], "dims": []})
    return rows


def _norm_list(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _get_mapping(cfg: dict, key: str) -> tuple[Path, str]:
    m = cfg.get("data", {}).get("era5_vars", {}).get(key)
    if not m:
        raise KeyError(f"[映射缺失] data.era5_vars 中没有条目：{key}")
    f = m.get("file"); v = m.get("var")
    if not f or not v:
        raise KeyError(f"[映射不完整] {key} 需要 file 与 var：{m}")
    era5_dir = Path(cfg["data"]["era5_path"])
    fpath = era5_dir / f
    if not fpath.exists():
        raise FileNotFoundError(f"[文件不存在] {key}: {fpath}")
    # 立即验证 var 是否在文件中；若不在，列出实际可用变量
    with xr.open_dataset(fpath) as ds:
        if v not in ds.data_vars:
            raise KeyError(
                f"[变量不存在] {key}: 期望 var='{v}' 不在文件 {fpath.name} 中。\n"
                f"可用变量：{list(ds.data_vars.keys())}"
            )
    return fpath, v


def main():
    args = parse_args()
    cfg = load_config(str(ROOT / "configs" / "config.yaml"))

    # 列清单 / 校验：保持原逻辑
    if args.list:
        base = Path(cfg["data"]["era5_path"])
        rows = _list_nc_vars(base)
        if not rows:
            print(f"(空) 目录下没有 .nc：{base}")
            return
        print(f"列出目录：{base}")
        for r in rows:
            print(f"- {r['file']}: vars={r['vars']}")
        return

    if args.check:
        base = Path(cfg["data"]["era5_path"])
        maps = cfg.get("data", {}).get("era5_vars", {})
        if not maps:
            raise SystemExit("config.yaml 中 data.era5_vars 为空。")
        ok = True
        for k, spec in maps.items():
            try:
                _get_mapping(cfg, k)
                print(f"[OK] {k} -> {spec}")
            except Exception as e:
                ok = False
                print(f"[ERR] {k} -> {spec}\n  {e}")
        if not ok:
            raise SystemExit("有错误。请按上面提示修正 data.era5_vars。")
        print("全部映射校验通过。")
        return

    # 计算模式
    if not args.vars:
        raise SystemExit("计算模式必须提供 --vars。")
    req_vars = [x.strip() for x in args.vars.split(",") if x.strip()]

    logger = setup_logger(cfg, name="region_mean_era5")
    y0, y1 = cfg["project"]["time_span"]

    # === B. 决定要跑的区域列表（一次或五区） ===
    regions_to_run = _regions_to_run(cfg, args)

    # === C. 把原“单区流水线”包进循环里（每个区域独立算、独立写盘） ===
    for region_key, box in regions_to_run:
        logger.info(f"[RUN] region={region_key} | box={box} | span: {y0}-{y1}")

        series_list = []
        harmonics_rows = []

        # ---- 你原来的“逐变量”循环，保持不变（严格映射&落盘 harmonics） ----
        for key in req_vars:
            try:
                fpath, varname = _get_mapping(cfg, key)
            except Exception as e:
                logger.error(f"[{key}] 映射无效：{e}")
                raise

            logger.info(f"[{key}] using file={fpath.name} varname={varname}")
            try:
                ds = xr.open_dataset(fpath)
            except Exception as e:
                logger.exception(f"[{key}] 打开失败：{e}")
                continue

            da = ds[varname]
            ts = select_region_mean(da, box)
            ts = ts.where((ts['time'].dt.year >= y0) & (ts['time'].dt.year <= y1), drop=True)
            if ts.size == 0:
                logger.warning(f"[{key}] 在 {y0}-{y1} 无数据，跳过。")
                continue

            ser = ts.to_series(); ser.name = f"ERA5_{key}"
            series_list.append(ser)

            clim = ts.groupby("time.month").mean("time", skipna=True).sel(month=np.arange(1,13))
            fit = climu.first_harmonic_fit(clim, min_valid=8, soft=True)

            span = f"{y0}-{y1}"
            harm_name = f"ERA5_{key}_harmonics_{span}_{region_key}.csv"
            harm_path = make_output_path(cfg, "harmonics", harm_name)
            pd.DataFrame([{
                "source": "ERA5","var": key,"region": region_key,"span": span,
                "a": fit["a"], "b": fit["b"], "c": fit["c"],
                "amplitude": fit["amplitude"], "phase_deg": fit["phase_deg"],
                "peak_month": fit["peak_month"], "trough_month": fit["trough_month"],
                "r2": fit["r2"], "n_valid": fit.get("n_valid", np.nan),
            }]).to_csv(harm_path, index=False)
            logger.info(f"[{key}] Saved harmonics → {harm_path}")

            harmonics_rows.append({
                "source":"ERA5","var":key,"region":region_key,"span":span,
                "A": float(fit["amplitude"]), "phi_deg": float(fit["phase_deg"]),
                "R2": float(fit["r2"]), "n_valid": float(fit.get("n_valid", np.nan)),
            })

        if not series_list:
            logger.warning(f"[{region_key}] 本区没有成功产出的变量，跳过写月表。")
            continue

        df_monthly = pd.concat(series_list, axis=1).sort_index()
        out_csv = make_output_path(cfg, "regional_monthly", f"ERA5_{region_key}_monthly.csv")
        df_monthly.to_csv(out_csv, index=True)
        logger.info(f"[{region_key}] Saved monthly table → {out_csv}")

        # 小概览（可留可删）
        small_path = make_output_path(cfg, "harmonics", f"ERA5_harmonics_{region_key}.csv")
        pd.DataFrame(harmonics_rows).to_csv(small_path, index=False)
        logger.info(f"[{region_key}] Saved harmonics overview → {small_path}")

    logger.info("DONE.")


'''
import xarray as xr
f = xr.open_dataset("/gws/nopw/j04/csgap/ceppi/data/era5/5x5/tos_197901-202212.nc")

# NEP: 15–30N, 155–115W  → 经度 -155..-115, 纬度 15..30
print(f.tos.sel(lon=slice(-155,-115), lat=slice(15,30)).mean().values)

# NEA: 15–30N, 30W–15E   → 经度 -30..15,   纬度 15..30
print(f.tos.sel(lon=slice(-30,15),    lat=slice(15,30)).mean().values)
'''
if __name__ == "__main__":
    main()

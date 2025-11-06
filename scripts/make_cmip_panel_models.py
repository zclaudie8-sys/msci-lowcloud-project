#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_cmip_panel_models.py — 逐模型 CMIP6 AMIP 月度面板 (model, region, time, value)

用法示例：
  python -m scripts.make_cmip_panel_models --var clswlow --start 2003 --end 2014 \
    --out output/tables/cmip_panel_clswlow_2003-2014_models.csv
"""
from __future__ import annotations
from pathlib import Path
import re, argparse
import pandas as pd
import xarray as xr
import numpy as np

# JASMIN 路径（项目约定）
CMIP_ROOT = Path("/gws/nopw/j04/csgap/ceppi/data/cmip6/5x5")

# 变量子目录与数据中变量名
VAR_MAP = {
    "cllisccp": ("cllisccp", "cllisccp"),
    "clswlow" : ("clswlow" , "clswlow"),
    "cllwlow" : ("cllwlow" , "cllwlow"),
    "ts"      : ("ts"      , "ts"),
    "tas"     : ("tas"     , "tas"),
    "eislts"  : ("eislts"  , "eislts"),  # Paulo 口径
    "eis"     : ("eislts"  , "eislts"),
    "lts"     : ("eislts"  , "eislts"),
}

DEFAULT_REGIONS = ["SEP","SEA","NEP","NEA","SEI"]

def parse_model_from_name(p: Path) -> str:
    s = p.stem
    m = re.search(r"_([A-Za-z0-9\-\.]+)_amip_", s)
    if m: return m.group(1)
    if "_amip_" in s:
        return s.split("_amip_")[0].split("_")[-1]
    # 兜底：从 attrs 中取
    try:
        with xr.open_dataset(p) as ds:
            for k in ["source_id","model_id","model","source"]:
                if k in ds.attrs: return str(ds.attrs[k]).split()[0]
    except Exception:
        pass
    return "UNKNOWN"

def load_regions_from_config() -> list[str]:
    try:
        import yaml  # type: ignore
        y1 = Path("configs/regions.yaml")
        if y1.exists():
            data = yaml.safe_load(y1.read_text())
            if isinstance(data, dict): return [str(k).upper() for k in data.keys()]
        y2 = Path("configs/config.yaml")
        if y2.exists():
            data = yaml.safe_load(y2.read_text())
            if isinstance(data, dict):
                if "regions" in data and isinstance(data["regions"], dict):
                    return [str(k).upper() for k in data["regions"].keys()]
                proj = data.get("project")
                if isinstance(proj, dict) and isinstance(proj.get("regions"), list):
                    return [str(r).upper() for r in proj["regions"]]
    except Exception:
        pass
    return DEFAULT_REGIONS
    
def _region_mean_flex(da, R):
    """兼容不同版本的 select_region_mean 签名：
       - select_region_mean(da, region='SEP')
       - select_region_mean(da, 'SEP')                # 位置参数
       - select_region_mean(da, bbox=[S,N,W,E])      # 若仅支持 bbox
    """
    # 先尝试 region 关键词
    try:
        return select_region_mean(da, region=R)
    except TypeError:
        pass
    # 再尝试位置参数
    try:
        return select_region_mean(da, R)
    except TypeError:
        pass
    # 最后回退 bbox（需要你 utils_region 里有区域字典；若没有就自己写一个简表）
    REGBOX = {
        "SEP": [-90, -60, -140, -70],
        "SEA": [-30,   0,  -90, -30],
        "NEP": [  0,  30, -140, -70],
        "NEA": [  0,  30,  -90, -30],
        "SEI": [-30,   0,   60, 120],
    }
    if R in REGBOX:
        return select_region_mean(da, bbox=REGBOX[R])
    raise TypeError("select_region_mean does not accept region name and no bbox mapping available.")

# ---- 如果项目里的 select_region_mean 导入失败，就用本地实现 ----
def _guess_latlon(ds):
    for k in ["lat","latitude","Latitude","y"]: 
        if k in ds.coords or k in ds.dims: lat=k; break
    else:
        raise KeyError("No latitude coord")
    for k in ["lon","longitude","Longitude","x"]:
        if k in ds.coords or k in ds.dims: lon=k; break
    else:
        raise KeyError("No longitude coord")
    return lat, lon

REGBOX = {
    "SEP": [-90, -60, -140, -70],
    "SEA": [-30,   0,  -90,  -30],
    "NEP": [  0,  30, -140,  -70],
    "NEA": [  0,  30,  -90,  -30],
    "SEI": [-30,   0,   60,  120],
}

def _wrap_lon_if_needed(da, lon_name):
    lon = da[lon_name]
    if lon.min() < -180 or lon.max() > 180:
        # 0–360 → -180–180
        lon2 = ((lon + 180) % 360) - 180
        da = da.assign_coords({lon_name: lon2}).sortby(lon_name)
    return da

def _subset_bbox(da, bbox, lat_name, lon_name):
    s, n, w, e = bbox
    da = _wrap_lon_if_needed(da, lon_name)
    if w <= e:
        da = da.sel({lat_name: slice(s, n), lon_name: slice(w, e)})
    else:
        # 跨经度 180 度（不常用，这里给个兜底）
        left  = da.sel({lon_name: slice(w, 180), lat_name: slice(s, n)})
        right = da.sel({lon_name: slice(-180, e), lat_name: slice(s, n)})
        da = xr.concat([left, right], dim=lon_name)
    return da

def _area_weighted_mean(da, lat_name, lon_name):
    # cos(lat) 权重；广播到 lat,lon
    w = xr.ufuncs.cos(xr.apply_ufunc(np.deg2rad, da[lat_name]))
    w = w / w.mean()  # 归一化（避免绝对量级）
    # 在空间维上聚合，保留 time
    dims = [d for d in da.dims if d in (lat_name, lon_name)]
    return (da * w).mean(dim=dims, skipna=True)

def _select_region_mean_local(da, region=None, bbox=None):
    lat_name, lon_name = _guess_latlon(da)
    if bbox is None:
        if region is None:
            raise ValueError("Need region or bbox")
        bbox = REGBOX[region]
    sub = _subset_bbox(da, bbox, lat_name, lon_name)
    return _area_weighted_mean(sub, lat_name, lon_name)

# 一个统一入口：尽量用项目里的函数；否则用本地实现
def _region_mean_flex(da, R):
    func = globals().get("select_region_mean", None)
    if func is not None:
        try:
            return func(da, region=R)
        except TypeError:
            try:
                return func(da, R)
            except TypeError:
                pass
        return _select_region_mean_local(da, region=R)
    else:
        return _select_region_mean_local(da, region=R)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", required=True, help="clswlow/ts/cllisccp/eislts 等")
    ap.add_argument("--start", type=int, default=2003)
    ap.add_argument("--end",   type=int, default=2014)
    ap.add_argument("--regions", help="逗号分隔（默认读配置或五区）")
    ap.add_argument("--out", required=True, help="输出 CSV 路径")
    args = ap.parse_args()

    key = args.var.lower()
    if key not in VAR_MAP:
        raise SystemExit(f"Unknown var '{args.var}'. Supported: {sorted(VAR_MAP)}")
    subdir, varname = VAR_MAP[key]

    # 读区域函数（用你现有的 utils_region.select_region_mean）
    import sys
    HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent))
    try:
        from scripts.utils_region import select_region_mean
    except ModuleNotFoundError:
        from utils_region import select_region_mean

    regions = [r.strip().upper() for r in args.regions.split(",")] if args.regions \
              else load_regions_from_config()

    nc_dir = CMIP_ROOT / subdir
    files = sorted(nc_dir.glob("*_amip_*.nc"))
    if not files:
        raise SystemExit(f"No files under {nc_dir}")

    rows = []
    for f in files:
        model = parse_model_from_name(f)
        try:
            ds = xr.open_dataset(f)
        except Exception as e:
            print(f"[WARN] open failed: {f} ({e})"); continue
        if varname not in ds:
            print(f"[WARN] skip {f.name}: missing var '{varname}'"); continue

        da = ds[varname]
        # 时间裁剪
        if "time" not in da.coords:
            print(f"[WARN] skip {f.name}: no time coord"); continue
        # 时间筛选（兼容 360_day）
        da = da.where(
            (da["time"].dt.year >= args.start) & (da["time"].dt.year <= args.end),
            drop=True
        )
        
        for R in regions:
            try:
                reg = _region_mean_flex(da, R)   # 按上面的弹性函数
            except Exception as e:
                print(f"[WARN] region {R} failed on {f.name}: {e}")
                continue
            if "time" not in reg.coords:
                print(f"[WARN] region {R} on {f.name}: no time coord after selection")
                continue
            vals   = reg.values
            years  = reg["time"].dt.year.values   # xarray 提供的矢量化取年
            months = reg["time"].dt.month.values  # xarray 提供的矢量化取月
            
            rows_local = []
            for y, m, v in zip(years, months, vals):
                if pd.isna(v):
                    continue
                ts = pd.Timestamp(year=int(y), month=int(m), day=1)  # 月首时间戳
                rows_local.append((model, R, ts, float(v)))
            rows.extend(rows_local)



        ds.close()

    if not rows:
        raise SystemExit("No rows collected.")
    out = pd.DataFrame(rows, columns=["model","region","time","value"])
    out["region"] = out["region"].astype(str).str.upper()
    out = out.sort_values(["model","region","time"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out} (rows={len(out)}, models={out.model.nunique()}, regions={out.region.nunique()})")

if __name__ == "__main__":
    main()

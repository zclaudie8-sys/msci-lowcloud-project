#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import sys, argparse
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
try:
    from scripts.utils_config import load_config, make_output_path, setup_logger
    from scripts.utils_region  import select_region_mean
except ModuleNotFoundError:
    from utils_config import load_config, make_output_path, setup_logger
    from utils_region import select_region_mean

def parse_args():
    p = argparse.ArgumentParser(description="Add MODIS_RE and/or MODIS_TAU to MODIS_<REGION>_monthly.csv")
    p.add_argument("--region", required=True, help="SEP/SEA/NEP/NEA/SEI")
    p.add_argument("--add", default="RE,TAU", help="Which to add: RE,TAU or subset, comma-separated")
    return p.parse_args()

def _ensure_latlon(ds: xr.Dataset, varname: str, logger=None) -> xr.DataArray | None:
    """
    从 Dataset 中提取变量 varname，并尽力为其挂上 1D 的 lat/lon 坐标：
    - 支持坐标名在 coords 里或 data_vars 里（只在 dims 中也行）
    - 支持维度名是 lat/lon 或 y/x（或 latitude/longitude）
    - 优先寻找 1D lat/lon；若 2D 曲线网格，不在此脚本处理（返回 None -> 跳过）
    返回：带 lat/lon 坐标的 DataArray（维度统一重命名为 'lat','lon'），或 None 表示放弃。
    """
    if varname not in ds:
        if logger: logger.warning(f"[ensure_latlon] var '{varname}' not in dataset; vars={list(ds.data_vars)}")
        return None
    da = ds[varname]

    # 1) 先猜测经纬维度名（在 dims 里）
    dims = list(da.dims)
    lat_dim = None
    lon_dim = None
    for cand in ["lat", "latitude", "y"]:
        if cand in dims:
            lat_dim = cand; break
    for cand in ["lon", "longitude", "x"]:
        if cand in dims:
            lon_dim = cand; break

    # 2) 在 ds 里找 1D 的 lat/lon 坐标变量（可能在 coords 或 data_vars）
    def _find_1d_coord(names, fallback_dims):
        # 首先按常用名字找 1D 变量
        for n in names:
            if n in ds.coords and ds.coords[n].ndim == 1:
                return n, ds.coords[n].values
            if n in ds.data_vars and ds.data_vars[n].ndim == 1:
                return n, ds.data_vars[n].values
        # 再按 units/standard_name 找
        for v, obj in ds.variables.items():
            if getattr(obj, "ndim", 0) != 1: 
                continue
            units = str(getattr(obj, "units", "")).lower()
            stdn  = str(getattr(obj, "standard_name", "")).lower()
            if any(k in units for k in ["degrees_north","degree_north","degrees_n"]) or "latitude" in stdn:
                return v, ds[v].values
            if any(k in units for k in ["degrees_east","degree_east","degrees_e"]) or "longitude" in stdn:
                return v, ds[v].values
        # 最后，如果 dims 已知，尝试把该 dim 自身当作坐标（很多文件 dim 名=coord 名）
        for d in fallback_dims:
            if d in ds.coords and ds.coords[d].ndim == 1:
                return d, ds.coords[d].values
            if d in ds.data_vars and ds.data_vars[d].ndim == 1:
                return d, ds.data_vars[d].values
        return None, None

    lat_var, lat_vals = _find_1d_coord(["lat","latitude"], [lat_dim] if lat_dim else [])
    lon_var, lon_vals = _find_1d_coord(["lon","longitude"], [lon_dim] if lon_dim else [])

    # 3) 若维度是 y/x 但找到了 1D lat/lon，则把它们挂到 da 上并重命名维度
    if lat_var is not None and lon_var is not None:
        # 如果维度名还不确定，尽量猜：找和 lat_vals 长度一致的维度作为纬度维
        if lat_dim is None:
            for d in dims:
                if ds.dims.get(d, None) == len(lat_vals):
                    lat_dim = d; break
        if lon_dim is None:
            for d in dims:
                if ds.dims.get(d, None) == len(lon_vals) and d != lat_dim:
                    lon_dim = d; break

        # 仍没法确定经纬维 或 出现 2D 网格：放弃
        if lat_dim is None or lon_dim is None:
            if logger: logger.warning(f"[ensure_latlon] cannot deduce lat/lon dims for var '{varname}': dims={dims}")
            return None
        if da[lat_dim].ndim != 1 or da[lon_dim].ndim != 1:
            if logger: logger.warning(f"[ensure_latlon] 2D/curvilinear grid not supported here for '{varname}'.")
            return None

        # 挂坐标并重命名为标准 'lat','lon'
        da = da.assign_coords({lat_dim: lat_vals, lon_dim: lon_vals})
        rename_map = {}
        if lat_dim != "lat": rename_map[lat_dim] = "lat"
        if lon_dim != "lon": rename_map[lon_dim] = "lon"
        if rename_map:
            da = da.rename(rename_map)
            if logger: logger.info(f"[ensure_latlon] renamed dims {rename_map}")

        return da

    # 4) 如果原本就有 lat/lon 作为 coords（极少数情况）：
    if "lat" in da.coords and "lon" in da.coords:
        return da

    # 5) 走到这里还是不行，就放弃（返回 None）
    if logger:
        logger.warning(f"[ensure_latlon] failed to ensure lat/lon for '{varname}'. dims={dims}, coords={list(da.coords)}")
    return None



def _to_ms(df_or_ser):
    x = df_or_ser.copy()
    x.index = pd.to_datetime(x.index).to_period("M").start_time
    return x.sort_index()

def _load_or_make_modis_monthly(cfg, region, logger):
    regdir = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["regional_monthly"]
    f = regdir / f"MODIS_{region}_monthly.csv"
    if f.exists():
        df = pd.read_csv(f)
        if "time" not in [c.lower() for c in df.columns]:
            df = df.rename(columns={df.columns[0]: "time"})
        df["time"]=pd.to_datetime(df["time"], errors="coerce"); df=df.dropna(subset=["time"]).set_index("time").sort_index()
        # 规范 LCF 列名
        if "MODIS_LCF" not in df.columns:
            for c in df.columns:
                if c.lower() in {"lcf","cllmodis","modis_lcf"}:
                    df = df.rename(columns={c:"MODIS_LCF"}); break
        if "MODIS_LCF" not in df.columns:
            logger.warning(f"{f.name} has no MODIS_LCF; keeping as container for optics columns.")
        return df
    # fallback from cllmodis_region_mean
    cand = sorted(regdir.glob(f"cllmodis_region_mean_*_{region}.csv"))
    if not cand:
        raise FileNotFoundError(f"No MODIS monthly nor fallback cllmodis_region_mean for region {region}")
    df = pd.read_csv(cand[-1])
    if "time" not in [c.lower() for c in df.columns]:
        # likely 2-col without header
        df.columns = ["time", "MODIS_LCF"]
    df["time"]=pd.to_datetime(df["time"], errors="coerce"); df=df.dropna(subset=["time"]).set_index("time").sort_index()
    logger.info(f"[MODIS] built minimal monthly from {cand[-1].name}")
    return df

def main():
    args = parse_args()
    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    logger = setup_logger(cfg, name="augment_modis_optics")
    region = args.region
    want = {s.strip().upper() for s in args.add.split(",") if s.strip()}
    y0, y1 = cfg["project"]["time_span"]
    from pathlib import Path  # 顶部已import则不必重复
    modis_dir = Path(cfg["data"]["modis_path"])

    # region box
    from scripts.utils_config import get_regions
    raw_box = get_regions(cfg, [region])[region]
    # 显式重排为 W,E,S,N
    def _wesn(b): latmin, latmax, lonmin, lonmax = map(float, b); return [lonmin, lonmax, latmin, latmax]
    box = _wesn(raw_box)

    # 1) load or make MODIS monthly shell
    df = _load_or_make_modis_monthly(cfg, region, logger)

    # 2) add RE
    # RE 文件选择（优先精确，其次模糊）
    cands = list(modis_dir.glob("mcd06cosp_reff_cld_200207-202210.nc"))
    if not cands:
        cands = list(modis_dir.glob("mcd06cosp_reff*_*.nc")) + list(modis_dir.glob("mcd06cosp_reff*.nc"))
    if not cands:
        logger.warning(f"[{region}] RE file not found under {modis_dir} -> skip RE")
    else:
        refile = sorted(cands)[-1]
        ds = xr.open_dataset(refile)
        da = _ensure_latlon(ds, "reff", logger)
        if da is None:
            logger.warning(f"[{region}] RE: cannot infer lat/lon; skip RE")
        else:
            ts = select_region_mean(da, box)
            ts = ts.where((ts.time.dt.year>=y0)&(ts.time.dt.year<=y1), drop=True)
            s = _to_ms(ts.to_series().rename("MODIS_RE"))
            df = df.join(s, how="outer")



    # 3) add TAUxf
    # TAU 文件选择
    cands = list(modis_dir.glob("cllmodis_tau1p3_200207-202210.nc"))
    if not cands:
        cands = list(modis_dir.glob("cllmodis_tau*.nc"))
    if not cands:
        logger.warning(f"[{region}] TAU file not found under {modis_dir} -> skip TAU")
    else:
        taufile = sorted(cands)[-1]
        ds = xr.open_dataset(taufile)
        vname = "tau" if "tau" in ds.data_vars else list(ds.data_vars)[0]
        da = _ensure_latlon(ds, vname, logger)
        if da is None:
            logger.warning(f"[{region}] TAU: cannot infer lat/lon; skip TAU")
        else:
            ts = select_region_mean(da, box)
            ts = ts.where((ts.time.dt.year>=y0)&(ts.time.dt.year<=y1), drop=True)
            s = _to_ms(ts.to_series().rename("MODIS_TAU"))
            df = df.join(s, how="outer")


    # 4) clip to project window and save back
    df = df.loc[(df.index.year>=y0)&(df.index.year<=y1)]
    out = Path(cfg["output"]["root"]) / cfg["output"]["subdirs"]["regional_monthly"] / f"MODIS_{region}_monthly.csv"
    df.reset_index().rename(columns={"index":"time"}).to_csv(out, index=False)
    logger.info(f"[{region}] updated {out} with columns: {list(df.columns)}")

if __name__ == "__main__":
    main()

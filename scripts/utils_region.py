from __future__ import annotations
from typing import Sequence, Optional
import numpy as np
import xarray as xr

# ----------------------------
# helpers: coord detection
# ----------------------------
def _guess_lat_name(obj: xr.Dataset | xr.DataArray) -> str:
    for name in ["lat", "latitude", "Lat", "Latitude", "y"]:
        if name in obj.coords or name in obj.dims:
            return name
    for c in obj.coords:
        if obj[c].attrs.get("standard_name", "").lower() == "latitude":
            return c
    raise KeyError("未找到纬度坐标（尝试过 lat/latitude/y 等）。")

def _guess_lon_name(obj: xr.Dataset | xr.DataArray) -> str:
    for name in ["lon", "longitude", "Lon", "Longitude", "x"]:
        if name in obj.coords or name in obj.dims:
            return name
    for c in obj.coords:
        if obj[c].attrs.get("standard_name", "").lower() == "longitude":
            return c
    raise KeyError("未找到经度坐标（尝试过 lon/longitude/x 等）。")

# ----------------------------
# helpers: longitude handling
# ----------------------------
def _to_range_180(lon: xr.DataArray) -> xr.DataArray:
    """[-180, 180)"""
    return (((lon + 180) % 360) - 180)

def _to_range_360(lon: xr.DataArray) -> xr.DataArray:
    """[0, 360)"""
    return lon % 360

def _match_lon_convention(lon: xr.DataArray, target_min: float, target_max: float) -> xr.DataArray:
    """根据目标 box 的范围自动选择 0–360 或 -180–180"""
    if (0 <= target_min < 360) and (0 < target_max <= 360):
        return _to_range_360(lon)
    return _to_range_180(lon)

# ----------------------------
# core: region slicing  (已修复：坐标排序 + 跨 180°)
# ----------------------------
def _slice_region(obj: xr.Dataset | xr.DataArray,
                  box: Sequence[float],
                  lat_name: Optional[str] = None,
                  lon_name: Optional[str] = None) -> xr.Dataset | xr.DataArray:
    """
    box: [lat_min, lat_max, lon_min, lon_max]
    先统一经度约定并 sortby(lon/lat)，确保 slice() 正常；
    支持跨 180° 的经度拼接。
    """
    if lat_name is None:
        lat_name = _guess_lat_name(obj)
    if lon_name is None:
        lon_name = _guess_lon_name(obj)

    lat_min, lat_max, lon_min, lon_max = box
    da_lon = obj[lon_name]
    da_lat = obj[lat_name]

    # 统一经度约定并排序（关键！）
    lon_std = _match_lon_convention(da_lon, lon_min, lon_max)
    obj_ = obj.assign_coords({lon_name: lon_std})
    obj_ = obj_.sortby(lon_name)
    obj_ = obj_.sortby(lat_name)

    # 纬度切片
    obj_lat = obj_.sel({lat_name: slice(min(lat_min, lat_max), max(lat_min, lat_max))})

    # 经度切片
    if lon_min <= lon_max:
        obj_reg = obj_lat.sel({lon_name: slice(lon_min, lon_max)})
    else:
        # 跨 180°
        if float(obj_lat[lon_name].min()) < 0:   # -180..180 体系
            left  = obj_lat.sel({lon_name: slice(lon_min, 180)})
            right = obj_lat.sel({lon_name: slice(-180, lon_max)})
        else:                                    # 0..360 体系
            left  = obj_lat.sel({lon_name: slice(lon_min, 360)})
            right = obj_lat.sel({lon_name: slice(0, lon_max)})
        obj_reg = xr.concat([left, right], dim=lon_name).sortby(lon_name)

    return obj_reg

# ----------------------------
# weights & mean
# ----------------------------
def _lat_weights(lat: xr.DataArray) -> xr.DataArray:
    """cos(lat) 权重（弧度）"""
    return np.cos(np.deg2rad(lat))

def area_weighted_mean(da: xr.DataArray,
                       lat_name: Optional[str] = None,
                       lon_name: Optional[str] = None,
                       keep_attrs: bool = True) -> xr.DataArray:
    """
    对 2D/3D(含 time) DataArray 在 (lat, lon) 上一次性做加权平均：
    - 权重：cos(lat) 展成二维
    - 对 NaN 做显式掩膜，分母按有效格点权重归一（不同月份缺测也稳健）
    """
    if lat_name is None:
        lat_name = _guess_lat_name(da)
    if lon_name is None:
        lon_name = _guess_lon_name(da)

    # 1D -> 2D 权重
    w_lat = _lat_weights(da[lat_name])
    # broadcast 到 da 的形状（除去非 lat/lon 维）
    w2d = xr.ones_like(da.isel({lat_name: 0, lon_name: 0})) * w_lat
    w2d = w2d.broadcast_like(da)

    # 有效掩膜（哪儿有数据就取哪儿的权重）
    valid = xr.where(np.isfinite(da), 1.0, np.nan)

    # 分子/分母（可带其他维度，如 time）
    num = (da * w2d).sum(dim=(lat_name, lon_name), skipna=True)
    den = (w2d * valid).sum(dim=(lat_name, lon_name), skipna=True)

    res = num / den
    if keep_attrs:
        res.attrs.update(da.attrs)
    return res


# ----------------------------
# public: region -> weighted mean timeseries
# ----------------------------
'''
def select_region_mean(data: xr.DataArray | xr.Dataset,
                       region_box: Sequence[float],
                       var: Optional[str] = None,
                       lat_name: Optional[str] = None,
                       lon_name: Optional[str] = None,
                       time_name: str = "time") -> xr.DataArray:
    """
    按 box 切片并做 cos(lat) 加权区域平均。
    - data 为 Dataset 时需提供 var
    - 返回时间序列（若存在 time 维）
    """
    if isinstance(data, xr.Dataset):
        if var is None:
            raise ValueError("data 是 Dataset，请通过 var 指定变量名。")
        da = data[var]
    else:
        da = data

    reg = _slice_region(da, region_box, lat_name=lat_name, lon_name=lon_name)
    ts = area_weighted_mean(reg, lat_name=lat_name, lon_name=lon_name)

    if time_name in ts.dims:
        ts = ts.sortby(time_name)
    return ts
'''


def _wrap_lon(da, lon_min, lon_max):
    # 兼容 0–360 / -180–180
    lon = da['lon']
    if lon.max() > 180:
        f = lambda x: (x % 360 + 360) % 360
        L1, L2 = f(lon_min), f(lon_max)
        if L1 <= L2:
            return da.sel(lon=slice(L1, L2))
        else:
            return xr.concat([
                da.sel(lon=slice(0, L2)),
                da.sel(lon=slice(L1, 360))
            ], dim="lon")
    else:
        return da.sel(lon=slice(lon_min, lon_max))

def select_region_mean(da: xr.DataArray, box) -> xr.DataArray:
    """
    box: [W, E, S, N] （经度西到东，纬度南到北；可为负经度）
    自动适配：
      - 坐标名 lat/lon or latitude/longitude
      - 经度系 -180..180 与 0..360
      - 纬度升序/降序（slice 顺序自动翻转）
    """
    import numpy as np
    # --- 坐标名 ---
    lat_name = "lat" if "lat" in da.coords else "latitude"
    lon_name = "lon" if "lon" in da.coords else "longitude"

    W, E, S, N = map(float, box)

    # --- 经度系统一 ---
    lon = da[lon_name]
    lon_vals = lon.values
    # 判断数据是不是 0..360
    is_0360 = (lon_vals.min() >= 0.0) and (lon_vals.max() > 180.0)
    W2, E2 = W, E
    if is_0360:
        # 把请求的经度也映射到 0..360
        W2 = W % 360.0
        E2 = E % 360.0

    # --- 选择经度 ---
    if is_0360 and W2 > E2:
        # 跨越 0 度经：拼两段
        lon_sel = xr.concat([
            da.sel({lon_name: slice(W2, 360)}),
            da.sel({lon_name: slice(0, E2)})
        ], dim=lon_name)
    else:
        lon_sel = da.sel({lon_name: slice(W2, E2)})

    # --- 选择纬度（根据升/降序决定 slice 方向） ---
    lat = lon_sel[lat_name]
    lat_asc = lat.values[0] < lat.values[-1]
    if lat_asc:
        lat_sel = lon_sel.sel({lat_name: slice(S, N)})
    else:
        lat_sel = lon_sel.sel({lat_name: slice(N, S)})

    # --- 面积权重（cos φ）并做经纬均值 ---
    w = np.cos(np.deg2rad(lat_sel[lat_name]))
    # broadcast 到 (lat,lon)
    w2 = w / w.mean()  # 归一化（不影响平均值，仅稳住数量级）
    out = lat_sel.weighted(w2).mean(dim=(lat_name, lon_name), skipna=True)

    return out



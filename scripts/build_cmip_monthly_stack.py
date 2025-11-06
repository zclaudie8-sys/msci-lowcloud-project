#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把两张 CMIP AMIP 的“月度面板表”按 (model, region, time) 合并为回归输入：
  - y 文件：例如 SWCRE/CLSWLOW 的月度面板
  - x 文件：例如 TS/TAS 或 EIS 的月度面板
输出列：model, region, time, y, x, [family(若存在)]

示例：
  python -m scripts.build_cmip_monthly_stack \
    --y output/tables/cmip_panel_clswlow_2003-2014.csv \
    --x output/tables/cmip_panel_ts_2003-2014.csv \
    --out output/cmip_amip_monthly_2003-2014.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def _read_monthly_csv(path: Path, value_alias: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")
    cols = {c.lower(): c for c in df.columns}

    def pick(names):
        for n in names:
            if n in cols:
                return cols[n]
        raise KeyError(f"{path}: missing one of {names}")

    model_c  = pick(["model"])
    region_c = pick(["region"])

    # time 或 year+month
    time_c = cols.get("time") or cols.get("date")
    if time_c:
        df["time"] = pd.to_datetime(df[time_c], errors="coerce")
    else:
        year_c  = pick(["year"])
        month_c = pick(["month"])
        df["time"] = pd.to_datetime(
            {"year": pd.to_numeric(df[year_c], errors="coerce").astype("Int64"),
             "month": pd.to_numeric(df[month_c], errors="coerce").astype("Int64"),
             "day": 1}
        )
    df = df.dropna(subset=["time"])

    # 取值列
    val_c = None
    for cand in [value_alias, "value", "val", "data", "series", "mean", "y", "x"]:
        if cand in cols:
            val_c = cols[cand]
            break
    if val_c is None:
        blacklist = {model_c, region_c, "time"}
        candidates = [c for c in df.columns if c not in blacklist]
        if not candidates:
            raise KeyError(f"{path}: cannot find value column")
        val_c = candidates[-1]

    out = df[[model_c, region_c, "time", val_c]].copy()
    out.columns = ["model", "region", "time", value_alias]

    # 透传 family（若存在）
    if "family" in cols:
        out["family"] = df[cols["family"]]

    # 限定 2003–2014
    out = out[(out["time"].dt.year >= 2003) & (out["time"].dt.year <= 2014)]
    out["model"]  = out["model"].astype(str)
    out["region"] = out["region"].astype(str).str.upper()
    var_tokens = {"clswlow", "swcre", "ts", "tas", "cllisccp", "lcf", "eislts", "eis"}
    if out["model"].nunique() == 1:
        only = out["model"].iloc[0].strip().lower()
        if only in var_tokens:
            out["model"] = "MMM"
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y",   required=True, help="CSV for Y variable (e.g., SWCRE/CLSWLOW or CLLISCCP monthly panel)")
    ap.add_argument("--x",   required=True, help="CSV for X variable (e.g., TS/TAS or EIS monthly panel)")
    ap.add_argument("--out", required=True, help="Output merged CSV path")
    ap.add_argument("--y-name", default=None, help="Rename output y column to this name (e.g., clswlow)")
    ap.add_argument("--x-name", default=None, help="Rename output x column to this name (e.g., ts)")

    args = ap.parse_args()

    y_df = _read_monthly_csv(Path(args.y), "y")
    x_df = _read_monthly_csv(Path(args.x), "x")
    
    on_cols = ["model", "region", "time"]
    
    def _collapse(df: pd.DataFrame, val: str) -> pd.DataFrame:
        # 如果存在 family，多值时取最常见（mode），否则保留第一条
        has_family = "family" in df.columns
        if has_family:
            fam = (
                df.groupby(on_cols)["family"]
                  .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
                  .reset_index(name="family")
            )
        # 对同键的多行做聚合（默认均值；可按需改 median/first）
        agg = df.groupby(on_cols, as_index=False)[val].mean()
        if has_family:
            agg = agg.merge(fam, on=on_cols, how="left")
        return agg
    
    y_c = _collapse(y_df, "y")
    x_c = _collapse(x_df, "x")
   

    
    # 现在两侧都是一键一行，可以 one_to_one 合并
    merged = pd.merge(y_c, x_c, on=on_cols, how="inner", validate="one_to_one")

    # 若两侧都有 family，则合并为单列
    if "family_x" in merged.columns or "family_y" in merged.columns:
        fam_cols = [c for c in ["family_x", "family_y"] if c in merged.columns]
        if fam_cols:
            merged["family"] = np.nan
            for c in fam_cols:
                merged["family"] = merged["family"].fillna(merged[c])
            merged = merged.drop(columns=fam_cols)

    merged = merged.sort_values(on_cols).reset_index(drop=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.y_name:
        merged = merged.rename(columns={"y": args.y_name})
    if args.x_name:
        merged = merged.rename(columns={"x": args.x_name})
    merged.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}  (rows={len(merged)})")

if __name__ == "__main__":
    main()

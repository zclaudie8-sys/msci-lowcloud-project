#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

BASE = Path("output/tables")
FILES = {
    "SWCRE": BASE/"cmip_panel_clswlow_2003-2014_models.csv",
    "EIS":   BASE/"cmip_panel_eislts_2003-2014_models.csv",
    "Ts":    BASE/"cmip_panel_ts_2003-2014_models.csv",
}

def load(path, name):
    df = pd.read_csv(path, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df["time"] = df["time"].dt.to_period("M").dt.to_timestamp("M")
    df = df[["model","region","time","value"]].rename(columns={"value": name})
    return df

dfs = {k: load(p, k) for k,p in FILES.items()}

# ---- 每个面板自身的时间范围 ----
for k, df in dfs.items():
    span = (
        df.groupby(["model","region"])["time"]
        .agg(n="count", tmin="min", tmax="max")
        .reset_index()
        .sort_values("n")
        .head(10)
    )
    print(f"\n=== {k} per (model,region) worst 10 by count ===")
    print(span)

# ---- 三者求交集后的样本数 ----
w = (
    dfs["SWCRE"]
    .merge(dfs["EIS"], on=["model","region","time"], how="inner")
    .merge(dfs["Ts"],  on=["model","region","time"], how="inner")
)
full = (
    w.groupby(["region","model"])
    .size()
    .reset_index(name="n_full")
    .sort_values("n_full")
    .head(20)
)
print("\n=== Intersection (SWCRE ∩ EIS ∩ Ts) worst 20 ===")
print(full)

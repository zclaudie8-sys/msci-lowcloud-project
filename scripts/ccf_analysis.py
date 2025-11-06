"""Cloud-controlling factor (CCF) regression and feedback analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


DEFAULT_FACTORS = ("EIS", "Ts")

FACTOR_ALIASES: Dict[str, Tuple[str, ...]] = {
    "SWCRE": ("swcre", "sw_cld", "swcre_mean"),
    "EIS": ("eis", "eislts", "lower_tropospheric_stability"),
    "Ts": ("ts", "tas", "surface_temperature"),
    "LCF": ("lcf", "low_cld_frac", "low_cloud_fraction"),
    "omega500": ("omega500", "omega_500", "wap500"),
    "RH700": ("rh700", "q700", "rh_700", "q_700"),
    "sstgrad": ("sstgrad", "sst_grad"),
    "u10": ("u10", "uas", "u_10"),
    "v10": ("v10", "vas", "v_10"),
}


@dataclass
class RegressionConfig:
    method: str
    factors: Sequence[str]
    hac_lags: int = 12
    pcr_var: float = 0.9
    standardize: str = "zscore"
    mbb: int = 1000
    mbb_block: int = 3
    importance_block: int = 3


@dataclass
class DatasetResult:
    dataset: str
    model: str
    region: str
    season: str
    method: str
    factors: Sequence[str]
    beta: Dict[str, float]
    se: Dict[str, float]
    tvalue: Dict[str, float]
    pvalue: Dict[str, float]
    r2: float
    n: int
    standardized: bool


def parse_list_argument(value: str) -> List[str]:
    if value is None:
        return []
    parts = [v.strip() for v in value.split(",")]
    return [p for p in parts if p]


def normalise_factor_name(name: str) -> str:
    name_upper = name.upper()
    if name_upper in FACTOR_ALIASES:
        return name_upper
    lowered = name.lower()
    for canonical, aliases in FACTOR_ALIASES.items():
        if lowered == canonical.lower() or any(alias in lowered for alias in aliases):
            return canonical
    return name_upper


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def setup_logger(log_path: Path) -> logging.Logger:
    ensure_output_dir(log_path)
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    handler = logging.FileHandler(log_path)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    return logger


def find_obs_file(base_dir: Path, region: str, factor: str) -> Optional[Path]:
    if not base_dir.exists():
        return None
    region_lower = region.lower()
    factor_aliases = FACTOR_ALIASES.get(factor, (factor.lower(),))
    for candidate in base_dir.glob(f"*{region}*.csv"):
        lower = candidate.name.lower()
        if region_lower not in lower:
            continue
        if any(alias in lower for alias in factor_aliases):
            return candidate
    return None


def read_obs_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"OBS file {path} is empty")
    cols = {col.lower(): col for col in df.columns}
    time_col = None
    for key in ("time", "month", "date"):
        if key in cols:
            time_col = cols[key]
            break
    if time_col is None:
        raise ValueError(f"Unable to infer time column for {path}")
    value_cols = [c for c in df.columns if c != time_col]
    if not value_cols:
        raise ValueError(f"No value column found in {path}")
    value_col = value_cols[0]
    series = df[[time_col, value_col]].copy()
    series[value_col] = pd.to_numeric(series[value_col], errors="coerce")
    try:
        series["time"] = pd.to_datetime(series[time_col])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unable to parse time column in {path}: {exc}") from exc
    series = series.rename(columns={value_col: "value"})[["time", "value"]]
    return series.sort_values("time").dropna(subset=["value"])


def load_obs_dataset(
    region: str,
    required: Sequence[str],
    optional: Sequence[str],
    base_dir: Path,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    needed = list(dict.fromkeys(list(required) + list(optional)))
    data: Optional[pd.DataFrame] = None
    found: Dict[str, Path] = {}
    for factor in needed:
        file_path = find_obs_file(base_dir, region, factor)
        if file_path is None:
            continue
        series = read_obs_series(file_path)
        series = series.rename(columns={"value": factor})
        if data is None:
            data = series
        else:
            data = pd.merge(data, series, on="time", how="outer")
        found[factor] = file_path
    if data is None:
        raise FileNotFoundError(f"No observational data for region {region}")
    data = data.sort_values("time").reset_index(drop=True)
    missing = [f for f in required if f not in data.columns]
    if missing:
        raise FileNotFoundError(
            f"Missing required observational factors {missing} for region {region}"
        )
    for factor in optional:
        if factor not in data.columns:
            logger.warning(
                "Optional observational factor %s missing for region %s", factor, region
            )
    return data, found


def standardise_predictors(
    X: np.ndarray, mode: str
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if mode.lower() != "zscore":
        return X.copy(), None, None
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0, ddof=1)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def compute_monthly_anomalies(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    if "time" not in df.columns:
        raise ValueError("Input dataframe requires a 'time' column")
    df["month"] = df["time"].dt.month
    for col in columns:
        clim = df.groupby("month")[col].transform("mean")
        df[col] = df[col] - clim
    return df.drop(columns=["month"])


def season_mask(times: pd.Series, season: str) -> np.ndarray:
    if season.upper() == "ALL":
        return np.ones(len(times), dtype=bool)
    months = times.dt.month
    if season.upper() == "JJA":
        return months.isin([6, 7, 8]).to_numpy()
    if season.upper() == "DJF":
        return months.isin([12, 1, 2]).to_numpy()
    raise ValueError(f"Unsupported season {season}")


def require_minimum_samples(n: int, season: str) -> bool:
    if season.upper() == "ALL":
        return n >= 24
    return n >= 18


def hac_ols(y: np.ndarray, X: np.ndarray, hac_lags: int):
    model = sm.OLS(y, sm.add_constant(X))
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def fit_mlr(
    y: np.ndarray,
    X: np.ndarray,
    factor_names: Sequence[str],
    config: RegressionConfig,
) -> Tuple[DatasetResult, Dict[str, float]]:
    X_proc, mean, std = standardise_predictors(X, config.standardize)
    res = hac_ols(y, X_proc, config.hac_lags)
    params = res.params[1:]
    ses = res.bse[1:]
    tvals = res.tvalues[1:]
    pvals = res.pvalues[1:]
    beta = {f: float(params[i]) for i, f in enumerate(factor_names)}
    se = {f: float(ses[i]) for i, f in enumerate(factor_names)}
    tvalue = {f: float(tvals[i]) for i, f in enumerate(factor_names)}
    pvalue = {f: float(pvals[i]) for i, f in enumerate(factor_names)}
    info = {
        "mean": mean,
        "std": std,
        "pred": res.predict(),
        "intercept": float(res.params[0]),
        "r2": float(res.rsquared),
    }
    dataset_result = DatasetResult(
        dataset="",
        model="",
        region="",
        season="",
        method="MLR",
        factors=factor_names,
        beta=beta,
        se=se,
        tvalue=tvalue,
        pvalue=pvalue,
        r2=float(res.rsquared),
        n=int(res.nobs),
        standardized=config.standardize.lower() == "zscore",
    )
    return dataset_result, info


def _fit_pcr_once(
    y: np.ndarray,
    X: np.ndarray,
    factor_names: Sequence[str],
    config: RegressionConfig,
) -> Dict[str, object]:
    X_proc, mean, std = standardise_predictors(X, config.standardize)
    X_center = X_proc - np.nanmean(X_proc, axis=0, keepdims=True)
    pca = PCA()
    scores = pca.fit_transform(X_center)
    if scores.size == 0:
        raise ValueError("Unable to perform PCA on the provided predictors")
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    threshold = float(config.pcr_var)
    k = int(np.searchsorted(cumvar, threshold) + 1)
    k = max(1, min(k, scores.shape[1]))
    scores_k = scores[:, :k]
    res = sm.OLS(y, sm.add_constant(scores_k)).fit()
    beta_pc = res.params[1:]
    intercept_pc = float(res.params[0])
    components = pca.components_[:k, :]
    beta = components.T @ beta_pc
    intercept = intercept_pc - np.nanmean(X_proc, axis=0) @ beta
    y_hat = intercept + X_proc @ beta
    y_mean = y.mean()
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "beta": beta,
        "intercept": intercept,
        "mean": mean,
        "std": std,
        "r2": r2,
        "k": k,
        "cumvar": float(cumvar[k - 1]),
    }


def moving_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if block_size <= 1:
        return rng.integers(0, n, size=n)
    blocks: List[int] = []
    starts = np.arange(0, max(n - block_size + 1, 1))
    while len(blocks) < n:
        start = int(rng.choice(starts))
        block = list(range(start, min(start + block_size, n)))
        blocks.extend(block)
    return np.array(blocks[:n], dtype=int)


def fit_pcr(
    y: np.ndarray,
    X: np.ndarray,
    factor_names: Sequence[str],
    config: RegressionConfig,
    rng: np.random.Generator,
) -> Tuple[DatasetResult, Dict[str, float]]:
    base = _fit_pcr_once(y, X, factor_names, config)
    beta = {f: float(base["beta"][i]) for i, f in enumerate(factor_names)}
    info = {
        "k": base["k"],
        "cumvar": base["cumvar"],
        "r2": base["r2"],
    }
    boot: List[np.ndarray] = []
    for _ in range(config.mbb):
        idx = moving_block_indices(len(y), config.mbb_block, rng)
        try:
            res = _fit_pcr_once(y[idx], X[idx], factor_names, config)
        except Exception:  # noqa: BLE001
            continue
        boot.append(res["beta"])
    if boot:
        boot_arr = np.vstack(boot)
        lower = np.percentile(boot_arr, 2.5, axis=0)
        upper = np.percentile(boot_arr, 97.5, axis=0)
        se = 0.5 * (upper - lower) / 1.96
        med = np.median(boot_arr, axis=0)
        beta = {f: float(med[i]) for i, f in enumerate(factor_names)}
        se_dict = {f: float(se[i]) for i, f in enumerate(factor_names)}
    else:
        se_dict = {f: np.nan for f in factor_names}
    dataset_result = DatasetResult(
        dataset="",
        model="",
        region="",
        season="",
        method="PCR",
        factors=factor_names,
        beta=beta,
        se=se_dict,
        tvalue={f: np.nan for f in factor_names},
        pvalue={f: np.nan for f in factor_names},
        r2=float(base["r2"]),
        n=len(y),
        standardized=config.standardize.lower() == "zscore",
    )
    return dataset_result, info


def block_permutation(arr: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n = len(arr)
    if block_size <= 1 or block_size >= n:
        return arr[rng.permutation(n)]
    blocks = [np.arange(i, min(i + block_size, n)) for i in range(0, n, block_size)]
    rng.shuffle(blocks)
    idx = np.concatenate(blocks)
    return arr[idx]


def permutation_importance(
    y: np.ndarray,
    X: np.ndarray,
    factor_names: Sequence[str],
    config: RegressionConfig,
    base_r2: float,
    fit_func,
    rng: np.random.Generator,
) -> Dict[str, float]:
    delta: Dict[str, float] = {}
    for col, name in enumerate(factor_names):
        X_perm = X.copy()
        X_perm[:, col] = block_permutation(X_perm[:, col], config.importance_block, rng)
        try:
            result = fit_func(y, X_perm, factor_names)
            shuffled_r2 = float(result["r2"])
        except Exception:  # noqa: BLE001
            shuffled_r2 = np.nan
        delta[name] = base_r2 - shuffled_r2 if np.isfinite(shuffled_r2) else np.nan
    return delta


def fit_func_mlr(y, X, factor_names, config):
    res, _ = fit_mlr(y, X, factor_names, config)
    return {"r2": res.r2}


def fit_func_pcr(y, X, factor_names, config, rng):
    base = _fit_pcr_once(y, X, factor_names, config)
    return {"r2": base["r2"]}


def fit_feedback_chain(
    df: pd.DataFrame,
    logger: logging.Logger,
    season: str,
    hac_lags: int,
) -> Optional[Dict[str, float]]:
    needed = ["SWCRE", "LCF", "EIS", "Ts"]
    if any(col not in df.columns for col in needed):
        logger.warning("Feedback decomposition skipped – missing required columns: %s", needed)
        return None
    dfs = compute_monthly_anomalies(df[["time"] + needed], needed)
    mask = season_mask(dfs["time"], season)
    subset = dfs.loc[mask].dropna()
    if not require_minimum_samples(len(subset), season):
        logger.warning("Insufficient samples for feedback decomposition: %s", len(subset))
        return None
    swcre = subset["SWCRE"].to_numpy()
    lcf = subset["LCF"].to_numpy()
    eis = subset["EIS"].to_numpy()
    ts = subset["Ts"].to_numpy()
    try:
        res_sw_lcf = hac_ols(swcre, lcf[:, None], hac_lags)
        res_lcf_eis = hac_ols(lcf, eis[:, None], hac_lags)
        res_eis_ts = hac_ols(eis, ts[:, None], hac_lags)
        res_sw_ts = hac_ols(swcre, ts[:, None], hac_lags)
    except Exception as exc:  # noqa: BLE001
        logger.error("Feedback decomposition failed: %s", exc)
        return None
    beta_sw_lcf = float(res_sw_lcf.params[1])
    beta_lcf_eis = float(res_lcf_eis.params[1])
    beta_eis_ts = float(res_eis_ts.params[1])
    lambda_chain = beta_sw_lcf * beta_lcf_eis * beta_eis_ts
    logger.info(
        "Feedback decomposition slopes: dSWCRE/dLCF=%.3f, dLCF/dEIS=%.3f, dEIS/dTs=%.3f -> lambda_prod=%.3f",
        beta_sw_lcf,
        beta_lcf_eis,
        beta_eis_ts,
        lambda_chain,
    )
    direct = float(res_sw_ts.params[1])
    logger.info("Direct lambda_SW (SWCRE vs Ts) = %.3f", direct)
    return {
        "dSWCRE_dLCF": beta_sw_lcf,
        "dLCF_dEIS": beta_lcf_eis,
        "dEIS_dTs": beta_eis_ts,
        "lambda_prod": lambda_chain,
        "lambda_direct": direct,
    }


def build_cmip_table(cmip_csv: Path) -> pd.DataFrame:
    if not cmip_csv.exists():
        raise FileNotFoundError(f"CMIP CSV not found: {cmip_csv}")
    df = pd.read_csv(cmip_csv)
    if df.empty:
        raise ValueError(f"CMIP CSV {cmip_csv} is empty")
    lower_map = {col.lower(): col for col in df.columns}
    required = ["model", "region", "time", "swcre", "eis", "ts"]
    missing = [key for key in required if key not in lower_map]
    if missing:
        raise KeyError(f"Missing required CMIP columns: {missing}")
    rename: Dict[str, str] = {}
    for canonical, aliases in FACTOR_ALIASES.items():
        for alias in aliases:
            if alias in lower_map:
                rename[lower_map[alias]] = canonical
                break
    rename[lower_map["model"]] = "model"
    rename[lower_map["region"]] = "region"
    rename[lower_map["time"]] = "time"
    df = df.rename(columns=rename)
    df["time"] = pd.to_datetime(df["time"])
    return df


def plot_betas(df: pd.DataFrame, output_path: Path, region: str, method: str, season: str) -> None:
    ensure_output_dir(output_path)
    if df.empty:
        return
    factors = list(dict.fromkeys(df["factor"]))
    obs = df[df["dataset"] == "obs"]
    cmip = df[df["dataset"] == "cmip"]
    fig, ax = plt.subplots(figsize=(max(6, len(factors) * 1.2), 4))
    offsets = np.linspace(-0.2, 0.2, max(len(cmip["model"].unique()), 2)) if not cmip.empty else np.array([0.0])
    for i, factor in enumerate(factors):
        x_base = i
        obs_row = obs[obs["factor"] == factor]
        if not obs_row.empty:
            row = obs_row.iloc[0]
            ax.errorbar(
                x_base - 0.25,
                row["beta"],
                yerr=row["se"],
                fmt="o",
                color="black",
                label="OBS" if i == 0 else "",
            )
        models_for_factor = cmip[cmip["factor"] == factor]
        for j, (_, row) in enumerate(models_for_factor.iterrows()):
            off = offsets[j % len(offsets)]
            ax.errorbar(
                x_base + off,
                row["beta"],
                yerr=row["se"],
                fmt="s",
                color="tab:blue",
                alpha=0.7,
                label=row["model"] if i == 0 else "",
            )
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels(factors)
    ax.set_ylabel("Beta (W m$^{-2}$ per unit factor)")
    ax.set_title(f"CCF betas – {region} – {method} – {season}")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), fontsize="small", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_contributions(df: pd.DataFrame, output_path: Path, region: str, method: str, season: str) -> None:
    ensure_output_dir(output_path)
    if df.empty:
        return
    df_plot = df.sort_values("delta_R2", ascending=False)
    fig, ax = plt.subplots(figsize=(6, max(3, len(df_plot) * 0.3)))
    y_pos = np.arange(len(df_plot))
    colors = ["black" if row.dataset == "obs" else "tab:blue" for row in df_plot.itertuples()]
    ax.barh(y_pos, df_plot["delta_R2"], color=colors)
    ax.set_yticks(y_pos)
    labels = [f"{row.model} – {row.factor}" for row in df_plot.itertuples()]
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("ΔR²")
    ax.set_title(f"Permutation importance – {region} – {method} – {season}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_feedback(df: pd.DataFrame, output_path: Path, region: str, season: str) -> None:
    ensure_output_dir(output_path)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    if "lambda_direct" in df.columns and df["lambda_direct"].notna().any():
        colors = ["black" if d == "obs" else "tab:blue" for d in df["dataset"]]
        ax.scatter(df["lambda_direct"], df["lambda_prod"], c=colors)
        values = df[["lambda_direct", "lambda_prod"]].to_numpy()
        finite = np.isfinite(values)
        if finite.any():
            min_val = float(np.nanmin(values))
            max_val = float(np.nanmax(values))
            ax.plot([min_val, max_val], [min_val, max_val], "--", color="grey")
        ax.set_xlabel("Direct $\\lambda_{SW}$ (W m$^{-2}$ K$^{-1}$)")
        ax.set_ylabel("Decomposed $\\lambda_{prod}$ (W m$^{-2}$ K$^{-1}$)")
    else:
        y_pos = np.arange(len(df))
        colors = ["black" if d == "obs" else "tab:blue" for d in df["dataset"]]
        ax.barh(y_pos, df["lambda_prod"], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row.model}" for row in df.itertuples()])
        ax.set_xlabel("$\\lambda_{prod}$ (W m$^{-2}$ K$^{-1}$)")
        ax.set_ylabel("Model")
    ax.set_title(f"Feedback decomposition – {region} – {season}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cloud controlling factor analysis")
    parser.add_argument("--regions", required=True, help="Comma-separated region list")
    parser.add_argument("--method", choices=["MLR", "PCR"], default="MLR")
    parser.add_argument("--factors", default=",".join(DEFAULT_FACTORS))
    parser.add_argument("--cmip-csv", default="output/cmip_amip_monthly_2003-2014.csv")
    parser.add_argument("--hac-lags", type=int, default=12)
    parser.add_argument("--pcr-var", type=float, default=0.9)
    parser.add_argument("--standardize", choices=["none", "zscore"], default="zscore")
    parser.add_argument("--season", choices=["ALL", "JJA", "DJF"], default="ALL")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mbb", type=int, default=1000, help="Bootstrap samples for PCR")
    parser.add_argument("--mbb-block", type=int, default=3, help="Bootstrap block size")
    parser.add_argument(
        "--importance-block", type=int, default=3, help="Block size for permutation importance"
    )

    args = parser.parse_args(argv)
    regions = parse_list_argument(args.regions)
    if not regions:
        parser.error("At least one region must be specified via --regions")
    factors = [normalise_factor_name(f) for f in parse_list_argument(args.factors)]
    factors = [f for f in factors if f != "SWCRE"]
    if not factors:
        parser.error("No predictor factors specified")

    config = RegressionConfig(
        method=args.method,
        factors=factors,
        hac_lags=args.hac_lags,
        pcr_var=args.pcr_var,
        standardize=args.standardize,
        mbb=args.mbb,
        mbb_block=args.mbb_block,
        importance_block=args.importance_block,
    )

    obs_dir = Path("output/regional_monthly")
    cmip_csv = Path(args.cmip_csv)
    rng = np.random.default_rng(1234)

    try:
        cmip_df = build_cmip_table(cmip_csv)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load CMIP data: {exc}", file=sys.stderr)
        return 2

    for region in regions:
        beta_tables: List[pd.DataFrame] = []
        contrib_tables: List[pd.DataFrame] = []
        feedback_entries: List[pd.Series] = []

        log_path = Path("logs") / f"ccf_{region}_{config.method}_{args.season}.log"
        logger = setup_logger(log_path)
        logger.info("Starting analysis for region %s", region)

        required = ["SWCRE"] + list(config.factors)
        optional = ["LCF", "EIS", "Ts", "omega500", "RH700", "sstgrad", "u10", "v10"]
        try:
            obs_df, _ = load_obs_dataset(region, required, optional, obs_dir, logger)
        except Exception as exc:  # noqa: BLE001
            logger.error("Skipping region %s due to observational data error: %s", region, exc)
            continue

        obs_anom = compute_monthly_anomalies(obs_df, [c for c in obs_df.columns if c != "time"])
        obs_mask = season_mask(obs_anom["time"], args.season)
        obs_subset = obs_anom.loc[obs_mask].dropna(subset=required)
        n_obs = len(obs_subset)
        logger.info("OBS sample size after filtering: %s", n_obs)

        if args.dry_run:
            print(f"[DRY-RUN] Region={region} dataset=OBS factors={config.factors} n={n_obs}")
            print(obs_subset.head())
        elif require_minimum_samples(n_obs, args.season):
            X_obs = obs_subset[config.factors].to_numpy()
            y_obs = obs_subset["SWCRE"].to_numpy()
            if config.method == "MLR":
                obs_result, _ = fit_mlr(y_obs, X_obs, config.factors, config)
                base_r2 = obs_result.r2
                delta = permutation_importance(
                    y_obs,
                    X_obs,
                    config.factors,
                    config,
                    base_r2,
                    lambda y, X, names: fit_func_mlr(y, X, names, config),
                    rng,
                )
            else:
                obs_result, info = fit_pcr(y_obs, X_obs, config.factors, config, rng)
                logger.info(
                    "OBS PCR retained PCs=%d (cumVar=%.2f)", info["k"], info["cumvar"]
                )
                base_r2 = obs_result.r2
                delta = permutation_importance(
                    y_obs,
                    X_obs,
                    config.factors,
                    config,
                    base_r2,
                    lambda y, X, names: fit_func_pcr(y, X, names, config, rng),
                    rng,
                )
            obs_result.dataset = "obs"
            obs_result.model = "OBS"
            obs_result.region = region
            obs_result.season = args.season
            beta_tables.append(
                pd.DataFrame(
                    [
                        {
                            "dataset": obs_result.dataset,
                            "model": obs_result.model,
                            "region": region,
                            "season": args.season,
                            "method": config.method,
                            "factor": factor,
                            "beta": obs_result.beta.get(factor, np.nan),
                            "se": obs_result.se.get(factor, np.nan),
                            "t": obs_result.tvalue.get(factor, np.nan),
                            "p": obs_result.pvalue.get(factor, np.nan),
                            "R2": obs_result.r2,
                            "n": obs_result.n,
                            "standardized": obs_result.standardized,
                        }
                        for factor in config.factors
                    ]
                )
            )
            contrib_tables.append(
                pd.DataFrame(
                    [
                        {
                            "dataset": "obs",
                            "model": "OBS",
                            "region": region,
                            "season": args.season,
                            "method": config.method,
                            "factor": factor,
                            "delta_R2": delta.get(factor, np.nan),
                            "base_R2": base_r2,
                            "n": n_obs,
                        }
                        for factor in config.factors
                    ]
                )
            )
            feedback = fit_feedback_chain(obs_df, logger, args.season, config.hac_lags)
            if feedback:
                feedback_entries.append(
                    pd.Series(
                        {
                            "dataset": "obs",
                            "model": "OBS",
                            "region": region,
                            "season": args.season,
                            **feedback,
                        }
                    )
                )
        else:
            logger.warning("Insufficient OBS samples for regression; skipping region %s", region)

        cmip_region = cmip_df[cmip_df["region"].str.upper() == region.upper()]
        if cmip_region.empty:
            logger.warning("No CMIP data found for region %s", region)
        for model in sorted(cmip_region["model"].unique()):
            model_df = cmip_region[cmip_region["model"] == model].copy()
            available_factors = [f for f in config.factors if f in model_df.columns]
            missing_factors = [f for f in config.factors if f not in available_factors]
            if not available_factors:
                logger.warning("Model %s lacks requested predictors; skipped", model)
                continue
            if missing_factors:
                logger.warning(
                    "Model %s missing predictors %s; proceeding with available factors",
                    model,
                    missing_factors,
                )
            feedback_cols = [c for c in ["LCF", "EIS", "Ts"] if c in model_df.columns]
            model_raw = model_df[["time", "SWCRE"] + available_factors + feedback_cols]
            model_anom = compute_monthly_anomalies(
                model_df[["time", "SWCRE"] + available_factors],
                ["SWCRE"] + available_factors,
            )
            mask = season_mask(model_anom["time"], args.season)
            model_subset = model_anom.loc[mask].dropna(subset=["SWCRE"] + available_factors)
            n_mod = len(model_subset)
            if args.dry_run:
                print(
                    f"[DRY-RUN] Region={region} dataset=CMIP model={model} factors={available_factors} n={n_mod}"
                )
                print(model_subset.head())
                continue
            if not require_minimum_samples(n_mod, args.season):
                logger.warning(
                    "Model %s insufficient samples after filtering (n=%s); skipped",
                    model,
                    n_mod,
                )
                continue
            X_mod = model_subset[available_factors].to_numpy()
            y_mod = model_subset["SWCRE"].to_numpy()
            if config.method == "MLR":
                mod_result, _ = fit_mlr(y_mod, X_mod, available_factors, config)
                base_r2 = mod_result.r2
                delta_mod = permutation_importance(
                    y_mod,
                    X_mod,
                    available_factors,
                    config,
                    base_r2,
                    lambda y, X, names: fit_func_mlr(y, X, names, config),
                    rng,
                )
            else:
                mod_result, info = fit_pcr(y_mod, X_mod, available_factors, config, rng)
                logger.info(
                    "Model %s PCR retained PCs=%d (cumVar=%.2f)",
                    model,
                    info["k"],
                    info["cumvar"],
                )
                base_r2 = mod_result.r2
                delta_mod = permutation_importance(
                    y_mod,
                    X_mod,
                    available_factors,
                    config,
                    base_r2,
                    lambda y, X, names: fit_func_pcr(y, X, names, config, rng),
                    rng,
                )
            mod_result.dataset = "cmip"
            mod_result.model = model
            mod_result.region = region
            mod_result.season = args.season
            beta_tables.append(
                pd.DataFrame(
                    [
                        {
                            "dataset": "cmip",
                            "model": model,
                            "region": region,
                            "season": args.season,
                            "method": config.method,
                            "factor": factor,
                            "beta": mod_result.beta.get(factor, np.nan),
                            "se": mod_result.se.get(factor, np.nan),
                            "t": mod_result.tvalue.get(factor, np.nan),
                            "p": mod_result.pvalue.get(factor, np.nan),
                            "R2": mod_result.r2,
                            "n": mod_result.n,
                            "standardized": mod_result.standardized,
                        }
                        for factor in available_factors
                    ]
                )
            )
            contrib_tables.append(
                pd.DataFrame(
                    [
                        {
                            "dataset": "cmip",
                            "model": model,
                            "region": region,
                            "season": args.season,
                            "method": config.method,
                            "factor": factor,
                            "delta_R2": delta_mod.get(factor, np.nan),
                            "base_R2": base_r2,
                            "n": n_mod,
                        }
                        for factor in available_factors
                    ]
                )
            )
            if all(c in model_raw.columns for c in ["LCF", "EIS", "Ts"]):
                feedback = fit_feedback_chain(model_raw, logger, args.season, config.hac_lags)
                if feedback:
                    feedback_entries.append(
                        pd.Series(
                            {
                                "dataset": "cmip",
                                "model": model,
                                "region": region,
                                "season": args.season,
                                **feedback,
                            }
                        )
                    )

        if args.dry_run:
            continue

        beta_df = pd.concat(beta_tables, ignore_index=True) if beta_tables else pd.DataFrame()
        contrib_df = (
            pd.concat(contrib_tables, ignore_index=True) if contrib_tables else pd.DataFrame()
        )
        feedback_df = (
            pd.DataFrame(feedback_entries) if feedback_entries else pd.DataFrame()
        )

        if not beta_df.empty:
            beta_path = Path("tables") / f"ccf_betas_{region}_{config.method}_{args.season}.csv"
            ensure_output_dir(beta_path)
            beta_df.to_csv(beta_path, index=False)
            plot_betas(
                beta_df,
                Path("figures")
                / f"ccf_betas_obs_vs_models_{region}_{config.method}_{args.season}.png",
                region,
                config.method,
                args.season,
            )
        if not contrib_df.empty:
            contrib_path = (
                Path("tables") / f"ccf_contrib_{region}_{config.method}_{args.season}.csv"
            )
            ensure_output_dir(contrib_path)
            contrib_df.to_csv(contrib_path, index=False)
            plot_contributions(
                contrib_df,
                Path("figures")
                / f"ccf_contrib_{region}_{config.method}_{args.season}.png",
                region,
                config.method,
                args.season,
            )
        if not feedback_df.empty:
            feedback_path = (
                Path("tables") / f"ccf_feedback_decomp_{region}_{args.season}.csv"
            )
            ensure_output_dir(feedback_path)
            feedback_df.to_csv(feedback_path, index=False)
            plot_feedback(
                feedback_df,
                Path("figures")
                / f"ccf_feedback_decomposition_{region}_{args.season}.png",
                region,
                args.season,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

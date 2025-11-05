# scripts/utils_config.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

# ---- 依赖：PyYAML ----
try:
    import yaml
except ModuleNotFoundError:
    raise SystemExit(
        "PyYAML 未安装。请在 Terminal 里执行：\n"
        "  pipInstall -y PyYAML\n"
        "或：pip install --user pyyaml\n"
        "然后重启内核再试。"
    )

# ---------- 私有：确保输出树 ----------
def _ensure_output_tree(cfg: dict) -> None:
    """
    统一输出目录结构：
      output/
        regional_monthly/  harmonics/  sensitivity_beta/  gamma/  dLCF_dTs/
        lambda/  tables/  figures/  logs/
    允许老配置的 results_dir/figures_dir/logs_dir 共存（向后兼容）。
    """
    root = Path(cfg.get("output", {}).get("root", "output"))
    # 默认子目录
    defaults = {
        "regional_monthly": "regional_monthly",
        "harmonics": "harmonics",
        "sensitivity_beta": "sensitivity_beta",
        "gamma": "gamma",
        "dLCF_dTs": "dLCF_dTs",
        "lambda": "lambda",
        "tables": "tables",
        "figures": "figures",
        "logs": "logs",
    }
    sub = dict(defaults)
    # 若用户在 config 里自定义 subdirs，则覆盖
    user_sub = cfg.get("output", {}).get("subdirs", {})
    if isinstance(user_sub, dict):
        sub.update(user_sub)

    # 兼容老字段：results_dir/figures_dir/logs_dir
    legacy_results = cfg.get("output", {}).get("results_dir")
    legacy_figs    = cfg.get("output", {}).get("figures_dir")
    legacy_logs    = cfg.get("output", {}).get("logs_dir")
    if legacy_results: sub["regional_monthly"] = Path(legacy_results).name
    if legacy_figs:    sub["figures"]          = Path(legacy_figs).name
    if legacy_logs:    sub["logs"]             = Path(legacy_logs).name

    cfg.setdefault("output", {})
    cfg["output"]["root"] = str(root)
    cfg["output"]["subdirs"] = sub

    # 创建目录
    for v in sub.values():
        Path(root, v).mkdir(parents=True, exist_ok=True)

# ---------- 基础：加载配置 ----------
def load_config(cfg_path: str = "configs/config.yaml") -> dict:
    cfg_p = Path(cfg_path)
    if not cfg_p.exists():
        raise FileNotFoundError(f"配置文件不存在：{cfg_p.resolve()}")
    with cfg_p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("配置解析失败：不是有效的 YAML 字典。")
    _ensure_output_tree(cfg)
    return cfg

# ---------- 数据与输出路径 ----------
def get_paths(cfg: dict) -> dict:
    root = Path(cfg["output"]["root"])
    sub  = cfg["output"]["subdirs"]
    return {
        "root": root,
        "subdirs": sub,
        "save_format": cfg["output"].get("save_format", "png"),
    }

def get_outdir(cfg: dict, kind: str) -> Path:
    """
    kind ∈ {'regional_monthly','harmonics','sensitivity_beta','gamma',
            'dLCF_dTs','lambda','tables','figures','logs'}
    """
    root = Path(cfg["output"]["root"])
    sub  = cfg["output"]["subdirs"]
    if kind not in sub:
        raise KeyError(f"Unknown output kind: {kind}")
    p = root / sub[kind]
    p.mkdir(parents=True, exist_ok=True)
    return p

def make_output_path(cfg: dict, kind: str, name: str) -> Path:
    """返回 output/<kind>/<name> 的完整路径。"""
    return get_outdir(cfg, kind) / name

# ---------- 区域字典（可选筛选） ----------
def get_regions(cfg: dict, keys=None) -> dict:
    regions = cfg["regions"].copy()
    if keys is None:
        return regions
    return {k: regions[k] for k in keys if k in regions}

# ---------- 生成 tag（时间段 / 区域 / 自定义）（供你自己拼文件名时用） ----------
def make_tag(cfg: dict, region: str | None = None, extra: str | None = None) -> str:
    y0, y1 = cfg["project"]["time_span"]
    parts = [f"{y0}-{y1}"]
    if region:
        parts.append(region)
    if extra:
        parts.append(extra)
    return "_".join(parts)

# ---------- 日志封装（文件 + 控制台） ----------
def setup_logger(cfg: dict, name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    log_dir = get_outdir(cfg, "logs")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt); fh.setLevel(level)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt); ch.setLevel(level)

    logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"Logger started. Writing to: {log_file}")
    return logger

# ---------- 小结工具（可选） ----------
def summarize_config(cfg: dict) -> str:
    y0, y1 = cfg["project"]["time_span"]
    return "\n".join([
        f"Project: {cfg['project']['name']}",
        f"Time span: {y0}-{y1}",
        f"Regions: {', '.join(cfg['project']['regions'])}",
        f"Data paths: MODIS={cfg['data']['modis_path']}, CERES={cfg['data']['ceres_path']}, ERA5={cfg['data']['era5_path']}",
        f"Save format: {cfg['output'].get('save_format','png')}",
        f"Output root: {cfg['output']['root']}",
        f"Subdirs: {cfg['output']['subdirs']}",
    ])

"""Repository summarizer CLI.

Scans a local Git repository to build architecture and logic documentation.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class FileRecord:
    path: Path
    rel_path: str
    size: int
    language: str
    sample: str
    truncated: bool = False


@dataclass
class FunctionRecord:
    name: str
    args: List[str]
    lineno: int
    docstring: Optional[str]


@dataclass
class ClassRecord:
    name: str
    lineno: int
    docstring: Optional[str]


@dataclass
class PythonModuleInfo:
    module: str
    path: Path
    rel_path: str
    functions: List[FunctionRecord] = field(default_factory=list)
    classes: List[ClassRecord] = field(default_factory=list)
    imports: Set[str] = field(default_factory=set)
    cli_entries: List[Dict[str, str]] = field(default_factory=list)
    docstring: Optional[str] = None


class RepoSummarizer:
    IGNORE_DIRS = {".git", "__pycache__", "data", "figs", "outputs", "output", "build", "dist"}
    TEXT_EXTENSIONS = {".py", ".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".ini"}
    NOTEBOOK_EXTENSIONS = {".ipynb"}
    CONFIG_EXTENSIONS = {".yaml", ".yml", ".toml", ".ini"}
    DOC_EXTENSIONS = {".md", ".rst"}

    CANONICAL_VARIABLES = {
        "SST": ["sst", "sea_surface_temperature", "sstadv"],
        "EIS": ["eis", "estimated_inversion_strength"],
        "RH700": ["rh700", "relative_humidity", "rh_700"],
        "OMEGA700": ["omega700", "omega_700", "w700", "omega"],
        "WS": ["ws", "wind_speed", "u10", "v10", "wind"],
        "SSTADV": ["sstadv", "adv_sst", "sst_advection"],
        "SWCRE": ["swcre", "shortwave_cre"],
        "LWCRE": ["lwcre", "longwave_cre"],
        "LCF": ["lcf", "low_cloud_fraction"],
        "NETCRE": ["netcre", "net_cre"],
    }

    ALIGNMENT_ITEMS = {
        "six_factors": {
            "label": "六因子特征 (SST, EIS, RH700, Ω700, WS, SSTADV)",
            "keywords": ["SST", "EIS", "RH700", "OMEGA", "WS", "SSTADV"],
        },
        "five_by_five": {
            "label": "非局地 5×5 邻域特征",
            "keywords": ["5x5", "5\u00d75", "(2k+1)", "neighborhood"],
        },
        "leave_one_year": {
            "label": "留一年 (GroupKFold) 交叉验证",
            "keywords": ["GroupKFold", "leave-one-year", "leave_one_year"],
        },
        "deseason": {
            "label": "去季节 / deseasonalize 处理",
            "keywords": ["deseason", "seasonal", "climatology"],
        },
        "lambda_assembly": {
            "label": "反馈 λ 组装",
            "keywords": ["lambda", "feedback", "dR2"],
        },
    }

    FIGURE_NAMES = ["fig1", "fig2", "fig3", "fig4"]

    def __init__(self, repo: Path, out_dir: Path, max_files: int, max_lines: int,
                 detect_alias: bool, with_mermaid: bool, dry_run: bool) -> None:
        self.repo = repo
        self.out_dir = out_dir
        self.max_files = max_files
        self.max_lines = max_lines
        self.detect_alias = detect_alias
        self.with_mermaid = with_mermaid
        self.dry_run = dry_run
        self.files: List[FileRecord] = []
        self.python_modules: Dict[str, PythonModuleInfo] = {}
        self.cli_entries: List[Dict[str, str]] = []
        self.aliases: Dict[str, Set[str]] = defaultdict(set)
        self.todo_items: Dict[str, List[str]] = defaultdict(list)
        self.module_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.notebook_summaries: Dict[str, List[Tuple[str, str]]] = {}
        self.config_entries: Dict[str, Dict[str, List[str]]] = {}
        self.documentation_snippets: Dict[str, str] = {}
        self.alignment_hits: Dict[str, Dict[str, str]] = {}
        self.figure_scripts: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self.pipeline_hits: Dict[str, Dict[str, str]] = {}
        self.repo_packages: Set[str] = set()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        self._discover_repo_packages()
        self._gather_files()
        self._process_python_files()
        self._process_notebooks()
        self._process_configs()
        self._process_docs()
        self._detect_alignment()
        self._detect_pipeline_features()
        self._write_outputs()
        self._print_console_summary()

    # ------------------------------------------------------------------
    def _discover_repo_packages(self) -> None:
        for path in self.repo.rglob("__init__.py"):
            rel = path.relative_to(self.repo)
            parts = rel.parts
            if parts:
                self.repo_packages.add(parts[0])

    # ------------------------------------------------------------------
    def _should_skip_dir(self, name: str) -> bool:
        return name in self.IGNORE_DIRS or name.startswith(".") and name not in {".git"}

    def _gather_files(self) -> None:
        count = 0
        for root, dirs, files in os.walk(self.repo):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if not self._should_skip_dir(d)]
            for filename in files:
                if count >= self.max_files:
                    return
                path = root_path / filename
                if path.is_symlink():
                    continue
                if path.stat().st_size > 2 * 1024 * 1024:
                    continue
                rel_path = str(path.relative_to(self.repo))
                ext = path.suffix.lower()
                language = self._detect_language(ext)
                sample, truncated = self._read_sample(path)
                record = FileRecord(path=path, rel_path=rel_path, size=path.stat().st_size,
                                    language=language, sample=sample, truncated=truncated)
                self.files.append(record)
                self._extract_todos(record)
                count += 1

    def _detect_language(self, ext: str) -> str:
        if ext == ".py":
            return "python"
        if ext in self.NOTEBOOK_EXTENSIONS:
            return "notebook"
        if ext in self.CONFIG_EXTENSIONS:
            return "config"
        if ext in self.DOC_EXTENSIONS:
            return "markdown"
        return "text"

    def _read_sample(self, path: Path) -> Tuple[str, bool]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ("", False)
        lines = text.splitlines()
        truncated = False
        if len(lines) > self.max_lines:
            head = "\n".join(lines[:200])
            tail = "\n".join(lines[-200:])
            sample = head + "\n...\n" + tail
            truncated = True
        else:
            sample = "\n".join(lines[:200])
        return (sample, truncated)

    def _extract_todos(self, record: FileRecord) -> None:
        text = record.sample
        items: List[str] = []
        for line in text.splitlines():
            if "TODO" in line or "FIXME" in line:
                snippet = line.strip()
                items.append(snippet)
        if items:
            self.todo_items[record.rel_path].extend(items)

    # ------------------------------------------------------------------
    def _process_python_files(self) -> None:
        for record in self.files:
            if record.language != "python":
                continue
            module_name = self._module_name_from_path(record.rel_path)
            try:
                source = record.path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            try:
                tree = ast.parse(source, filename=record.rel_path)
            except SyntaxError:
                continue

            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    setattr(child, "parent", parent)

            functions: List[FunctionRecord] = []
            classes: List[ClassRecord] = []
            imports: Set[str] = set()
            cli_entries: List[Dict[str, str]] = []
            docstring = ast.get_docstring(tree)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if isinstance(getattr(node, "parent", None), ast.ClassDef):
                        continue
                    args = [arg.arg for arg in node.args.args]
                    functions.append(FunctionRecord(name=node.name, args=args,
                                                     lineno=node.lineno,
                                                     docstring=ast.get_docstring(node)))
                elif isinstance(node, ast.ClassDef):
                    classes.append(ClassRecord(name=node.name, lineno=node.lineno,
                                               docstring=ast.get_docstring(node)))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imports.add(self._resolve_from_import(module_name, module, node.level))

            cli_entries.extend(self._detect_cli_entries(tree, source, record.rel_path))

            info = PythonModuleInfo(module=module_name, path=record.path, rel_path=record.rel_path,
                                    functions=functions, classes=classes, imports=imports,
                                    cli_entries=cli_entries, docstring=docstring)
            self.python_modules[record.rel_path] = info

            for imported in imports:
                internal = self._normalize_internal_module(imported)
                if internal:
                    self.module_dependencies[module_name].add(internal)

            for cli in cli_entries:
                self.cli_entries.append(cli)

            if self.detect_alias:
                self._detect_aliases(source)

            self._collect_figure_scripts(record.rel_path, source)

    def _module_name_from_path(self, rel_path: str) -> str:
        module = rel_path.replace(os.sep, ".")
        if module.endswith(".__init__.py"):
            module = module[:-12]
        elif module.endswith(".py"):
            module = module[:-3]
        return module

    def _resolve_from_import(self, current_module: str, module: str, level: int) -> str:
        if level == 0:
            return module or current_module
        parts = current_module.split(".")
        if "__init__" in parts:
            parts = parts[:-1]
        if level > len(parts):
            base: List[str] = []
        else:
            base = parts[:-level]
        if module:
            base.append(module)
        return ".".join(part for part in base if part)

    def _normalize_internal_module(self, module: str) -> Optional[str]:
        if not module:
            return None
        module = module.split(" as ")[0]
        module_root = module.split(".")[0]
        if module_root in self.repo_packages or (self.repo / f"{module_root}.py").exists():
            return module
        return None

    def _detect_cli_entries(self, tree: ast.AST, source: str, rel_path: str) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []
        argparse_hits: List[str] = []
        click_hits: List[str] = []

        class CliVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
                func = node.func
                name = None
                if isinstance(func, ast.Attribute):
                    name = f"{self._attr_to_name(func)}"
                elif isinstance(func, ast.Name):
                    name = func.id
                if name and "ArgumentParser" in name:
                    argparse_hits.append(name)
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
                decorators = [self._attr_to_name(dec) for dec in node.decorator_list]
                if any("click.command" in (dec or "") for dec in decorators):
                    click_hits.append(node.name)
                self.generic_visit(node)

            def _attr_to_name(self, node: ast.AST) -> Optional[str]:
                if isinstance(node, ast.Attribute):
                    parent = self._attr_to_name(node.value)
                    if parent:
                        return f"{parent}.{node.attr}"
                    return node.attr
                if isinstance(node, ast.Name):
                    return node.id
                return None

        CliVisitor().visit(tree)

        if argparse_hits:
            help_text = self._extract_argparse_help(source)
            entries.append({
                "type": "argparse",
                "module": rel_path,
                "parser": argparse_hits[0],
                "help": help_text.strip(),
            })

        for func_name in click_hits:
            entries.append({
                "type": "click",
                "module": rel_path,
                "command": func_name,
            })
        return entries

    def _extract_argparse_help(self, source: str) -> str:
        lines = []
        for line in source.splitlines():
            if "ArgumentParser" in line and "description" in line:
                lines.append(line.strip())
            if "add_argument" in line and "help" in line:
                lines.append(line.strip())
        return "\n".join(lines)

    def _detect_aliases(self, source: str) -> None:
        lowered = source.lower()
        for canonical, patterns in self.CANONICAL_VARIABLES.items():
            for pattern in patterns:
                if pattern in lowered:
                    self.aliases[canonical].add(pattern)

    def _collect_figure_scripts(self, rel_path: str, source: str) -> None:
        rel_lower = rel_path.lower()
        for fig in self.FIGURE_NAMES:
            if fig in rel_lower:
                entry = {
                    "script": rel_path,
                }
                if "if __name__ == \"__main__\"" in source:
                    entry["entry"] = "__main__"
                if "ArgumentParser" in source:
                    entry["cli"] = "argparse"
                self.figure_scripts[fig].append(entry)

    # ------------------------------------------------------------------
    def _process_notebooks(self) -> None:
        for record in self.files:
            if record.language != "notebook":
                continue
            try:
                data = json.loads(record.path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            cells = data.get("cells", [])
            summary: List[Tuple[str, str]] = []
            for cell in cells:
                if cell.get("cell_type") != "markdown":
                    continue
                source = "".join(cell.get("source", []))
                for line in source.splitlines():
                    if line.startswith("#"):
                        title = line.lstrip("# ")
                        summary.append((title, source.strip()))
                        break
                if summary:
                    break
            self.notebook_summaries[record.rel_path] = summary

    # ------------------------------------------------------------------
    def _process_configs(self) -> None:
        for record in self.files:
            ext = Path(record.rel_path).suffix.lower()
            if ext not in self.CONFIG_EXTENSIONS:
                continue
            text = record.path.read_text(encoding="utf-8", errors="ignore")
            entries: Dict[str, List[str]] = defaultdict(list)
            if ext in {".yaml", ".yml"} and yaml is not None:
                try:
                    parsed = yaml.safe_load(text)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    for key in ["regions", "factors", "paths", "variables", "features"]:
                        value = parsed.get(key)
                        if isinstance(value, dict):
                            entries[key].extend([f"{k}: {v}" for k, v in value.items()])
                        elif isinstance(value, list):
                            entries[key].extend([str(v) for v in value])
                        elif value is not None:
                            entries[key].append(str(value))
            elif ext == ".toml":
                try:
                    import tomllib  # type: ignore
                except Exception:
                    tomllib = None
                if tomllib is not None:
                    try:
                        parsed = tomllib.loads(text)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, dict):
                        for key in parsed:
                            value = parsed[key]
                            if isinstance(value, dict):
                                entries[key].extend([f"{k}: {v}" for k, v in value.items()])
            else:
                for line in text.splitlines():
                    if "=" in line:
                        entries["values"].append(line.strip())
            if entries:
                self.config_entries[record.rel_path] = dict(entries)

    # ------------------------------------------------------------------
    def _process_docs(self) -> None:
        for record in self.files:
            ext = Path(record.rel_path).suffix.lower()
            if ext not in self.DOC_EXTENSIONS:
                continue
            snippet = "\n".join(record.sample.splitlines()[:100])
            self.documentation_snippets[record.rel_path] = snippet

    # ------------------------------------------------------------------
    def _detect_alignment(self) -> None:
        for key, info in self.ALIGNMENT_ITEMS.items():
            hit = self._find_keyword_hits(info["keywords"])
            self.alignment_hits[key] = hit

    def _find_keyword_hits(self, keywords: Sequence[str]) -> Dict[str, str]:
        hits: Dict[str, str] = {}
        for record in self.files:
            text = record.sample.lower()
            for kw in keywords:
                if kw.lower() in text:
                    hits[kw] = record.rel_path
        return hits

    # ------------------------------------------------------------------
    def _detect_pipeline_features(self) -> None:
        items = {
            "data_loading": ["load", "dataset", "read_csv", "open_dataset"],
            "seasonal_split": ["season", "deseason", "climatology"],
            "feature_engineering": ["feature", "sst", "eis", "omega"],
            "nonlocal_features": ["neighborhood", "5x5", "rolling"],
            "ridge_regression": ["Ridge", "GroupKFold", "alpha"],
            "delta_r2": ["delta", "dR2", "permutation"],
            "lambda_feedback": ["lambda", "feedback", "partial"],
            "figures": ["fig", "plot", "savefig"],
        }
        for name, keywords in items.items():
            hit = self._find_keyword_hits(keywords)
            self.pipeline_hits[name] = hit

    # ------------------------------------------------------------------
    def _write_outputs(self) -> None:
        if self.dry_run:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        index_dir = self.out_dir / "index"
        index_dir.mkdir(exist_ok=True)

        self._write_json(index_dir / "files.json", [self._record_to_dict(r) for r in self.files])
        self._write_json(index_dir / "python_api.json", self._python_api_payload())
        if self.detect_alias:
            alias_path = self.out_dir / "GLOSSARY_aliases.json"
            alias_dict = {k: sorted(v) for k, v in self.aliases.items()}
            self._write_json(alias_path, alias_dict)

        self._write_markdown(self.out_dir / "ARCHITECTURE.md", self._render_architecture())
        self._write_markdown(self.out_dir / "PIPELINE.md", self._render_pipeline())
        self._write_markdown(self.out_dir / "FIGURES.md", self._render_figures())
        self._write_markdown(self.out_dir / "GLOSSARY.md", self._render_glossary())
        self._write_markdown(self.out_dir / "ALIGNMENT_CEPPI2024.md", self._render_alignment())
        self._write_markdown(self.out_dir / "TODO.md", self._render_todo())

    def _record_to_dict(self, record: FileRecord) -> Dict[str, object]:
        return {
            "path": record.rel_path,
            "size": record.size,
            "language": record.language,
            "sample": record.sample,
            "truncated": record.truncated,
        }

    def _python_api_payload(self) -> Dict[str, object]:
        payload = {}
        for rel_path, info in self.python_modules.items():
            payload[rel_path] = {
                "module": info.module,
                "functions": [
                    {"name": fn.name, "args": fn.args, "lineno": fn.lineno, "docstring": fn.docstring}
                    for fn in info.functions
                ],
                "classes": [
                    {"name": cls.name, "lineno": cls.lineno, "docstring": cls.docstring}
                    for cls in info.classes
                ],
                "imports": sorted(info.imports),
                "cli_entries": info.cli_entries,
            }
        return payload

    def _write_json(self, path: Path, payload: object) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def _write_markdown(self, path: Path, content: str) -> None:
        with path.open("w", encoding="utf-8") as fh:
            fh.write(content)

    # ------------------------------------------------------------------
    def _metadata_table(self, title: str) -> str:
        rows = [
            ("Generated", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            ("Repository", str(self.repo.resolve())),
            ("Files scanned", str(len(self.files))),
            ("Python", sys.version.split()[0]),
            ("Title", title),
        ]
        headers = "| Key | Value |\n| --- | --- |\n"
        body = "\n".join(f"| {k} | {v} |" for k, v in rows)
        return headers + body + "\n\n"

    def _render_architecture(self) -> str:
        parts = ["# ARCHITECTURE", self._metadata_table("ARCHITECTURE")]
        parts.append("## 目录树 (≤3 层)")
        parts.append(self._render_directory_tree())
        parts.append("\n## 模块与依赖")
        parts.append(self._render_module_table())
        if self.with_mermaid:
            parts.append("\n```mermaid\n" + self._render_dependency_mermaid() + "\n```")
        parts.append("\n## CLI 入口")
        if self.cli_entries:
            parts.append("| Type | Module | Entry | Details |\n| --- | --- | --- | --- |")
            for cli in self.cli_entries:
                entry = cli.get("parser") or cli.get("command") or ""
                help_text = cli.get("help", "")
                parts.append(f"| {cli['type']} | {cli['module']} | {entry} | {help_text} |")
        else:
            parts.append("未检测到 CLI 入口。")
        return "\n".join(parts) + "\n"

    def _render_directory_tree(self) -> str:
        root = self.repo
        prefix = ""
        lines: List[str] = []

        def walk(current: Path, depth: int) -> None:
            if depth > 2:
                return
            entries = sorted([p for p in current.iterdir() if p.is_dir() and not self._should_skip_dir(p.name)])
            files = sorted([p for p in current.iterdir() if p.is_file()])
            for directory in entries:
                rel = directory.relative_to(root)
                indent = "  " * depth
                lines.append(f"{indent}- {rel}/")
                walk(directory, depth + 1)
            for file in files:
                if file.suffix in {".py", ".md", ".yaml", ".yml", ".toml"}:
                    rel = file.relative_to(root)
                    indent = "  " * depth
                    lines.append(f"{indent}- {rel}")

        walk(root, 0)
        return "\n".join(lines)

    def _render_module_table(self) -> str:
        if not self.python_modules:
            return "(无 Python 模块信息)"
        lines = ["| Module | Functions | Classes |", "| --- | --- | --- |"]
        for info in sorted(self.python_modules.values(), key=lambda x: x.module):
            fn_count = len(info.functions)
            cls_count = len(info.classes)
            lines.append(f"| {info.module} | {fn_count} | {cls_count} |")
        return "\n".join(lines)

    def _render_dependency_mermaid(self) -> str:
        lines = ["graph TD"]
        for src, targets in sorted(self.module_dependencies.items()):
            sanitized_src = src.replace("/", "_").replace(".", "__")
            for tgt in targets:
                sanitized_tgt = tgt.replace("/", "_").replace(".", "__")
                lines.append(f"  {sanitized_src} --> {sanitized_tgt}")
        if len(lines) == 1:
            lines.append("  noop")
        return "\n".join(lines)

    def _render_pipeline(self) -> str:
        parts = ["# PIPELINE", self._metadata_table("PIPELINE")]
        parts.append("## 数据→预处理→特征→建模→评估→反馈→出图")
        if self.with_mermaid:
            parts.append("```mermaid")
            parts.append("flowchart TD")
            parts.append("  A[数据加载] --> B[季节/去季节]")
            parts.append("  B --> C[特征工程]")
            parts.append("  C --> D[Ridge + 留一年]")
            parts.append("  D --> E[ΔR² 评估]")
            parts.append("  E --> F[反馈 λ 组装]")
            parts.append("  F --> G[图形生成]")
            parts.append("```")
        parts.append("\n## 关键节点定位")
        for name, hits in self.pipeline_hits.items():
            display = ", ".join(f"{kw}→{path}" for kw, path in hits.items()) or "未检测到"
            parts.append(f"- **{name}**: {display}")
        return "\n".join(parts) + "\n"

    def _render_figures(self) -> str:
        parts = ["# FIGURES", self._metadata_table("FIGURES")]
        for fig in self.FIGURE_NAMES:
            entries = self.figure_scripts.get(fig, [])
            parts.append(f"## {fig.upper()}")
            if entries:
                parts.append("| Script | Entry | CLI |\n| --- | --- | --- |")
                for entry in entries:
                    parts.append(f"| {entry.get('script', '')} | {entry.get('entry', '函数')} | {entry.get('cli', '-') } |")
            else:
                parts.append("未检测到，建议参考脚手架：scripts/{}_*.py".format(fig))
        return "\n".join(parts) + "\n"

    def _render_glossary(self) -> str:
        parts = ["# GLOSSARY", self._metadata_table("GLOSSARY")]
        parts.append("## 区域 & 配置条目")
        if self.config_entries:
            for path, entries in self.config_entries.items():
                parts.append(f"### {path}")
                for key, values in entries.items():
                    parts.append(f"- {key}: {', '.join(values)}")
        else:
            parts.append("未检测到区域/变量配置。")
        if self.detect_alias:
            parts.append("\n## 变量别名")
            for canonical, aliases in sorted(self.aliases.items()):
                display = ", ".join(sorted(set(aliases))) or "(无)"
                parts.append(f"- **{canonical}**: {display}")
        return "\n".join(parts) + "\n"

    def _render_alignment(self) -> str:
        parts = ["# ALIGNMENT_CEPPI2024", self._metadata_table("ALIGNMENT")]
        parts.append("## 对齐检查表")
        parts.append("| 要点 | 关键词定位 | 状态 |\n| --- | --- | --- |")
        for key, info in self.ALIGNMENT_ITEMS.items():
            hits = self.alignment_hits.get(key, {})
            status = "✅" if hits else "⚠️"
            display = ", ".join(f"{kw}→{path}" for kw, path in hits.items()) or "未找到"
            parts.append(f"| {info['label']} | {display} | {status} |")
        parts.append("\n## 差异或缺失建议")
        missing = [info["label"] for key, info in self.ALIGNMENT_ITEMS.items() if not self.alignment_hits.get(key)]
        if missing:
            for item in missing:
                parts.append(f"- ⚠️ {item} 在仓库中未明确定位，请补充代码标注。")
        else:
            parts.append("- ✅ 所有关键要点均检测到相关实现片段。")
        return "\n".join(parts) + "\n"

    def _render_todo(self) -> str:
        parts = ["# TODO", self._metadata_table("TODO")]
        if not self.todo_items:
            parts.append("无 TODO/FIXME 标记。")
            return "\n".join(parts) + "\n"
        for path, items in sorted(self.todo_items.items()):
            parts.append(f"## {path}")
            for item in items:
                parts.append(f"- {item}")
        return "\n".join(parts) + "\n"

    # ------------------------------------------------------------------
    def _print_console_summary(self) -> None:
        print("Top module dependencies:")
        counter = Counter({src: len(tgts) for src, tgts in self.module_dependencies.items()})
        for module, degree in counter.most_common(10):
            print(f"  {module}: {degree} edges")
        print("\nCLI entries:")
        for cli in self.cli_entries:
            entry = cli.get("parser") or cli.get("command")
            print(f"  {cli['type']} -> {cli['module']} ({entry})")
        print("\nSeason/deseason functions:")
        hits = self.pipeline_hits.get("seasonal_split", {})
        for kw, path in hits.items():
            print(f"  {kw}: {path}")
        print("\n非局地/邻域函数:")
        hits = self.pipeline_hits.get("nonlocal_features", {})
        for kw, path in hits.items():
            print(f"  {kw}: {path}")
        print("\nCeppi(2024) 对齐:")
        for key, info in self.ALIGNMENT_ITEMS.items():
            hits = self.alignment_hits.get(key, {})
            status = "✅" if hits else "⚠️"
            print(f"  {status} {info['label']} -> {', '.join(f'{k}:{v}' for k, v in hits.items()) or '未找到'}")


# ----------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a repository architecture")
    parser.add_argument("--repo", type=str, default=".", help="Repository path")
    parser.add_argument("--out", type=str, default="docs/_autogen", help="Output directory")
    parser.add_argument("--max-files", type=int, default=2000, help="Maximum files to scan")
    parser.add_argument("--max-lines", type=int, default=4000, help="Maximum lines per file")
    parser.add_argument("--detect-alias", action="store_true", help="Detect variable aliases")
    parser.add_argument("--with-mermaid", action="store_true", help="Include Mermaid diagrams")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    repo = Path(args.repo).resolve()
    out_dir = Path(args.out)
    summarizer = RepoSummarizer(
        repo=repo,
        out_dir=out_dir,
        max_files=args.max_files,
        max_lines=args.max_lines,
        detect_alias=args.detect_alias,
        with_mermaid=args.with_mermaid,
        dry_run=args.dry_run,
    )
    summarizer.run()


if __name__ == "__main__":
    main()

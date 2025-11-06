#!/usr/bin/env python3
"""Audit output paths used across analysis scripts."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

TARGET_ATTR_CALLS = {
    "to_csv": 0,
    "to_netcdf": 0,
    "savefig": 0,
    "to_parquet": 0,
    "save": 0,
    "mkdir": 0,
    "makedirs": 0,
    "dump": 1,
}

TARGET_NAME_CALLS = {
    "open": 0,
    "dump": 1,
    "makedirs": 0,
}

KEYWORD_FALLBACKS = [
    "path",
    "path_or_buf",
    "fname",
    "fp",
    "file",
    "filepath_or_buffer",
    "filename",
]

HEURISTIC_PATTERNS = [
    re.compile(r"output/", re.IGNORECASE),
    re.compile(r"fig(?:ures)?/", re.IGNORECASE),
    re.compile(r"tables?/", re.IGNORECASE),
    re.compile(r"logs?/", re.IGNORECASE),
]

ARTIFACT_EXTENSIONS = {
    "table": {".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".feather", ".nc", ".netcdf"},
    "figure": {".png", ".pdf", ".jpg", ".jpeg", ".svg", ".eps"},
    "log": {".log", ".txt"},
}

ROOT_CONSTANT_CANDIDATES = {"OUTPUT", "FIG", "TABLE", "LOG"}


@dataclass
class ManifestRecord:
    script: str
    function: str
    artifact_type: str
    path_pattern: str
    root_constant: Optional[str]
    notes: str


@dataclass
class ScriptSummary:
    path: Path
    docstring: Optional[str] = None
    constants: Dict[str, str] = field(default_factory=dict)
    records: List[ManifestRecord] = field(default_factory=list)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit output paths across scripts")
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "scripts",
        help="Directory containing analysis scripts",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    logs_dir = root / "logs"
    tables_dir = root / "tables"
    docs_dir = root / "docs"
    for directory in (logs_dir, tables_dir, docs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"audit_outputs_{timestamp}.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("Starting output path audit for %s", args.scripts_dir)

    summaries: List[ScriptSummary] = []

    for script_path in sorted(args.scripts_dir.glob("*.py")):
        logging.info("Scanning %s", script_path)
        try:
            with script_path.open("r", encoding="utf-8") as handle:
                source = handle.read()
        except OSError as exc:
            logging.exception("Failed to read %s: %s", script_path, exc)
            continue

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            logging.exception("Failed to parse %s: %s", script_path, exc)
            continue

        docstring = ast.get_docstring(tree)
        summary = ScriptSummary(path=script_path, docstring=docstring)
        constants = collect_constants(tree, summary)
        summary.constants.update(constants)

        auditor = OutputVisitor(summary, source, constants)
        auditor.visit(tree)
        auditor.flush_heuristics()

        summaries.append(summary)

    manifest_records = [record for summary in summaries for record in summary.records]
    write_csv(manifest_records, tables_dir / "output_manifest.csv")
    write_json(manifest_records, root / "output_manifest.json")
    write_docs(summaries, docs_dir / "OUTPUT_MAP.md")

    logging.info("Completed audit. %d records captured.", len(manifest_records))
    print(f"Audit complete. See {log_path.relative_to(root)} for details.")


def collect_constants(tree: ast.AST, summary: ScriptSummary) -> Dict[str, str]:
    constants: Dict[str, str] = {}

    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            if not targets:
                continue
            name = targets[0].id
            if not name.isupper():
                continue
            value, root_constant, notes = extract_path_pattern(
                node.value, constants
            )
            if value is None:
                continue
            constants[name] = value
            note_parts = ["constant definition"]
            if notes:
                note_parts.append(notes)
            summary.records.append(
                ManifestRecord(
                    script=str(summary.path.relative_to(summary.path.parents[1])),
                    function="<module>",
                    artifact_type=classify_artifact(value, None, name),
                    path_pattern=value,
                    root_constant=name,
                    notes="; ".join(note_parts),
                )
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if not name.isupper() or node.value is None:
                continue
            value, root_constant, notes = extract_path_pattern(node.value, constants)
            if value is None:
                continue
            constants[name] = value
            note_parts = ["constant definition"]
            if notes:
                note_parts.append(notes)
            summary.records.append(
                ManifestRecord(
                    script=str(summary.path.relative_to(summary.path.parents[1])),
                    function="<module>",
                    artifact_type=classify_artifact(value, None, name),
                    path_pattern=value,
                    root_constant=name,
                    notes="; ".join(note_parts),
                )
            )
    return constants


class OutputVisitor(ast.NodeVisitor):
    def __init__(self, summary: ScriptSummary, source: str, constants: Dict[str, str]):
        self.summary = summary
        self.source = source
        self.constants = constants
        self.context: List[str] = ["<module>"]
        self.heuristic_strings: Set[Tuple[str, str]] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.context.append(node.name)
        self.generic_visit(node)
        self.context.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.context.append(node.name)
        self.generic_visit(node)
        self.context.pop()

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                self._maybe_capture_call(item.context_expr)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._maybe_capture_call(node)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            text = node.value
            for pattern in HEURISTIC_PATTERNS:
                if pattern.search(text):
                    key = (text, self.current_function)
                    self.heuristic_strings.add(key)
                    break
        self.generic_visit(node)

    @property
    def current_function(self) -> str:
        return self.context[-1]

    def flush_heuristics(self) -> None:
        script_rel = str(self.summary.path.relative_to(self.summary.path.parents[1]))
        for text, function in sorted(self.heuristic_strings):
            artifact_type = classify_artifact(text, None, None)
            notes = "heuristic string match"
            self.summary.records.append(
                ManifestRecord(
                    script=script_rel,
                    function=function,
                    artifact_type=artifact_type,
                    path_pattern=text,
                    root_constant=None,
                    notes=notes,
                )
            )

    def _maybe_capture_call(self, node: ast.Call) -> None:
        func_name, is_attr = get_call_name(node.func)
        if func_name is None:
            return

        arg_index = None
        if func_name == "open" and is_attr:
            # skip Path.open and similar helpers; these typically operate on handles
            return
        if func_name in TARGET_ATTR_CALLS:
            arg_index = TARGET_ATTR_CALLS[func_name]
        elif func_name in TARGET_NAME_CALLS and not is_attr:
            arg_index = TARGET_NAME_CALLS[func_name]
        else:
            return

        target_arg = None
        if func_name in {"mkdir", "makedirs"} and not node.args:
            target_arg = node.func.value if isinstance(node.func, ast.Attribute) else None
        else:
            if len(node.args) > arg_index:
                target_arg = node.args[arg_index]
            else:
                for keyword in node.keywords:
                    if keyword.arg in KEYWORD_FALLBACKS and keyword.value is not None:
                        target_arg = keyword.value
                        break
        if target_arg is None:
            return

        path_pattern, root_constant, note = extract_path_pattern(target_arg, self.constants)
        if path_pattern is None:
            return
        if path_pattern.lower() in {"w", "r", "a", "wb", "rb", "ab"}:
            return

        artifact_type = classify_artifact(path_pattern, func_name, root_constant)
        notes = []
        if note:
            notes.append(note)
        if func_name in {"mkdir", "makedirs"}:
            notes.append("directory creation")
        record = ManifestRecord(
            script=str(self.summary.path.relative_to(self.summary.path.parents[1])),
            function=self.current_function,
            artifact_type=artifact_type,
            path_pattern=path_pattern,
            root_constant=root_constant,
            notes="; ".join(notes) if notes else "",
        )
        self.summary.records.append(record)


def get_call_name(func: ast.AST) -> Tuple[Optional[str], bool]:
    if isinstance(func, ast.Attribute):
        return func.attr, True
    if isinstance(func, ast.Name):
        return func.id, False
    return None, False


def extract_path_pattern(
    node: ast.AST, known_constants: Dict[str, str]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    notes: List[str] = []
    root_constants: Set[str] = set()

    def handle(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Str):  # pragma: no cover - legacy
            return node.s
        if isinstance(node, ast.JoinedStr):
            parts: List[str] = []
            for value in node.values:
                if isinstance(value, ast.Str):
                    parts.append(value.s)
                elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                elif isinstance(value, ast.FormattedValue):
                    placeholder, rc = placeholder_for_expression(value.value, known_constants)
                    if rc:
                        root_constants.add(rc)
                    parts.append(placeholder)
                    notes.append("from f-string")
                else:
                    parts.append("<expr>")
            return "".join(parts)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                left = handle(node.left)
                right = handle(node.right)
                if left is None or right is None:
                    return None
                notes.append("string concatenation")
                return left + right
            if isinstance(node.op, ast.Div):
                left = handle(node.left)
                right = handle(node.right)
                if left is None or right is None:
                    return None
                notes.append("pathlib division")
                return combine_path_parts([left, right])
        if isinstance(node, ast.Call):
            func_name, is_attr = get_call_name(node.func)
            if func_name in {"join", "joinpath"} and is_attr:
                base = handle(node.func.value)
                args = [handle(arg) for arg in node.args]
                if base is None or any(arg is None for arg in args):
                    return None
                notes.append("path join")
                parts = [base] + [arg for arg in args if arg is not None]
                return combine_path_parts(parts)
            if func_name == "Path":
                parts = [handle(arg) for arg in node.args]
                if any(part is None for part in parts if part is not None):
                    return None
                notes.append("Path constructor")
                return combine_path_parts([part for part in parts if part is not None])
            if func_name == "os" and isinstance(node.func, ast.Attribute) and node.func.attr == "path":
                # os.path(...) pattern is unlikely; skip
                return None
            if func_name == "join" and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute):
                value = node.func.value
                if isinstance(value.value, ast.Name) and value.attr == "path" and value.value.id == "os":
                    parts = [handle(arg) for arg in node.args]
                    if any(part is None for part in parts):
                        return None
                    notes.append("os.path.join")
                    return combine_path_parts([part for part in parts if part is not None])
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            # reference to constant attribute
            placeholder = f"<{node.value.id.upper()}_{node.attr.upper()}>"
            notes.append("attribute placeholder")
            return placeholder
        if isinstance(node, ast.Name):
            name = node.id
            if name in known_constants:
                root_constants.add(name)
                return known_constants[name]
            if name.isupper():
                root_constants.add(name)
                notes.append("uses constant placeholder")
                return f"<{name}>"
            notes.append("variable placeholder")
            return f"<{name.upper()}>"
        if isinstance(node, ast.Subscript):
            placeholder, rc = placeholder_for_expression(node, known_constants)
            if rc:
                root_constants.add(rc)
            return placeholder
        return None

    path = handle(node)
    if path is None:
        return None, None, None

    path = path.replace("\\", "/")
    # compact duplicate slashes
    path = re.sub(r"/{2,}", "/", path)

    root_constant = ",".join(sorted(root_constants)) if root_constants else None
    note_text = "; ".join(dict.fromkeys(notes)) if notes else None
    return path, root_constant, note_text


def placeholder_for_expression(
    expr: ast.AST, known_constants: Dict[str, str]
) -> Tuple[str, Optional[str]]:
    if isinstance(expr, ast.Name):
        name = expr.id
        if name in known_constants:
            return known_constants[name], name
        if name.isupper():
            return f"<{name}>", name
        return f"<{name.upper()}>", None
    if isinstance(expr, ast.Attribute):
        base, rc = placeholder_for_expression(expr.value, known_constants)
        name = expr.attr.upper()
        return f"<{base.strip('<>').strip('>').upper()}_{name}>", rc
    if isinstance(expr, ast.Call):
        func_name, _ = get_call_name(expr.func)
        if func_name == "Path":
            parts = []
            root_constant = None
            for arg in expr.args:
                placeholder, rc = placeholder_for_expression(arg, known_constants)
                if root_constant is None and rc is not None:
                    root_constant = rc
                parts.append(placeholder)
            path = combine_path_parts(parts)
            return path, root_constant
        return "<expr>", None
    if isinstance(expr, ast.Subscript):
        base, rc = placeholder_for_expression(expr.value, known_constants)
        return f"<{base.strip('<>').strip('>').upper()}>", rc
    return "<expr>", None


def combine_path_parts(parts: Sequence[str]) -> str:
    cleaned = [part.strip("/") if part not in ("/", "") else part for part in parts if part is not None]
    if not cleaned:
        return ""
    result_parts: List[str] = []
    for part in cleaned:
        if part in {"", "/"}:
            continue
        if part.startswith("<") and part.endswith(">"):
            result_parts.append(part)
        else:
            result_parts.append(part)
    return "/".join(result_parts)


def classify_artifact(path: str, func_name: Optional[str], root_constant: Optional[str]) -> str:
    lower_path = path.lower()
    extension = Path(lower_path).suffix

    for artifact, extensions in ARTIFACT_EXTENSIONS.items():
        if extension in extensions:
            return artifact

    if func_name in {"savefig"}:
        return "figure"
    if func_name in {"to_csv", "to_parquet", "to_netcdf"}:
        return "table"
    if "log" in lower_path:
        return "log"
    if any(token in (root_constant or "") for token in ROOT_CONSTANT_CANDIDATES):
        for candidate, artifact in (("FIG", "figure"), ("TABLE", "table"), ("LOG", "log")):
            if candidate in (root_constant or ""):
                return artifact
    return "other"


def write_csv(records: Sequence[ManifestRecord], csv_path: Path) -> None:
    fieldnames = [
        "script",
        "function",
        "artifact_type",
        "path_pattern",
        "root_constant",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def write_json(records: Sequence[ManifestRecord], json_path: Path) -> None:
    data = [record.__dict__ for record in records]
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_docs(summaries: Sequence[ScriptSummary], md_path: Path) -> None:
    lines: List[str] = ["# Output Map", ""]
    for summary in sorted(summaries, key=lambda s: s.path.name):
        rel_path = summary.path.relative_to(summary.path.parents[1])
        lines.append(f"## {rel_path}")
        brief = summary.docstring.splitlines()[0] if summary.docstring else "No module docstring detected."
        lines.append(f"Brief: {brief}")
        if summary.constants:
            const_items = ", ".join(
                f"{name} = {value}" for name, value in sorted(summary.constants.items())
            )
            lines.append(f"Constants: {const_items}")
        lines.append("")
        for record in sorted(summary.records, key=lambda r: (r.artifact_type, r.path_pattern)):
            lines.append(
                f"- {record.artifact_type} | {record.path_pattern} | {record.function} | {record.notes or '—'}"
            )
        lines.append("")
    lines.append("## Consolidated conventions")
    lines.append(
        "- Prefer storing tabular data in `tables/` with descriptive filenames."
    )
    lines.append("- Figures should be written to `fig/` or `figures/` folders with informative names.")
    lines.append("- Use `output/` for derived datasets and diagnostics.")
    lines.append("- Write execution logs to `logs/` with timestamps or identifiers.")
    lines.append("")

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    main()

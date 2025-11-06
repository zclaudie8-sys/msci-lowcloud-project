"""Utilities for working with ``output_manifest.json`` artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

Manifest = Dict[str, object]


def load_manifest(path: str | Path | None) -> Optional[Manifest]:
    """Load and return the manifest dictionary, or ``None`` if unavailable."""
    if not path:
        return None

    manifest_path = Path(path)
    if not manifest_path.exists():
        return None

    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    base_dir = str(manifest_path.parent.resolve())

    if isinstance(data, dict):
        data.setdefault("__base_dir__", base_dir)
        return data
    if isinstance(data, list):
        return {"artifacts": data, "__base_dir__": base_dir}
    return None


def _iter_artifacts(manifest: Manifest) -> Iterator[Dict[str, object]]:
    if not isinstance(manifest, dict):
        return

    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, list):
        for item in artifacts:
            if isinstance(item, dict):
                yield item
        return

    outputs = manifest.get("outputs")
    if isinstance(outputs, list):
        for item in outputs:
            if isinstance(item, dict):
                yield item
        return

    # Some manifests may directly map artifact names to metadata dictionaries.
    for value in manifest.values():
        if isinstance(value, dict) and {"path", "type"}.issubset(value.keys()):
            yield value


def _apply_placeholders(value: str, placeholders: Dict[str, str]) -> str:
    result = value
    for key, replacement in placeholders.items():
        result = result.replace(f"<{key}>", replacement)
    return result


def resolve_from_manifest(
    manifest: Optional[Manifest],
    want: Dict[str, object],
) -> List[str]:
    """Return manifest paths that match ``want`` filters."""
    if not manifest:
        return []

    script_contains = want.get("script_contains")
    if script_contains is None:
        script_filters = []
    elif isinstance(script_contains, str):
        script_filters = [script_contains]
    elif isinstance(script_contains, Iterable):
        script_filters = list(script_contains)
    else:
        script_filters = [str(script_contains)]

    path_filters = want.get("path_like_contains") or []
    if isinstance(path_filters, str):
        path_filters = [path_filters]
    elif not isinstance(path_filters, Iterable):
        path_filters = [path_filters]

    placeholders = want.get("placeholders") or {}
    if not isinstance(placeholders, dict):
        placeholders = {}

    path_filters = [_apply_placeholders(str(item), placeholders) for item in path_filters]

    script_filters = [_apply_placeholders(str(item), placeholders) for item in script_filters]

    artifact_type = want.get("artifact_type")
    results: List[str] = []
    seen: set[str] = set()

    base_dir = Path(manifest.get("__base_dir__", "."))

    for artifact in _iter_artifacts(manifest):
        path_value = artifact.get("path") or artifact.get("filepath") or artifact.get("file")
        if not isinstance(path_value, str):
            continue

        path_lower = path_value.lower()
        if path_filters and not all(sub.lower() in path_lower for sub in path_filters):
            continue

        if artifact_type:
            type_value = artifact.get("artifact_type") or artifact.get("type")
            if isinstance(type_value, str):
                if str(artifact_type).lower() != type_value.lower():
                    continue
            else:
                continue

        if script_filters:
            script_value = artifact.get("script") or artifact.get("source") or artifact.get("producer")
            if isinstance(script_value, str):
                script_lower = script_value.lower()
                if not all(sub.lower() in script_lower for sub in script_filters):
                    continue
            else:
                continue

        candidate_path = Path(path_value)
        if not candidate_path.is_absolute():
            candidate_path = base_dir / candidate_path

        normalized = str(candidate_path)
        if normalized in seen:
            continue
        seen.add(normalized)
        results.append(normalized)

    return results

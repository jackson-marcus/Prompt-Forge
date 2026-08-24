"""Versioned prompt registry backed by a JSON file."""

from __future__ import annotations

import json
from pathlib import Path

from promptforge.settings import get_config, resolve_path


def _registry_path() -> Path:
    return resolve_path(get_config()["data"]["registry_path"])


def load_registry() -> dict:
    path = _registry_path()
    if not path.exists():
        return {"prompts": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(registry: dict) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def register_prompt(task: str, name: str, template: str, set_baseline: bool = False) -> dict:
    registry = load_registry()
    entry = registry["prompts"].setdefault(task, {"variants": {}, "baseline": None})
    version = len(entry["variants"].get(name, {}).get("versions", [])) + 1
    variant = entry["variants"].setdefault(name, {"versions": []})
    variant["versions"].append({"version": version, "template": template})
    if set_baseline or entry["baseline"] is None:
        entry["baseline"] = name
    save_registry(registry)
    return {"task": task, "name": name, "version": version, "baseline": entry["baseline"]}


def get_template(task: str, name: str) -> str:
    registry = load_registry()
    return registry["prompts"][task]["variants"][name]["versions"][-1]["template"]


def list_variants(task: str) -> dict:
    registry = load_registry()
    entry = registry["prompts"].get(task, {"variants": {}, "baseline": None})
    return {
        "baseline": entry["baseline"],
        "variants": {name: v["versions"][-1]["template"] for name, v in entry["variants"].items()},
    }

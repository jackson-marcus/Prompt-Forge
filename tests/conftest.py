"""Offline fixtures: task suite + registry evaluated into tmp artifacts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_tasks import TASKS, build_cases  # noqa: E402

from promptforge.settings import get_config, get_settings  # noqa: E402


@pytest.fixture(scope="session")
def evaluated(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("promptforge")
    (tmp / "processed").mkdir()

    cfg = get_config()
    originals = (
        cfg["data"]["processed_dir"],
        cfg["data"]["artifacts_dir"],
        cfg["data"]["registry_path"],
        cfg["eval"]["bootstrap_iters"],
    )
    cfg["data"]["processed_dir"] = str(tmp / "processed")
    cfg["data"]["artifacts_dir"] = str(tmp / "artifacts")
    cfg["data"]["registry_path"] = str(tmp / "processed" / "registry.json")
    cfg["eval"]["bootstrap_iters"] = 500

    (tmp / "processed" / "registry.json").write_text(json.dumps({"prompts": {}}), encoding="utf-8")

    from promptforge.registry.store import register_prompt

    rng = np.random.default_rng(7)
    frames = []
    for task, spec in TASKS.items():
        frames.append(build_cases(task, spec, 50, rng))
        for i, (name, template) in enumerate(spec["variants"].items()):
            register_prompt(task, name, template, set_baseline=(i == 0))
    pd.concat(frames, ignore_index=True).to_parquet(
        tmp / "processed" / "cases.parquet", index=False
    )

    old_uri = os.environ.get("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tmp / 'mlflow.db'}"
    get_settings.cache_clear()

    from promptforge.evaluation.run import evaluate

    report = evaluate()
    yield {"report": report, "artifacts": tmp / "artifacts"}

    (
        cfg["data"]["processed_dir"],
        cfg["data"]["artifacts_dir"],
        cfg["data"]["registry_path"],
        cfg["eval"]["bootstrap_iters"],
    ) = originals
    if old_uri is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = old_uri
    get_settings.cache_clear()


@pytest.fixture
def api_client(evaluated):
    from fastapi.testclient import TestClient

    from promptforge.api import routes
    from promptforge.api.main import app

    routes._report.cache_clear()
    try:
        yield TestClient(app)
    finally:
        routes._report.cache_clear()

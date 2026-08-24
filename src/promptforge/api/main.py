"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from promptforge import __version__
from promptforge.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    app = FastAPI(
        title="promptforge",
        description="Prompt engineering workbench: a versioned prompt registry, task test-suites with assertion-based scoring, A/B evaluation across prompt variants with a bootstrap win-rate CI, regression gating, and token-cost accounting.",
        version=__version__,
    )
    app.include_router(router)
    return app


app = create_app()

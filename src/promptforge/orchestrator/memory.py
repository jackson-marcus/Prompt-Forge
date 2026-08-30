"""Short-term scratchpad shared by agents in one run."""

from __future__ import annotations

from typing import Any


class Memory:
    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def write(self, key: str, value: Any) -> None:
        self._store[key] = value

    def read(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._store)

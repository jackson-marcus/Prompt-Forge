"""Caretaker: an append-only, versioned repository of prompt snapshots.

:class:`PromptRepository` is the interface every caller programs against. It
owns the *policy* - version numbering, the content-addressed no-op, optimistic
concurrency, restore-as-append, diffing - and delegates only four raw storage
primitives to a backend. :class:`JsonPromptRepository` is one such backend; a
Postgres or S3 one would implement the same four methods and nothing else would
change.

The repository is append-only by construction: there is no update and no
delete. Restoring an old prompt appends a new head whose parent is the restored
version, so "we shipped v2, rolled back to v1, then shipped v4" stays legible
months later.
"""

from __future__ import annotations

import difflib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from promptforge.registry.memento import PromptSnapshot, PromptVariant
from promptforge.settings import get_config, resolve_path

SCHEMA_VERSION = 2


class UnknownPromptError(KeyError):
    """No such task, variant, or version in the repository."""


class VersionConflictError(RuntimeError):
    """A snapshot was captured from a working copy that is no longer the head."""


class PromptRepository(ABC):
    """Versioned, append-only store of :class:`PromptSnapshot` mementos."""

    # --- storage primitives: the entire surface a backend must implement ---

    @abstractmethod
    def _append(self, snapshot: PromptSnapshot) -> None:
        """Persist ``snapshot`` as the new head of its variant's history."""

    @abstractmethod
    def history(self, task: str, name: str) -> list[PromptSnapshot]:
        """Every snapshot of one variant, oldest first. Empty if unknown."""

    @abstractmethod
    def names(self, task: str) -> list[str]:
        """Variant names registered for ``task``, in registration order."""

    @abstractmethod
    def baseline(self, task: str) -> str | None:
        """The variant designated as this task's comparison baseline."""

    @abstractmethod
    def set_baseline(self, task: str, name: str) -> None:
        """Designate ``name`` as the baseline variant for ``task``."""

    @abstractmethod
    def clear(self) -> None:
        """Drop all history. Used by the seed script to rebuild from scratch."""

    # --- policy: storage-independent, shared by every backend ---

    def save(self, snapshot: PromptSnapshot) -> PromptSnapshot:
        """Append ``snapshot``, unless it says nothing new.

        Returns the snapshot that is now the head. Saving a template identical
        to the current head is a no-op that does **not** bump the version -
        content addressing means an unchanged prompt has no new state to
        remember. A snapshot captured from a stale working copy (someone else
        saved in between) raises :class:`VersionConflictError` rather than
        silently clobbering their version.
        """
        head = self.head(snapshot.task, snapshot.name)
        if head is None:
            if snapshot.version != 1:
                raise VersionConflictError(
                    f"{snapshot.task}/{snapshot.ref}: first version must be v1"
                )
        elif head.has_same_content_as(snapshot):
            return head
        elif snapshot.version != head.version + 1:
            raise VersionConflictError(
                f"{snapshot.task}/{snapshot.ref} was captured from a stale copy; "
                f"head is now v{head.version}"
            )
        self._append(snapshot)
        return snapshot

    def head(self, task: str, name: str) -> PromptSnapshot | None:
        """The newest snapshot of a variant, or None if it has no history."""
        snapshots = self.history(task, name)
        return snapshots[-1] if snapshots else None

    def get(self, task: str, name: str, version: int | None = None) -> PromptSnapshot:
        """One snapshot: the head by default, or an exact historical version."""
        snapshots = self.history(task, name)
        if not snapshots:
            raise UnknownPromptError(f"no variant {name!r} registered for task {task!r}")
        if version is None:
            return snapshots[-1]
        for snapshot in snapshots:
            if snapshot.version == version:
                return snapshot
        raise UnknownPromptError(f"{task}/{name} has no version {version}")

    def checkout(self, task: str, name: str) -> PromptVariant:
        """A working copy on top of the current head (empty if brand new)."""
        head = self.head(task, name)
        return PromptVariant.from_snapshot(head) if head else PromptVariant(task, name)

    def register(self, task: str, name: str, template: str, created_by: str) -> PromptSnapshot:
        """Check out, edit, capture, save - the ordinary write path."""
        working = self.checkout(task, name).edit(template)
        return self.save(working.capture(created_by))

    def restore(
        self, task: str, name: str, version: int, created_by: str = "restore"
    ) -> PromptSnapshot:
        """Bring an old version back as a *new* head whose parent is that old version.

        History is never rewritten: the versions in between remain readable.
        Restoring a version whose template already matches the head is a no-op,
        for the same content-addressed reason a re-save is.
        """
        target = self.get(task, name, version)
        working = self.checkout(task, name).restore(target)
        return self.save(working.capture(created_by))

    def heads(self, task: str) -> dict[str, PromptSnapshot]:
        """Current snapshot of every variant of a task, in registration order."""
        return {name: self.get(task, name) for name in self.names(task)}

    @staticmethod
    def diff(a: PromptSnapshot, b: PromptSnapshot) -> str:
        """Unified line diff between two snapshots' templates ('' if identical)."""
        if a.has_same_content_as(b):
            return ""
        return "\n".join(
            difflib.unified_diff(
                a.template.splitlines(),
                b.template.splitlines(),
                fromfile=a.ref,
                tofile=b.ref,
                lineterm="",
            )
        )


class JsonPromptRepository(PromptRepository):
    """A JSON-file backend. Every detail of the on-disk shape lives in here.

    The document is read once per instance and written through on append, so a
    repository instance owns the file for its lifetime; construct a fresh one
    per request or per script run (see :func:`get_repository`).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._doc: dict[str, Any] | None = None

    # --- file plumbing ---

    def _document(self) -> dict[str, Any]:
        if self._doc is None:
            if self.path.exists():
                self._doc = json.loads(self.path.read_text(encoding="utf-8"))
            else:
                self._doc = {"schema": SCHEMA_VERSION, "prompts": {}}
        return self._doc

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._document(), indent=2), encoding="utf-8")

    def _task_entry(self, task: str) -> dict[str, Any]:
        return self._document()["prompts"].get(task, {"baseline": None, "variants": {}})

    # --- storage primitives ---

    def _append(self, snapshot: PromptSnapshot) -> None:
        prompts = self._document()["prompts"]
        entry = prompts.setdefault(snapshot.task, {"baseline": None, "variants": {}})
        variant = entry["variants"].setdefault(snapshot.name, {"snapshots": []})
        variant["snapshots"].append(snapshot.to_dict())
        if entry["baseline"] is None:
            entry["baseline"] = snapshot.name
        self._flush()

    def history(self, task: str, name: str) -> list[PromptSnapshot]:
        variant = self._task_entry(task)["variants"].get(name)
        if variant is None:
            return []
        return [PromptSnapshot.from_dict(payload) for payload in variant["snapshots"]]

    def names(self, task: str) -> list[str]:
        return list(self._task_entry(task)["variants"])

    def baseline(self, task: str) -> str | None:
        return self._task_entry(task)["baseline"]

    def set_baseline(self, task: str, name: str) -> None:
        if name not in self.names(task):
            raise UnknownPromptError(f"cannot baseline unknown variant {task}/{name}")
        self._document()["prompts"][task]["baseline"] = name
        self._flush()

    def clear(self) -> None:
        self._doc = {"schema": SCHEMA_VERSION, "prompts": {}}
        self._flush()


def get_repository() -> PromptRepository:
    """The configured repository. Swap this one line to swap the backend."""
    return JsonPromptRepository(resolve_path(get_config()["data"]["registry_path"]))

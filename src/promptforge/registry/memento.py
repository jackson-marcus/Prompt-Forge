"""Memento: immutable prompt snapshots and the working copy that produces them.

Two roles live here, and only these two:

* ``PromptSnapshot`` is the **memento** - a frozen, content-addressed value
  object recording one state of one prompt variant. It validates its own
  ``content_hash`` on construction, so a snapshot can never claim to be a
  template it is not, whether it was just captured or read back off disk.
* ``PromptVariant`` is the **originator** - the mutable working copy you edit.
  ``capture()`` freezes its current state into a snapshot; ``restore()`` rolls
  it back to an earlier one while remembering that earlier version as the
  lineage parent, so the *next* capture appends rather than rewrites.

Neither role knows anything about storage. Persistence is the caretaker's job
(see :mod:`promptforge.registry.repository`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

HASH_ALGORITHM = "sha256"
HASH_DIGITS = 16


def content_hash(template: str) -> str:
    """Address a template by its content: identical text -> identical hash."""
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return f"{HASH_ALGORITHM}:{digest[:HASH_DIGITS]}"


class SnapshotIntegrityError(ValueError):
    """A snapshot's content_hash does not address its own template."""


@dataclass(frozen=True, slots=True)
class PromptSnapshot:
    """One immutable, content-addressed state of a prompt variant."""

    task: str
    name: str
    version: int
    template: str
    created_by: str
    parent_version: int | None
    content_hash: str

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError(f"version must start at 1, got {self.version}")
        if self.parent_version is not None and self.parent_version >= self.version:
            raise ValueError(
                f"parent v{self.parent_version} must precede v{self.version} ({self.ref})"
            )
        expected = content_hash(self.template)
        if self.content_hash != expected:
            raise SnapshotIntegrityError(
                f"{self.ref}: content_hash {self.content_hash!r} does not address its template "
                f"(expected {expected!r})"
            )

    @classmethod
    def create(
        cls,
        *,
        task: str,
        name: str,
        version: int,
        template: str,
        created_by: str,
        parent_version: int | None = None,
    ) -> PromptSnapshot:
        """Build a snapshot, computing the content hash from the template."""
        return cls(
            task=task,
            name=name,
            version=version,
            template=template,
            created_by=created_by,
            parent_version=parent_version,
            content_hash=content_hash(template),
        )

    @property
    def ref(self) -> str:
        """Human-readable reference, e.g. ``engineered@v3``."""
        return f"{self.name}@v{self.version}"

    @property
    def key(self) -> tuple[str, str]:
        return (self.task, self.name)

    def has_same_content_as(self, other: PromptSnapshot) -> bool:
        return self.content_hash == other.content_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "name": self.name,
            "version": self.version,
            "template": self.template,
            "created_by": self.created_by,
            "parent_version": self.parent_version,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PromptSnapshot:
        """Rehydrate a snapshot, re-verifying its content hash."""
        return cls(
            task=payload["task"],
            name=payload["name"],
            version=int(payload["version"]),
            template=payload["template"],
            created_by=payload["created_by"],
            parent_version=(
                None if payload["parent_version"] is None else int(payload["parent_version"])
            ),
            content_hash=payload["content_hash"],
        )


class PromptVariant:
    """Originator: the mutable working copy of one prompt variant.

    ``version`` is the stored head this copy was taken from (0 = never saved);
    ``parent_version`` is the lineage parent the next capture will record. They
    differ exactly when the copy has been rolled back with :meth:`restore`,
    which is what lets a restore append a new version instead of rewriting one.
    """

    __slots__ = ("name", "parent_version", "task", "template", "version")

    def __init__(
        self,
        task: str,
        name: str,
        template: str = "",
        *,
        version: int = 0,
        parent_version: int | None = None,
    ) -> None:
        self.task = task
        self.name = name
        self.template = template
        self.version = version
        self.parent_version = parent_version

    @classmethod
    def from_snapshot(cls, snapshot: PromptSnapshot) -> PromptVariant:
        """Check out a working copy sitting on top of ``snapshot``."""
        return cls(
            snapshot.task,
            snapshot.name,
            snapshot.template,
            version=snapshot.version,
            parent_version=snapshot.version,
        )

    def edit(self, template: str) -> PromptVariant:
        """Change the working template. Nothing is versioned until capture()."""
        self.template = template
        return self

    def capture(self, created_by: str) -> PromptSnapshot:
        """Freeze the working state into the next snapshot in the lineage."""
        return PromptSnapshot.create(
            task=self.task,
            name=self.name,
            version=self.version + 1,
            template=self.template,
            created_by=created_by,
            parent_version=self.parent_version,
        )

    def restore(self, snapshot: PromptSnapshot) -> PromptVariant:
        """Roll the working copy back to ``snapshot``.

        The head version is deliberately left alone - only the lineage parent
        moves - so the next capture lands *after* the current head with the
        restored version as its parent. History is never rewritten.
        """
        if snapshot.key != (self.task, self.name):
            raise ValueError(
                f"cannot restore {snapshot.task}/{snapshot.name} into {self.task}/{self.name}"
            )
        if snapshot.version > self.version:
            raise ValueError(f"{snapshot.ref} is ahead of the checked-out head v{self.version}")
        self.template = snapshot.template
        self.parent_version = snapshot.version
        return self

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PromptVariant({self.task!r}, {self.name!r}, head=v{self.version}, "
            f"parent={self.parent_version})"
        )

"""Memento + versioned repository: immutability, content addressing, restore, diff."""

from __future__ import annotations

import dataclasses

import pytest

from promptforge.registry.memento import (
    PromptSnapshot,
    PromptVariant,
    SnapshotIntegrityError,
    content_hash,
)
from promptforge.registry.repository import (
    JsonPromptRepository,
    UnknownPromptError,
    VersionConflictError,
)

V1 = "Classify the sentiment: {input}"
V2 = "Classify the sentiment.\nAnswer with only one word.\nReview: {input}"
V3 = "Classify the sentiment.\nAnswer with only one word.\nDo not explain.\nReview: {input}"


def _snapshot(template: str = V1, version: int = 1, parent: int | None = None) -> PromptSnapshot:
    return PromptSnapshot.create(
        task="sentiment",
        name="bare",
        version=version,
        template=template,
        created_by="test",
        parent_version=parent,
    )


# --- the memento ---------------------------------------------------------


def test_snapshot_is_frozen_and_content_addressed():
    snap = _snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.template = "mutated"  # type: ignore[misc]

    # identical templates address identically, different ones do not
    assert content_hash(V1) == _snapshot(V1, version=9).content_hash
    assert content_hash(V1) != content_hash(V2)
    assert snap.has_same_content_as(_snapshot(V1, version=4, parent=3))
    assert not snap.has_same_content_as(_snapshot(V2))


def test_snapshot_rejects_a_hash_that_does_not_address_its_template():
    forged = _snapshot().to_dict() | {"template": V2}
    with pytest.raises(SnapshotIntegrityError):
        PromptSnapshot.from_dict(forged)


def test_snapshot_roundtrips_through_its_dict_form():
    snap = _snapshot(V2, version=2, parent=1)
    assert PromptSnapshot.from_dict(snap.to_dict()) == snap
    assert snap.ref == "bare@v2"


def test_snapshot_rejects_impossible_lineage():
    with pytest.raises(ValueError):
        _snapshot(version=2, parent=2)  # a parent cannot be its own child's version
    with pytest.raises(ValueError):
        _snapshot(version=0)


# --- the originator ------------------------------------------------------


def test_originator_captures_and_restores_working_state():
    variant = PromptVariant("sentiment", "bare")
    first = variant.edit(V1).capture("alice")
    assert (first.version, first.parent_version) == (1, None)

    working = PromptVariant.from_snapshot(first).edit(V2)
    second = working.capture("bob")
    assert (second.version, second.parent_version, second.template) == (2, 1, V2)

    # restore rolls the working template back but keeps the head position
    working.restore(first)
    assert working.template == V1
    third = working.capture("bob")
    assert (third.version, third.parent_version) == (2, 1)


def test_originator_refuses_a_snapshot_from_another_variant():
    variant = PromptVariant("sentiment", "bare", V1, version=1, parent_version=1)
    alien = PromptSnapshot.create(
        task="sentiment",
        name="engineered",
        version=1,
        template=V2,
        created_by="test",
        parent_version=None,
    )
    with pytest.raises(ValueError):
        variant.restore(alien)


# --- the repository ------------------------------------------------------


def test_saving_an_unchanged_template_is_a_noop(repo):
    repo.register("sentiment", "bare", V1, created_by="seed")
    head = repo.register("sentiment", "bare", V1, created_by="seed-again")

    assert head.version == 1, "an unchanged template must not bump the version"
    assert head.created_by == "seed", "the original snapshot must survive untouched"
    assert len(repo.history("sentiment", "bare")) == 1


def test_history_is_append_only_and_versions_link_to_parents(repo):
    for template in (V1, V2, V3):
        repo.register("sentiment", "bare", template, created_by="alice")

    history = repo.history("sentiment", "bare")
    assert [s.version for s in history] == [1, 2, 3]
    assert [s.parent_version for s in history] == [None, 1, 2]
    assert [s.template for s in history] == [V1, V2, V3]
    assert repo.get("sentiment", "bare").version == 3  # default is the head
    assert repo.get("sentiment", "bare", 1).template == V1


def test_restore_appends_a_new_version_parented_on_the_restored_one(repo):
    for template in (V1, V2, V3):
        repo.register("sentiment", "bare", template, created_by="alice")

    restored = repo.restore("sentiment", "bare", 1, created_by="oncall")

    assert restored.version == 4, "restore appends; it never rewinds the counter"
    assert restored.parent_version == 1, "lineage records what was restored"
    assert restored.template == V1
    assert restored.content_hash == content_hash(V1)
    # nothing was rewritten: the versions that were rolled back are still readable
    history = repo.history("sentiment", "bare")
    assert [s.version for s in history] == [1, 2, 3, 4]
    assert repo.get("sentiment", "bare", 2).template == V2
    assert repo.get("sentiment", "bare", 3).template == V3


def test_restoring_the_current_head_changes_nothing(repo):
    repo.register("sentiment", "bare", V1, created_by="alice")
    repo.register("sentiment", "bare", V2, created_by="alice")

    head = repo.restore("sentiment", "bare", 2)

    assert head.version == 2
    assert len(repo.history("sentiment", "bare")) == 2


def test_stale_working_copy_is_rejected(repo):
    repo.register("sentiment", "bare", V1, created_by="alice")
    stale = repo.checkout("sentiment", "bare")  # alice's copy, based on v1
    repo.register("sentiment", "bare", V2, created_by="bob")  # bob lands v2 first

    with pytest.raises(VersionConflictError):
        repo.save(stale.edit(V3).capture("alice"))
    assert [s.template for s in repo.history("sentiment", "bare")] == [V1, V2]


def test_diff_is_a_unified_line_diff(repo):
    repo.register("sentiment", "bare", V1, created_by="alice")
    repo.register("sentiment", "bare", V2, created_by="alice")

    diff = repo.diff(repo.get("sentiment", "bare", 1), repo.get("sentiment", "bare", 2))

    assert "--- bare@v1" in diff and "+++ bare@v2" in diff
    assert f"-{V1}" in diff
    assert "+Answer with only one word." in diff
    # identical snapshots have nothing to say
    assert repo.diff(repo.get("sentiment", "bare", 1), repo.get("sentiment", "bare", 1)) == ""


def test_unknown_lookups_raise(repo):
    repo.register("sentiment", "bare", V1, created_by="alice")
    with pytest.raises(UnknownPromptError):
        repo.get("sentiment", "nope")
    with pytest.raises(UnknownPromptError):
        repo.get("sentiment", "bare", 99)
    with pytest.raises(UnknownPromptError):
        repo.set_baseline("sentiment", "nope")
    assert repo.history("nope", "bare") == []
    assert repo.names("nope") == [] and repo.baseline("nope") is None


def test_baseline_defaults_to_first_registered_and_can_be_moved(repo):
    repo.register("sentiment", "bare", V1, created_by="alice")
    repo.register("sentiment", "engineered", V3, created_by="alice")
    assert repo.baseline("sentiment") == "bare"
    assert list(repo.heads("sentiment")) == ["bare", "engineered"]

    repo.set_baseline("sentiment", "engineered")
    assert repo.baseline("sentiment") == "engineered"


def test_storage_is_swappable_behind_the_interface(repo, tmp_path):
    """State lives in the backend, not the instance: a fresh one reads it all back."""
    for template in (V1, V2):
        repo.register("sentiment", "bare", template, created_by="alice")
    repo.restore("sentiment", "bare", 1)

    reopened = JsonPromptRepository(tmp_path / "registry.json")

    assert reopened.history("sentiment", "bare") == repo.history("sentiment", "bare")
    assert reopened.get("sentiment", "bare").parent_version == 1
    assert reopened.baseline("sentiment") == "bare"

    reopened.clear()
    assert reopened.names("sentiment") == []

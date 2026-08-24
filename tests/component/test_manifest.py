"""Run manifests and the pooling guard."""

from __future__ import annotations

import pytest

from agent_newton import manifest as manifest_module
from agent_newton.config import Config
from agent_newton.manifest import (
    IncomparableRunsError,
    RunManifest,
    assert_poolable,
    hash_content,
)


def _manifest(run_id: str, **overrides) -> RunManifest:
    manifest = RunManifest.create(Config(), run_id=run_id)
    manifest.catalogue_hash = "cat0"
    manifest.item_bank_hash = "item0"
    manifest.concept_graph_hash = "graph0"
    for key, value in overrides.items():
        setattr(manifest, key, value)
    return manifest


class TestRunManifest:
    def test_records_provenance(self) -> None:
        manifest = RunManifest.create(Config(run_name="demo", seed=42), run_id="r1")
        assert manifest.run_name == "demo"
        assert manifest.seed == 42
        assert manifest.config_hash

    def test_takes_the_sha_from_the_environment_when_declared(self, monkeypatch) -> None:
        # A container built from a copy of the source has no .git to ask, and a
        # manifest recording null there breaks the chain from a reported number
        # back to the commit that produced it.
        monkeypatch.setenv(manifest_module.SHA_ENV, "abc123")
        monkeypatch.delenv(manifest_module.DIRTY_ENV, raising=False)
        monkeypatch.setattr(manifest_module, "_git", lambda *args: None)

        manifest = RunManifest.create(Config(), run_id="r1")
        assert manifest.git_sha == "abc123"
        assert manifest.git_dirty is False

    def test_declared_sha_carries_its_own_dirty_flag(self, monkeypatch) -> None:
        monkeypatch.setenv(manifest_module.SHA_ENV, "abc123")
        monkeypatch.setenv(manifest_module.DIRTY_ENV, "1")
        monkeypatch.setattr(manifest_module, "_git", lambda *args: None)

        assert RunManifest.create(Config(), run_id="r1").git_dirty is True

    def test_reads_the_working_tree_when_nothing_is_declared(self, monkeypatch) -> None:
        # The other half of the guard: the environment does not shadow a real
        # repository, and an empty value is not a declaration.
        monkeypatch.setenv(manifest_module.SHA_ENV, "")
        monkeypatch.setenv(manifest_module.DIRTY_ENV, "1")
        monkeypatch.setattr(
            manifest_module,
            "_git",
            lambda *args: "deadbeef" if args[0] == "rev-parse" else "",
        )

        manifest = RunManifest.create(Config(), run_id="r1")
        assert manifest.git_sha == "deadbeef"
        # From `git status`, which reported nothing — not from DIRTY_ENV.
        assert manifest.git_dirty is False

    def test_records_model_free_roles_by_impl_not_model_name(self) -> None:
        # An oracle diagnostic never calls gemma4:12b, so recording that model
        # name would misdescribe the run in the manifest.
        config = Config.model_validate(
            {
                "agents": {"diagnostic": {"impl": "oracle", "model": "gemma4:12b"}},
                "simulator": {"surface": "symbolic"},
            }
        )
        manifest = RunManifest.create(config, run_id="r1")
        assert manifest.models["diagnostic"] == "<oracle>"
        assert manifest.models["simulator_surface"] == "<symbolic>"

    def test_records_models_when_llm_backed(self) -> None:
        config = Config.model_validate(
            {"agents": {"tutor": {"impl": "llm", "provider": "ollama", "model": "gemma4:12b"}}}
        )
        assert RunManifest.create(config, run_id="r1").models["tutor"] == "ollama/gemma4:12b"

    def test_roundtrips_through_disk(self, tmp_path) -> None:
        original = _manifest("r1")
        original.write(tmp_path)
        assert RunManifest.read(tmp_path) == original


class TestAssertPoolable:
    """The guard that stops a changed domain silently corrupting a comparison."""

    def test_allows_matching_runs(self) -> None:
        assert_poolable([_manifest("r1"), _manifest("r2")])

    def test_allows_a_single_run(self) -> None:
        assert_poolable([_manifest("r1")])

    def test_rejects_changed_misconception_catalogue(self) -> None:
        # The scenario this exists for: a misconception is added part-way
        # through a project, so the diagnostic's label space grew and earlier
        # accuracy figures are no longer on the same scale.
        with pytest.raises(IncomparableRunsError, match="label space"):
            assert_poolable([_manifest("r1"), _manifest("r2", catalogue_hash="cat1")])

    def test_rejects_changed_item_bank(self) -> None:
        with pytest.raises(IncomparableRunsError, match="different items"):
            assert_poolable([_manifest("r1"), _manifest("r2", item_bank_hash="item1")])

    def test_rejects_changed_concept_graph(self) -> None:
        with pytest.raises(IncomparableRunsError, match="ZPD frontier"):
            assert_poolable([_manifest("r1"), _manifest("r2", concept_graph_hash="graph1")])

    def test_rejects_different_domains(self) -> None:
        with pytest.raises(IncomparableRunsError, match="different domains"):
            assert_poolable([_manifest("r1"), _manifest("r2", domain="calculus")])

    def test_error_names_the_offending_runs(self) -> None:
        with pytest.raises(IncomparableRunsError, match="alpha"):
            assert_poolable([_manifest("alpha"), _manifest("beta", catalogue_hash="cat1")])


class TestHashContent:
    def test_is_stable(self) -> None:
        assert hash_content("a", "b") == hash_content("a", "b")

    def test_is_not_concatenation_ambiguous(self) -> None:
        # ("ab", "c") and ("a", "bc") must differ, or a reordered catalogue
        # could hash identically to the original.
        assert hash_content("ab", "c") != hash_content("a", "bc")

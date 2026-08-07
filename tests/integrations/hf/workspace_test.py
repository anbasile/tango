import itertools

import pytest

from tango.integrations.hf.common import HfBucketNotFound
from tango.integrations.hf.workspace import HfBucketWorkspace
from tango.step import Step
from tango.step_caches.remote_step_cache import RemoteNotFoundError
from tango.step_info import StepState
from tango.workspace import Workspace

from .fake_hub import FakeHfApi, install_fakes


@Step.register("hf_test_add")
class AddStep(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, a: int, b: int) -> int:  # type: ignore[override]
        return a + b


@Step.register("hf_test_boom")
class BoomStep(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self) -> int:  # type: ignore[override]
        raise ValueError("boom")


@pytest.fixture
def s3(monkeypatch, tmp_path):
    return install_fakes(monkeypatch, tmp_path / "cache")


@pytest.fixture
def workspace(s3):
    return HfBucketWorkspace("org/bucket")


@pytest.fixture
def other_machine(monkeypatch, tmp_path, s3):
    """
    Build a workspace over the same bucket but with an empty local cache, which is what a
    second machine sees. Without this the on-disk mirror answers first and the bucket is
    never consulted.
    """
    counter = itertools.count()

    def _make(bucket: str = "org/bucket") -> HfBucketWorkspace:
        from tango.integrations.hf import step_cache

        cache_dir = tmp_path / f"machine-{next(counter)}"
        monkeypatch.setattr(step_cache, "tango_cache_dir", lambda: cache_dir)
        return HfBucketWorkspace(bucket)

    return _make


class TestUrls:
    def test_url_round_trips(self, workspace):
        assert workspace.url == "hf://buckets/org/bucket"
        assert Workspace.from_url(workspace.url).url == workspace.url

    def test_canonical_hub_uri(self, s3):
        assert Workspace.from_url("hf://buckets/org/bucket").url == "hf://buckets/org/bucket"

    def test_short_form(self, s3):
        assert Workspace.from_url("hf://org/bucket").url == "hf://buckets/org/bucket"

    def test_prefix_within_a_bucket(self, s3):
        ws = Workspace.from_url("hf://buckets/org/bucket/experiments/v2")
        assert ws.url == "hf://buckets/org/bucket/experiments/v2"

    def test_rejects_a_bucketless_url(self, s3):
        with pytest.raises(Exception):
            Workspace.from_url("hf://buckets/")


class TestStepInfo:
    def test_unknown_step_id_raises(self, workspace):
        with pytest.raises(KeyError):
            workspace.step_info("nosuchstep")

    def test_new_step_is_recorded_as_incomplete(self, workspace):
        step = AddStep(a=1, b=2)
        info = workspace.step_info(step)
        assert info.state == StepState.INCOMPLETE
        # And it is now durable, retrievable by id alone.
        assert workspace.step_info(step.unique_id).unique_id == step.unique_id


class TestStepExecution:
    def test_result_is_cached_and_readable(self, workspace):
        step = AddStep(a=1, b=2)
        step.ensure_result(workspace)

        assert workspace.step_info(step).state == StepState.COMPLETED
        assert workspace.step_cache[step] == 3
        assert step in workspace.step_cache
        assert len(workspace.step_cache) == 1

    def test_another_machine_reuses_the_cached_result(self, workspace, other_machine):
        """
        The whole point of a bucket-backed workspace: another machine skips the work.
        """
        step = AddStep(a=2, b=3)
        step.ensure_result(workspace)

        fresh = other_machine()
        assert AddStep(a=2, b=3) in fresh.step_cache
        assert fresh.step_cache[AddStep(a=2, b=3)] == 5

    def test_the_lock_is_released_after_a_step_finishes(self, workspace, s3):
        step = AddStep(a=1, b=1)
        step.ensure_result(workspace)
        assert s3.objects == {}
        assert workspace.locks == {}

    def test_a_failed_step_records_the_error_and_releases_the_lock(self, workspace, s3):
        step = BoomStep()
        with pytest.raises(ValueError, match="boom"):
            step.ensure_result(workspace)

        info = workspace.step_info(step)
        assert info.state == StepState.FAILED
        assert "boom" in (info.error or "")
        assert s3.objects == {}

    def test_an_uncommitted_artifact_is_not_a_cache_hit(self, workspace, other_machine):
        """
        A step whose upload died half-way must not be served as a finished result.
        """
        step = AddStep(a=4, b=4)
        step.ensure_result(workspace)
        assert step in workspace.step_cache

        cache = workspace.step_cache
        artifact = cache.Constants.step_artifact_name(step)
        cache.client.put_bytes(f"{artifact}/{cache.Constants.UNCOMMITTED_FNAME}", b"")

        fresh = other_machine()
        assert AddStep(a=4, b=4) not in fresh.step_cache
        assert len(fresh.step_cache) == 0


class TestProbedHubBehaviour:
    """
    Cases where the real Hub is more forgiving than you would guess, each confirmed by probing
    it. The fakes reproduce the real behaviour, so these exercise the code paths that actually
    run in production.
    """

    def test_a_missing_object_is_reported_even_though_the_hub_does_not_raise(self, workspace):
        # `download_bucket_files` warns and skips rather than raising, so `get_bytes` can only
        # tell by checking whether the file appeared.
        with pytest.raises(HfBucketNotFound):
            workspace.step_cache.client.get_bytes("nowhere/absent.json")

    def test_a_vanished_artifact_is_a_cache_miss_not_an_empty_directory(self, workspace, tmp_path):
        # `sync_bucket` treats a prefix with no objects as "nothing to do". Without an explicit
        # check the caller would get an empty directory and a confusing failure much later.
        with pytest.raises(RemoteNotFoundError):
            workspace.step_cache._download_step_remote("tango-step-never-existed", tmp_path / "out")

    def test_a_partial_artifact_is_a_cache_miss(self, workspace, tmp_path):
        step = AddStep(a=5, b=5)
        step.ensure_result(workspace)

        # Lose the metadata but keep the payload, as an interrupted delete would.
        cache = workspace.step_cache
        artifact = cache.Constants.step_artifact_name(step)
        cache.client.delete(f"{artifact}/{cache.METADATA_FILE_NAME}")

        with pytest.raises(RemoteNotFoundError):
            cache._download_step_remote(artifact, tmp_path / "out")


class TestRuns:
    def test_register_and_read_back(self, workspace):
        step = AddStep(a=1, b=2)
        run = workspace.register_run([step], name="my-run")

        assert run.name == "my-run"
        assert workspace.registered_run("my-run").steps.keys() == run.steps.keys()
        assert set(workspace.registered_runs()) == {"my-run"}

    def test_the_returned_run_matches_the_stored_one(self, workspace):
        # The serialised timestamp has no sub-second field, so an untruncated start_date here
        # would not survive a round trip.
        run = workspace.register_run([AddStep(a=1, b=2)], name="round-trip")
        assert workspace.registered_run("round-trip").start_date == run.start_date

    def test_generated_names_are_unique(self, workspace):
        first = workspace.register_run([AddStep(a=1, b=2)])
        second = workspace.register_run([AddStep(a=3, b=4)])
        assert first.name != second.name
        assert set(workspace.registered_runs()) == {first.name, second.name}

    def test_duplicate_name_is_rejected(self, workspace):
        workspace.register_run([AddStep(a=1, b=2)], name="taken")
        with pytest.raises(ValueError, match="already in use"):
            workspace.register_run([AddStep(a=3, b=4)], name="taken")

    def test_unknown_run_raises(self, workspace):
        with pytest.raises(KeyError):
            workspace.registered_run("nope")

    def test_search_falls_back_to_the_base_implementation(self, workspace):
        workspace.register_run([AddStep(a=1, b=2)], name="alpha")
        workspace.register_run([AddStep(a=3, b=4)], name="beta")

        # Not overridden by this workspace; `Workspace` implements it over `registered_runs()`.
        names = [run.name for run in workspace.search_registered_runs(match="al")]
        assert names == ["alpha"]
        assert workspace.num_registered_runs() == 2


class TestBucketLayout:
    def test_objects_land_where_documented(self, workspace):
        step = AddStep(a=1, b=2)
        step.ensure_result(workspace)
        workspace.register_run([step], name="my-run")

        keys = set(FakeHfApi.STORE["org/bucket"])
        assert "settings.json" in keys
        assert f"stepinfo/{step.unique_id}.json" in keys
        assert "runs/my-run.json" in keys
        assert any(key.startswith(f"tango-step-{step.unique_id}/result/") for key in keys)
        assert not any(key.endswith(".uncommitted") for key in keys)

    def test_a_prefix_scopes_every_object(self, s3):
        workspace = HfBucketWorkspace("org/bucket/experiments")
        AddStep(a=1, b=2).ensure_result(workspace)

        keys = FakeHfApi.STORE["org/bucket"]
        assert keys, "nothing was written"
        assert all(key.startswith("experiments/") for key in keys)

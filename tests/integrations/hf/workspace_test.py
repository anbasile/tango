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

    def test_re_registering_the_same_graph_is_a_no_op(self, workspace):
        """
        A detached run depends on this: the client registers the run, then the driver job runs
        `tango run -n <name>` again inside its own container.
        """
        first = workspace.register_run([AddStep(a=1, b=2)], name="same")
        again = workspace.register_run([AddStep(a=1, b=2)], name="same")
        assert again.name == first.name
        assert again.start_date == first.start_date
        assert set(workspace.registered_runs()) == {"same"}

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


class TestLockIsNotAResult:
    """
    The lock `tango-step-<id>-lock` sits next to the result folder `tango-step-<id>/`, and a
    bucket listing matches by string prefix.
    """

    def _lock_only(self, workspace, step):
        workspace.step_starting(step)
        return workspace.locks.pop(step)  # held, as by a job that died

    def test_a_lock_alone_is_not_a_cached_step(self, workspace, other_machine, s3):
        step = AddStep(a=1, b=2)
        lock = self._lock_only(workspace, step)
        assert s3.objects, "the lock should be in the bucket"

        assert step not in other_machine().step_cache
        assert len(other_machine().step_cache) == 0
        lock.release()

    def test_removing_a_step_leaves_a_lock_alone(self, workspace, s3):
        done = AddStep(a=1, b=2)
        done.ensure_result(workspace)
        # A second run has the same step locked while it is being removed here.
        other = HfBucketWorkspace("org/bucket")
        lock = other._remote_lock(done)
        lock.acquire(timeout=5)

        workspace.remove_step(done.unique_id)
        assert list(s3.objects) == [f"tango-step-{done.unique_id}-lock"]
        lock.release()


class TestStepAbandoned:
    def _started_by_job(self, monkeypatch, workspace, step, job_id):
        monkeypatch.setenv("JOB_ID", job_id)
        workspace.step_starting(step)
        monkeypatch.delenv("JOB_ID")
        return workspace.locks.pop(step)

    def test_the_step_is_failed_and_unlocked(self, monkeypatch, workspace, s3):
        step = AddStep(a=1, b=2)
        lock = self._started_by_job(monkeypatch, workspace, step, "job-1")

        assert workspace.step_abandoned(step, "job ended in stage CANCELED", "job-1") is True
        info = workspace.step_info(step)
        assert info.state == StepState.FAILED
        assert info.error == "Abandoned: job ended in stage CANCELED"
        assert s3.objects == {}

        # And the step can simply be run again.
        assert step.result(workspace) == 3
        del lock

    def test_a_step_someone_else_is_running_is_left_alone(self, monkeypatch, workspace, s3):
        import huggingface_hub

        from .fake_hub import FakeJob

        step = AddStep(a=1, b=2)
        lock = self._started_by_job(monkeypatch, workspace, step, "another-job")
        monkeypatch.setattr(
            huggingface_hub, "inspect_job", lambda **kw: FakeJob(kw["job_id"], "RUNNING")
        )

        assert workspace.step_abandoned(step, "job ended in stage ERROR", "job-1") is False
        assert workspace.step_info(step).state == StepState.RUNNING
        assert len(s3.objects) == 1
        del lock

    def test_a_finished_step_stays_finished(self, workspace):
        step = AddStep(a=1, b=2)
        step.ensure_result(workspace)
        assert workspace.step_abandoned(step, "whatever", "job-1") is True
        assert workspace.step_info(step).state == StepState.COMPLETED


class TestRunNames:
    def test_the_same_graph_again_is_the_same_run(self, workspace):
        step = AddStep(a=1, b=2)
        first = workspace.register_run([step], name="main")
        again = workspace.register_run([AddStep(a=1, b=2)], name="main")
        assert again.start_date == first.start_date

    def test_a_changed_graph_updates_the_run(self, workspace, caplog):
        # Refusing this is why the runs of one experiment were called main, main2 ... main6.
        kept = AddStep(a=1, b=2, step_name="kept")
        workspace.register_run(
            [kept, AddStep(a=3, b=4, step_name="changed"), AddStep(a=5, b=6, step_name="gone")],
            name="main",
        )
        old_id = AddStep(a=3, b=4, step_name="changed").unique_id

        with caplog.at_level("WARNING", logger="tango.integrations.hf.workspace"):
            run = workspace.register_run(
                [kept, AddStep(a=3, b=40, step_name="changed"), AddStep(a=7, b=8, step_name="new")],
                name="main",
            )
        # A step the new graph does not mention stays: `tango run -s <step>` registers only
        # part of the graph, and must not make the run forget the rest.
        assert set(run.steps) == {"kept", "changed", "new", "gone"}
        assert set(workspace.registered_run("main").steps) == {"kept", "changed", "new", "gone"}
        # What changed is said, since a changed identity is a step that gets paid for again.
        assert "added: new" in caplog.text
        assert "changed identity: changed" in caplog.text

        import json

        record = json.loads(FakeHfApi.STORE["org/bucket"]["runs/main.json"])
        # Only what was replaced is kept, not the whole mapping once per relaunch.
        assert record["history"] == [
            {
                "start_date": record["history"][0]["start_date"],
                "added": ["new"],
                "replaced": {"changed": old_id},
            }
        ]
        assert workspace.run_step_ids("main") == record["steps"]
        assert workspace.run_step_ids("no-such-run") is None


class TestFlakyConnection:
    def test_a_dropped_connection_does_not_fail_a_cache_lookup(self, monkeypatch, workspace):
        # Seen on the first live run of the new executor: one SSL error while listing the
        # bucket failed a step before it was submitted.
        from tango.integrations.hf import common

        monkeypatch.setattr(common, "RETRY_BASE_SECONDS", 0.01)
        step = AddStep(a=1, b=2)
        step.ensure_result(workspace)

        original = FakeHfApi.list_bucket_tree
        failures = [ConnectionError("EOF occurred in violation of protocol")] * 2

        def flaky(self, *args, **kwargs):
            if failures:
                raise failures.pop()
            return original(self, *args, **kwargs)

        monkeypatch.setattr(FakeHfApi, "list_bucket_tree", flaky)
        assert workspace.step_cache._step_result_remote(step) is not None
        assert failures == []

    def test_running_part_of_a_graph_keeps_the_rest_of_the_run(self, workspace, caplog):
        first = AddStep(a=1, b=2, step_name="first")
        second = AddStep(a=3, b=4, step_name="second")
        workspace.register_run([first, second], name="main")

        with caplog.at_level("WARNING", logger="tango.integrations.hf.workspace"):
            workspace.register_run([second], name="main")
        assert "updating" not in caplog.text
        assert set(workspace.registered_run("main").steps) == {"first", "second"}

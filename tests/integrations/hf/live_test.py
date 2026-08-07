"""
Tests that talk to the real Hugging Face Hub.

These exist because the offline suite can only prove the integration is consistent with the
Hub's *documentation*. Everything here checks an assumption that would silently be wrong if the
docs and the implementation disagree — above all that ``If-None-Match: *`` on ``PutObject``
really does fail when the object exists, which is the only thing making the step lock a lock.

Nothing here costs money: buckets are free to create within the storage allowance, and no Job or
Inference Endpoint is ever started. A scratch bucket is created and deleted around each session.

To run::

    export HF_TOKEN=hf_...                    # a *write* token
    export HF_S3_ACCESS_KEY_ID=HFAK...        # Settings > Tokens > Generate S3 credentials
    export HF_S3_SECRET_ACCESS_KEY=...
    pytest -v tests/integrations/hf/live_test.py

Tests skip rather than fail when credentials are absent.
"""

import json
import os
import uuid

import pytest

from tango.integrations.hf.common import (
    Constants,
    HfBucketClient,
    HfBucketNotFound,
    HfStepLock,
)
from tango.integrations.hf.step_cache import HfBucketStepCache
from tango.integrations.hf.workspace import HfBucketWorkspace
from tango.step import Step
from tango.step_info import StepState

pytestmark = pytest.mark.hf_live

needs_token = pytest.mark.skipif(
    not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    reason="needs HF_TOKEN",
)
needs_s3 = pytest.mark.skipif(
    not (os.environ.get("HF_S3_ACCESS_KEY_ID") and os.environ.get("HF_S3_SECRET_ACCESS_KEY")),
    reason="needs HF_S3_ACCESS_KEY_ID and HF_S3_SECRET_ACCESS_KEY",
)


@Step.register("hf_live_add")
class AddStep(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, a: int, b: int) -> int:  # type: ignore[override]
        return a + b


@pytest.fixture(scope="session")
def bucket_path():
    """
    A scratch bucket under the token's own namespace, removed when the session ends.
    """
    from huggingface_hub import delete_bucket, get_token, whoami

    token = get_token() or os.environ.get("HF_TOKEN")
    namespace = whoami(token=token)["name"]
    bucket_id = f"{namespace}/tango-live-test-{uuid.uuid4().hex[:8]}"

    from huggingface_hub import create_bucket

    create_bucket(bucket_id, private=True, exist_ok=True, token=token)
    try:
        yield bucket_id
    finally:
        delete_bucket(bucket_id, missing_ok=True, token=token)


@pytest.fixture
def client(bucket_path):
    return HfBucketClient(bucket_path)


@needs_token
class TestBucketApiShapes:
    """
    Every assumption `HfBucketClient` makes about ``huggingface_hub``.
    """

    def test_put_and_get_json(self, client):
        client.put_json("probe/value.json", {"hello": "world"})
        assert client.get_json("probe/value.json") == {"hello": "world"}

    def test_read_after_write_is_immediate(self, client):
        # The lock depends on being able to read back what was just written.
        for index in range(5):
            key = f"probe/rw-{index}.json"
            client.put_json(key, {"index": index})
            assert client.get_json(key)["index"] == index

    def test_exists(self, client):
        client.put_json("probe/there.json", {})
        assert client.exists("probe/there.json")
        assert not client.exists("probe/not-there.json")

    def test_exists_is_not_fooled_by_a_longer_key(self, client):
        client.put_json("probe/name.json.bak", {})
        assert not client.exists("probe/name.json")

    def test_missing_object_raises_our_error(self, client):
        # If this leaks a different exception, `step_info` would propagate it instead of
        # treating the step as new.
        with pytest.raises(HfBucketNotFound):
            client.get_bytes("probe/definitely-absent.json")

    def test_listing_reports_files_and_directories(self, client):
        client.put_bytes("tree/a.txt", b"a")
        client.put_bytes("tree/nested/b.txt", b"b")

        recursive = {entry.path for entry in client.ls("tree", recursive=True)}
        assert client.key("tree/a.txt") in recursive
        assert client.key("tree/nested/b.txt") in recursive

        shallow = {entry.path: entry.type for entry in client.ls("tree", recursive=False)}
        assert shallow.get(client.key("tree/a.txt")) == "file"
        assert shallow.get(client.key("tree/nested")) == "directory"

    def test_directory_round_trip(self, client, tmp_path):
        source = tmp_path / "out"
        (source / "sub").mkdir(parents=True)
        (source / "one.txt").write_text("one")
        (source / "sub" / "two.txt").write_text("two")

        client.upload_dir("dirs/round-trip", source)
        target = tmp_path / "back"
        client.download_dir("dirs/round-trip", target)

        assert (target / "one.txt").read_text() == "one"
        assert (target / "sub" / "two.txt").read_text() == "two"

    def test_delete_and_delete_prefix(self, client):
        client.put_bytes("gone/one.txt", b"1")
        client.put_bytes("gone/two.txt", b"2")
        client.delete("gone/one.txt")
        assert not client.exists("gone/one.txt")
        assert client.exists("gone/two.txt")

        client.delete_prefix("gone")
        assert not client.exists("gone/two.txt")


@needs_token
@needs_s3
class TestLockAgainstTheRealGateway:
    """
    The claim the whole workspace rests on.
    """

    def test_conditional_put_is_exclusive(self, client):
        first = HfStepLock(client, "live-step-exclusive")
        second = HfStepLock(client, "live-step-exclusive")
        try:
            first.acquire(timeout=30)
            with pytest.raises(TimeoutError):
                second.acquire(timeout=5, poll_interval=1.0)
        finally:
            first.release()

    def test_the_lock_is_reusable_after_release(self, client):
        lock = HfStepLock(client, "live-step-reuse")
        lock.acquire(timeout=30)
        lock.release()

        again = HfStepLock(client, "live-step-reuse")
        again.acquire(timeout=30)
        again.release()

    def test_the_holder_record_is_readable(self, client):
        lock = HfStepLock(client, "live-step-record")
        try:
            lock.acquire(timeout=30)
            raw = lock._s3.get_object(Bucket=lock._bucket, Key=lock._key)["Body"].read()
            record = json.loads(raw)
            assert record["step"] == "live-step-record"
            assert record["heartbeat"] > 0
        finally:
            lock.release()

    def test_a_stale_lock_is_broken(self, client):
        stale = HfStepLock(client, "live-step-stale", ttl=1.0, heartbeat_interval=3600.0)
        stale.acquire(timeout=30)
        # Heartbeat interval is longer than the run, so the record goes stale on its own.
        import time

        time.sleep(2.0)

        breaker = HfStepLock(client, "live-step-stale", ttl=1.0)
        try:
            breaker.acquire(timeout=30, poll_interval=1.0)
        finally:
            breaker.release()
            stale._held = False


@needs_token
@needs_s3
class TestProjectSyncAgainstTheRealHub:
    """
    The executor mirrors your project into the bucket with exclusions. Submitting no Job, this
    costs nothing, but it is the check that stops a routine run uploading a multi-gigabyte
    virtualenv.
    """

    def test_default_exclusions_keep_the_junk_out(self, bucket_path, tmp_path, monkeypatch):
        from tango.integrations.hf import step_cache as step_cache_module
        from tango.integrations.hf.executor import HfJobsExecutor
        from tango.workspaces import remote_workspace

        monkeypatch.setattr(step_cache_module, "tango_cache_dir", lambda: tmp_path / "cache")
        monkeypatch.setattr(remote_workspace, "tango_cache_dir", lambda: tmp_path / "cache")

        project = tmp_path / "project"
        for relative in (
            "main.py",
            "pkg/mod.py",
            "pkg/__pycache__/mod.pyc",
            ".venv/lib/site-packages/torch/big.so",
            ".git/objects/ab/cdef",
            "data/keep.jsonl",
        ):
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")

        workspace = HfBucketWorkspace(f"{bucket_path}/sync")
        executor = HfJobsExecutor(workspace, project_dir=str(project))
        volume = executor._sync_project()

        client = HfBucketClient(f"{bucket_path}/sync")
        uploaded = sorted(
            entry.path[len(volume.path) + 1 :]
            for entry in client.api.list_bucket_tree(
                client.bucket_id, prefix=volume.path, recursive=True
            )
        )
        assert uploaded == ["data/keep.jsonl", "main.py", "pkg/mod.py"]
        assert volume.type == "bucket" and volume.read_only


@needs_token
@needs_s3
class TestWorkspaceEndToEnd:
    def test_step_round_trip(self, bucket_path, tmp_path, monkeypatch):
        from tango.integrations.hf import step_cache as step_cache_module
        from tango.workspaces import remote_workspace

        monkeypatch.setattr(step_cache_module, "tango_cache_dir", lambda: tmp_path / "cache-a")
        monkeypatch.setattr(remote_workspace, "tango_cache_dir", lambda: tmp_path / "cache-a")

        workspace = HfBucketWorkspace(f"{bucket_path}/e2e")
        step = AddStep(a=40, b=2)
        step.ensure_result(workspace)

        assert workspace.step_info(step).state == StepState.COMPLETED
        assert workspace.step_cache[step] == 42

        # A second machine: same bucket, empty local cache.
        monkeypatch.setattr(step_cache_module, "tango_cache_dir", lambda: tmp_path / "cache-b")
        monkeypatch.setattr(remote_workspace, "tango_cache_dir", lambda: tmp_path / "cache-b")
        elsewhere = HfBucketWorkspace(f"{bucket_path}/e2e")
        assert AddStep(a=40, b=2) in elsewhere.step_cache
        assert elsewhere.step_cache[AddStep(a=40, b=2)] == 42

    def test_run_registration(self, bucket_path, tmp_path, monkeypatch):
        from tango.integrations.hf import step_cache as step_cache_module
        from tango.workspaces import remote_workspace

        monkeypatch.setattr(step_cache_module, "tango_cache_dir", lambda: tmp_path / "cache")
        monkeypatch.setattr(remote_workspace, "tango_cache_dir", lambda: tmp_path / "cache")

        workspace = HfBucketWorkspace(f"{bucket_path}/runs-test")
        run = workspace.register_run([AddStep(a=1, b=2)], name="live-run")
        assert workspace.registered_run("live-run").start_date == run.start_date
        assert "live-run" in workspace.registered_runs()

    def test_bucket_layout(self, bucket_path, tmp_path, monkeypatch):
        from tango.integrations.hf import step_cache as step_cache_module
        from tango.workspaces import remote_workspace

        monkeypatch.setattr(step_cache_module, "tango_cache_dir", lambda: tmp_path / "cache")
        monkeypatch.setattr(remote_workspace, "tango_cache_dir", lambda: tmp_path / "cache")

        workspace = HfBucketWorkspace(f"{bucket_path}/layout")
        step = AddStep(a=7, b=7)
        step.ensure_result(workspace)

        client = HfBucketClient(f"{bucket_path}/layout")
        assert client.exists(Constants.SETTINGS_FNAME)
        assert client.exists(Constants.step_info_key(step.unique_id))
        artifact = Constants.step_artifact_name(step)
        assert not client.exists(f"{artifact}/{Constants.UNCOMMITTED_FNAME}")
        assert isinstance(workspace.step_cache, HfBucketStepCache)
        assert len(workspace.step_cache) == 1


# Unlike everything above, this one costs money -- an Inference Endpoint bills by the hour
# while it is up. It needs a second, explicit opt-in so it can never run by accident.
needs_paid = pytest.mark.skipif(
    os.environ.get("TANGO_HF_PAID_TESTS") != "1",
    reason="costs money; set TANGO_HF_PAID_TESTS=1 to run",
)


@needs_token
@needs_paid
class TestEndpointBatchStep:
    """
    Verified end-to-end for about $0.001: gpt2 on the cheapest CPU instance stays up for
    roughly a minute. The endpoint is deleted afterwards whatever happens.
    """

    def test_create_generate_pause(self, tmp_path):
        from huggingface_hub import delete_inference_endpoint, get_inference_endpoint

        from tango.integrations.hf.endpoint import EndpointBatchStep

        name = f"tango-test-{uuid.uuid4().hex[:8]}"
        step = EndpointBatchStep(
            prompts=["The capital of France is", "The opposite of hot is"],
            endpoint_name=name,
            repository="openai-community/gpt2",
            task="text-generation",
            accelerator="cpu",
            vendor="aws",
            region="us-east-1",
            # Names go stale: `intel-icl` from Hugging Face's docs no longer exists.
            instance_type="intel-spr",
            instance_size="x1",
            generation_kwargs={"max_new_tokens": 8, "do_sample": False},
            max_concurrency=2,
            timeout=900.0,
        )
        try:
            results = step.result()
            assert len(results) == 2
            assert all(isinstance(text, str) and text for text in results)
            # Paused in a `finally`, so it stops billing even if generation had failed.
            assert get_inference_endpoint(name).status in {"paused", "pending"}
        finally:
            delete_inference_endpoint(name)

    def test_missing_hardware_is_a_clear_error(self):
        from tango.common.exceptions import ConfigurationError
        from tango.integrations.hf.endpoint import EndpointBatchStep

        step = EndpointBatchStep(
            prompts=["x"],
            endpoint_name=f"tango-test-{uuid.uuid4().hex[:8]}",
            repository="openai-community/gpt2",
        )
        # Without this check huggingface_hub raises an opaque TypeError instead.
        with pytest.raises(ConfigurationError, match="instance_size and instance_type"):
            step.result()

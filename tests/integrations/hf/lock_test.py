import json
import time

import pytest

from tango.integrations.hf.common import HfBucketClient, HfStepLock

from .fake_hub import FakeJob, install_fakes


@pytest.fixture
def s3(monkeypatch, tmp_path):
    return install_fakes(monkeypatch, tmp_path)


@pytest.fixture
def client(s3):
    return HfBucketClient("org/bucket")


LOCK_KEY = "tango-step-step123-lock"


def _holder(s3):
    return json.loads(s3.objects[LOCK_KEY].decode("utf-8"))


class TestHfStepLock:
    def test_acquire_and_release(self, client, s3):
        lock = HfStepLock(client, "step123", s3_client=s3)
        lock.acquire(timeout=5)
        assert LOCK_KEY in s3.objects
        assert _holder(s3)["step"] == "step123"

        lock.release()
        assert LOCK_KEY not in s3.objects

    def test_acquire_is_idempotent(self, client, s3):
        lock = HfStepLock(client, "step123", s3_client=s3)
        lock.acquire(timeout=5)
        before = s3.put_calls
        lock.acquire(timeout=5)
        assert s3.put_calls == before
        lock.release()

    def test_second_holder_is_locked_out(self, client, s3):
        first = HfStepLock(client, "step123", s3_client=s3)
        first.acquire(timeout=5)

        second = HfStepLock(client, "step123", s3_client=s3)
        with pytest.raises(TimeoutError, match="step123"):
            second.acquire(timeout=1, poll_interval=0.05)

        first.release()
        # Once the first holder lets go, the second gets in.
        second.acquire(timeout=5)
        assert LOCK_KEY in s3.objects
        second.release()

    def test_a_different_step_is_a_different_lock(self, client, s3):
        first = HfStepLock(client, "step123", s3_client=s3)
        second = HfStepLock(client, "step456", s3_client=s3)
        first.acquire(timeout=5)
        second.acquire(timeout=5)
        assert len(s3.objects) == 2
        first.release()
        second.release()

    def test_stale_local_lock_is_broken_after_the_ttl(self, client, s3):
        dead = HfStepLock(client, "step123", s3_client=s3, ttl=60.0)
        dead.acquire(timeout=5)
        # Simulate a process that died without releasing: its heartbeat stops advancing.
        record = _holder(s3)
        record["heartbeat"] = time.time() - 3600
        s3.objects[LOCK_KEY] = json.dumps(record).encode("utf-8")

        live = HfStepLock(client, "step123", s3_client=s3, ttl=60.0)
        live.acquire(timeout=5, poll_interval=0.05)
        assert _holder(s3)["pid"] == record["pid"]  # same process here, but the record is fresh
        assert _holder(s3)["heartbeat"] > record["heartbeat"]
        live.release()

    def test_a_fresh_local_heartbeat_is_respected(self, client, s3):
        holder = HfStepLock(client, "step123", s3_client=s3, ttl=3600.0)
        holder.acquire(timeout=5)

        other = HfStepLock(client, "step123", s3_client=s3, ttl=3600.0)
        with pytest.raises(TimeoutError):
            other.acquire(timeout=1, poll_interval=0.05)
        holder.release()

    def test_lock_held_by_a_finished_job_is_broken(self, client, s3, monkeypatch):
        monkeypatch.setenv("JOB_ID", "job-abc")
        dead = HfStepLock(client, "step123", s3_client=s3)
        dead.acquire(timeout=5)
        assert _holder(s3)["job_id"] == "job-abc"
        monkeypatch.delenv("JOB_ID")

        import huggingface_hub

        monkeypatch.setattr(
            huggingface_hub, "inspect_job", lambda **kw: FakeJob(kw["job_id"], "COMPLETED")
        )

        live = HfStepLock(client, "step123", s3_client=s3)
        live.acquire(timeout=5, poll_interval=0.05)
        assert _holder(s3)["job_id"] is None
        live.release()

    def test_lock_held_by_a_running_job_is_respected(self, client, s3, monkeypatch):
        monkeypatch.setenv("JOB_ID", "job-abc")
        running = HfStepLock(client, "step123", s3_client=s3)
        running.acquire(timeout=5)
        monkeypatch.delenv("JOB_ID")

        import huggingface_hub

        monkeypatch.setattr(
            huggingface_hub, "inspect_job", lambda **kw: FakeJob(kw["job_id"], "RUNNING")
        )

        other = HfStepLock(client, "step123", s3_client=s3)
        with pytest.raises(TimeoutError):
            other.acquire(timeout=1, poll_interval=0.05)

    def test_an_unreachable_jobs_api_does_not_break_the_lock(self, client, s3, monkeypatch):
        """
        A transient failure asking about the holder must not be read as "the holder is dead" —
        that would hand the same step to two runners at once.
        """
        monkeypatch.setenv("JOB_ID", "job-abc")
        holder = HfStepLock(client, "step123", s3_client=s3)
        holder.acquire(timeout=5)
        monkeypatch.delenv("JOB_ID")

        import huggingface_hub

        def explode(**kwargs):
            raise ConnectionError("the Hub is down")

        monkeypatch.setattr(huggingface_hub, "inspect_job", explode)

        other = HfStepLock(client, "step123", s3_client=s3)
        with pytest.raises(TimeoutError):
            other.acquire(timeout=1, poll_interval=0.05)

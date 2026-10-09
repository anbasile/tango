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

    def test_acquired_at_stays_put_while_the_heartbeat_moves(self, client, s3):
        # Both used to be rewritten together on every beat, so the record could not say how
        # long a lock had been held.
        lock = HfStepLock(client, "step123", s3_client=s3, heartbeat_interval=0.05)
        lock.acquire(timeout=5)
        first = _holder(s3)
        deadline = time.time() + 5
        while _holder(s3)["heartbeat"] == first["heartbeat"] and time.time() < deadline:
            time.sleep(0.02)
        later = _holder(s3)
        assert later["heartbeat"] > first["heartbeat"]
        assert later["acquired_at"] == first["acquired_at"]
        lock.release()

    def test_no_heartbeat_writes_the_lock_back_after_release(self, client, s3):
        lock = HfStepLock(client, "step123", s3_client=s3, heartbeat_interval=0.01)
        lock.acquire(timeout=5)
        time.sleep(0.05)
        lock.release()
        time.sleep(0.05)
        assert LOCK_KEY not in s3.objects

    def test_a_failed_delete_is_not_silent(self, client, s3, caplog):
        # Inside a Job only warnings and above are visible, and a lock that stays behind
        # blocks the step for every later run.
        lock = HfStepLock(client, "step123", s3_client=s3)
        lock.acquire(timeout=5)
        s3.fail_deletes = True
        with caplog.at_level("WARNING", logger="tango.integrations.hf.common"):
            lock.release()
        assert "Failed to delete the lock" in caplog.text
        s3.fail_deletes = False

    def test_the_job_is_looked_up_in_its_own_namespace(self, client, s3, monkeypatch):
        from tango.integrations.hf.common import JOB_NAMESPACE_ENV_VAR

        monkeypatch.setenv("JOB_ID", "job-abc")
        monkeypatch.setenv(JOB_NAMESPACE_ENV_VAR, "my-org")
        dead = HfStepLock(client, "step123", s3_client=s3)
        dead.acquire(timeout=5)
        monkeypatch.delenv("JOB_ID")
        monkeypatch.delenv(JOB_NAMESPACE_ENV_VAR)

        import huggingface_hub

        asked = []

        def inspect_job(**kwargs):
            asked.append(kwargs)
            return FakeJob(kwargs["job_id"], "CANCELED")

        monkeypatch.setattr(huggingface_hub, "inspect_job", inspect_job)
        live = HfStepLock(client, "step123", s3_client=s3)
        live.acquire(timeout=5, poll_interval=0.05)
        assert asked[0]["namespace"] == "my-org"
        live.release()


class TestBreakIfDead:
    def _held_by_job(self, client, s3, monkeypatch, job_id="job-abc"):
        monkeypatch.setenv("JOB_ID", job_id)
        lock = HfStepLock(client, "step123", s3_client=s3)
        lock.acquire(timeout=5)
        monkeypatch.delenv("JOB_ID")
        return lock

    def test_no_lock_means_free(self, client, s3):
        assert HfStepLock(client, "step123", s3_client=s3).break_if_dead() is True

    def test_the_named_job_loses_its_lock_without_asking(self, client, s3, monkeypatch):
        holder = self._held_by_job(client, s3, monkeypatch)
        import huggingface_hub

        def explode(**kwargs):
            raise AssertionError("the caller has already seen this job end")

        monkeypatch.setattr(huggingface_hub, "inspect_job", explode)
        assert HfStepLock(client, "step123", s3_client=s3).break_if_dead("job-abc") is True
        assert LOCK_KEY not in s3.objects
        del holder

    def test_another_live_holder_keeps_the_lock(self, client, s3, monkeypatch):
        holder = self._held_by_job(client, s3, monkeypatch, job_id="someone-else")
        import huggingface_hub

        monkeypatch.setattr(
            huggingface_hub, "inspect_job", lambda **kw: FakeJob(kw["job_id"], "RUNNING")
        )
        assert HfStepLock(client, "step123", s3_client=s3).break_if_dead("job-abc") is False
        assert LOCK_KEY in s3.objects
        del holder

    def test_a_lock_taken_over_in_between_is_left_alone(self, client, s3, monkeypatch):
        # Between judging a holder dead and deleting, another run may have taken the lock.
        holder = self._held_by_job(client, s3, monkeypatch)
        lock = HfStepLock(client, "step123", s3_client=s3)
        judged_dead = lock.holder()
        newcomer = dict(judged_dead, job_id="job-new", acquired_at="2030-01-01T00:00:00+00:00")
        s3.objects[LOCK_KEY] = json.dumps(newcomer).encode("utf-8")

        assert lock._force_release(expected=judged_dead) is False
        assert _holder(s3)["job_id"] == "job-new"
        del holder

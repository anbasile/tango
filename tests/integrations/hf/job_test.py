import sys

import pytest

from tango.integrations.hf import job
from tango.integrations.hf.common import HfBucketClient

from .fake_hub import FakeHfApi, install_fakes


@pytest.fixture
def bucket(monkeypatch, tmp_path):
    install_fakes(monkeypatch, tmp_path / "cache")
    monkeypatch.setenv("JOB_ID", "job-42")
    HfBucketClient("org/bucket")
    return FakeHfApi.STORE["org/bucket"]


def run(*command, result_of=None, owner="step-abc"):
    argv = ["--bucket", "org/bucket", "--log-owner", owner]
    if result_of:
        argv += ["--result-of", result_of]
    return job.main([*argv, "--", sys.executable, "-c", *command])


class TestJobEntryPoint:
    def test_the_output_is_kept_in_the_bucket(self, bucket, capfd):
        # The Hub returned nothing for the jobs whose logs mattered.
        code = run("import sys; print('to stdout'); print('to stderr', file=sys.stderr)")

        assert code == 0
        log = bucket["logs/step-abc/job-42.log"].decode("utf-8")
        assert "to stdout" in log and "to stderr" in log
        assert "disk at" in log
        assert "command exited with code 0" in log
        # And it still goes to the job's own output.
        assert "to stdout" in capfd.readouterr().out

    def test_the_log_is_uploaded_while_the_command_runs(self, bucket, monkeypatch, tmp_path):
        # A job that hangs or is killed never reaches the final upload.
        monkeypatch.setenv(job.LOG_UPLOAD_INTERVAL_ENV_VAR, "0.05")
        seen = tmp_path / "seen"
        code = run(
            "import time, pathlib; print('early line', flush=True); time.sleep(1.0); "
            f"pathlib.Path({str(seen)!r}).write_text('x')",
        )
        assert code == 0
        assert "early line" in bucket["logs/step-abc/job-42.log"].decode("utf-8")

        uploads = []
        original = FakeHfApi.batch_bucket_files

        def record(self, bucket_id, add=None, delete=None, **kwargs):
            uploads.extend(remote for _, remote in add or [])
            return original(self, bucket_id, add=add, delete=delete, **kwargs)

        monkeypatch.setattr(FakeHfApi, "batch_bucket_files", record)
        run("import time; print('one', flush=True); time.sleep(0.5); print('two')")
        assert uploads.count("logs/step-abc/job-42.log") >= 2

    def test_the_exit_code_is_the_commands(self, bucket):
        assert run("raise SystemExit(7)") == 7
        assert "command exited with code 7" in bucket["logs/step-abc/job-42.log"].decode("utf-8")

    def test_success_without_a_result_is_a_failure(self, bucket):
        # Jobs once exited 0 without having run their step, and were trusted.
        assert run("print('nothing done')", result_of="abc") == job.NO_RESULT_EXIT_CODE
        assert "has no result" in bucket["logs/step-abc/job-42.log"].decode("utf-8")

    def test_a_lock_is_not_a_result(self, bucket):
        bucket["tango-step-abc-lock"] = b"{}"
        assert run("print('nothing done')", result_of="abc") == job.NO_RESULT_EXIT_CODE

    def test_an_upload_in_flight_is_not_a_result(self, bucket):
        bucket["tango-step-abc/.uncommitted"] = b""
        bucket["tango-step-abc/cache-metadata.json"] = b"{}"
        assert run("print('half way')", result_of="abc") == job.NO_RESULT_EXIT_CODE

    def test_success_with_a_result(self, bucket):
        bucket["tango-step-abc/cache-metadata.json"] = b"{}"
        assert run("print('done')", result_of="abc") == 0

    def test_a_failed_log_upload_does_not_fail_the_step(self, bucket, monkeypatch):
        def explode(self, *args, **kwargs):
            raise ConnectionError("the Hub is down")

        monkeypatch.setattr(FakeHfApi, "batch_bucket_files", explode)
        assert run("print('still fine')") == 0

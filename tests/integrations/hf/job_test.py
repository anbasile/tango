import os
import signal
import sys
import threading

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
        monkeypatch.setattr(job, "LOG_RETRY_BUDGET", 0.05)
        assert run("print('still fine')") == 0

    def test_a_cancelled_job_still_uploads_its_log(self, bucket):
        # A cancelled job gets SIGTERM. The command is stopped and the log goes up.
        timer = threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGTERM))
        timer.start()
        try:
            code = run("import time; print('before the signal', flush=True); time.sleep(60)")
        finally:
            timer.cancel()

        assert code == 128 + signal.SIGTERM
        log = bucket["logs/step-abc/job-42.log"].decode("utf-8")
        assert "before the signal" in log
        assert f"received signal {int(signal.SIGTERM)}" in log
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL, "handlers are put back"

    def test_a_log_copy_that_cannot_be_written_does_not_stop_the_command(
        self, bucket, monkeypatch, capfd
    ):
        # A full disk is what the wrapper is there to report; it must not die of it.
        real_open = open

        class FullDisk:
            def __init__(self, file):
                self._file = file

            def write(self, text):
                raise OSError(28, "No space left on device")

            def __getattr__(self, name):
                return getattr(self._file, name)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self._file.close()

        def fake_open(path, mode="r", *args, **kwargs):
            file = real_open(path, mode, *args, **kwargs)
            return FullDisk(file) if mode == "w" else file

        monkeypatch.setattr(job, "open", fake_open, raising=False)
        assert run("print('still printed')") == 0
        out = capfd.readouterr().out
        assert "still printed" in out
        assert out.count("could not write to the log copy") == 1

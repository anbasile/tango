"""
What every Job submitted by :class:`~tango.integrations.hf.executor.HfJobsExecutor` runs its
command through::

    python -m tango.integrations.hf.job --bucket <ns>/<bucket> --log-owner <step id> \\
        [--result-of <step id>] -- tango --called-by-executor run ...

It does three things the bare command did not.

**It keeps the output.** The Hub returns roughly the last thousand lines of a job's log, and
for some jobs nothing at all. The output is copied to ``logs/<owner>/<job id>.log`` in the
workspace bucket every few minutes and when the command ends, so a job that hangs or is killed
leaves behind what it printed up to then.

**It checks the result.** A command that exits 0 has not necessarily run its step. With
``--result-of``, a clean exit without the step's result in the bucket becomes exit code
:data:`NO_RESULT_EXIT_CODE`, so the job ends in stage ``ERROR`` and not ``COMPLETED``.

**It reports the disk**, at the start and at the end. A job that fills its disk is evicted
without a word in its log.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, List, Optional, Sequence

#: Seconds between uploads of the log. Set in the job's environment to change it.
LOG_UPLOAD_INTERVAL_ENV_VAR = "TANGO_HF_LOG_UPLOAD_INTERVAL"
DEFAULT_LOG_UPLOAD_INTERVAL = 300.0

#: The exit code of a job whose command succeeded without producing the step's result.
NO_RESULT_EXIT_CODE = 3

#: Seconds an upload of the log keeps trying when the Hub does not answer. Short on purpose:
#: the next upload carries everything again, and the step must not wait on its log.
LOG_RETRY_BUDGET = 30.0

#: Only the end of a very long log is kept.
MAX_LOG_BYTES = 20 * 1024 * 1024

_PREFIX = "[tango-hf]"


def _say(message: str) -> None:
    print(f"{_PREFIX} {message}", flush=True)


def _disk(path: str = "/tmp") -> str:
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:  # pragma: no cover - depends on the container
        return f"disk at {path}: unknown ({exc})"
    gb = 1024**3
    return (
        f"disk at {path}: {usage.used / gb:.1f} GB used of {usage.total / gb:.1f} GB, "
        f"{usage.free / gb:.1f} GB free"
    )


class _LogUploader:
    """
    Copies the log file to the bucket, on a timer and on demand. Never raises: losing the
    log must not fail the step.
    """

    def __init__(self, bucket: str, key: str, path: Path, interval: float) -> None:
        self._bucket = bucket
        self._key = key
        self._path = path
        self._interval = interval
        self._client: Optional[Any] = None
        self._uploaded_size = -1
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="tango-hf-log", daemon=True)

    @property
    def client(self) -> Any:
        if self._client is None:
            from .common import HfBucketClient

            self._client = HfBucketClient(self._bucket, create=False)
        return self._client

    def upload(self) -> None:
        with self._lock:
            try:
                size = self._path.stat().st_size
                if size == self._uploaded_size:
                    return
                with open(self._path, "rb") as log:
                    if size > MAX_LOG_BYTES:
                        log.seek(size - MAX_LOG_BYTES)
                    data = log.read()
                from .common import hub_call

                client = self.client
                hub_call(
                    client.api.batch_bucket_files,
                    client.bucket_id,
                    add=[(data, client.key(self._key))],
                    budget=LOG_RETRY_BUDGET,
                )
                self._uploaded_size = size
            except Exception as exc:
                _say(f"could not upload the log: {type(exc).__name__}: {exc}")

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.upload()

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> None:
        self._stop.set()
        self.upload()


def _has_result(client: Any, step_id: str) -> bool:
    from .common import Constants, hub_call

    artifact = Constants.step_artifact_name(step_id)
    entries = hub_call(client.ls_dir, artifact, budget=120.0)
    uncommitted = client.key(f"{artifact}/{Constants.UNCOMMITTED_FNAME}")
    return bool(entries) and all(entry.path != uncommitted for entry in entries)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m {__name__}", description=__doc__)
    parser.add_argument("--bucket", required=True, help="<namespace>/<bucket>[/<prefix>]")
    parser.add_argument("--log-owner", required=True, help="Folder of the log under logs/.")
    parser.add_argument("--result-of", help="Unique id of the step whose result must exist.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command: List[str] = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given after --")

    from .common import Constants

    job_id = os.environ.get("JOB_ID") or f"local-{os.getpid()}"
    interval = float(os.environ.get(LOG_UPLOAD_INTERVAL_ENV_VAR) or DEFAULT_LOG_UPLOAD_INTERVAL)
    log_dir = Path(tempfile.mkdtemp(prefix="tango-hf-job-"))
    log_path = log_dir / "job.log"
    log_key = Constants.job_log_key(args.log_owner, job_id)
    uploader = _LogUploader(args.bucket, log_key, log_path, interval)

    with open(log_path, "w", encoding="utf-8", errors="replace") as log:

        def emit(line: str) -> None:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()

        emit(f"{_PREFIX} job {job_id}, flavor {os.environ.get('TANGO_HF_FLAVOR', 'unknown')}\n")
        emit(f"{_PREFIX} log kept at hf://buckets/{args.bucket.strip('/')}/{log_key}\n")
        emit(f"{_PREFIX} {_disk()}\n")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )

        def forward(signum: int, frame: Any) -> None:
            # A cancelled job gets SIGTERM. Pass it on, then fall through to the final upload.
            emit(f"{_PREFIX} received signal {signum}; stopping the command\n")
            process.terminate()

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, forward)
            signal.signal(signal.SIGINT, forward)

        uploader.start()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                emit(line)
            code = process.wait()
        finally:
            emit(f"{_PREFIX} {_disk()}\n")

        if code < 0:
            code = 128 - code
        emit(f"{_PREFIX} command exited with code {code}\n")

        if code == 0 and args.result_of:
            try:
                found = _has_result(uploader.client, args.result_of)
            except Exception as exc:
                # Could not look. The executor checks again from outside.
                emit(f"{_PREFIX} could not check for the result: {type(exc).__name__}: {exc}\n")
                found = True
            if not found:
                emit(
                    f"{_PREFIX} the command succeeded but step {args.result_of} has no result "
                    f"in the bucket; failing the job\n"
                )
                code = NO_RESULT_EXIT_CODE

    uploader.finish()
    shutil.rmtree(log_dir, ignore_errors=True)
    return code


if __name__ == "__main__":
    sys.exit(main())

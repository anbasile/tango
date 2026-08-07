"""
Classes and utility functions shared by the Hugging Face workspace, step cache and executor.
"""

import atexit
import json
import logging
import os
import re
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tango.common.aliases import PathOrStr
from tango.common.exceptions import (
    ConfigurationError,
    IntegrationMissingError,
    TangoError,
)
from tango.common.remote_utils import RemoteConstants
from tango.common.util import utc_now_datetime
from tango.step import Step, StepResources
from tango.step_info import StepInfo

logger = logging.getLogger(__name__)


class Constants(RemoteConstants):
    """
    Object keys used inside the bucket, on top of the shared
    :class:`~tango.common.remote_utils.RemoteConstants`.
    """

    STEP_INFO_DIR: str = "stepinfo"
    RUNS_DIR: str = "runs"
    SETTINGS_FNAME: str = "settings.json"
    UNCOMMITTED_FNAME: str = ".uncommitted"
    RUN_LOG_FNAME: str = "out.log"

    @classmethod
    def step_info_key(cls, step_or_unique_id: Union[str, Step, StepInfo]) -> str:
        unique_id = (
            step_or_unique_id if isinstance(step_or_unique_id, str) else step_or_unique_id.unique_id
        )
        return f"{cls.STEP_INFO_DIR}/{unique_id}.json"

    @classmethod
    def run_key(cls, name: str) -> str:
        return f"{cls.RUNS_DIR}/{name}.json"

    @classmethod
    def run_log_key(cls, name: str) -> str:
        return f"{cls.RUNS_DIR}/{name}.log"


class HfBucketNotFound(TangoError):
    """
    Raised when an object is not present in the bucket.
    """


def _not_found_errors() -> Tuple[type, ...]:
    """
    The exception types that mean "this object isn't there", as opposed to a transient failure.

    Getting this distinction wrong is not cosmetic: :meth:`HfBucketWorkspace.step_info` treats a
    missing record as "this step has never run" and creates a fresh one, so swallowing a network
    error as a miss would silently discard a step's history.
    """
    errors: List[type] = [FileNotFoundError, KeyError]
    try:
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

        errors += [EntryNotFoundError, RepositoryNotFoundError]
    except ImportError:  # pragma: no cover - depends on the installed huggingface_hub
        pass
    return tuple(errors)


def _is_not_found(exc: BaseException) -> bool:
    if isinstance(exc, _not_found_errors()):
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 404


def split_bucket_path(path: str) -> Tuple[str, str]:
    """
    Split ``"<namespace>/<bucket>[/<prefix>]"`` into the bucket id and the prefix within it.

    :raises ConfigurationError: If the path doesn't name both a namespace and a bucket.

    :examples:

    .. testcode::

        from tango.integrations.hf.common import split_bucket_path

        print(split_bucket_path("my-org/my-bucket/experiments"))

    .. testoutput::

        ('my-org/my-bucket', 'experiments')
    """
    parts = [part for part in path.strip("/").split("/") if part]
    if len(parts) < 2:
        raise ConfigurationError(
            f"Expected a bucket path of the form '<namespace>/<bucket>[/<prefix>]', got '{path}'."
        )
    return "/".join(parts[:2]), "/".join(parts[2:])


class HfBucketClient:
    """
    A thin wrapper around the ``huggingface_hub`` bucket API, scoped to one bucket and
    (optionally) one prefix inside it.

    :param bucket_path: ``"<namespace>/<bucket>[/<prefix>]"``.
    :param token: A Hugging Face token. Falls back to the ambient login.
    :param create: Create the bucket if it doesn't exist yet.
    """

    def __init__(self, bucket_path: str, token: Optional[str] = None, create: bool = True):
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError:
            raise IntegrationMissingError("hf", dependencies={"huggingface_hub"})

        self.bucket_id, self.prefix = split_bucket_path(bucket_path)
        self.namespace = self.bucket_id.split("/")[0]
        self._api = HfApi(token=token)
        self._token = token

        if create:
            self._api.create_bucket(self.bucket_id, exist_ok=True)
        self._ensure_settings()

    @property
    def api(self) -> Any:
        return self._api

    @property
    def token(self) -> Optional[str]:
        return self._token

    def key(self, *parts: str) -> str:
        """
        Join ``parts`` onto the client's prefix to give a key within the bucket.
        """
        return "/".join(part.strip("/") for part in (self.prefix, *parts) if part.strip("/"))

    def url(self, artifact: Optional[str] = None) -> str:
        """
        The ``hf://buckets/...`` URL of an object, or of the workspace root.
        """
        key = self.key(artifact) if artifact is not None else self.key()
        return f"hf://buckets/{self.bucket_id}" + (f"/{key}" if key else "")

    def _ensure_settings(self) -> None:
        if not self.exists(Constants.SETTINGS_FNAME):
            self.put_json(Constants.SETTINGS_FNAME, {"version": 1})

    def ls(self, prefix: str = "", recursive: bool = True) -> List[Any]:
        """
        List entries under ``prefix``. Returned ``path`` values are keys within the bucket,
        prefix included.
        """
        return list(
            self._api.list_bucket_tree(self.bucket_id, prefix=self.key(prefix), recursive=recursive)
        )

    def exists(self, path: str) -> bool:
        full = self.key(path)
        # `prefix` is a plain string match, so a file's own key is a valid prefix for itself.
        # Compare exactly, otherwise "foo.json" would be reported present by "foo.json.bak".
        return any(item.path == full for item in self.ls(path, recursive=True))

    def put_bytes(self, path: str, data: bytes) -> None:
        self._api.batch_bucket_files(self.bucket_id, add=[(data, self.key(path))])

    def get_bytes(self, path: str) -> bytes:
        """
        :raises HfBucketNotFound: If the object isn't in the bucket.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            local = Path(tmp_dir) / "object"
            try:
                self._api.download_bucket_files(
                    self.bucket_id, files=[(self.key(path), str(local))]
                )
            except Exception as exc:
                if _is_not_found(exc):
                    raise HfBucketNotFound(self.url(path)) from exc
                raise
            if not local.is_file():
                raise HfBucketNotFound(self.url(path))
            return local.read_bytes()

    def put_json(self, path: str, obj: Any) -> None:
        self.put_bytes(path, json.dumps(obj).encode("utf-8"))

    def get_json(self, path: str) -> Any:
        return json.loads(self.get_bytes(path).decode("utf-8"))

    def upload_dir(self, path: str, local_dir: PathOrStr) -> None:
        self._api.sync_bucket(str(local_dir), f"hf://buckets/{self.bucket_id}/{self.key(path)}")

    def download_dir(self, path: str, local_dir: PathOrStr) -> None:
        self._api.sync_bucket(f"hf://buckets/{self.bucket_id}/{self.key(path)}", str(local_dir))

    def delete(self, *paths: str) -> None:
        keys = [self.key(path) for path in paths]
        if keys:
            self._api.batch_bucket_files(self.bucket_id, delete=keys)

    def delete_prefix(self, prefix: str) -> None:
        keys = [item.path for item in self.ls(prefix, recursive=True) if item.type == "file"]
        if keys:
            self._api.batch_bucket_files(self.bucket_id, delete=keys)


#
# Hardware.
#


@dataclass(frozen=True)
class Flavor:
    """
    One Hugging Face Jobs hardware flavor.
    """

    name: str
    vcpu: int
    ram_gb: int
    gpus: int
    accelerator: Optional[str]
    cost_per_hour: float


# Mirrors `hf jobs hardware`. Kept as a table rather than fetched because matching needs the
# structured fields, and `list_jobs_hardware()` is only used to warn when this drifts.
FLAVORS: Tuple[Flavor, ...] = (
    Flavor("cpu-basic", 2, 16, 0, None, 0.01),
    Flavor("cpu-upgrade", 8, 32, 0, None, 0.03),
    Flavor("cpu-xl", 16, 124, 0, None, 1.00),
    Flavor("cpu-performance", 32, 256, 0, None, 1.90),
    Flavor("t4-small", 4, 15, 1, "T4", 0.40),
    Flavor("t4-medium", 8, 30, 1, "T4", 0.60),
    Flavor("l4x1", 8, 30, 1, "L4", 0.80),
    Flavor("a10g-small", 4, 15, 1, "A10G", 1.00),
    Flavor("a10g-large", 12, 46, 1, "A10G", 1.50),
    Flavor("l40sx1", 8, 62, 1, "L40S", 1.80),
    Flavor("a100-large", 12, 142, 1, "A100", 2.50),
    Flavor("rtx-pro-6000", 23, 256, 1, "RTX PRO 6000", 2.75),
    Flavor("a10g-largex2", 24, 92, 2, "A10G", 3.00),
    Flavor("l4x4", 48, 186, 4, "L4", 3.80),
    Flavor("h200", 23, 256, 1, "H200", 5.00),
    Flavor("a10g-largex4", 48, 184, 4, "A10G", 5.00),
    Flavor("rtx-pro-6000x2", 46, 512, 2, "RTX PRO 6000", 5.50),
    Flavor("l40sx4", 48, 382, 4, "L40S", 8.30),
    Flavor("a100x4", 48, 568, 4, "A100", 10.00),
    Flavor("h200x2", 46, 512, 2, "H200", 10.00),
    Flavor("rtx-pro-6000x4", 92, 1024, 4, "RTX PRO 6000", 11.00),
    Flavor("a100x8", 96, 1136, 8, "A100", 20.00),
    Flavor("h200x4", 92, 1024, 4, "H200", 20.00),
    Flavor("rtx-pro-6000x8", 184, 2048, 8, "RTX PRO 6000", 22.00),
    Flavor("l40sx8", 192, 1534, 8, "L40S", 23.50),
    Flavor("h200x8", 184, 2048, 8, "H200", 40.00),
)

_MEMORY_UNITS: Dict[str, float] = {
    "": 1 / 1024**3,
    "k": 1000 / 1024**3,
    "ki": 1 / 1024**2,
    "m": 1000**2 / 1024**3,
    "mi": 1 / 1024,
    "g": 1000**3 / 1024**3,
    "gi": 1.0,
    "t": 1000**4 / 1024**3,
    "ti": 1024.0,
}


def parse_memory(value: Optional[str]) -> Optional[float]:
    """
    Parse a memory string into GiB, accepting the Kubernetes-style suffixes that
    :class:`~tango.step.StepResources` documents.

    :examples:

    .. testcode::

        from tango.integrations.hf.common import parse_memory

        print(parse_memory("2.5GiB"), parse_memory("1024Mi"), parse_memory("32G"))

    .. testoutput::

        2.5 1.0 29.802322387695312
    """
    if value is None:
        return None
    match = re.fullmatch(r"\s*([0-9.]+)\s*([a-zA-Z]*)\s*", str(value))
    if match is None:
        raise ConfigurationError(f"Could not parse '{value}' as an amount of memory.")
    amount, suffix = match.groups()
    suffix = suffix.lower().rstrip("b")
    if suffix not in _MEMORY_UNITS:
        raise ConfigurationError(f"Unknown memory unit '{suffix}' in '{value}'.")
    return float(amount) * _MEMORY_UNITS[suffix]


def _normalise_gpu(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def _satisfies(flavor: Flavor, resources: StepResources) -> bool:
    if resources.cpu_count is not None and flavor.vcpu < resources.cpu_count:
        return False
    if resources.gpu_count is not None and flavor.gpus < resources.gpu_count:
        return False
    memory = parse_memory(resources.memory)
    if memory is not None and flavor.ram_gb < memory:
        return False
    if resources.gpu_type is not None:
        if flavor.accelerator is None:
            return False
        if _normalise_gpu(flavor.accelerator) not in _normalise_gpu(resources.gpu_type):
            return False
    return True


def resolve_flavor(
    resources: Optional[StepResources], flavors: Tuple[Flavor, ...] = FLAVORS
) -> Optional[str]:
    """
    Pick the cheapest flavor that satisfies a step's declared resources.

    Returns ``None`` when the step asks for nothing in particular, which leaves the choice
    to the executor's own default.

    :raises ConfigurationError: If no flavor can satisfy the request.

    :examples:

    .. testcode::

        from tango.step import StepResources
        from tango.integrations.hf.common import resolve_flavor

        print(resolve_flavor(StepResources(gpu_count=1)))
        print(resolve_flavor(StepResources(cpu_count=16, memory="100GiB")))
        print(resolve_flavor(StepResources()))

    .. testoutput::

        t4-small
        cpu-xl
        None
    """
    if resources is None:
        return None
    if all(
        value is None
        for value in (
            resources.cpu_count,
            resources.gpu_count,
            resources.gpu_type,
            resources.memory,
        )
    ):
        return None

    candidates = [flavor for flavor in flavors if _satisfies(flavor, resources)]
    if not candidates:
        raise ConfigurationError(
            f"No Hugging Face Jobs flavor satisfies the requested resources ({resources}). "
            f"Run `hf jobs hardware` to see what is available."
        )
    # Ties on price go to the smaller machine, so a GPU request doesn't grab a 96-core box.
    return min(candidates, key=lambda flavor: (flavor.cost_per_hour, flavor.vcpu)).name


#
# Locking.
#

#: Job stages that mean the job is over, whatever the outcome.
TERMINAL_JOB_STAGES = frozenset({"COMPLETED", "ERROR", "CANCELED", "DELETED"})


def get_s3_client(
    namespace: str,
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
) -> Any:
    """
    Build a boto3 client for the Hugging Face S3 gateway, scoped to ``namespace``.

    The step lock needs an atomic claim, and ``If-None-Match`` on ``PutObject`` through this
    gateway is the only one the Hub exposes — the native bucket API is explicitly
    non-transactional. Credentials are separate from your HF token: generate them from
    https://huggingface.co/settings/tokens with "Generate S3 credentials".

    :raises ConfigurationError: If credentials aren't available.
    """
    try:
        import boto3
        from botocore.config import Config
    except ModuleNotFoundError:
        raise IntegrationMissingError("hf", dependencies={"boto3"})

    access_key_id = access_key_id or os.environ.get("HF_S3_ACCESS_KEY_ID")
    secret_access_key = secret_access_key or os.environ.get("HF_S3_SECRET_ACCESS_KEY")
    if not access_key_id or not secret_access_key:
        raise ConfigurationError(
            "The Hugging Face workspace needs S3 gateway credentials to lock steps. Generate them "
            "at https://huggingface.co/settings/tokens ('Generate S3 credentials' on a write "
            "token) and set HF_S3_ACCESS_KEY_ID and HF_S3_SECRET_ACCESS_KEY."
        )

    return boto3.client(
        "s3",
        endpoint_url=f"https://s3.hf.co/{namespace}",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(
            region_name="us-east-1",
            s3={"addressing_style": "path"},
            # Recent botocore sends trailing CRC32 checksums in `aws-chunked` framing, which the
            # gateway does not parse. These keep them to operations that strictly require them.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


class HfStepLock:
    """
    A mutual-exclusion lock over one step, held as a single object in the bucket.

    Acquisition is a conditional ``PutObject`` with ``If-None-Match: *``, so exactly one client
    can create the object. The body records who holds it, which is what makes a stale lock
    recoverable: if the holder was a Job, its liveness is settled by asking the Jobs API, and if
    it was a local process, a heartbeat thread refreshes the record until the lock is released.

    This is the part that :class:`GCSStepLock` in the old Google Storage integration got wrong —
    it had no notion of a dead holder, so a crashed run left a lock that spun forever until
    somebody deleted it by hand.
    """

    def __init__(
        self,
        client: HfBucketClient,
        step: Union[str, Step, StepInfo],
        s3_client: Optional[Any] = None,
        ttl: float = 300.0,
        heartbeat_interval: float = 60.0,
    ):
        self._client = client
        self._step_id = step if isinstance(step, str) else step.unique_id
        self._key = client.key(Constants.step_lock_artifact_name(step))
        self._bucket = client.bucket_id.split("/", 1)[1]
        self._s3 = s3_client if s3_client is not None else get_s3_client(client.namespace)
        self._ttl = ttl
        self._heartbeat_interval = heartbeat_interval
        self._held = False
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self.lock_url = client.url(Constants.step_lock_artifact_name(step))

    def _payload(self) -> bytes:
        return json.dumps(
            {
                "step": self._step_id,
                # Set inside a Job container, absent locally. Which one it is decides how a
                # stale lock gets detected.
                "job_id": os.environ.get("JOB_ID"),
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "acquired_at": utc_now_datetime().isoformat(),
                "heartbeat": time.time(),
            }
        ).encode("utf-8")

    def _read_holder(self) -> Optional[Dict[str, Any]]:
        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=self._key)
            return json.loads(response["Body"].read().decode("utf-8"))
        except Exception:
            # Either the lock was released between the failed put and this read, or the record
            # is unreadable. Both are handled by the caller retrying the conditional put.
            return None

    def _holder_is_dead(self, holder: Optional[Dict[str, Any]]) -> bool:
        if holder is None:
            return False

        job_id = holder.get("job_id")
        if job_id:
            try:
                from huggingface_hub import inspect_job

                job = inspect_job(job_id=job_id, token=self._client.token)
            except Exception:
                logger.debug("Could not inspect job %s holding the lock.", job_id, exc_info=True)
                return False
            stage = getattr(getattr(job, "status", None), "stage", None)
            return str(stage) in TERMINAL_JOB_STAGES

        heartbeat = holder.get("heartbeat")
        if not isinstance(heartbeat, (int, float)):
            return False
        return (time.time() - heartbeat) > self._ttl

    def _heartbeat_loop(self) -> None:
        while not self._stop_heartbeat.wait(self._heartbeat_interval):
            try:
                self._s3.put_object(Bucket=self._bucket, Key=self._key, Body=self._payload())
            except Exception:  # pragma: no cover - best effort
                logger.debug("Failed to refresh the lock for '%s'.", self._step_id, exc_info=True)

    def acquire(
        self,
        timeout: Optional[float] = None,
        poll_interval: float = 2.0,
        log_interval: float = 30.0,
    ) -> None:
        """
        Block until the lock is held.

        :raises TimeoutError: If ``timeout`` elapses first.
        """
        if self._held:
            return

        start = time.monotonic()
        last_logged: Optional[float] = None
        while timeout is None or (time.monotonic() - start < timeout):
            try:
                self._s3.put_object(
                    Bucket=self._bucket, Key=self._key, Body=self._payload(), IfNoneMatch="*"
                )
            except Exception as exc:
                if not _is_precondition_failed(exc):
                    raise

                holder = self._read_holder()
                if self._holder_is_dead(holder):
                    logger.warning(
                        "Breaking the lock on step '%s': its holder (%s) is gone.",
                        self._step_id,
                        (holder or {}).get("job_id") or (holder or {}).get("host"),
                    )
                    self._force_release()
                    continue

                if last_logged is None or time.monotonic() - last_logged >= log_interval:
                    logger.warning(
                        "Waiting on the lock for step '%s', held by %s.\n%s",
                        self._step_id,
                        (holder or {}).get("job_id") or (holder or {}).get("host") or "another run",
                        self.lock_url,
                    )
                    last_logged = time.monotonic()
                time.sleep(poll_interval)
            else:
                self._held = True
                atexit.register(self.release)
                # A Job's liveness is settled by the Jobs API, so only a local holder needs to
                # prove it is still alive.
                if not os.environ.get("JOB_ID"):
                    self._stop_heartbeat.clear()
                    self._heartbeat_thread = threading.Thread(
                        target=self._heartbeat_loop,
                        name=f"HfStepLock-{self._step_id[:8]}",
                        daemon=True,
                    )
                    self._heartbeat_thread.start()
                return

        raise TimeoutError(
            f"Timed out waiting for the lock on step '{self._step_id}'.\n\n{self.lock_url}\n\n"
            f"This usually means the step is running elsewhere. If you are sure it isn't, "
            f"delete the lock object above."
        )

    def _force_release(self) -> None:
        try:
            self._s3.delete_object(Bucket=self._bucket, Key=self._key)
        except Exception:  # pragma: no cover - best effort
            logger.debug("Failed to delete the lock for '%s'.", self._step_id, exc_info=True)

    def release(self) -> None:
        if not self._held:
            return
        self._stop_heartbeat.set()
        self._heartbeat_thread = None
        self._force_release()
        self._held = False
        atexit.unregister(self.release)

    def __del__(self) -> None:
        self.release()


def _is_precondition_failed(exc: BaseException) -> bool:
    """
    Did a conditional ``PutObject`` fail because the object already existed?
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error") or {}
    if str(error.get("Code")) in {"PreconditionFailed", "412"}:
        return True
    metadata = response.get("ResponseMetadata") or {}
    return metadata.get("HTTPStatusCode") == 412

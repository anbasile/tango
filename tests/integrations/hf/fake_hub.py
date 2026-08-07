"""
In-memory stand-ins for the parts of the Hugging Face Hub this integration talks to.

These exist so the workspace, step cache and lock can be exercised end-to-end without a token,
an S3 credential pair, or a bill. They implement the semantics the integration actually relies
on — notably that ``If-None-Match: *`` on ``PutObject`` fails when the object already exists,
which is the whole basis of the step lock.
"""

import io
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union


@dataclass
class FakeEntry:
    """
    Shaped like a real ``list_bucket_tree`` entry, which carries
    ``type/path/size/xet_hash/mtime/uploaded_at``.
    """

    path: str
    type: str
    size: int = 0
    xet_hash: str = "0" * 64
    mtime: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)
    uploaded_at: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _glob_matches(pattern: str, path: str) -> bool:
    """
    Reproduce how ``sync_bucket`` actually matches exclude patterns, as probed against the Hub:

    - patterns are anchored at both ends, so ``__pycache__/*`` does not match
      ``pkg/__pycache__/x.pyc``;
    - ``*`` crosses ``/``, so ``.venv/*`` matches ``.venv/lib/site-packages/torch/x.so``;
    - consequently a bare ``.venv`` matches nothing at all.
    """
    regex = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", ".*").replace(r"\?", ".")
    return re.fullmatch(regex, path) is not None


def _split_uri(uri: str) -> Optional[Tuple[str, str]]:
    """
    Split ``hf://buckets/<namespace>/<bucket>/<key>`` into the bucket id and the key.
    """
    if not uri.startswith("hf://buckets/"):
        return None
    rest = uri[len("hf://buckets/") :].strip("/")
    parts = rest.split("/")
    return "/".join(parts[:2]), "/".join(parts[2:])


class FakeHfApi:
    """
    A minimal ``HfApi`` covering the bucket calls used by :class:`HfBucketClient`.
    """

    #: Shared across instances, because the workspace and its cache each build a client.
    STORE: Dict[str, Dict[str, bytes]] = {}

    def __init__(self, token: Optional[str] = None, **kwargs: Any) -> None:
        self.token = token

    @classmethod
    def reset(cls) -> None:
        cls.STORE = {}

    def _files(self, bucket_id: str) -> Dict[str, bytes]:
        return self.STORE.setdefault(bucket_id, {})

    def create_bucket(self, bucket_id: str, exist_ok: bool = False, **kwargs: Any) -> str:
        if bucket_id in self.STORE and not exist_ok:
            # Probed against the Hub: a conflict is a 409 HfHubHTTPError, not a ValueError.
            # It insists on a real httpx.Response, so build one.
            import httpx
            from huggingface_hub.errors import HfHubHTTPError

            raise HfHubHTTPError(
                f"409 Conflict: you already created {bucket_id}",
                response=httpx.Response(409, request=httpx.Request("POST", "https://hf.co")),
            )
        self.STORE.setdefault(bucket_id, {})
        return f"hf://buckets/{bucket_id}"

    def list_bucket_tree(
        self, bucket_id: str, prefix: str = "", recursive: bool = True, **kwargs: Any
    ) -> Iterator[FakeEntry]:
        files = self._files(bucket_id)
        prefix = prefix.strip("/")
        seen_dirs = set()
        for key in sorted(files):
            if prefix and not key.startswith(prefix):
                continue
            if recursive:
                yield FakeEntry(key, "file", len(files[key]))
                continue
            remainder = key[len(prefix) :].strip("/") if prefix else key
            head, separator, _ = remainder.partition("/")
            if separator:
                directory = f"{prefix}/{head}".strip("/") if prefix else head
                if directory not in seen_dirs:
                    seen_dirs.add(directory)
                    yield FakeEntry(directory, "directory")
            else:
                yield FakeEntry(key, "file", len(files[key]))

    def batch_bucket_files(
        self,
        bucket_id: str,
        add: Optional[Sequence[Tuple[Union[bytes, str], str]]] = None,
        delete: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        files = self._files(bucket_id)
        for source, remote in add or []:
            data = source if isinstance(source, bytes) else Path(source).read_bytes()
            files[remote.strip("/")] = data
        for remote in delete or []:
            files.pop(remote.strip("/"), None)

    def get_bucket_paths_info(
        self, bucket_id: str, paths: Sequence[str], **kwargs: Any
    ) -> List[FakeEntry]:
        # Probed against the Hub: missing paths are omitted, not reported.
        store = self._files(bucket_id)
        return [
            FakeEntry(path.strip("/"), "file", len(store[path.strip("/")]))
            for path in paths
            if path.strip("/") in store
        ]

    def download_bucket_files(
        self, bucket_id: str, files: Sequence[Tuple[Any, str]], **kwargs: Any
    ) -> None:
        # Probed against the Hub: a missing key warns and is skipped -- it does *not* raise, and
        # no local file appears. Reproducing that here is the point; a fake that raised would
        # make `HfBucketClient.get_bytes` pass on an exception path the real API never takes.
        store = self._files(bucket_id)
        for remote, local in files:
            key = (remote.path if hasattr(remote, "path") else str(remote)).strip("/")
            if key not in store:
                warnings.warn(f"File '{key}' not found in bucket '{bucket_id}'. Skipping.")
                continue
            target = Path(local)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(store[key])

    def sync_bucket(
        self,
        source: str,
        destination: str,
        exclude: Optional[Sequence[str]] = None,
        delete: bool = False,
        **kwargs: Any,
    ) -> None:
        source_parts = _split_uri(source)
        destination_parts = _split_uri(destination)

        if destination_parts is not None and source_parts is None:  # upload
            bucket_id, prefix = destination_parts
            store = self._files(bucket_id)
            root = Path(source)
            if not root.is_dir():
                raise FileNotFoundError(source)

            uploaded = set()
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                relative = str(path.relative_to(root))
                if any(_glob_matches(pattern, relative) for pattern in exclude or []):
                    continue
                key = f"{prefix}/{relative}".strip("/")
                store[key] = path.read_bytes()
                uploaded.add(key)

            if delete:
                stale = [
                    key
                    for key in store
                    if key.startswith(prefix.strip("/") + "/") and key not in uploaded
                ]
                for key in stale:
                    del store[key]
        elif source_parts is not None and destination_parts is None:  # download
            # Probed against the Hub: a prefix holding no objects is "Nothing to sync", not an
            # error. So this must not raise either, or `_download_step_remote` would look
            # correct offline while silently producing an empty directory in production.
            bucket_id, prefix = source_parts
            store = self._files(bucket_id)
            root = Path(destination)
            for key, data in store.items():
                if not key.startswith(prefix.strip("/")):
                    continue
                relative = key[len(prefix.strip("/")) :].strip("/")
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        else:
            raise ValueError(f"Cannot sync {source} -> {destination}")


class FakeClientError(Exception):
    """
    Shaped like ``botocore.exceptions.ClientError`` for the fields the lock inspects.
    """

    def __init__(self, code: str, status: int) -> None:
        super().__init__(code)
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class FakeS3Client:
    """
    Implements the one S3 behaviour the lock depends on: a conditional create.
    """

    def __init__(self) -> None:
        self.objects: Dict[str, bytes] = {}
        self.put_calls = 0

    def put_object(
        self, Bucket: str, Key: str, Body: bytes, IfNoneMatch: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        self.put_calls += 1
        if IfNoneMatch == "*" and Key in self.objects:
            raise FakeClientError("PreconditionFailed", 412)
        self.objects[Key] = Body
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def get_object(self, Bucket: str, Key: str, **kwargs: Any) -> Dict[str, Any]:
        if Key not in self.objects:
            raise FakeClientError("NoSuchKey", 404)
        return {"Body": io.BytesIO(self.objects[Key])}

    def delete_object(self, Bucket: str, Key: str, **kwargs: Any) -> Dict[str, Any]:
        self.objects.pop(Key, None)
        return {"ResponseMetadata": {"HTTPStatusCode": 204}}


class FakeJob:
    def __init__(self, job_id: str, stage: str, url: str = "https://hf.co/jobs/fake") -> None:
        self.id = job_id
        self.url = url
        self.status = type("JobStatus", (), {"stage": stage, "message": None})()


class FakeJobsApi:
    """
    Records what the executor submits and reports a configurable outcome.
    """

    def __init__(self, stage: str = "COMPLETED") -> None:
        self.stage = stage
        self.submitted: List[Dict[str, Any]] = []
        self.cancelled: List[str] = []
        self.volumes: List[Tuple[str, str]] = []
        self.running: List[FakeJob] = []

    def run_job(self, **kwargs: Any) -> FakeJob:
        self.submitted.append(kwargs)
        job_id = f"job-{len(self.submitted)}"
        return FakeJob(job_id, "SCHEDULING", url=f"https://hf.co/jobs/{job_id}")

    def inspect_job(self, job_id: str, **kwargs: Any) -> FakeJob:
        return FakeJob(job_id, self.stage, url=f"https://hf.co/jobs/{job_id}")

    def list_jobs(self, **kwargs: Any) -> List[FakeJob]:
        return list(self.running)

    def cancel_job(self, job_id: str, **kwargs: Any) -> None:
        self.cancelled.append(job_id)

    def sync_job_volume(self, local_dir: str, mount_path: str, **kwargs: Any) -> str:
        self.volumes.append((local_dir, mount_path))
        return f"volume::{mount_path}"

    @property
    def commands(self) -> List[str]:
        """
        The shell script of each submitted job, for asserting on what actually gets run.
        """
        return [job["command"][-1] for job in self.submitted]


def install_job_fakes(monkeypatch, stage: str = "COMPLETED") -> FakeJobsApi:
    import huggingface_hub

    jobs = FakeJobsApi(stage=stage)
    for name in ("run_job", "inspect_job", "list_jobs", "cancel_job", "sync_job_volume"):
        monkeypatch.setattr(huggingface_hub, name, getattr(jobs, name))
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_faketoken")
    return jobs


def install_fakes(
    monkeypatch, cache_dir: Path, s3_client: Optional[FakeS3Client] = None
) -> FakeS3Client:
    """
    Point the integration at the fakes and hand back the S3 client so tests can inspect it.

    ``cache_dir`` redirects :func:`~tango.common.util.tango_cache_dir`, which is otherwise
    hardcoded to ``~/.cache/tango``. Without that, tests would write into the developer's real
    cache, and — because the remote step cache checks its local mirror before the bucket — a
    result cached by one test would be found by the next even after the fake bucket is reset.
    """
    import huggingface_hub

    from tango.integrations.hf import common, step_cache, workspace
    from tango.workspaces import remote_workspace

    FakeHfApi.reset()
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    s3 = s3_client if s3_client is not None else FakeS3Client()
    monkeypatch.setattr(common, "get_s3_client", lambda *args, **kwargs: s3)
    # `workspace` imported the name directly, so patching `common` alone would miss it.
    monkeypatch.setattr(workspace, "get_s3_client", lambda *args, **kwargs: s3)
    # Both the step cache and `RemoteWorkspace.steps_dir` hang off the cache dir.
    monkeypatch.setattr(step_cache, "tango_cache_dir", lambda: Path(cache_dir))
    monkeypatch.setattr(remote_workspace, "tango_cache_dir", lambda: Path(cache_dir))
    monkeypatch.setenv("HF_S3_ACCESS_KEY_ID", "HFAKTEST")
    monkeypatch.setenv("HF_S3_SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("JOB_ID", raising=False)
    os.environ.setdefault("HF_TOKEN", "hf_faketoken")
    return s3

import logging
from pathlib import Path
from typing import Iterator, Optional, Union

from tango.common.aliases import PathOrStr
from tango.common.util import make_safe_filename, tango_cache_dir
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_caches.remote_step_cache import RemoteNotFoundError, RemoteStepCache
from tango.step_info import StepInfo

from .common import Constants, HfBucketClient, _is_not_found

logger = logging.getLogger(__name__)


@StepCache.register("hf")
class HfBucketStepCache(RemoteStepCache):
    """
    A :class:`~tango.step_cache.StepCache` that stores step results in a
    `Hugging Face Storage Bucket <https://huggingface.co/docs/hub/storage-buckets>`_.

    Used by :class:`HfBucketWorkspace`. It inherits an in-memory cache and an on-disk mirror
    from :class:`~tango.step_caches.remote_step_cache.RemoteStepCache`, so reading a step a
    second time doesn't hit the network.

    .. tip::
        Registered as a :class:`~tango.step_cache.StepCache` under the name "hf".

    :param bucket: ``"<namespace>/<bucket>[/<prefix>]"``.
    :param client: An existing client to share with a workspace, instead of building a new one.
    """

    Constants = Constants

    def __init__(self, bucket: str, client: Optional[HfBucketClient] = None):
        self._client = client if client is not None else HfBucketClient(bucket)
        super().__init__(tango_cache_dir() / "hf_cache" / make_safe_filename(bucket))

    @property
    def client(self) -> HfBucketClient:
        return self._client

    def _uncommitted_key(self, artifact_name: str) -> str:
        return f"{artifact_name}/{self.Constants.UNCOMMITTED_FNAME}"

    def _step_result_remote(self, step: Union[Step, StepInfo]) -> Optional[str]:
        """
        The artifact name of a *finished* step, or ``None``.

        A step whose upload is still in flight has an ``.uncommitted`` marker and must not be
        reported as cached, or a concurrent reader would pick up a half-written result.
        """
        artifact_name = self.Constants.step_artifact_name(step)
        entries = self._client.ls(artifact_name, recursive=True)
        if not entries:
            return None
        if any(
            entry.path == self._client.key(self._uncommitted_key(artifact_name))
            for entry in entries
        ):
            return None
        return artifact_name

    def _upload_step_remote(self, step: Step, objects_dir: Path) -> str:
        artifact_name = self.Constants.step_artifact_name(step)
        # Mark the artifact as under construction *before* any of it lands, and clear the marker
        # only once everything has. The marker lives in the bucket rather than in `objects_dir`,
        # and `sync_bucket` doesn't delete at the destination by default, so the upload won't
        # remove it behind our back.
        self._client.put_bytes(self._uncommitted_key(artifact_name), b"")
        self._client.upload_dir(artifact_name, objects_dir)
        self._client.delete(self._uncommitted_key(artifact_name))
        return artifact_name

    def _download_step_remote(self, step_result: str, target_dir: PathOrStr) -> None:
        try:
            self._client.download_dir(step_result, target_dir)
        except Exception as exc:
            if _is_not_found(exc):
                raise RemoteNotFoundError(self._client.url(step_result)) from exc
            raise

    def committed_step_ids(self) -> Iterator[str]:
        """
        The unique ids of every finished step in the bucket.
        """
        root = self._client.key()
        offset = len(root) + 1 if root else 0
        for entry in self._client.ls("", recursive=False):
            if entry.type != "directory":
                continue
            artifact_name = entry.path[offset:]
            if not artifact_name.startswith(self.Constants.STEP_ARTIFACT_PREFIX):
                continue
            if self._client.exists(self._uncommitted_key(artifact_name)):
                continue
            yield artifact_name[len(self.Constants.STEP_ARTIFACT_PREFIX) :]

    def __len__(self) -> int:
        return sum(1 for _ in self.committed_step_ids())

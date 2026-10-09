import concurrent.futures
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import ParseResult

import petname

from tango.common.util import utc_now_datetime
from tango.step import Step
from tango.step_info import StepInfo, StepState
from tango.workspace import Run, Workspace
from tango.workspaces.remote_workspace import RemoteWorkspace

from .common import (
    Constants,
    HfBucketClient,
    HfBucketNotFound,
    HfStepLock,
    get_s3_client,
)
from .step_cache import HfBucketStepCache

logger = logging.getLogger(__name__)

# The format the rest of Tango serialises run timestamps in, per `Run.from_json_dict`.
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


@Workspace.register("hf")
class HfBucketWorkspace(RemoteWorkspace):
    """
    A :class:`~tango.workspace.Workspace` that keeps step results and run metadata in a
    `Hugging Face Storage Bucket <https://huggingface.co/docs/hub/storage-buckets>`_.

    Buckets are mutable and unversioned, which is what step info needs — a step's record is
    rewritten on every state change, and a git-backed dataset repo would accumulate a commit
    each time.

    .. tip::
        Registered as a :class:`~tango.workspace.Workspace` under the name "hf", so its URL is
        the Hub's own: ``hf://buckets/<namespace>/<bucket>[/<prefix>]``.

    .. important::
        Locking steps needs Hugging Face S3 gateway credentials in addition to your token.
        See :func:`~tango.integrations.hf.common.get_s3_client`. They are checked when the
        workspace is built, not when the first step runs.

    :param bucket: ``"<namespace>/<bucket>[/<prefix>]"``.
    :param token: A Hugging Face token. Falls back to the ambient login.
    :param s3_access_key_id: Overrides ``HF_S3_ACCESS_KEY_ID``.
    :param s3_secret_access_key: Overrides ``HF_S3_SECRET_ACCESS_KEY``.

    :examples:

    .. code-block::

        tango run config.jsonnet -w hf://buckets/my-org/my-workspace
    """

    Constants = Constants
    NUM_CONCURRENT_WORKERS: int = 16
    #: Starts the error of a step that :meth:`step_abandoned` failed, which tells it apart from
    #: an error the step raised itself.
    ABANDONED_PREFIX: str = "Abandoned: "

    def __init__(
        self,
        bucket: str,
        token: Optional[str] = None,
        s3_access_key_id: Optional[str] = None,
        s3_secret_access_key: Optional[str] = None,
    ):
        self._client = HfBucketClient(bucket, token=token)
        self._cache = HfBucketStepCache(bucket, client=self._client)
        self._locks: Dict[Step, HfStepLock] = {}
        # Build the S3 client now so missing credentials surface here rather than part-way
        # through a run, when a step is already in flight.
        self._s3 = get_s3_client(self._client.namespace, s3_access_key_id, s3_secret_access_key)
        super().__init__()

    @property
    def cache(self) -> HfBucketStepCache:
        return self._cache

    @property
    def locks(self) -> Dict[Step, HfStepLock]:
        return self._locks

    @property
    def steps_dir_name(self) -> str:
        return "hf_workspace"

    @property
    def url(self) -> str:
        return self._client.url()

    @classmethod
    def from_parsed_url(cls, parsed_url: ParseResult) -> Workspace:
        # Accepts the Hub's canonical "hf://buckets/<ns>/<bucket>" and the shorter
        # "hf://<ns>/<bucket>".
        if parsed_url.netloc == "buckets":
            bucket = parsed_url.path
        else:
            bucket = f"{parsed_url.netloc}{parsed_url.path}"
        bucket = bucket.strip("/")
        if not bucket:
            raise ValueError(f"Bad URL for a Hugging Face workspace: '{parsed_url.geturl()}'")
        return cls(bucket)

    def _remote_lock(self, step: Step) -> HfStepLock:
        return HfStepLock(self._client, step, s3_client=self._s3)

    def _step_location(self, step: Step) -> str:
        return self._client.url(self.Constants.step_artifact_name(step))

    #
    # Step info.
    #

    def step_info(self, step_or_unique_id: Union[Step, str]) -> StepInfo:
        key = self.Constants.step_info_key(step_or_unique_id)
        try:
            return StepInfo.from_json_dict(self._client.get_json(key))
        except HfBucketNotFound:
            if not isinstance(step_or_unique_id, Step):
                raise KeyError(step_or_unique_id)
            step_info = StepInfo.new_from_step(step_or_unique_id)
            self._update_step_info(step_info)
            return step_info

    def _update_step_info(self, step_info: StepInfo) -> None:
        self._client.put_json(
            self.Constants.step_info_key(step_info.unique_id), step_info.to_json_dict()
        )

    def _remove_step_info(self, step_info: StepInfo) -> None:
        # The folder only: a prefix match would also delete the lock of a step that is running.
        self._client.delete_dir(self.Constants.step_artifact_name(step_info))
        try:
            self._client.delete(self.Constants.step_info_key(step_info.unique_id))
        except Exception:
            logger.debug("No step info to remove for '%s'.", step_info.unique_id, exc_info=True)

    def step_abandoned(self, step: Step, error: str, job_id: Optional[str] = None) -> bool:
        """
        Record that whatever was running ``step`` is gone without having finished it.

        A step's state and lock are written by the process that runs it. When that process is
        a Job that gets cancelled, evicted or killed at its timeout, nothing writes the end:
        the step reads as running forever and its lock blocks the next attempt. The executor
        calls this once it has seen the Job end.

        :param error: What happened, stored as the step's error.
        :param job_id: The Job that was running the step, whose lock can go without asking.
        :returns: ``False`` when a holder that still looks alive has the lock, in which case
            nothing is changed.
        """
        if not step.cache_results:
            return True
        if not self._remote_lock(step).break_if_dead(job_id):
            return False
        step_info = self.step_info(step)
        if step_info.state == StepState.RUNNING:
            step_info.end_time = utc_now_datetime()
            step_info.error = f"{self.ABANDONED_PREFIX}{error}"
            self._update_step_info(step_info)
        return True

    #
    # Runs.
    #

    def _save_run(
        self, steps: Dict[str, StepInfo], run_data: Dict[str, str], name: Optional[str] = None
    ) -> Run:
        history: List[Dict[str, Any]] = []
        kept: Dict[str, str] = {}
        if name is None:
            while True:
                name = petname.generate() + str(random.randint(0, 100))
                if not self._client.exists(self.Constants.run_key(name)):
                    break
        else:
            try:
                existing = self._client.get_json(self.Constants.run_key(name))
            except HfBucketNotFound:
                pass
            else:
                # Registering the very same graph under the same name again is a no-op. A
                # detached run relies on this: the client registers the run, then the driver
                # job re-runs `tango run -n <name>` in its own container.
                previous_steps: Dict[str, str] = existing.get("steps") or {}
                if previous_steps == run_data:
                    return self._run_from_json(existing)

                # A changed graph updates the run. Refusing it meant a new name for every
                # relaunch after a fix (`main`, `main2`, ...), with nothing gained: results are
                # shared between runs by step identity anyway. Say what changed, since a
                # changed identity is also how a finished step gets paid for twice.
                #
                # Steps the new graph does not mention stay in the run. `tango run -s <step>`
                # registers only that step and its dependencies, and resuming part of a run
                # must not make it forget the rest.
                added = sorted(set(run_data) - set(previous_steps))
                changed = sorted(
                    step_name
                    for step_name in set(run_data) & set(previous_steps)
                    if run_data[step_name] != previous_steps[step_name]
                )
                if not added and not changed:
                    return self._run_from_json(existing)
                logger.warning(
                    "Run '%s' exists with a different graph; updating it.%s%s",
                    name,
                    f"\n  added: {', '.join(added)}" if added else "",
                    f"\n  changed identity: {', '.join(changed)}" if changed else "",
                )
                kept = {
                    step_name: unique_id
                    for step_name, unique_id in previous_steps.items()
                    if step_name not in run_data
                }
                history = list(existing.get("history") or [])
                history.append({"start_date": existing.get("start_date"), "steps": previous_steps})

        # Truncate to the second before returning, not just before writing: the serialised form
        # has no sub-second field, so keeping microseconds here would make the Run handed back
        # differ from the one any later `registered_run()` reads.
        start_date = utc_now_datetime().replace(microsecond=0)
        record: Dict[str, Any] = {
            "name": name,
            "start_date": start_date.strftime(_DATE_FORMAT),
            "steps": {**kept, **run_data},
        }
        if history:
            record["history"] = history
        self._client.put_json(self.Constants.run_key(name), record)
        if kept:
            return self._run_from_json(record)
        return Run(name=name, steps=steps, start_date=start_date)

    def run_step_ids(self, name: str) -> Optional[Dict[str, str]]:
        """
        The ``{step name: unique id}`` mapping a run was last registered with, or ``None`` when
        there is no run of that name. Unlike :meth:`registered_run` this reads one object.
        """
        try:
            return self._client.get_json(self.Constants.run_key(name)).get("steps") or {}
        except HfBucketNotFound:
            return None

    def _run_from_json(self, run_json: Dict[str, Any]) -> Run:
        start_date = datetime.strptime(run_json["start_date"], _DATE_FORMAT).replace(
            tzinfo=timezone.utc
        )
        unique_ids: Dict[str, str] = run_json.get("steps") or {}

        steps: Dict[str, StepInfo] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.NUM_CONCURRENT_WORKERS,
            thread_name_prefix="HfBucketWorkspace._run_from_json()-",
        ) as executor:
            futures = {
                executor.submit(self.step_info, unique_id): step_name
                for step_name, unique_id in unique_ids.items()
            }
            for future in concurrent.futures.as_completed(futures):
                step_name = futures[future]
                try:
                    steps[step_name] = future.result()
                except KeyError:
                    # The run references a step whose info has since been removed. Report the
                    # rest of the run rather than failing the whole listing.
                    logger.warning(
                        "Run '%s' references step '%s', which is no longer in the workspace.",
                        run_json["name"],
                        step_name,
                    )

        return Run(name=run_json["name"], steps=steps, start_date=start_date)

    def registered_run(self, name: str) -> Run:
        try:
            run_json = self._client.get_json(self.Constants.run_key(name))
        except HfBucketNotFound:
            raise KeyError(f"Run '{name}' not found in workspace")
        return self._run_from_json(run_json)

    def registered_runs(self) -> Dict[str, Run]:
        root = self._client.key()
        offset = len(root) + 1 if root else 0

        names = []
        for entry in self._client.ls(self.Constants.RUNS_DIR, recursive=False):
            if entry.type != "file" or not entry.path.endswith(".json"):
                continue
            names.append(Path(entry.path[offset:]).stem)

        runs: Dict[str, Run] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.NUM_CONCURRENT_WORKERS,
            thread_name_prefix="HfBucketWorkspace.registered_runs()-",
        ) as executor:
            futures = {executor.submit(self.registered_run, name): name for name in names}
            for future in concurrent.futures.as_completed(futures):
                try:
                    run = future.result()
                except KeyError:
                    continue
                runs[run.name] = run
        return runs

    def _save_run_log(self, name: str, log_file: Path) -> None:
        self._client.put_bytes(self.Constants.run_log_key(name), Path(log_file).read_bytes())

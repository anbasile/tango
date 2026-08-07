import concurrent.futures
import hashlib
import logging
import os
import shlex
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from tango.common.exceptions import (
    CancellationError,
    ConfigurationError,
    ExecutorError,
    RunCancelled,
)
from tango.common.logging import cli_logger, log_exception
from tango.executor import ExecutionMetadata, Executor, ExecutorOutput
from tango.step import Step
from tango.step_graph import StepGraph
from tango.workspace import Workspace

from .common import TERMINAL_JOB_STAGES, Constants, resolve_flavor

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "python:3.12"
"""
Used when no ``image`` is given. Steps that need a GPU should point at an image with a matching
CUDA build of PyTorch, e.g. ``pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel``.
"""

PROJECT_MOUNT = "/tango/project"
CONFIG_MOUNT = "/tango/config"
CONFIG_FILENAME = "config.jsonnet"
SETTINGS_FILENAME = "tango.yml"

#: Set in the driver job so the executor inside it fans out instead of detaching again.
NO_DETACH_ENV_VAR = "TANGO_HF_NO_DETACH"

DEFAULT_PROJECT_EXCLUDE: Tuple[str, ...] = tuple(
    pattern
    for name in (
        ".venv",
        "venv",
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "node_modules",
        "*.egg-info",
    )
    # Both forms are needed. Probing the Hub showed that a bare name like ".venv" matches
    # nothing at all, "X/**" only matches at the root, and "**/X/**" only matches below it.
    for pattern in (f"{name}/**", f"**/{name}/**")
) + ("*.pyc", "**/*.pyc", ".DS_Store", "**/.DS_Store")
"""
Excluded from the project upload by default. Without these, a routine ``tango run`` would push
the virtualenv and the whole git history to the Hub on every invocation.
"""


class StepFailedError(ExecutorError):
    def __init__(self, msg: str, job_url: str):
        super().__init__(msg)
        self.job_url = job_url


@dataclass
class _RunContext:
    """
    The per-run state shared by every step job: the mounted volumes and the run name.
    """

    volumes: List[Any]
    run_name: Optional[str] = None
    temp_dirs: List[Any] = field(default_factory=list)


@Executor.register("hf")
class HfJobsExecutor(Executor):
    """
    An :class:`~tango.executor.Executor` that runs each step as a
    `Hugging Face Job <https://huggingface.co/docs/hub/jobs>`_.

    .. tip::
        Registered as an :class:`~tango.executor.Executor` under the name "hf".

    .. important::
        Jobs are ephemeral, so results must go somewhere durable. Use this with
        :class:`~tango.integrations.hf.workspace.HfBucketWorkspace`; a
        :class:`~tango.workspaces.LocalWorkspace` would lose every result.

    Unlike the old Beaker executor, your code does not have to be committed and pushed
    anywhere. ``project_dir`` is mirrored into the workspace bucket and mounted read-only into
    the container, so uncommitted work runs as-is. The mirror is incremental and skips
    :data:`DEFAULT_PROJECT_EXCLUDE` — without which a routine run would upload your virtualenv.

    :param workspace: The workspace to use. Must be an
        :class:`~tango.integrations.hf.workspace.HfBucketWorkspace`.
    :param include_package: Packages to import before running steps.
    :param parallelism: Maximum number of steps in flight at once.
    :param image: Docker image to run steps in.
    :param install_cmd: How to install your code inside the container. By default this is
        inferred from ``project_dir``: ``pip install -e .`` when there's a ``pyproject.toml``
        or ``setup.py``, ``pip install -r requirements.txt`` when there's one of those, and
        nothing otherwise. Pass ``""`` to skip it.
    :param flavor: Hardware to use for steps that don't declare any resources.
        See ``hf jobs hardware``.
    :param timeout: How long a step may run before the platform kills it. The platform's own
        default is 30 minutes, which is too short for most training steps.
    :param namespace: Run jobs under an organization instead of your own account.
    :param env: Extra environment variables for each job.
    :param secrets: Extra secrets for each job. Encrypted by the Hub.
    :param project_dir: The directory to sync into the container.
    :param detach: Submit one cheap driver job that runs the whole graph, and return
        immediately, so you can close your laptop.
    :param token: A Hugging Face token. Falls back to the ambient login.
    :param poll_interval: Seconds between job status checks.

    :examples:

    .. code:: yaml

        executor:
          type: hf
          image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
          parallelism: 4
          timeout: 4h
    """

    def __init__(
        self,
        workspace: Workspace,
        include_package: Optional[Sequence[str]] = None,
        parallelism: Optional[int] = 4,
        image: str = DEFAULT_IMAGE,
        install_cmd: Optional[str] = None,
        flavor: str = "cpu-basic",
        timeout: str = "1h",
        namespace: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        secrets: Optional[Dict[str, str]] = None,
        project_dir: str = ".",
        project_exclude: Optional[Sequence[str]] = None,
        detach: bool = False,
        token: Optional[str] = None,
        poll_interval: float = 10.0,
    ) -> None:
        super().__init__(workspace, include_package=include_package, parallelism=parallelism)

        from .workspace import HfBucketWorkspace

        if not isinstance(workspace, HfBucketWorkspace):
            # Jobs are ephemeral: whatever a step writes to local disk is gone when the
            # container exits. Beaker's executor only warned about this in its docstring and
            # let you lose a day's results; refuse instead.
            raise ConfigurationError(
                f"{type(self).__name__} needs an HfBucketWorkspace, because results have to "
                f"outlive the container that produced them. Got "
                f"{type(workspace).__name__}. Use `-w hf://buckets/<namespace>/<bucket>`."
            )

        self.image = image
        self.flavor = flavor
        self.timeout = timeout
        self.namespace = namespace
        self.env = dict(env or {})
        self.project_dir = Path(project_dir).resolve()
        self.project_exclude = list(
            project_exclude if project_exclude is not None else DEFAULT_PROJECT_EXCLUDE
        )
        self.poll_interval = poll_interval
        self.max_thread_workers = max(1, parallelism or 1)

        if not self.project_dir.is_dir():
            raise ConfigurationError(f"project_dir '{self.project_dir}' is not a directory.")

        self.install_cmd = (
            install_cmd if install_cmd is not None else self._infer_install_cmd(self.project_dir)
        )

        # A driver job must not detach again, or it would submit a driver job of its own,
        # forever.
        self.detach = detach and not os.environ.get(NO_DETACH_ENV_VAR)

        from huggingface_hub import get_token

        self.token = token or get_token()
        if not self.token:
            raise ConfigurationError(
                "No Hugging Face token found. Run `hf auth login`, or pass `token`."
            )

        self.secrets = {"HF_TOKEN": self.token, **self._s3_secrets(), **(secrets or {})}

        self._is_cancelled = threading.Event()
        self._submitted_job_ids: Set[str] = set()

    @staticmethod
    def _infer_install_cmd(project_dir: Path) -> str:
        if (project_dir / "pyproject.toml").is_file() or (project_dir / "setup.py").is_file():
            return "pip install -e ."
        if (project_dir / "requirements.txt").is_file():
            return "pip install -r requirements.txt"
        return ""

    @staticmethod
    def _s3_secrets() -> Dict[str, str]:
        # The step lock inside the job needs these just as much as the client does.
        secrets = {}
        for name in ("HF_S3_ACCESS_KEY_ID", "HF_S3_SECRET_ACCESS_KEY"):
            value = os.environ.get(name)
            if value:
                secrets[name] = value
        return secrets

    #
    # Job construction.
    #

    def _build_command(self, step_name: str, run_name: Optional[str]) -> List[str]:
        tango_cmd = [
            "tango",
            "--called-by-executor",
            "run",
            f"{CONFIG_MOUNT}/{CONFIG_FILENAME}",
            "-s",
            step_name,
            "-w",
            self.workspace.url,
        ]
        if run_name is not None:
            tango_cmd += ["-n", run_name]
        for package in self.include_package or []:
            tango_cmd += ["-i", package]

        # The mounts are read-only and shared between concurrent jobs, so copy the project onto
        # the container's own disk before installing it. An editable install writes metadata
        # into the source tree, and several jobs doing that to one mount would collide.
        script = [
            "set -euo pipefail",
            f"cp -a {PROJECT_MOUNT} /tmp/project",
            "cd /tmp/project",
        ]
        if self.install_cmd:
            script.append(self.install_cmd)
        script.append(" ".join(shlex.quote(part) for part in tango_cmd))

        return ["bash", "-c", "\n".join(script)]

    @property
    def _client(self) -> Any:
        # Guaranteed by the workspace type check in __init__.
        return self.workspace.step_cache.client  # type: ignore[attr-defined]

    def _mount(self, prefix: str, mount_path: str) -> Any:
        from huggingface_hub import Volume

        return Volume(
            type="bucket",
            source=self._client.bucket_id,
            mount_path=mount_path,
            path=self._client.key(prefix),
            read_only=True,
        )

    def _sync_project(self) -> Any:
        """
        Mirror the project into the workspace bucket and return a read-only mount of it.

        This goes through ``sync_bucket`` rather than ``sync_job_volume`` for one reason:
        ``sync_job_volume`` takes no exclusions, so it would upload ``.venv`` and ``.git`` on
        every run. The prefix is keyed on the project path, so re-runs are incremental.
        """
        digest = hashlib.sha256(str(self.project_dir).encode()).hexdigest()[:12]
        prefix = f"{Constants.PROJECT_DIR}/{digest}"

        cli_logger.info("[blue]Syncing %s to the workspace bucket...[/]", self.project_dir)
        self._client.api.sync_bucket(
            str(self.project_dir),
            f"hf://buckets/{self._client.bucket_id}/{self._client.key(prefix)}",
            exclude=self.project_exclude,
            # Keep the remote a mirror, so a file deleted locally stops being importable.
            delete=True,
            quiet=True,
        )
        return self._mount(prefix, PROJECT_MOUNT)

    def _upload_config(
        self, step_graph: StepGraph, run_name: Optional[str], with_settings: bool = False
    ) -> Any:
        with tempfile.TemporaryDirectory(prefix="tango-hf-config-") as config_dir:
            local = Path(config_dir) / CONFIG_FILENAME
            step_graph.to_file(local, include_unique_id=True)
            prefix = f"{Constants.CONFIG_DIR}/{run_name or 'run'}"
            self._client.put_bytes(f"{prefix}/{CONFIG_FILENAME}", local.read_bytes())
            if with_settings:
                self._client.put_bytes(f"{prefix}/{SETTINGS_FILENAME}", self._settings_yaml())
        return self._mount(prefix, CONFIG_MOUNT)

    def _settings_yaml(self) -> bytes:
        """
        A ``tango.yml`` for the driver job to run under.

        The driver invokes a plain ``tango run``, which picks its executor out of the settings
        file. Shipping one means detaching does not require the user's repo to contain a
        ``tango.yml`` naming this executor — without it the driver would silently fall back to
        the default executor and run every step inside the driver container.
        """
        import yaml

        settings: Dict[str, Any] = {
            "executor": {
                "type": "hf",
                "image": self.image,
                "install_cmd": self.install_cmd,
                "flavor": self.flavor,
                "timeout": self.timeout,
                "parallelism": self.max_thread_workers,
                "poll_interval": self.poll_interval,
                # The driver's own copy of the project, already installed by its entrypoint.
                "project_dir": "/tmp/project",
            }
        }
        if self.namespace is not None:
            settings["executor"]["namespace"] = self.namespace
        return yaml.safe_dump(settings).encode("utf-8")

    def _prepare_run_context(self, step_graph: StepGraph, run_name: Optional[str]) -> _RunContext:
        volumes = [self._sync_project(), self._upload_config(step_graph, run_name)]
        return _RunContext(volumes=volumes, run_name=run_name)

    def _find_running_job(self, step: Step) -> Optional[Any]:
        """
        Reattach to a job already running this exact step, so an interrupted client doesn't
        pay for the same work twice.
        """
        from huggingface_hub import list_jobs

        try:
            jobs = list_jobs(
                status=["RUNNING", "SCHEDULING"],
                labels={"tango-step": step.unique_id},
                namespace=self.namespace,
                token=self.token,
            )
            return next(iter(jobs), None)
        except Exception:
            logger.debug("Could not list running jobs for '%s'.", step.name, exc_info=True)
            return None

    def _submit(self, step: Step, context: _RunContext) -> Any:
        from huggingface_hub import run_job

        flavor = resolve_flavor(step.resources) or self.flavor
        return run_job(
            image=self.image,
            command=self._build_command(step.name, context.run_name),
            flavor=flavor,
            timeout=self.timeout,
            env=self.env or None,
            secrets=self.secrets,
            volumes=context.volumes,
            namespace=self.namespace,
            labels={
                "tango-step": step.unique_id,
                "tango-run": context.run_name or "",
                "name": step.name,
            },
            token=self.token,
        )

    #
    # Execution.
    #

    def _check_if_cancelled(self) -> None:
        if self._is_cancelled.is_set():
            raise RunCancelled

    def _execute_step_job(
        self, step_graph: StepGraph, step_name: str, context: _RunContext
    ) -> Optional[str]:
        from huggingface_hub import inspect_job

        self._check_if_cancelled()
        step = step_graph[step_name]

        if step.cache_results and step in self.workspace.step_cache:
            cli_logger.info(
                '[green]\N{CHECK MARK} Found output for step [bold]"%s"[/] in cache...[/]',
                step_name,
            )
            return None

        if step.resources.machine == "local":
            # The documented escape hatch for steps too small to be worth a container.
            self.execute_step(step)
            return None

        job = self._find_running_job(step) if step.cache_results else None
        if job is not None:
            cli_logger.info(
                '[blue]\N{BLACK RIGHTWARDS ARROW} Reattaching to job [b]%s[/] for step [b]"%s"[/]...[/]',
                job.url,
                step_name,
            )
        else:
            self._check_if_cancelled()
            step.log_starting()
            job = self._submit(step, context)
            cli_logger.info(
                '[blue]\N{BLACK RIGHTWARDS ARROW} Submitted job [b]%s[/] for step [b]"%s"[/]...[/]',
                job.url,
                step_name,
            )
        self._submitted_job_ids.add(job.id)

        try:
            while True:
                self._check_if_cancelled()
                time.sleep(self.poll_interval)
                job = inspect_job(job_id=job.id, token=self.token)
                stage = str(getattr(getattr(job, "status", None), "stage", ""))
                if stage in TERMINAL_JOB_STAGES:
                    break
        finally:
            self._submitted_job_ids.discard(job.id)

        if stage != "COMPLETED":
            message = getattr(getattr(job, "status", None), "message", None)
            raise StepFailedError(
                f"Job for step '{step_name}' finished with stage {stage}"
                + (f": {message}" if message else "")
                + f".\nLogs: {job.url}",
                job.url,
            )
        return job.url

    def _cancel_submitted_jobs(self) -> None:
        from huggingface_hub import cancel_job

        for job_id in list(self._submitted_job_ids):
            try:
                cancel_job(job_id=job_id, token=self.token)
            except Exception:  # pragma: no cover - best effort
                logger.debug("Could not cancel job %s.", job_id, exc_info=True)

    def _execute_detached(self, step_graph: StepGraph, run_name: Optional[str]) -> ExecutorOutput:
        from huggingface_hub import run_job

        volumes = [
            self._sync_project(),
            self._upload_config(step_graph, run_name, with_settings=True),
        ]

        # No `--called-by-executor` here: the driver is a full `tango run`, so it picks its
        # executor out of the settings file and fans out per-step jobs of its own. The settings
        # file is one we ship alongside the config, so this works whether or not the project
        # itself has a tango.yml.
        driver_cmd = [
            "tango",
            "--settings",
            f"{CONFIG_MOUNT}/{SETTINGS_FILENAME}",
            "run",
            f"{CONFIG_MOUNT}/{CONFIG_FILENAME}",
            "-w",
            self.workspace.url,
        ]
        if run_name is not None:
            driver_cmd += ["-n", run_name]
        for package in self.include_package or []:
            driver_cmd += ["-i", package]

        script = [
            "set -euo pipefail",
            f"cp -a {PROJECT_MOUNT} /tmp/project",
            "cd /tmp/project",
        ]
        if self.install_cmd:
            script.append(self.install_cmd)
        script.append(" ".join(shlex.quote(part) for part in driver_cmd))

        job = run_job(
            image=self.image,
            command=["bash", "-c", "\n".join(script)],
            flavor="cpu-basic",
            timeout=self.timeout,
            env={**self.env, NO_DETACH_ENV_VAR: "1"},
            secrets=self.secrets,
            volumes=volumes,
            namespace=self.namespace,
            labels={"tango-run": run_name or "", "name": f"tango-driver-{run_name or 'run'}"},
            token=self.token,
        )

        cli_logger.info(
            "[green]\N{CHECK MARK} Submitted driver job [bold]%s[/]. "
            "It will run the whole graph; you can disconnect now.[/]",
            job.url,
        )
        return ExecutorOutput(
            successful={},
            failed={},
            not_run={name: ExecutionMetadata(logs_location=job.url) for name in step_graph},
        )

    def execute_step_graph(
        self, step_graph: StepGraph, run_name: Optional[str] = None
    ) -> ExecutorOutput:
        """
        Run every step of the graph, each as its own Job, respecting dependencies.

        Steps whose dependencies failed are not run. Failures are logged rather than raised,
        matching the base :class:`~tango.executor.Executor`.
        """
        if self.detach:
            return self._execute_detached(step_graph, run_name)

        self._is_cancelled.clear()

        successful: Dict[str, ExecutionMetadata] = {}
        failed: Dict[str, ExecutionMetadata] = {}
        not_run: Dict[str, ExecutionMetadata] = {}

        steps_to_run: Set[str] = set()
        submitted_steps: Set[str] = set()
        step_futures: List[concurrent.futures.Future] = []

        uncacheable_leaf_steps = step_graph.uncacheable_leaf_steps()
        steps_left_to_run = uncacheable_leaf_steps | {
            step for step in step_graph.values() if step.cache_results
        }

        def update_steps_to_run() -> None:
            nonlocal steps_to_run
            for step_name, step in step_graph.items():
                if (
                    step_name in submitted_steps
                    or step_name in successful
                    or step_name in failed
                    or step_name in not_run
                ):
                    steps_to_run.discard(step_name)
                else:
                    for dependency in step.dependencies:
                        if dependency.name not in successful and dependency.cache_results:
                            if dependency.name in failed or dependency.name in not_run:
                                not_run[step_name] = ExecutionMetadata()
                                steps_to_run.discard(step_name)
                                steps_left_to_run.discard(step)
                            break
                    else:
                        if step.cache_results or step in uncacheable_leaf_steps:
                            steps_to_run.add(step_name)

        def make_done_callback(step_name: str):
            def done_callback(future: concurrent.futures.Future) -> None:
                step = step_graph[step_name]
                try:
                    exc = future.exception()
                except concurrent.futures.CancelledError:
                    failed[step_name] = ExecutionMetadata()
                    steps_left_to_run.discard(step)
                    return

                if exc is None:
                    successful[step_name] = ExecutionMetadata(
                        result_location=(
                            None
                            if not step.cache_results
                            else self.workspace.step_info(step).result_location
                        ),
                        logs_location=future.result(),
                    )
                elif isinstance(exc, StepFailedError):
                    failed[step_name] = ExecutionMetadata(logs_location=exc.job_url)
                elif isinstance(exc, (ExecutorError, CancellationError)):
                    failed[step_name] = ExecutionMetadata()
                else:
                    log_exception(exc, logger)
                    failed[step_name] = ExecutionMetadata()
                steps_left_to_run.discard(step)

            return done_callback

        context = self._prepare_run_context(step_graph, run_name)
        update_steps_to_run()

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_thread_workers, thread_name_prefix="HfJobsExecutor-"
            ) as pool:
                while steps_left_to_run:
                    for step_name in list(steps_to_run):
                        future = pool.submit(self._execute_step_job, step_graph, step_name, context)
                        future.add_done_callback(make_done_callback(step_name))
                        step_futures.append(future)
                        submitted_steps.add(step_name)

                    if step_futures:
                        _, not_done = concurrent.futures.wait(
                            step_futures,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                            timeout=2.0,
                        )
                        step_futures = list(not_done)
                    else:
                        time.sleep(2.0)

                    update_steps_to_run()
        except (KeyboardInterrupt, CancellationError):
            cli_logger.warning("Received interrupt, cancelling jobs...")
            self._is_cancelled.set()
            self._cancel_submitted_jobs()
            concurrent.futures.wait(step_futures)
            raise
        finally:
            self._is_cancelled.clear()
            for temp_dir in context.temp_dirs:
                temp_dir.cleanup()

        # Done-callbacks run on worker threads and may land after the last loop iteration, so
        # refresh `not_run` once more before reporting.
        update_steps_to_run()

        return ExecutorOutput(successful=successful, failed=failed, not_run=not_run)

import concurrent.futures
import hashlib
import logging
import os
import shlex
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from tango.common.exceptions import (
    CancellationError,
    ConfigurationError,
    ExecutorError,
    RunCancelled,
)
from tango.common.logging import cli_logger, log_exception
from tango.common.util import utc_now_datetime
from tango.executor import ExecutionMetadata, Executor, ExecutorOutput
from tango.step import Step
from tango.step_graph import StepGraph
from tango.step_info import StepState
from tango.workspace import Workspace

from .common import (
    JOB_NAMESPACE_ENV_VAR,
    TERMINAL_JOB_STAGES,
    Constants,
    flavor_info,
    http_status,
    hub_call,
    job_stage,
    parse_timeout,
    resolve_flavor,
)

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

#: Set to ``1`` in every Job this executor submits, for code that needs to know it runs in one.
IN_JOB_ENV_VAR = "TANGO_HF_JOB"
#: The hardware flavor of the Job, e.g. ``t4-small``.
FLAVOR_ENV_VAR = "TANGO_HF_FLAVOR"

#: The module every Job runs its command through. See :mod:`tango.integrations.hf.job`.
JOB_MODULE = "tango.integrations.hf.job"

#: Label shared by the jobs of one executor, so they can all be polled with one request.
SESSION_LABEL = "tango-session"

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


class OutOfCreditError(ExecutorError):
    """
    The Hub refused to start a job for lack of credit (HTTP 402).
    """


@dataclass
class _Outcome:
    """
    How one job ended, as seen from the client.
    """

    job: Any
    stage: str
    #: Why the executor cancelled the job itself, if it did.
    reason: Optional[str] = None
    #: Whether submitting the step again could end differently.
    retryable: bool = True
    running_seconds: float = 0.0


class _JobPoller:
    """
    Answers "what stage is this job in" for every worker thread with one request per interval.

    Polling each job on its own costs a request per job per interval, twice that when the
    namespace has to be looked up, and two runs at once went over the Hub's 1,000 requests per
    five minutes. The jobs an executor submits share a label, so one listing covers them all. A
    job missing from the listing (one this executor reattached to, or one too new to be listed)
    is asked for by itself.
    """

    def __init__(self, executor: "HfJobsExecutor") -> None:
        self._executor = executor
        self._lock = threading.Lock()
        self._jobs: Dict[str, Any] = {}
        self._refreshed: Optional[float] = None
        self._listing = True

    def _refresh(self) -> None:
        from huggingface_hub import list_jobs

        executor = self._executor
        try:
            jobs = hub_call(
                lambda: list(
                    list_jobs(
                        labels={SESSION_LABEL: executor.session},
                        namespace=executor.namespace,
                        token=executor.token,
                    )
                ),
                stop=executor._is_cancelled,
            )
        except Exception:
            # Not worth failing a run over: fall back to asking about each job.
            logger.warning(
                "Could not list this run's jobs; polling them one by one from now on.",
                exc_info=True,
            )
            self._listing = False
            return
        self._jobs = {job.id: job for job in jobs}

    def get(self, job_id: str) -> Any:
        from huggingface_hub import inspect_job

        executor = self._executor
        with self._lock:
            now = time.monotonic()
            stale = self._refreshed is None or now - self._refreshed >= executor.poll_interval / 2
            if self._listing and stale:
                self._refresh()
                self._refreshed = time.monotonic()
            job = self._jobs.get(job_id) if self._listing else None
        if job is None:
            job = hub_call(
                inspect_job,
                job_id=job_id,
                namespace=executor.namespace,
                token=executor.token,
                stop=executor._is_cancelled,
            )
        return job


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
    :param timeout: How long a step may run: seconds, or a number followed by ``s``, ``m``,
        ``h`` or ``d``. It is given to the platform, and enforced here as well: a job still
        running a minute past it is cancelled and its step fails. (The platform's own default is
        30 minutes, which is too short for most training steps, and it once let a job with a
        four-hour timeout run for five and a half.)
    :param scheduling_timeout: How long a job may wait to start before it is cancelled. No
        limit by default.
    :param attempts: How many times to submit a step whose job dies without the step itself
        having failed: a volume that would not mount, an eviction, a job that never starts.
        A step that raises is not submitted again, nor is one that ran into ``timeout``.
    :param namespace: Run jobs under an organization instead of your own account.
    :param env: Extra environment variables for each job. They override the defaults listed
        below. Values are turned into strings, so ``OMP_NUM_THREADS: 1`` in YAML is fine.
    :param secrets: Extra secrets for each job. Encrypted by the Hub.
    :param secrets_from_env: Names of environment variables to pass on as secrets, which keeps
        their values out of ``tango.yml``.
    :param project_dir: The directory to sync into the container.
    :param project_exclude: Glob patterns left out of the upload, *instead of*
        :data:`DEFAULT_PROJECT_EXCLUDE`.
    :param extra_project_exclude: Glob patterns left out of the upload *in addition to*
        ``project_exclude`` (or the defaults). This is the one to use for a data directory.
    :param detach: Submit one cheap driver job that runs the whole graph, and return
        immediately, so you can close your laptop.
    :param driver_timeout: How long the driver job of a detached run may live. It has to
        outlive the whole graph, not one step.
    :param driver_flavor: Hardware for the driver job.
    :param token: A Hugging Face token. Falls back to the ambient login.
    :param poll_interval: Seconds between job status checks. One request covers all the jobs
        of a run.

    Every job gets these environment variables unless ``env`` says otherwise:
    ``TANGO_HF_JOB=1`` and ``TANGO_HF_FLAVOR`` (to tell that code runs in a job, and on what);
    ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` set to the flavor's vCPU count (PyTorch
    otherwise starts a thread per core of the *host*, which made a small CPU training loop
    thousands of times slower); ``TANGO_LOG_LEVEL=info`` and ``FILE_FRIENDLY_LOGGING=true`` (so
    a step's ``self.logger`` output reaches the job log); ``HF_HUB_DISABLE_PROGRESS_BARS=1``
    and ``TQDM_DISABLE=1`` (the Hub keeps about the last thousand lines of a job's log, and
    progress bars fill them); ``GIT_PYTHON_REFRESH=quiet`` (most images have no ``git``).

    The output of every job is also kept in the workspace bucket, under
    ``logs/<step unique id>/<job id>.log``, and a record of each job (flavor, times, stage,
    estimated cost) under ``jobs/<run name>/<job id>.json``.

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
        timeout: Union[int, float, str] = "1h",
        namespace: Optional[str] = None,
        env: Optional[Dict[str, Any]] = None,
        secrets: Optional[Dict[str, str]] = None,
        project_dir: str = ".",
        project_exclude: Optional[Sequence[str]] = None,
        detach: bool = False,
        token: Optional[str] = None,
        poll_interval: float = 10.0,
        scheduling_timeout: Optional[Union[int, float, str]] = None,
        attempts: int = 1,
        secrets_from_env: Optional[Sequence[str]] = None,
        extra_project_exclude: Optional[Sequence[str]] = None,
        driver_timeout: Union[int, float, str] = "24h",
        driver_flavor: str = "cpu-basic",
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
        self.timeout_seconds = parse_timeout(timeout)
        self.scheduling_timeout_seconds = (
            None if scheduling_timeout is None else parse_timeout(scheduling_timeout)
        )
        self.driver_timeout_seconds = parse_timeout(driver_timeout)
        self.driver_flavor = driver_flavor
        if attempts < 1:
            raise ConfigurationError(f"attempts has to be at least 1, got {attempts}.")
        self.attempts = attempts
        self.env = {str(key): str(value) for key, value in (env or {}).items()}
        self.project_dir = Path(project_dir).resolve()
        self.project_exclude = list(
            project_exclude if project_exclude is not None else DEFAULT_PROJECT_EXCLUDE
        ) + list(extra_project_exclude or [])
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

        user_secrets = dict(secrets or {})
        for name in secrets_from_env or []:
            if name not in os.environ:
                raise ConfigurationError(
                    f"secrets_from_env names '{name}', which is not set in the environment."
                )
            user_secrets.setdefault(name, os.environ[name])
        #: Names only: the driver of a detached run reads the values from its own environment.
        self.user_secret_names = sorted(user_secrets)
        self.secrets = {"HF_TOKEN": self.token, **self._s3_secrets(), **user_secrets}

        self._is_cancelled = threading.Event()
        self._out_of_credit = threading.Event()
        self._submitted_job_ids: Set[str] = set()

        # Looked up once. Left to the Hub, every `inspect_job` and `cancel_job` without a
        # namespace costs a `whoami` request of its own, and looks under the token's account
        # even when the jobs run under an organization.
        if namespace is None:
            from huggingface_hub import whoami

            namespace = hub_call(whoami, token=self.token)["name"]
        self.namespace: str = namespace

        self.session = uuid.uuid4().hex[:12]
        self._poller = _JobPoller(self)

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

    def _job_env(self, flavor: str) -> Dict[str, str]:
        """
        The environment of a job on ``flavor``: the defaults, then the user's ``env`` on top.
        """
        env = {
            IN_JOB_ENV_VAR: "1",
            FLAVOR_ENV_VAR: flavor,
            JOB_NAMESPACE_ENV_VAR: self.namespace,
            # The stock images have no `git` binary, and GitPython refuses to import without
            # one, which fails the step before it starts.
            "GIT_PYTHON_REFRESH": "quiet",
            # At the default level (warning) nothing a step logs reaches the job's output.
            "TANGO_LOG_LEVEL": "info",
            "TANGO_CLI_LOGGER_ENABLED": "true",
            "FILE_FRIENDLY_LOGGING": "true",
            "PYTHONUNBUFFERED": "1",
            # The Hub keeps roughly the last thousand lines of a job's log.
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TQDM_DISABLE": "1",
        }
        info = flavor_info(flavor)
        if info is not None:
            # The container sees every core of the host; the flavor is what it may use.
            env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(info.vcpu)
        env.update(self.env)
        return env

    def _script(self, tango_cmd: List[str], log_owner: str, step: Optional[Step] = None) -> str:
        """
        The shell script a job runs: copy the project, install it, then run ``tango_cmd``
        through :mod:`tango.integrations.hf.job`, which keeps the output and checks the result.
        """
        bucket_path = "/".join(
            part for part in (self._client.bucket_id, self._client.prefix) if part
        )
        wrapper = ["python", "-m", JOB_MODULE, "--bucket", bucket_path, "--log-owner", log_owner]
        if step is not None and step.cache_results:
            wrapper += ["--result-of", step.unique_id]

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
        script.append(" ".join(shlex.quote(part) for part in [*wrapper, "--", *tango_cmd]))
        return "\n".join(script)

    def _build_command(
        self, step_name: str, run_name: Optional[str], step: Optional[Step] = None
    ) -> List[str]:
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

        log_owner = step.unique_id if step is not None else step_name
        return ["bash", "-c", self._script(tango_cmd, log_owner, step)]

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

        # Everything that shapes a step job has to be here, or the jobs of a detached run
        # differ from those of an attached one. Secrets go by name: the driver has the values
        # in its environment, and a settings file in the bucket is no place for them.
        settings: Dict[str, Any] = {
            "executor": {
                "type": "hf",
                "image": self.image,
                "install_cmd": self.install_cmd,
                "flavor": self.flavor,
                "timeout": self.timeout_seconds,
                "attempts": self.attempts,
                "parallelism": self.max_thread_workers,
                "poll_interval": self.poll_interval,
                "namespace": self.namespace,
                "env": dict(self.env),
                "secrets_from_env": list(self.user_secret_names),
                "project_exclude": list(self.project_exclude),
                # The driver's own copy of the project, already installed by its entrypoint.
                "project_dir": "/tmp/project",
            }
        }
        if self.scheduling_timeout_seconds is not None:
            settings["executor"]["scheduling_timeout"] = self.scheduling_timeout_seconds
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
            jobs = hub_call(
                lambda: list(
                    list_jobs(
                        status=["RUNNING", "SCHEDULING"],
                        labels={"tango-step": step.unique_id},
                        namespace=self.namespace,
                        token=self.token,
                    )
                ),
                stop=self._is_cancelled,
            )
            return next(iter(jobs), None)
        except Exception:
            logger.debug("Could not list running jobs for '%s'.", step.name, exc_info=True)
            return None

    def flavor_for(self, step: Step) -> str:
        """
        The hardware a step's job runs on: from its ``step_resources``, else the default.
        """
        return resolve_flavor(step.resources) or self.flavor

    def describe_step(self, step: Step) -> str:
        if step.resources.machine == "local":
            return "local"
        flavor = self.flavor_for(step)
        info = flavor_info(flavor)
        if info is None:
            return f"job on {flavor}"
        ceiling = info.cost_per_hour * self.timeout_seconds / 3600
        return f"job on {flavor}, ${info.cost_per_hour:.2f}/h, at most ${ceiling:.2f}"

    def _submit(self, step: Step, context: _RunContext) -> Any:
        from huggingface_hub import run_job

        if self._out_of_credit.is_set():
            raise OutOfCreditError("Not submitted: the Hub has refused a job for lack of credit.")

        flavor = self.flavor_for(step)
        try:
            # Only the rate limit is retried here. After any other failure the job may exist,
            # and submitting again would pay for it twice.
            return hub_call(
                run_job,
                image=self.image,
                command=self._build_command(step.name, context.run_name, step),
                flavor=flavor,
                timeout=self.timeout_seconds,
                env=self._job_env(flavor),
                secrets=self.secrets,
                volumes=context.volumes,
                namespace=self.namespace,
                labels={
                    "tango-step": step.unique_id,
                    "tango-run": context.run_name or "",
                    SESSION_LABEL: self.session,
                    "name": step.name,
                },
                token=self.token,
                statuses=frozenset({429}),
                stop=self._is_cancelled,
            )
        except Exception as exc:
            if http_status(exc) != 402:
                raise
            if not self._out_of_credit.is_set():
                self._out_of_credit.set()
                cli_logger.error(
                    "[red]\N{BALLOT X} The Hub refused to start a job: out of credit (402). "
                    "No more jobs will be submitted; the jobs already running are left to "
                    "finish.[/]"
                )
            raise OutOfCreditError(
                f"The Hub refused the job for step '{step.name}' (402)."
            ) from exc

    #
    # Execution.
    #

    #: Seconds a job may run past ``timeout`` before the executor cancels it. The platform is
    #: given the first chance to stop it.
    TIMEOUT_GRACE = 60.0
    #: Seconds to wait for a cancelled job to report a terminal stage.
    CANCEL_WAIT = 120.0

    def _check_if_cancelled(self) -> None:
        if self._is_cancelled.is_set():
            raise RunCancelled

    def _cancel_job(self, job_id: str) -> None:
        from huggingface_hub import cancel_job

        hub_call(
            cancel_job,
            job_id=job_id,
            namespace=self.namespace,
            token=self.token,
            budget=60.0,
        )

    def _watch_job(self, step_name: str, job: Any, record: Dict[str, Any]) -> _Outcome:
        """
        Poll ``job`` until it is over, cancelling it if it outlives its timeout.
        """
        submitted = time.monotonic()
        running_since: Optional[float] = None
        reason: Optional[str] = None
        retryable = True
        cancel_deadline = 0.0

        def running_seconds() -> float:
            return 0.0 if running_since is None else time.monotonic() - running_since

        while True:
            self._check_if_cancelled()
            time.sleep(self.poll_interval)
            job = self._poller.get(job.id)
            stage = job_stage(job)
            now = time.monotonic()

            if stage == "RUNNING" and running_since is None:
                running_since = now
                record["running_at"] = utc_now_datetime().isoformat()

            if stage in TERMINAL_JOB_STAGES:
                if reason is None and stage in {"CANCELED", "DELETED"}:
                    # Somebody cancelled it by hand. Starting it again would undo that.
                    retryable = False
                return _Outcome(job, stage, reason, retryable, running_seconds())

            if reason is not None:
                if now > cancel_deadline:
                    # The cancellation did not show. Stop waiting; the step is failed either way.
                    return _Outcome(job, "CANCELED", reason, retryable, running_seconds())
                continue

            if (
                running_since is not None
                and now - running_since > self.timeout_seconds + self.TIMEOUT_GRACE
            ):
                reason = f"timed out: still running after {timedelta(seconds=self.timeout_seconds)}"
                retryable = False
            elif (
                running_since is None
                and self.scheduling_timeout_seconds is not None
                and now - submitted > self.scheduling_timeout_seconds
            ):
                reason = (
                    "never started: still waiting for hardware after "
                    f"{timedelta(seconds=self.scheduling_timeout_seconds)}"
                )
            if reason is not None:
                cli_logger.warning(
                    '[yellow]Cancelling job [b]%s[/] of step [b]"%s"[/]: %s.[/]',
                    job.id,
                    step_name,
                    reason,
                )
                try:
                    self._cancel_job(job.id)
                except Exception:
                    logger.warning("Could not cancel job %s.", job.id, exc_info=True)
                cancel_deadline = now + self.CANCEL_WAIT

    def _result_in_bucket(self, step: Step) -> bool:
        cache = self.workspace.step_cache
        return hub_call(cache._step_result_remote, step) is not None  # type: ignore[attr-defined]

    def _failure_of(self, step: Step, outcome: _Outcome) -> Optional[str]:
        """
        What went wrong with a job that is over, or ``None`` when the step's result is there.
        """
        if outcome.reason is not None:
            return outcome.reason
        if outcome.stage != "COMPLETED":
            message = getattr(getattr(outcome.job, "status", None), "message", None)
            return f"job ended in stage {outcome.stage}" + (f": {message}" if message else "")
        if step.cache_results and not self._result_in_bucket(step):
            # A job that exits 0 has not necessarily run the step. Three once took the lock,
            # went silent for hours and exited cleanly, and were reported as succeeded.
            return "job completed without a result"
        return None

    def _own_failure(self, step: Step, since: Any) -> Optional[str]:
        """
        The error the step itself recorded while this job ran, if it did: the step raised,
        as opposed to the job dying under it.
        """
        if not step.cache_results:
            return None
        try:
            step_info = hub_call(self.workspace.step_info, step)
        except Exception:
            logger.debug("Could not read the state of '%s'.", step.name, exc_info=True)
            return None
        if step_info.state != StepState.FAILED or step_info.end_time is None:
            return None
        # Two minutes of slack for the clocks of this machine and the container.
        if step_info.end_time < since - timedelta(minutes=2):
            return None
        lines = [line for line in (step_info.error or "").strip().splitlines() if line.strip()]
        return lines[-1].strip() if lines else "the step raised"

    def _write_job_record(self, context: _RunContext, record: Dict[str, Any]) -> None:
        """
        Keep what is known about a job in the bucket. The Hub drops jobs from its list after a
        while and reports no running time for cancelled ones, so this is the only lasting
        account of what a run cost.
        """
        try:
            hub_call(
                self._client.put_json,
                Constants.job_record_key(context.run_name, record["job_id"]),
                record,
                budget=60.0,
            )
        except Exception:
            logger.warning("Could not write the record of job %s.", record["job_id"], exc_info=True)

    def _execute_step_job(
        self, step_graph: StepGraph, step_name: str, context: _RunContext
    ) -> Optional[str]:
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

        flavor = self.flavor_for(step)
        info = flavor_info(flavor)

        for attempt in range(1, self.attempts + 1):
            job = self._find_running_job(step) if step.cache_results and attempt == 1 else None
            since = utc_now_datetime()
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
                    '[blue]\N{BLACK RIGHTWARDS ARROW} Submitted job [b]%s[/] for step [b]"%s"[/] on %s...[/]',
                    job.url,
                    step_name,
                    flavor,
                )
            self._submitted_job_ids.add(job.id)

            record: Dict[str, Any] = {
                "job_id": job.id,
                "url": job.url,
                "step": step_name,
                "step_id": step.unique_id,
                "run": context.run_name,
                "attempt": attempt,
                "flavor": flavor,
                "price_per_hour": info.cost_per_hour if info is not None else None,
                "submitted_at": since.isoformat(),
                "running_at": None,
                "finished_at": None,
                "stage": job_stage(job),
            }
            self._write_job_record(context, record)

            try:
                outcome = self._watch_job(step_name, job, record)
            except BaseException:
                # Interrupted, or the Hub unreachable for longer than the retries cover. The
                # job itself may well be alive; say which one, since nothing watches it now.
                logger.warning("No longer watching job %s of step '%s'.", job.id, step_name)
                raise
            finally:
                self._submitted_job_ids.discard(job.id)

            failure = self._failure_of(step, outcome)
            record.update(
                finished_at=utc_now_datetime().isoformat(),
                stage=outcome.stage,
                failure=failure,
                running_seconds=round(outcome.running_seconds, 1),
                # As seen from here, to the nearest poll. The Hub bills somewhat more.
                estimated_cost_usd=(
                    None
                    if info is None
                    else round(outcome.running_seconds / 3600 * info.cost_per_hour, 4)
                ),
            )
            self._write_job_record(context, record)

            if failure is None:
                return outcome.job.url

            own_failure = self._own_failure(step, since)
            if own_failure is not None:
                failure = f"{failure}; the step failed with: {own_failure}"
            else:
                # The job is gone and wrote no end: do it here, or the step reads as running
                # forever and its lock blocks whoever tries next.
                try:
                    hub_call(self.workspace.step_abandoned, step, failure, job.id)  # type: ignore[attr-defined]
                except Exception:
                    logger.warning("Could not clear the state of '%s'.", step_name, exc_info=True)

            if own_failure is None and outcome.retryable and attempt < self.attempts:
                cli_logger.warning(
                    '[yellow]Job [b]%s[/] of step [b]"%s"[/] did not finish the step (%s). '
                    "Submitting it again (attempt %d of %d).[/]",
                    job.id,
                    step_name,
                    failure,
                    attempt + 1,
                    self.attempts,
                )
                continue

            raise StepFailedError(
                f"Step '{step_name}' failed: {failure}.\nLogs: {outcome.job.url}", outcome.job.url
            )

        raise AssertionError("unreachable")  # pragma: no cover

    def _cancel_submitted_jobs(self) -> None:
        for job_id in list(self._submitted_job_ids):
            try:
                self._cancel_job(job_id)
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

        # The driver's own timeout, not a step's: it has to outlive the whole graph, and a
        # driver killed at a step's timeout leaves its jobs running with nobody watching.
        job = hub_call(
            run_job,
            image=self.image,
            command=["bash", "-c", self._script(driver_cmd, Constants.DRIVER_LOGS_NAME)],
            flavor=self.driver_flavor,
            timeout=self.driver_timeout_seconds,
            env={**self._job_env(self.driver_flavor), NO_DETACH_ENV_VAR: "1"},
            secrets=self.secrets,
            volumes=volumes,
            namespace=self.namespace,
            labels={"tango-run": run_name or "", "name": f"tango-driver-{run_name or 'run'}"},
            token=self.token,
            statuses=frozenset({429}),
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
        self._out_of_credit.clear()

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
                elif isinstance(exc, OutOfCreditError):
                    # Never started, so there is nothing that failed.
                    not_run[step_name] = ExecutionMetadata()
                elif isinstance(exc, StepFailedError):
                    cli_logger.error("[red]\N{BALLOT X} %s[/]", exc)
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

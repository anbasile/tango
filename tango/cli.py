import logging
import multiprocessing as mp
import os
import sys
import warnings
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from tango.common.exceptions import CliRunError
from tango.common.logging import (
    cli_logger,
    initialize_logging,
    initialize_prefix_logging,
    teardown_logging,
)
from tango.common.params import Params
from tango.executor import Executor
from tango.settings import TangoGlobalSettings
from tango.step_graph import StepGraph
from tango.workspace import Workspace

if TYPE_CHECKING:
    from tango.executor import ExecutorOutput
    from tango.workspace import Run


logger = logging.getLogger(__name__)


def load_settings(settings: Union[str, Params, dict, None] = None) -> TangoGlobalSettings:
    return (
        TangoGlobalSettings.from_file(settings)
        if isinstance(settings, str)
        else (
            TangoGlobalSettings.from_params(settings)
            if isinstance(settings, (Params, dict))
            else TangoGlobalSettings.default()
        )
    )


@contextmanager
def tango_cli(settings: Union[TangoGlobalSettings, str, Params, dict, None] = None):
    if not isinstance(settings, TangoGlobalSettings):
        settings = load_settings(settings)

    try:
        initialize_cli(settings=settings, called_by_executor=False)
        yield
    finally:
        cleanup_cli()


def initialize_cli(
    settings: Optional[TangoGlobalSettings] = None,
    called_by_executor: bool = False,
):
    if settings is None:
        settings = TangoGlobalSettings.default()

    if not sys.warnoptions:
        warnings.simplefilter("default", category=DeprecationWarning)

    if settings.environment:
        from tango.common.aliases import EnvVarNames

        # These environment variables should not be set this way since they'll be ignored.
        blocked_env_variable_names = EnvVarNames.values()

        for key, value in settings.environment.items():
            if key not in blocked_env_variable_names:
                os.environ[key] = value
            else:
                warnings.warn(
                    f"Ignoring environment variable '{key}' from settings file. "
                    f"Please use the corresponding settings field instead.",
                    UserWarning,
                )

    mp.set_start_method(settings.multiprocessing_start_method)

    if not called_by_executor:
        initialize_logging(
            log_level=settings.log_level,
            file_friendly_logging=settings.file_friendly_logging,
            enable_cli_logs=True,
        )


def cleanup_cli():
    teardown_logging()


def prepare_workspace(
    settings: Optional[TangoGlobalSettings] = None,
    workspace_url: Optional[str] = None,
) -> Workspace:
    from tango.workspaces import default_workspace

    if settings is None:
        settings = TangoGlobalSettings.default()

    workspace: Workspace
    if workspace_url is not None:
        workspace = Workspace.from_url(workspace_url)
    elif settings.workspace is not None:
        workspace = Workspace.from_params(settings.workspace)
    else:
        workspace = default_workspace

    return workspace


def prepare_executor(
    workspace: Workspace,
    settings: Optional[TangoGlobalSettings] = None,
    include_package: Optional[Sequence[str]] = None,
    parallelism: Optional[int] = None,
    multicore: Optional[bool] = None,
    called_by_executor: bool = False,
    executor_options: Optional[Dict[str, Any]] = None,
) -> Executor:
    from tango.executors import MulticoreExecutor
    from tango.workspaces import MemoryWorkspace

    if settings is None:
        settings = TangoGlobalSettings.default()

    if executor_options and not called_by_executor:
        if settings.executor is None:
            raise CliRunError(
                "--executor-option changes the executor of the settings file, and "
                f"{settings.path or 'the settings'} names none."
            )
        settings.executor = {**settings.executor, **executor_options}

    executor: Executor
    if not called_by_executor and settings.executor is not None:
        if multicore is not None:
            logger.warning(
                "Ignoring argument 'multicore' since executor is defined in %s",
                settings.path or "setting",
            )
        executor = Executor.from_params(
            settings.executor,
            workspace=workspace,
            include_package=include_package,
            **(dict(parallelism=parallelism) if parallelism is not None else {}),  # type: ignore
        )
    else:
        # Determine if we can use the multicore executor.
        if multicore is None:
            if isinstance(workspace, MemoryWorkspace):
                # Memory workspace does not work with multiple cores.
                multicore = False
            elif "pydevd" in sys.modules:
                # Pydevd doesn't reliably follow child processes, so we disable multicore under the debugger.
                logger.warning("Debugger detected, disabling multicore.")
                multicore = False
            elif parallelism is None or parallelism == 0:
                multicore = False
            else:
                multicore = True

        if multicore:
            executor = MulticoreExecutor(
                workspace=workspace, include_package=include_package, parallelism=parallelism
            )
        else:
            executor = Executor(workspace=workspace, include_package=include_package)

    return executor


def dry_run(
    step_graph: StepGraph,
    workspace: Workspace,
    executor: Executor,
    name: Optional[str] = None,
    expect: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Say what ``tango run`` would do, and do none of it: no run is registered, nothing is
    uploaded, no step is started.

    Lists the steps that have no result in the workspace, each with its state and with where
    the executor would run it. With the ``name`` of an existing run, steps that are new or
    whose identity changed since that run are marked: a changed identity is how a finished
    step gets run, and paid for, a second time.

    :param expect: Shell-style patterns for the step names allowed to run. Raises
        :class:`~tango.common.exceptions.CliRunError` when a pending step matches none; an
        empty list means that nothing is expected to run.
    :returns: The names of the steps that would run.
    """
    from fnmatch import fnmatch

    previous: Optional[Dict[str, str]] = None
    if name is not None:
        reader = getattr(workspace, "run_step_ids", None)
        if reader is not None:
            previous = reader(name)
        else:
            try:
                previous = {
                    step_name: info.unique_id
                    for step_name, info in workspace.registered_run(name).steps.items()
                }
            except KeyError:
                previous = None

    uncacheable_leaves = step_graph.uncacheable_leaf_steps()
    pending: List[str] = []
    lines: List[str] = []
    for step_name, step in step_graph.items():
        if step.cache_results:
            if step in workspace.step_cache:
                continue
            try:
                state = workspace.step_info(step.unique_id).state.value
            except KeyError:
                state = "incomplete"
        elif step in uncacheable_leaves:
            state = "uncacheable"
        else:
            # Runs only inside the steps that depend on it.
            continue

        notes = [executor.describe_step(step), state]
        if previous is not None:
            if step_name not in previous:
                notes.append(f"new in run '{name}'")
            elif previous[step_name] != step.unique_id:
                notes.append(f"IDENTITY CHANGED since run '{name}'")
        pending.append(step_name)
        lines.append(f"  {step_name}  [{'; '.join(notes)}]")

    print(f"{len(pending)} of {len(step_graph)} steps would run")
    for line in lines:
        print(line)

    if expect is not None:
        unexpected = [
            step_name
            for step_name in pending
            if not any(fnmatch(step_name, pattern) for pattern in expect)
        ]
        if unexpected:
            print(f"UNEXPECTED: {', '.join(unexpected)}")
            raise CliRunError(f"{len(unexpected)} step(s) would run that --expect does not allow.")
    return pending


def execute_step_graph(
    step_graph: StepGraph,
    workspace: Optional[Workspace] = None,
    executor: Optional[Executor] = None,
    name: Optional[str] = None,
    called_by_executor: bool = False,
    step_names: Optional[Sequence[str]] = None,
) -> str:
    if workspace is None:
        workspace = prepare_workspace()
        executor = prepare_executor(workspace=workspace)
    elif executor is None:
        executor = prepare_executor(workspace=workspace)

    # Register run.
    run: "Run"
    if called_by_executor and name is not None:
        try:
            run = workspace.registered_run(name)
        except KeyError:
            raise RuntimeError(
                "The CLI was called by `MulticoreExecutor.execute_step_graph`, but "
                f"'{name}' is not already registered as a run. This should never happen!"
            )
    else:
        run = workspace.register_run((step for step in step_graph.values()), name)

    if called_by_executor:
        assert step_names is not None and len(step_names) == 1
        from tango.common.aliases import EnvVarNames

        # We set this environment variable so that any steps that contain multiprocessing
        # and call `initialize_worker_logging` also log the messages with the `step_name` prefix.
        os.environ[EnvVarNames.LOGGING_PREFIX.value] = f"step {step_names[0]}"
        # `MulticoreExecutor` runs each step as a child process that streams log records back
        # to its parent over a socket, and exports the port in this variable. A remote executor
        # has no such parent -- the step is alone in its own container -- so there is nothing to
        # connect to and this process is itself the main one. Without this check the step dies
        # immediately with "missing logging socket configuration".
        has_logging_socket = os.environ.get(EnvVarNames.LOGGING_PORT.value) is not None
        initialize_prefix_logging(
            prefix=f"step {step_names[0]}", main_process=not has_logging_socket
        )

    # Capture logs to file.
    with workspace.capture_logs_for_run(run.name) if not called_by_executor else nullcontext():
        if not called_by_executor:
            cli_logger.info("[green]Starting new run [bold]%s[/][/]", run.name)

        executor_output: ExecutorOutput = executor.execute_step_graph(step_graph, run_name=run.name)

        if executor_output.failed:
            cli_logger.error("[red]\N{BALLOT X} Run [bold]%s[/] finished with errors[/]", run.name)
        elif not called_by_executor:
            cli_logger.info("[green]\N{CHECK MARK} Finished run [bold]%s[/][/]", run.name)

        if executor_output is not None:
            if not called_by_executor:
                executor_output.display()
            if executor_output.failed:
                raise CliRunError

    return run.name

import os

import pytest

from tango.integrations.hf.executor import NO_DETACH_ENV_VAR, HfJobsExecutor
from tango.integrations.hf.workspace import HfBucketWorkspace
from tango.step import Step, StepResources
from tango.step_graph import StepGraph

from .fake_hub import FakeHfApi, install_fakes, install_job_fakes


@Step.register("hf_exec_add")
class AddStep(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, a: int, b: int) -> int:  # type: ignore[override]
        return a + b


@Step.register("hf_exec_double")
class DoubleStep(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, value: int) -> int:  # type: ignore[override]
        return value * 2


@pytest.fixture
def s3(monkeypatch, tmp_path):
    return install_fakes(monkeypatch, tmp_path / "cache")


@pytest.fixture
def jobs(monkeypatch):
    return install_job_fakes(monkeypatch)


@pytest.fixture
def workspace(s3):
    return HfBucketWorkspace("org/bucket")


def make_executor(workspace, tmp_path, **kwargs):
    kwargs.setdefault("project_dir", str(tmp_path))
    kwargs.setdefault("poll_interval", 0.01)
    return HfJobsExecutor(workspace, **kwargs)


class TestInstallCmd:
    def test_pyproject_implies_editable_install(self, workspace, jobs, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        assert make_executor(workspace, tmp_path).install_cmd == "pip install -e ."

    def test_requirements_txt(self, workspace, jobs, tmp_path):
        (tmp_path / "requirements.txt").write_text("tango\n")
        assert make_executor(workspace, tmp_path).install_cmd == "pip install -r requirements.txt"

    def test_nothing_to_install(self, workspace, jobs, tmp_path):
        assert make_executor(workspace, tmp_path).install_cmd == ""

    def test_explicit_override_wins(self, workspace, jobs, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        executor = make_executor(workspace, tmp_path, install_cmd="uv sync")
        assert executor.install_cmd == "uv sync"


class TestCommand:
    def test_invokes_tango_for_a_single_step(self, workspace, jobs, tmp_path):
        executor = make_executor(workspace, tmp_path, include_package=["my_steps"])
        script = executor._build_command("train", "my-run")[-1]

        assert "tango --called-by-executor run" in script
        assert "/tango/config/config.jsonnet" in script
        assert "-s train" in script
        assert f"-w {workspace.url}" in script
        assert "-n my-run" in script
        assert "-i my_steps" in script

    def test_copies_the_project_off_the_shared_mount(self, workspace, jobs, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        script = make_executor(workspace, tmp_path)._build_command("train", None)[-1]

        # Installing into the read-only mount shared by concurrent jobs would collide.
        assert "cp -a /tango/project /tmp/project" in script
        assert script.index("cp -a") < script.index("pip install -e .")
        assert "cd /tmp/project" in script

    def test_no_run_name_means_no_flag(self, workspace, jobs, tmp_path):
        script = make_executor(workspace, tmp_path)._build_command("train", None)[-1]
        assert "-n " not in script


class TestSubmission:
    def test_step_resources_choose_the_flavor(self, workspace, jobs, tmp_path):
        executor = make_executor(workspace, tmp_path, flavor="cpu-basic")
        graph = StepGraph({"train": AddStep(a=1, b=2, step_resources=StepResources(gpu_count=1))})
        executor.execute_step_graph(graph, run_name="r")

        assert jobs.submitted[0]["flavor"] == "t4-small"

    def test_default_flavor_when_a_step_asks_for_nothing(self, workspace, jobs, tmp_path):
        executor = make_executor(workspace, tmp_path, flavor="cpu-upgrade")
        executor.execute_step_graph(StepGraph({"train": AddStep(a=1, b=2)}), run_name="r")

        assert jobs.submitted[0]["flavor"] == "cpu-upgrade"

    def test_labels_identify_the_step(self, workspace, jobs, tmp_path):
        step = AddStep(a=1, b=2)
        executor = make_executor(workspace, tmp_path)
        executor.execute_step_graph(StepGraph({"train": step}), run_name="my-run")

        labels = jobs.submitted[0]["labels"]
        assert labels["tango-step"] == step.unique_id
        assert labels["tango-run"] == "my-run"

    def test_credentials_are_passed_as_secrets(self, workspace, jobs, tmp_path):
        executor = make_executor(workspace, tmp_path)
        executor.execute_step_graph(StepGraph({"train": AddStep(a=1, b=2)}), run_name="r")

        secrets = jobs.submitted[0]["secrets"]
        assert secrets["HF_TOKEN"] == "hf_faketoken"
        # The lock inside the job needs these just as much as the client does.
        assert secrets["HF_S3_ACCESS_KEY_ID"] == "HFAKTEST"
        assert secrets["HF_S3_SECRET_ACCESS_KEY"] == "secret"

    def test_both_mounts_are_attached_read_only(self, workspace, jobs, tmp_path):
        executor = make_executor(workspace, tmp_path)
        executor.execute_step_graph(StepGraph({"train": AddStep(a=1, b=2)}), run_name="r")

        volumes = jobs.submitted[0]["volumes"]
        assert [v.mount_path for v in volumes] == ["/tango/project", "/tango/config"]
        assert all(v.type == "bucket" and v.read_only for v in volumes)
        assert all(v.source == "org/bucket" for v in volumes)


class TestProjectSync:
    """
    `sync_job_volume` takes no exclusions, so the project goes through `sync_bucket` instead.
    Getting this wrong means uploading the virtualenv on every run.
    """

    def _tree(self, root):
        for relative in (
            "main.py",
            "pkg/mod.py",
            "pkg/__pycache__/mod.pyc",
            ".venv/lib/torch/x.so",
            ".git/objects/ab/cdef",
            "data/keep.jsonl",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")

    def _uploaded(self, prefix="_project/"):
        return sorted(
            key.split("/", 2)[-1] for key in FakeHfApi.STORE["org/bucket"] if key.startswith(prefix)
        )

    def test_venv_git_and_pycache_are_not_uploaded(self, workspace, jobs, tmp_path):
        self._tree(tmp_path)
        executor = make_executor(workspace, tmp_path)
        executor.execute_step_graph(StepGraph({"train": AddStep(a=1, b=2)}), run_name="r")

        assert self._uploaded() == ["data/keep.jsonl", "main.py", "pkg/mod.py"]

    def test_exclusions_can_be_overridden(self, workspace, jobs, tmp_path):
        self._tree(tmp_path)
        executor = make_executor(workspace, tmp_path, project_exclude=["data/**"])
        executor.execute_step_graph(StepGraph({"train": AddStep(a=1, b=2)}), run_name="r")

        uploaded = self._uploaded()
        assert "data/keep.jsonl" not in uploaded
        assert ".venv/lib/torch/x.so" in uploaded, "override should replace, not extend"

    def test_exclusions_can_be_added_to_the_defaults(self, workspace, jobs, tmp_path):
        self._tree(tmp_path)
        executor = make_executor(workspace, tmp_path, extra_project_exclude=["data/**"])
        executor.execute_step_graph(StepGraph({"train": AddStep(a=1, b=2)}), run_name="r")

        assert self._uploaded() == ["main.py", "pkg/mod.py"]

    def test_the_config_is_uploaded_where_the_command_looks_for_it(self, workspace, jobs, tmp_path):
        executor = make_executor(workspace, tmp_path)
        executor.execute_step_graph(StepGraph({"train": AddStep(a=1, b=2)}), run_name="my-run")

        assert "_config/my-run/config.jsonnet" in FakeHfApi.STORE["org/bucket"]
        config_volume = jobs.submitted[0]["volumes"][1]
        assert config_volume.path == "_config/my-run"

    def test_a_deleted_file_stops_being_mirrored(self, workspace, jobs, tmp_path):
        self._tree(tmp_path)
        executor = make_executor(workspace, tmp_path)
        executor.execute_step_graph(StepGraph({"train": AddStep(a=1, b=2)}), run_name="r")
        assert "pkg/mod.py" in self._uploaded()

        (tmp_path / "pkg" / "mod.py").unlink()
        executor.execute_step_graph(StepGraph({"train": AddStep(a=3, b=4)}), run_name="r2")
        assert "pkg/mod.py" not in self._uploaded()


class TestWorkspaceRequirement:
    def test_an_ephemeral_workspace_is_refused(self, jobs, tmp_path):
        from tango.common.exceptions import ConfigurationError
        from tango.workspaces import LocalWorkspace

        # A Job's local disk vanishes with the container, so a LocalWorkspace would silently
        # lose every result.
        with pytest.raises(ConfigurationError, match="HfBucketWorkspace"):
            HfJobsExecutor(LocalWorkspace(tmp_path / "ws"), project_dir=str(tmp_path))

    def test_timeout_is_an_hour_not_the_platform_default(self, workspace, jobs, tmp_path):
        # The platform kills jobs after 30 minutes by default, which is too short to train.
        assert make_executor(workspace, tmp_path).timeout == "1h"


class TestExecution:
    def test_dependencies_run_in_order(self, workspace, jobs, tmp_path):
        add = AddStep(a=1, b=2)
        graph = StepGraph({"add": add, "double": DoubleStep(value=add)})
        output = make_executor(workspace, tmp_path).execute_step_graph(graph, run_name="r")

        assert set(output.successful) == {"add", "double"}
        assert not output.failed
        submitted_steps = [job["labels"]["name"] for job in jobs.submitted]
        assert submitted_steps.index("add") < submitted_steps.index("double")

    def test_a_local_step_is_not_submitted(self, workspace, jobs, tmp_path):
        graph = StepGraph({"add": AddStep(a=1, b=2, step_resources=StepResources(machine="local"))})
        output = make_executor(workspace, tmp_path).execute_step_graph(graph, run_name="r")

        assert jobs.submitted == []
        assert set(output.successful) == {"add"}
        assert workspace.step_cache[graph["add"]] == 3

    def test_a_cached_step_is_not_submitted(self, workspace, jobs, tmp_path):
        step = AddStep(a=1, b=2)
        step.ensure_result(workspace)

        output = make_executor(workspace, tmp_path).execute_step_graph(
            StepGraph({"add": AddStep(a=1, b=2)}), run_name="r"
        )
        assert jobs.submitted == []
        assert set(output.successful) == {"add"}

    def test_a_failed_job_is_reported_with_its_url(self, monkeypatch, workspace, tmp_path):
        jobs = install_job_fakes(monkeypatch, stage="ERROR")
        output = make_executor(workspace, tmp_path).execute_step_graph(
            StepGraph({"add": AddStep(a=1, b=2)}), run_name="r"
        )

        assert set(output.failed) == {"add"}
        assert output.failed["add"].logs_location == "https://hf.co/jobs/job-1"

    def test_dependents_of_a_failed_step_are_not_run(self, monkeypatch, workspace, tmp_path):
        install_job_fakes(monkeypatch, stage="ERROR")
        add = AddStep(a=1, b=2)
        graph = StepGraph({"add": add, "double": DoubleStep(value=add)})
        output = make_executor(workspace, tmp_path).execute_step_graph(graph, run_name="r")

        assert set(output.failed) == {"add"}
        assert set(output.not_run) == {"double"}

    def test_reattaches_to_a_job_already_running_the_step(
        self, monkeypatch, workspace, jobs, tmp_path
    ):
        from .fake_hub import FakeJob

        step = AddStep(a=1, b=2)
        jobs.running = [FakeJob("existing-job", "RUNNING", url="https://hf.co/jobs/existing-job")]
        inspect_job = jobs.inspect_job

        def finish_then_inspect(**kwargs):
            # The job this executor did not submit finishes the step while it is watched.
            jobs.write_result("org/bucket", step.unique_id)
            return inspect_job(**kwargs)

        monkeypatch.setattr("huggingface_hub.inspect_job", finish_then_inspect)

        output = make_executor(workspace, tmp_path).execute_step_graph(
            StepGraph({"add": step}), run_name="r"
        )
        assert jobs.submitted == [], "should not pay to run the same step twice"
        assert output.successful["add"].logs_location == "https://hf.co/jobs/existing-job"


def started_by_job(monkeypatch, workspace, step, job_id):
    """
    Leave the workspace as a Job leaves it when it dies mid-step: the lock taken in the Job's
    name and the step marked as running. Returns the lock, which the caller has to keep alive
    (a lock object releases itself when collected).
    """
    monkeypatch.setenv("JOB_ID", job_id)
    workspace.step_starting(step)
    monkeypatch.delenv("JOB_ID")
    return workspace.locks.pop(step)


class TestJobOutcome:
    """
    What the executor concludes from how a job ended. Each of these cost money once.
    """

    def test_a_completed_job_without_a_result_is_a_failure(self, workspace, jobs, tmp_path):
        # Three jobs once took the lock, went silent and exited 0. They were reported as
        # succeeded, and the steps depending on them failed later.
        jobs.produce_results = False
        add = AddStep(a=1, b=2)
        graph = StepGraph({"add": add, "double": DoubleStep(value=add)})
        output = make_executor(workspace, tmp_path).execute_step_graph(graph, run_name="r")

        assert set(output.failed) == {"add"}
        assert set(output.not_run) == {"double"}

    def test_a_dead_job_leaves_neither_lock_nor_running_state(
        self, monkeypatch, workspace, s3, tmp_path
    ):
        from tango.step_info import StepState

        install_job_fakes(monkeypatch, stage="ERROR")
        step = AddStep(a=1, b=2)
        lock = started_by_job(monkeypatch, workspace, step, "job-1")
        assert workspace.step_info(step).state == StepState.RUNNING
        assert s3.objects

        output = make_executor(workspace, tmp_path).execute_step_graph(
            StepGraph({"add": step}), run_name="r"
        )
        assert set(output.failed) == {"add"}
        info = workspace.step_info(step)
        assert info.state == StepState.FAILED
        assert "ERROR" in info.error
        assert s3.objects == {}, "the lock of the dead job should be gone"
        del lock

    def test_a_lock_left_behind_does_not_make_the_step_look_cached(
        self, monkeypatch, workspace, s3, jobs, tmp_path
    ):
        # The lock is `tango-step-<id>-lock`, next to the result folder `tango-step-<id>/`.
        # Matching by prefix read the one as the other: the step was skipped as cached.
        step = AddStep(a=1, b=2)
        lock = started_by_job(monkeypatch, workspace, step, "some-dead-job")
        monkeypatch.setattr(
            "huggingface_hub.inspect_job", lambda **kw: FakeJobFor(kw["job_id"], "CANCELED")
        )

        output = make_executor(workspace, tmp_path).execute_step_graph(
            StepGraph({"add": step}), run_name="r"
        )
        assert len(jobs.submitted) == 1, "the step has no result, so it has to run"
        assert set(output.successful) == {"add"}
        del lock

    def test_a_job_past_its_timeout_is_cancelled(self, workspace, jobs, tmp_path):
        # The platform once let a job with a four-hour timeout run for five and a half.
        jobs.script = {"add": [["RUNNING"]]}
        executor = make_executor(workspace, tmp_path, attempts=2)
        executor.timeout_seconds = 0.05  # type: ignore[assignment]
        executor.TIMEOUT_GRACE = 0.0
        output = executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="r")

        assert jobs.cancelled == ["job-1"]
        assert set(output.failed) == {"add"}
        assert len(jobs.submitted) == 1, "a step that ran out of time is not submitted again"

    def test_a_job_that_ignores_cancellation_does_not_hang_the_run(self, workspace, jobs, tmp_path):
        jobs.script = {"add": [["RUNNING"]]}
        jobs.honour_cancel = False
        executor = make_executor(workspace, tmp_path)
        executor.timeout_seconds = 0.05  # type: ignore[assignment]
        executor.TIMEOUT_GRACE = 0.0
        executor.CANCEL_WAIT = 0.05
        output = executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="r")
        assert set(output.failed) == {"add"}

    def test_a_job_that_ignores_cancellation_keeps_its_lock(
        self, monkeypatch, workspace, s3, jobs, tmp_path
    ):
        # Nothing says that job is over, so its lock is not taken from it and no second job
        # is started next to it.
        jobs.script = {"add": [["SCHEDULING"]]}
        jobs.honour_cancel = False
        step = AddStep(a=1, b=2)
        lock = started_by_job(monkeypatch, workspace, step, "job-1")
        executor = make_executor(workspace, tmp_path, attempts=2)
        executor.scheduling_timeout_seconds = 0.05  # type: ignore[assignment]
        executor.CANCEL_WAIT = 0.05
        output = executor.execute_step_graph(StepGraph({"add": step}), run_name="r")

        assert set(output.failed) == {"add"}
        assert len(jobs.submitted) == 1
        assert s3.objects, "the lock of a job that may be alive stays"
        del lock

    def test_a_job_that_never_starts_is_cancelled_and_submitted_again(
        self, workspace, jobs, tmp_path
    ):
        jobs.script = {"add": [["SCHEDULING"], ["RUNNING", "COMPLETED"]]}
        executor = make_executor(workspace, tmp_path, attempts=2)
        executor.scheduling_timeout_seconds = 0.05  # type: ignore[assignment]
        output = executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="r")

        assert jobs.cancelled == ["job-1"]
        assert len(jobs.submitted) == 2
        assert set(output.successful) == {"add"}
        assert output.successful["add"].logs_location == "https://hf.co/jobs/job-2"


class TestInterrupt:
    def _interrupt_once_submitted(self, monkeypatch, jobs):
        import concurrent.futures

        real_wait = concurrent.futures.wait
        interrupted = []

        def wait(*args, **kwargs):
            if jobs.submitted and not interrupted:
                interrupted.append(True)
                raise KeyboardInterrupt
            return real_wait(*args, **kwargs)

        monkeypatch.setattr(concurrent.futures, "wait", wait)

    def test_an_interrupt_cancels_the_jobs_and_frees_their_steps(
        self, monkeypatch, workspace, s3, jobs, tmp_path
    ):
        from tango.step_info import StepState

        # A job that would run for ever: the run only ends if it is cancelled.
        jobs.script = {"add": [["RUNNING"]]}
        step = AddStep(a=1, b=2)
        lock = started_by_job(monkeypatch, workspace, step, "job-1")
        self._interrupt_once_submitted(monkeypatch, jobs)

        executor = make_executor(workspace, tmp_path)
        with pytest.raises(KeyboardInterrupt):
            executor.execute_step_graph(StepGraph({"add": step}), run_name="r")

        assert jobs.cancelled == ["job-1"]
        info = workspace.step_info(step)
        assert info.state == StepState.FAILED
        assert "cancelled with the run" in info.error
        assert s3.objects == {}, "the lock of the cancelled job should be gone"
        del lock

    def test_a_job_submitted_during_the_interrupt_is_cancelled_too(
        self, monkeypatch, workspace, jobs, tmp_path
    ):
        # The submission is in flight when the interrupt comes, so nobody knows the job yet.
        import threading

        jobs.script = {"add": [["RUNNING"]]}
        executor = make_executor(workspace, tmp_path)
        submitting, release = threading.Event(), threading.Event()
        real_run_job = jobs.run_job

        def slow_run_job(**kwargs):
            submitting.set()
            release.wait(5.0)
            return real_run_job(**kwargs)

        monkeypatch.setattr("huggingface_hub.run_job", slow_run_job)

        import concurrent.futures

        real_wait = concurrent.futures.wait
        interrupted = []

        def wait(*args, **kwargs):
            if not interrupted:
                submitting.wait(5.0)
                interrupted.append(True)
                threading.Timer(0.2, release.set).start()
                raise KeyboardInterrupt
            return real_wait(*args, **kwargs)

        monkeypatch.setattr(concurrent.futures, "wait", wait)
        with pytest.raises(KeyboardInterrupt):
            executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="r")
        assert jobs.cancelled == ["job-1"]


def FakeJobFor(job_id, stage):
    from .fake_hub import FakeJob

    return FakeJob(job_id, stage)


class TestAttempts:
    def test_a_job_that_dies_is_submitted_again(self, workspace, jobs, tmp_path):
        # A transient 401 while mounting the bucket failed a step before it started.
        jobs.script = {"add": [["RUNNING", "ERROR"], ["RUNNING", "COMPLETED"]]}
        output = make_executor(workspace, tmp_path, attempts=2).execute_step_graph(
            StepGraph({"add": AddStep(a=1, b=2)}), run_name="r"
        )
        assert len(jobs.submitted) == 2
        assert set(output.successful) == {"add"}

    def test_one_attempt_by_default(self, workspace, jobs, tmp_path):
        jobs.script = {"add": [["RUNNING", "ERROR"], ["RUNNING", "COMPLETED"]]}
        output = make_executor(workspace, tmp_path).execute_step_graph(
            StepGraph({"add": AddStep(a=1, b=2)}), run_name="r"
        )
        assert len(jobs.submitted) == 1
        assert set(output.failed) == {"add"}

    def test_a_step_that_raised_is_not_submitted_again(
        self, monkeypatch, workspace, jobs, tmp_path
    ):
        # The step's own error is in its record: running it again would raise again.
        step = AddStep(a=1, b=2)
        jobs.script = {"add": [["RUNNING", "ERROR"], ["RUNNING", "COMPLETED"]]}
        original = jobs.run_job

        def run_job_then_fail_the_step(**kwargs):
            job = original(**kwargs)
            workspace.step_starting(step)
            workspace.step_failed(step, ValueError("bad learning rate"))
            return job

        monkeypatch.setattr("huggingface_hub.run_job", run_job_then_fail_the_step)
        output = make_executor(workspace, tmp_path, attempts=3).execute_step_graph(
            StepGraph({"add": step}), run_name="r"
        )
        assert len(jobs.submitted) == 1
        assert set(output.failed) == {"add"}
        assert "bad learning rate" in workspace.step_info(step).error

    def test_a_job_cancelled_by_hand_is_not_submitted_again(self, workspace, jobs, tmp_path):
        jobs.script = {"add": [["RUNNING", "CANCELED"], ["RUNNING", "COMPLETED"]]}
        output = make_executor(workspace, tmp_path, attempts=2).execute_step_graph(
            StepGraph({"add": AddStep(a=1, b=2)}), run_name="r"
        )
        assert len(jobs.submitted) == 1
        assert set(output.failed) == {"add"}

    def test_attempts_must_be_positive(self, workspace, jobs, tmp_path):
        from tango.common.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="attempts"):
            make_executor(workspace, tmp_path, attempts=0)


class TestHubRequests:
    """
    The Hub allows 1,000 API requests per five minutes. Two runs at once went over it, and a
    single 429 failed both.
    """

    def test_all_jobs_are_polled_with_one_request(self, workspace, jobs, tmp_path):
        jobs.script = {name: [["RUNNING"] * 5 + ["COMPLETED"]] for name in "abcd"}
        graph = StepGraph({name: AddStep(a=i, b=1) for i, name in enumerate("abcd")})
        output = make_executor(workspace, tmp_path, parallelism=4).execute_step_graph(
            graph, run_name="r"
        )
        assert set(output.successful) == set("abcd")
        assert jobs.count("inspect_job") == 0
        listings = [kw for name, kw in jobs.calls if name == "list_jobs" and kw["status"] is None]
        # Six polls for each of four jobs would be 24 requests one by one.
        assert len(listings) <= 12
        assert all("tango-session" in kw["labels"] for kw in listings)

    def test_the_namespace_is_looked_up_once(self, workspace, jobs, tmp_path):
        executor = make_executor(workspace, tmp_path)
        executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="r")

        assert executor.namespace == "fake-user"
        assert jobs.count("whoami") == 1
        for name, kwargs in jobs.calls:
            if name in {"run_job", "list_jobs", "inspect_job", "cancel_job"}:
                assert kwargs["namespace"] == "fake-user", name

    def test_a_given_namespace_is_used_everywhere(self, workspace, jobs, tmp_path):
        jobs.script = {"add": [["RUNNING"]]}
        executor = make_executor(workspace, tmp_path, namespace="my-org")
        executor.timeout_seconds = 0.05  # type: ignore[assignment]
        executor.TIMEOUT_GRACE = 0.0
        executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="r")

        assert jobs.count("whoami") == 0
        # Without it the Hub looks a job up under the token's own account.
        assert [kw["namespace"] for name, kw in jobs.calls if name == "cancel_job"] == ["my-org"]

    def test_a_rate_limit_while_polling_is_waited_out(self, monkeypatch, workspace, jobs, tmp_path):
        from tango.integrations.hf import common

        from .fake_hub import fake_http_error

        monkeypatch.setattr(common, "RETRY_BASE_SECONDS", 0.01)
        jobs.script = {"add": [["RUNNING", "RUNNING", "COMPLETED"]]}
        jobs.failures["list_jobs"] = [
            fake_http_error(429),
            fake_http_error(429, {"Retry-After": "0.02"}),
            fake_http_error(503),
        ]
        # The first `list_jobs` is the "already running?" question, whose failure is ignored.
        output = make_executor(workspace, tmp_path).execute_step_graph(
            StepGraph({"add": AddStep(a=1, b=2)}), run_name="r"
        )
        assert set(output.successful) == {"add"}
        assert jobs.failures["list_jobs"] == []

    def test_polling_falls_back_to_one_job_at_a_time(self, workspace, jobs, tmp_path):
        from .fake_hub import fake_http_error

        jobs.script = {"add": [["RUNNING", "COMPLETED"]]}
        executor = make_executor(workspace, tmp_path)
        # Not an error worth retrying: the listing is given up, the run is not.
        jobs.failures["list_jobs"] = [fake_http_error(400), fake_http_error(400)]
        output = executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="r")

        assert set(output.successful) == {"add"}
        assert jobs.count("inspect_job") >= 1

    def test_a_submission_whose_answer_is_lost_is_not_repeated(self, workspace, jobs, tmp_path):
        # The job may exist. Submitting again would pay for the step twice.
        import httpx

        jobs.failures["run_job"] = [httpx.ReadTimeout("no answer")]
        output = make_executor(workspace, tmp_path).execute_step_graph(
            StepGraph({"add": AddStep(a=1, b=2)}), run_name="r"
        )
        assert jobs.count("run_job") == 1
        assert set(output.failed) == {"add"}

    def test_out_of_credit_stops_submitting(self, workspace, jobs, tmp_path):
        from .fake_hub import fake_http_error

        jobs.failures["run_job"] = [fake_http_error(402)]
        first = AddStep(a=1, b=2)
        graph = StepGraph(
            {"first": first, "second": AddStep(a=3, b=4), "third": DoubleStep(value=first)}
        )
        output = make_executor(workspace, tmp_path, parallelism=1).execute_step_graph(
            graph, run_name="r"
        )
        # One refusal is enough: nothing else is tried, and nothing is reported as failed.
        assert jobs.count("run_job") == 1
        assert not output.failed and not output.successful
        assert set(output.not_run) == {"first", "second", "third"}


class TestJobEnvironment:
    def _env(self, workspace, jobs, tmp_path, **kwargs):
        executor = make_executor(workspace, tmp_path, **kwargs)
        executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="r")
        return jobs.submitted[0]["env"]

    def test_defaults(self, workspace, jobs, tmp_path):
        env = self._env(workspace, jobs, tmp_path, flavor="cpu-upgrade")
        assert env["TANGO_HF_JOB"] == "1"
        assert env["TANGO_HF_FLAVOR"] == "cpu-upgrade"
        # The image has no git; GitPython refuses to import without this.
        assert env["GIT_PYTHON_REFRESH"] == "quiet"
        # At the default level a step's own logging never reached the job log.
        assert env["TANGO_LOG_LEVEL"] == "info"
        assert env["TQDM_DISABLE"] == "1"
        # cpu-upgrade has 8 vCPUs; the container sees every core of the host.
        assert env["OMP_NUM_THREADS"] == "8"
        assert env["MKL_NUM_THREADS"] == "8"

    def test_the_users_env_wins(self, workspace, jobs, tmp_path):
        env = self._env(workspace, jobs, tmp_path, env={"OMP_NUM_THREADS": 1, "MY_FLAG": "x"})
        assert env["OMP_NUM_THREADS"] == "1"
        assert env["MY_FLAG"] == "x"

    def test_the_command_goes_through_the_job_module(self, workspace, jobs, tmp_path):
        step = AddStep(a=1, b=2)
        executor = make_executor(workspace, tmp_path)
        executor.execute_step_graph(StepGraph({"add": step}), run_name="r")

        script = jobs.commands[0]
        # `exec`: the signal of a cancelled job has to reach the wrapper, not bash.
        assert "exec python -m tango.integrations.hf.job --bucket org/bucket" in script
        assert f"--log-owner {step.unique_id}" in script
        assert f"--result-of {step.unique_id}" in script
        assert "-- tango --called-by-executor run" in script

    def test_secrets_can_be_named_instead_of_written_down(
        self, monkeypatch, workspace, jobs, tmp_path
    ):
        from tango.common.exceptions import ConfigurationError

        monkeypatch.setenv("WANDB_API_KEY", "w-secret")
        executor = make_executor(workspace, tmp_path, secrets_from_env=["WANDB_API_KEY"])
        executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="r")
        assert jobs.submitted[0]["secrets"]["WANDB_API_KEY"] == "w-secret"

        with pytest.raises(ConfigurationError, match="NOT_SET_ANYWHERE"):
            make_executor(workspace, tmp_path, secrets_from_env=["NOT_SET_ANYWHERE"])


class TestTimeoutSetting:
    def test_seconds_reach_the_platform(self, workspace, jobs, tmp_path):
        executor = make_executor(workspace, tmp_path, timeout="4h")
        executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="r")
        assert jobs.submitted[0]["timeout"] == 4 * 3600

    def test_a_bare_number_is_accepted_from_settings(self, workspace, jobs, tmp_path):
        from tango.common.params import Params
        from tango.executor import Executor

        # `timeout: 3600` in tango.yml is an integer, and used to be rejected.
        executor = Executor.from_params(
            Params({"type": "hf", "timeout": 3600, "project_dir": str(tmp_path)}),
            workspace=workspace,
        )
        assert executor.timeout_seconds == 3600  # type: ignore[attr-defined]


class TestJobRecords:
    def test_every_job_leaves_a_record(self, workspace, jobs, tmp_path):
        import json

        jobs.script = {"add": [["RUNNING", "RUNNING", "ERROR"]]}
        executor = make_executor(workspace, tmp_path, flavor="cpu-upgrade", poll_interval=0.06)
        executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="my-run")

        # Cancelled and failed jobs report no running time on the Hub, and all jobs drop off
        # its list after a while. This is what is left to say what a run cost.
        record = json.loads(FakeHfApi.STORE["org/bucket"]["jobs/my-run/job-1.json"])
        assert record["step"] == "add"
        assert record["flavor"] == "cpu-upgrade"
        assert record["price_per_hour"] == 0.03
        assert record["stage"] == "ERROR"
        assert record["submitted_at"] and record["running_at"] and record["finished_at"]
        assert record["running_seconds"] > 0
        assert "ERROR" in record["failure"]


class TestDetach:
    def test_submits_one_driver_job(self, workspace, jobs, tmp_path):
        executor = make_executor(workspace, tmp_path, detach=True)
        add = AddStep(a=1, b=2)
        graph = StepGraph({"add": add, "double": DoubleStep(value=add)})
        output = executor.execute_step_graph(graph, run_name="my-run")

        assert len(jobs.submitted) == 1
        driver = jobs.submitted[0]
        assert driver["flavor"] == "cpu-basic"
        # No `--called-by-executor`: the driver is a full run that fans out on its own.
        assert "--called-by-executor" not in driver["command"][-1]
        assert "-s " not in driver["command"][-1]
        assert driver["env"][NO_DETACH_ENV_VAR] == "1"
        assert set(output.not_run) == {"add", "double"}

    def test_the_driver_gets_a_settings_file_naming_this_executor(self, workspace, jobs, tmp_path):
        """
        Without it the driver falls back to the default executor and runs every step inside
        the driver container instead of fanning out.
        """
        import yaml

        executor = make_executor(workspace, tmp_path, detach=True, flavor="t4-small")
        executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="my-run")

        raw = FakeHfApi.STORE["org/bucket"]["_config/my-run/tango.yml"]
        settings = yaml.safe_load(raw)
        assert settings["executor"]["type"] == "hf"
        assert settings["executor"]["flavor"] == "t4-small"
        # The driver's own copy, already installed by its entrypoint.
        assert settings["executor"]["project_dir"] == "/tmp/project"
        assert "--settings /tango/config/tango.yml" in jobs.submitted[0]["command"][-1]

    def test_the_driver_gets_every_setting_that_shapes_a_step_job(
        self, monkeypatch, workspace, jobs, tmp_path
    ):
        import yaml

        monkeypatch.setenv("WANDB_API_KEY", "w-secret")
        executor = make_executor(
            workspace,
            tmp_path,
            detach=True,
            timeout="4h",
            attempts=2,
            namespace="my-org",
            env={"OMP_NUM_THREADS": "1"},
            secrets={"OTHER": "o-secret"},
            secrets_from_env=["WANDB_API_KEY"],
            extra_project_exclude=["data-local/**"],
        )
        executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="my-run")

        raw = FakeHfApi.STORE["org/bucket"]["_config/my-run/tango.yml"]
        settings = yaml.safe_load(raw)["executor"]
        assert settings["timeout"] == 4 * 3600
        assert settings["attempts"] == 2
        assert settings["namespace"] == "my-org"
        assert settings["env"] == {"OMP_NUM_THREADS": "1"}
        assert "data-local/**" in settings["project_exclude"]
        # Names only. The values are the driver's own secrets, not text in the bucket.
        assert settings["secrets_from_env"] == ["OTHER", "WANDB_API_KEY"]
        assert b"w-secret" not in raw and b"o-secret" not in raw
        assert jobs.submitted[0]["secrets"]["OTHER"] == "o-secret"

    def test_the_driver_has_its_own_timeout_and_flavor(self, workspace, jobs, tmp_path):
        # A driver on a step's timeout is killed mid-graph and leaves its jobs unwatched.
        executor = make_executor(
            workspace, tmp_path, detach=True, timeout="1h", driver_timeout="2d"
        )
        executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="my-run")

        driver = jobs.submitted[0]
        assert driver["timeout"] == 2 * 86400
        assert driver["flavor"] == "cpu-basic"
        assert "--log-owner _driver" in driver["command"][-1]
        assert "--result-of" not in driver["command"][-1]

    def test_a_step_job_gets_no_settings_file(self, workspace, jobs, tmp_path):
        # Step jobs pass --called-by-executor, which ignores the settings executor anyway.
        executor = make_executor(workspace, tmp_path)
        executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="my-run")
        assert "_config/my-run/tango.yml" not in FakeHfApi.STORE["org/bucket"]

    def test_a_driver_job_does_not_detach_again(self, monkeypatch, workspace, jobs, tmp_path):
        """
        Without this guard the driver would submit a driver, forever.
        """
        monkeypatch.setenv(NO_DETACH_ENV_VAR, "1")
        executor = make_executor(workspace, tmp_path, detach=True)
        assert executor.detach is False

        executor.execute_step_graph(StepGraph({"add": AddStep(a=1, b=2)}), run_name="r")
        assert "--called-by-executor" in jobs.submitted[0]["command"][-1]


class TestStandaloneStepInvocation:
    """
    The executor's job command is `tango --called-by-executor run ... -s <step>`, which is the
    same entry point `MulticoreExecutor` uses. But multicore runs its children beside a parent
    that owns a logging socket, whereas a Job's step is alone in its container. Run the command
    the way a container does -- no TANGO_LOGGING_PORT -- and make sure it still works.
    """

    def test_runs_without_a_parent_logging_socket(self, tmp_path):
        import subprocess
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parents[3]
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"TANGO_LOGGING_PORT", "TANGO_LOGGING_PREFIX"}
        }
        env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"

        result = subprocess.run(
            [
                "tango",
                "--called-by-executor",
                "run",
                str(repo / "test_fixtures" / "integrations" / "hf" / "config.jsonnet"),
                "-s",
                "make",
                "-w",
                str(tmp_path / "ws"),
                "-i",
                "test_fixtures.integrations.hf.components",
            ],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert "missing logging socket configuration" not in result.stderr
        assert result.returncode == 0, result.stderr[-3000:]


class TestStepIdentity:
    def test_identities_are_what_they_were_in_v2_1_0(self):
        # A step's identity decides whether its stored result is found again. Nothing in the
        # executor may leak into it: these are the ids v2.1.0 gave the same steps.
        plain = AddStep(a=1, b=2)
        with_resources = AddStep(a=1, b=2, step_resources=StepResources(gpu_count=1))
        assert plain.unique_id == PLAIN_ID
        assert with_resources.unique_id == plain.unique_id


PLAIN_ID = "AddStep-2833Mxk7BZLPCofHUsfHbBWWZwurefYi"

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

    def test_reattaches_to_a_job_already_running_the_step(self, workspace, jobs, tmp_path):
        from .fake_hub import FakeJob

        step = AddStep(a=1, b=2)
        jobs.running = [FakeJob("existing-job", "RUNNING", url="https://hf.co/jobs/existing-job")]

        output = make_executor(workspace, tmp_path).execute_step_graph(
            StepGraph({"add": step}), run_name="r"
        )
        assert jobs.submitted == [], "should not pay to run the same step twice"
        assert output.successful["add"].logs_location == "https://hf.co/jobs/existing-job"


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

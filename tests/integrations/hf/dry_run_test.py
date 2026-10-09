import pytest

from tango.cli import dry_run, prepare_executor
from tango.common.exceptions import CliRunError
from tango.integrations.hf.workspace import HfBucketWorkspace
from tango.settings import TangoGlobalSettings
from tango.step import Step, StepResources
from tango.step_graph import StepGraph

from .fake_hub import FakeHfApi, install_fakes, install_job_fakes


@Step.register("hf_dry_add")
class AddStep(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, a: int, b: int) -> int:  # type: ignore[override]
        return a + b


@pytest.fixture
def jobs(monkeypatch, tmp_path):
    install_fakes(monkeypatch, tmp_path / "cache")
    return install_job_fakes(monkeypatch)


@pytest.fixture
def workspace(jobs):
    return HfBucketWorkspace("org/bucket")


def make_executor(workspace, tmp_path, **options):
    settings = TangoGlobalSettings(
        executor={"type": "hf", "project_dir": str(tmp_path), "poll_interval": 0.01, **options}
    )
    return prepare_executor(workspace, settings=settings)


def graph():
    return StepGraph(
        {
            "features": AddStep(a=1, b=2, step_resources=StepResources(gpu_count=1)),
            "sweep": AddStep(a=3, b=4, step_resources=StepResources(cpu_count=8)),
            "report": AddStep(a=5, b=6, step_resources=StepResources(machine="local")),
        }
    )


class TestDryRun:
    def test_it_says_what_each_step_would_cost(self, workspace, jobs, tmp_path, capsys):
        executor = make_executor(workspace, tmp_path, timeout="4h")
        pending = dry_run(graph(), workspace, executor)

        assert pending == ["features", "sweep", "report"]
        output = capsys.readouterr().out
        assert "3 of 3 steps would run" in output
        assert "features  [job on t4-small, $0.40/h, at most $1.60; incomplete]" in output
        assert "sweep  [job on cpu-upgrade, $0.03/h, at most $0.12; incomplete]" in output
        assert "report  [local; incomplete]" in output

    def test_nothing_is_registered_uploaded_or_started(self, workspace, jobs, tmp_path):
        before = set(FakeHfApi.STORE["org/bucket"])
        dry_run(graph(), workspace, make_executor(workspace, tmp_path), name="main")

        assert jobs.submitted == []
        added = set(FakeHfApi.STORE["org/bucket"]) - before
        assert not any(key.startswith(("runs/", "_project/", "_config/")) for key in added)

    def test_a_changed_identity_is_pointed_out(self, workspace, jobs, tmp_path, capsys):
        # Adding an argument to a step gave a finished grid new identities; the next run paid
        # for all of it again.
        executor = make_executor(workspace, tmp_path)
        steps = graph()
        workspace.register_run(steps.values(), name="main")
        for step in steps.values():
            jobs.write_result("org/bucket", step.unique_id)

        changed = StepGraph(
            {
                "features": steps["features"],
                "sweep": AddStep(a=3, b=40, step_resources=StepResources(cpu_count=8)),
                "extra": AddStep(a=7, b=8),
            }
        )
        pending = dry_run(changed, workspace, executor, name="main")
        output = capsys.readouterr().out

        assert pending == ["sweep", "extra"]
        assert "IDENTITY CHANGED since run 'main'" in output.split("sweep  [")[1].splitlines()[0]
        assert "new in run 'main'" in output.split("extra  [")[1].splitlines()[0]

    def test_expect(self, workspace, jobs, tmp_path, capsys):
        executor = make_executor(workspace, tmp_path)
        assert dry_run(graph(), workspace, executor, expect=["features", "sweep", "rep*"])

        with pytest.raises(CliRunError):
            dry_run(graph(), workspace, executor, expect=["sweep"])
        assert "UNEXPECTED: features, report" in capsys.readouterr().out


class TestSettings:
    def test_a_nested_env_block_is_accepted(self, workspace, jobs, tmp_path):
        # `env:` under `executor:` in tango.yml failed with "Any cannot be instantiated".
        (tmp_path / "tango.yml").write_text(
            "executor:\n"
            "  type: hf\n"
            f"  project_dir: {tmp_path}\n"
            "  timeout: 4h\n"
            "  env:\n"
            "    OMP_NUM_THREADS: 1\n"
            "    MY_FLAG: 'on'\n"
            "  extra_project_exclude:\n"
            "    - data-local/**\n"
        )
        settings = TangoGlobalSettings.from_file(tmp_path / "tango.yml")
        executor = prepare_executor(workspace, settings=settings)

        assert executor.env == {"OMP_NUM_THREADS": "1", "MY_FLAG": "on"}  # type: ignore[attr-defined]
        assert "data-local/**" in executor.project_exclude  # type: ignore[attr-defined]
        assert ".venv/**" in executor.project_exclude  # type: ignore[attr-defined]

    def test_executor_options_override_the_settings(self, workspace, jobs, tmp_path):
        settings = TangoGlobalSettings(executor={"type": "hf", "project_dir": str(tmp_path)})
        executor = prepare_executor(
            workspace, settings=settings, executor_options={"detach": True, "attempts": 3}
        )
        assert executor.detach is True  # type: ignore[attr-defined]
        assert executor.attempts == 3  # type: ignore[attr-defined]

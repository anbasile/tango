import pytest

from tango.common.exceptions import ConfigurationError
from tango.integrations.hf.common import (
    Constants,
    parse_memory,
    resolve_flavor,
    split_bucket_path,
)
from tango.step import StepResources


class TestSplitBucketPath:
    def test_with_prefix(self):
        assert split_bucket_path("org/bucket/exp/v2") == ("org/bucket", "exp/v2")

    def test_without_prefix(self):
        assert split_bucket_path("org/bucket") == ("org/bucket", "")

    def test_tolerates_surrounding_slashes(self):
        assert split_bucket_path("/org/bucket/") == ("org/bucket", "")

    def test_requires_a_namespace(self):
        with pytest.raises(ConfigurationError, match="namespace"):
            split_bucket_path("bucket")


class TestParseMemory:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("2.5GiB", 2.5),
            ("1024Mi", 1.0),
            ("1Ti", 1024.0),
            ("16", 16 / 1024**3),
            (None, None),
        ],
    )
    def test_units(self, value, expected):
        assert parse_memory(value) == expected

    def test_decimal_units_are_smaller_than_binary_ones(self):
        # 32GB is 32 * 10^9 bytes, which is less than 32GiB.
        assert parse_memory("32G") < parse_memory("32Gi")

    def test_rejects_nonsense(self):
        with pytest.raises(ConfigurationError):
            parse_memory("a lot")


class TestResolveFlavor:
    def test_no_requirements_defers_to_the_executor(self):
        assert resolve_flavor(StepResources()) is None
        assert resolve_flavor(None) is None

    def test_one_gpu_picks_the_cheapest_gpu_box(self):
        assert resolve_flavor(StepResources(gpu_count=1)) == "t4-small"

    def test_cpu_and_memory(self):
        assert resolve_flavor(StepResources(cpu_count=16, memory="100GiB")) == "cpu-xl"

    def test_gpu_type_is_matched_loosely(self):
        # The name a scheduler reports is far more specific than the flavor table's.
        assert resolve_flavor(StepResources(gpu_count=1, gpu_type="NVIDIA A100-SXM-80GB")) == (
            "a100-large"
        )

    def test_gpu_type_does_not_confuse_a10g_with_a100(self):
        assert resolve_flavor(StepResources(gpu_count=1, gpu_type="NVIDIA A10G")) == "a10g-small"

    def test_multiple_gpus(self):
        assert resolve_flavor(StepResources(gpu_count=8, gpu_type="A100")) == "a100x8"

    def test_impossible_request_is_an_error(self):
        with pytest.raises(ConfigurationError, match="No Hugging Face Jobs flavor"):
            resolve_flavor(StepResources(gpu_count=64))


class TestConstants:
    def test_keys(self):
        assert Constants.step_info_key("abc123") == "stepinfo/abc123.json"
        assert Constants.run_key("brave-moth") == "runs/brave-moth.json"
        assert Constants.run_log_key("brave-moth") == "runs/brave-moth.log"
        assert Constants.step_artifact_name("abc123") == "tango-step-abc123"
        assert Constants.step_lock_artifact_name("abc123") == "tango-step-abc123-lock"


class TestParseTimeout:
    @pytest.mark.parametrize(
        "value, seconds",
        [("4h", 14400), ("90s", 90), ("1.5m", 90), ("2d", 172800), ("30", 30), (45, 45), (2.9, 2)],
    )
    def test_units(self, value, seconds):
        from tango.integrations.hf.common import parse_timeout

        assert parse_timeout(value) == seconds

    @pytest.mark.parametrize("value", ["soon", "4 hours", "", "-1h", 0, True])
    def test_nonsense_is_refused(self, value):
        from tango.common.exceptions import ConfigurationError
        from tango.integrations.hf.common import parse_timeout

        with pytest.raises(ConfigurationError):
            parse_timeout(value)


class TestJobStage:
    def test_a_plain_string(self):
        from tango.integrations.hf.common import job_stage

        from .fake_hub import FakeJob

        assert job_stage(FakeJob("j", "RUNNING")) == "RUNNING"

    def test_an_enum(self):
        # The Hub annotates the stage as an enum, whose str() is "JobStage.COMPLETED".
        import enum

        from tango.integrations.hf.common import TERMINAL_JOB_STAGES, job_stage

        from .fake_hub import FakeJob

        class JobStage(enum.Enum):
            COMPLETED = "COMPLETED"

        assert job_stage(FakeJob("j", JobStage.COMPLETED)) in TERMINAL_JOB_STAGES  # type: ignore[arg-type]

    def test_no_status(self):
        from tango.integrations.hf.common import job_stage

        assert job_stage(object()) == ""


class TestHubCall:
    @pytest.fixture(autouse=True)
    def fast(self, monkeypatch):
        from tango.integrations.hf import common

        monkeypatch.setattr(common, "RETRY_BASE_SECONDS", 0.01)

    def _flaky(self, errors, result="ok"):
        calls = []

        def function(*args, **kwargs):
            calls.append((args, kwargs))
            if errors:
                raise errors.pop(0)
            return result

        return function, calls

    def test_a_rate_limit_is_waited_out(self):
        from tango.integrations.hf.common import hub_call

        from .fake_hub import fake_http_error

        function, calls = self._flaky([fake_http_error(429), fake_http_error(503)])
        assert hub_call(function, 1, key="value") == "ok"
        assert calls == [((1,), {"key": "value"})] * 3

    def test_a_dropped_connection_is_waited_out(self):
        from tango.integrations.hf.common import hub_call

        function, calls = self._flaky([ConnectionError("reset")])
        assert hub_call(function) == "ok"
        assert len(calls) == 2

    def test_other_errors_are_raised_at_once(self):
        from tango.integrations.hf.common import hub_call

        from .fake_hub import fake_http_error

        function, calls = self._flaky([fake_http_error(404)])
        with pytest.raises(Exception, match="404"):
            hub_call(function)
        assert len(calls) == 1

        function, calls = self._flaky([ValueError("a bug")])
        with pytest.raises(ValueError):
            hub_call(function)
        assert len(calls) == 1

    def test_only_the_named_statuses_are_retried(self):
        # Submitting a job is retried on the rate limit only: after a 500 it may exist.
        from tango.integrations.hf.common import hub_call

        from .fake_hub import fake_http_error

        function, calls = self._flaky([fake_http_error(500)])
        with pytest.raises(Exception, match="500"):
            hub_call(function, statuses=frozenset({429}))
        assert len(calls) == 1

    def test_it_gives_up_when_the_budget_is_spent(self):
        from tango.integrations.hf.common import hub_call

        from .fake_hub import fake_http_error

        function, calls = self._flaky([fake_http_error(429) for _ in range(100)])
        with pytest.raises(Exception, match="429"):
            hub_call(function, budget=0.05)
        assert 1 < len(calls) < 100

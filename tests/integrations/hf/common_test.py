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

"""
Steps for the smoke test in ``config.jsonnet``.

Deliberately trivial and dependency-free: the point is to prove the plumbing of
:class:`~tango.integrations.hf.executor.HfJobsExecutor` — that a Job starts, mounts the project
and config, installs the package, runs one step, writes its result to the bucket, and that a
dependent step picks that result up from a *different* container.
"""

from typing import List

from tango.step import Step


@Step.register("hf_smoke::make")
class MakeNumbers(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, n: int) -> List[int]:  # type: ignore[override]
        self.logger.info("Producing %d numbers.", n)
        return list(range(n))


@Step.register("hf_smoke::total")
class Total(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self, numbers: List[int]) -> int:  # type: ignore[override]
        # Reaching this at all proves the previous step's result came back out of the bucket.
        self.logger.info("Summing %d numbers from the previous step.", len(numbers))
        return sum(numbers)


@Step.register("hf_smoke::accelerator")
class ReportAccelerator(Step):
    """
    Reports the hardware the Job landed on, from the ``ACCELERATOR`` variable the platform sets.
    """

    DETERMINISTIC = False
    CACHEABLE = True

    def run(self) -> str:  # type: ignore[override]
        import os

        accelerator = os.environ.get("ACCELERATOR") or "none"
        self.logger.info("ACCELERATOR=%s CPU_CORES=%s", accelerator, os.environ.get("CPU_CORES"))
        return accelerator

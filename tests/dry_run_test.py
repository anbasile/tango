import subprocess

from tango.common.testing import TangoTestCase


class TestDryRun(TangoTestCase):
    def _tango(self, *args):
        cmd = [
            "tango",
            "run",
            str(self.FIXTURES_ROOT / "experiment" / "hello_world.jsonnet"),
            "-w",
            str(self.TEST_DIR),
            *args,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout + result.stderr

    def test_it_lists_the_steps_and_runs_nothing(self):
        code, output = self._tango("--dry-run")
        assert code == 0, output
        assert "2 of 2 steps would run" in output
        assert "hello  [local; incomplete]" in output
        assert "hello_world  [local; incomplete]" in output
        # Nothing ran and no run was registered.
        assert not (self.TEST_DIR / "cache").exists() or not list(
            (self.TEST_DIR / "cache").iterdir()
        )
        assert not (self.TEST_DIR / "runs").exists() or not list((self.TEST_DIR / "runs").iterdir())

    def test_finished_steps_are_not_listed(self):
        code, output = self._tango("-s", "hello")
        assert code == 0, output
        code, output = self._tango("--dry-run")
        assert code == 0, output
        assert "1 of 2 steps would run" in output
        assert "hello  [" not in output

    def test_an_unexpected_step_stops_the_run(self):
        # The guard against a changed step identity re-running a finished grid.
        code, output = self._tango("--expect", "hello")
        assert code != 0
        assert "UNEXPECTED: hello_world" in output
        assert not (self.TEST_DIR / "cache").exists() or not list(
            (self.TEST_DIR / "cache").iterdir()
        )

    def test_expecting_nothing(self):
        code, output = self._tango("--dry-run", "--expect", "")
        assert code != 0
        assert "UNEXPECTED: hello, hello_world" in output

    def test_expected_steps_let_the_run_go_ahead(self):
        code, output = self._tango("--expect", "hello*")
        assert code == 0, output
        assert "2 of 2 steps would run" in output
        assert len(list((self.TEST_DIR / "cache").iterdir())) == 2

        # And once everything is done, expecting nothing holds.
        code, output = self._tango("--dry-run", "--expect", "")
        assert code == 0, output
        assert "0 of 2 steps would run" in output

    def test_executor_options_need_an_executor_in_the_settings(self):
        code, output = self._tango("--executor-option", "detach=true")
        assert code != 0
        assert "--executor-option" in output

    def test_a_malformed_executor_option(self):
        code, output = self._tango("--executor-option", "detach")
        assert code != 0
        assert "KEY=VALUE" in output

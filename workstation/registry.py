"""The script allowlist.

Nothing runs on this machine unless it is described here first. The registry is
a YAML file rather than a database table so the allowlist is reviewable in git
and cannot be extended through the API: adding a script is a deliberate edit by
the owner, not something a conversation can do.

Each entry declares its executable and a typed parameter list. Values supplied
at call time are validated against those declarations and passed as argv, so a
parameter is always one argument no matter what it contains.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.logging_config import get_logger
from shared.enums import RiskLevel
from shared.errors import ConfigurationError, ValidationError

logger = get_logger(__name__)

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
PARAM_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
PARAM_TYPES = frozenset({"string", "integer", "number", "boolean", "choice", "path"})

MAX_TIMEOUT_SECONDS = 3600
MAX_PARAMETERS = 12
MAX_VALUE_LENGTH = 512


@dataclass(frozen=True)
class Parameter:
    name: str
    type: str = "string"
    required: bool = False
    default: Any = None
    choices: tuple[str, ...] = ()
    pattern: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    flag: str | None = None
    """Emitted before the value, e.g. ``--target``. Positional when absent."""
    description: str = ""

    def render(self, value: Any) -> list[str]:
        """Turn a validated value into argv fragments."""
        if self.type == "boolean":
            # A boolean is presence or absence of its flag; there is no value to
            # pass, and no way for it to carry a payload.
            if not value:
                return []
            if not self.flag:
                raise ConfigurationError(
                    f"Boolean parameter '{self.name}' needs a flag to be expressible."
                )
            return [self.flag]

        rendered = str(value)
        return [self.flag, rendered] if self.flag else [rendered]

    def coerce(self, value: Any) -> Any:
        """Validate one supplied value, returning the canonical form."""
        if self.type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in {"true", "false"}:
                return value.lower() == "true"
            raise ValidationError(f"Parameter '{self.name}' must be true or false.")

        if self.type in {"integer", "number"}:
            try:
                number = int(value) if self.type == "integer" else float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"Parameter '{self.name}' must be a number.") from exc
            if self.minimum is not None and number < self.minimum:
                raise ValidationError(f"Parameter '{self.name}' must be at least {self.minimum}.")
            if self.maximum is not None and number > self.maximum:
                raise ValidationError(f"Parameter '{self.name}' must be at most {self.maximum}.")
            return number

        text = str(value)
        if len(text) > MAX_VALUE_LENGTH:
            raise ValidationError(f"Parameter '{self.name}' is too long.")
        if "\x00" in text or "\n" in text or "\r" in text:
            # Not a shell concern - argv is safe - but a newline in an argument
            # is almost always a sign that something was misparsed upstream.
            raise ValidationError(f"Parameter '{self.name}' may not contain line breaks.")

        if self.type == "choice":
            if text not in self.choices:
                allowed = ", ".join(self.choices)
                raise ValidationError(f"Parameter '{self.name}' must be one of: {allowed}.")
            return text

        if self.type == "path":
            if ".." in Path(text).parts:
                raise ValidationError(f"Parameter '{self.name}' may not contain '..'.")

        if self.pattern and not re.fullmatch(self.pattern, text):
            raise ValidationError(f"Parameter '{self.name}' does not match the expected format.")
        return text


@dataclass(frozen=True)
class ScriptDefinition:
    name: str
    description: str
    executable: str
    parameters: tuple[Parameter, ...] = ()
    working_directory: str | None = None
    risk_level: RiskLevel = RiskLevel.REVERSIBLE_WRITE
    requires_approval: bool = True
    timeout_seconds: int = 300
    enabled: bool = True
    environment: dict[str, str] = field(default_factory=dict)

    def build_argv(self, arguments: dict[str, Any]) -> list[str]:
        """Resolve supplied arguments into the exact argv to execute.

        Unknown names are refused rather than ignored: silently dropping an
        argument would run something other than what was approved.
        """
        declared = {parameter.name: parameter for parameter in self.parameters}
        unknown = sorted(set(arguments) - set(declared))
        if unknown:
            raise ValidationError(
                f"Script '{self.name}' has no parameter(s): {', '.join(unknown)}."
            )

        argv = [self.executable]
        for parameter in self.parameters:
            if parameter.name in arguments:
                value = parameter.coerce(arguments[parameter.name])
            elif parameter.required:
                raise ValidationError(f"Script '{self.name}' requires '{parameter.name}'.")
            elif parameter.default is not None:
                value = parameter.coerce(parameter.default)
            else:
                continue
            argv.extend(parameter.render(value))
        return argv


@dataclass(frozen=True)
class LoadResult:
    scripts: dict[str, ScriptDefinition]
    errors: dict[str, str]
    checksum: str
    path: Path

    def get(self, name: str) -> ScriptDefinition:
        script = self.scripts.get(name)
        if script is None:
            if name in self.errors:
                raise ConfigurationError(
                    f"Script '{name}' is in the registry but did not load: {self.errors[name]}"
                )
            raise ValidationError(f"No registered script called '{name}'.")
        if not script.enabled:
            raise ValidationError(f"Script '{name}' is registered but disabled.")
        return script


def load_registry(path: Path) -> LoadResult:
    """Read and validate the registry.

    A malformed entry disables that one script and is reported; it does not stop
    Uses the configured workflow.
    take the whole capability offline.
    """
    if not path.exists():
        return LoadResult(scripts={}, errors={}, checksum="", path=path)

    raw_bytes = path.read_bytes()
    checksum = hashlib.sha256(raw_bytes).hexdigest()

    try:
        document = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Script registry {path} is not valid YAML: {exc}") from exc

    if not isinstance(document, dict):
        raise ConfigurationError(f"Script registry {path} must be a mapping at the top level.")

    entries = document.get("scripts") or []
    if not isinstance(entries, list):
        raise ConfigurationError("The 'scripts' key must be a list.")

    scripts: dict[str, ScriptDefinition] = {}
    errors: dict[str, str] = {}

    for index, entry in enumerate(entries):
        name = str(entry.get("name", f"entry-{index}")) if isinstance(entry, dict) else f"#{index}"
        try:
            script = _parse_entry(entry)
        except (ValidationError, ConfigurationError) as exc:
            errors[name] = str(exc)
            logger.warning("Rejected a registry entry", extra={"script": name, "reason": str(exc)})
            continue
        if script.name in scripts:
            errors[script.name] = "Duplicate name; the first definition wins."
            continue
        scripts[script.name] = script

    return LoadResult(scripts=scripts, errors=errors, checksum=checksum, path=path)


def _parse_entry(entry: Any) -> ScriptDefinition:
    if not isinstance(entry, dict):
        raise ValidationError("Each script entry must be a mapping.")

    name = str(entry.get("name", "")).strip()
    if not NAME_PATTERN.fullmatch(name):
        raise ValidationError(
            "A script name must be lowercase letters, digits, hyphens or underscores."
        )

    executable = str(entry.get("executable", "")).strip()
    if not executable:
        raise ValidationError("A script must declare an executable.")
    if not executable.startswith("/"):
        # A bare name would resolve through PATH, which the caller does not
        # control and which can change under the service.
        raise ValidationError("The executable must be an absolute path.")
    if ".." in Path(executable).parts:
        raise ValidationError("The executable path may not contain '..'.")

    raw_parameters = entry.get("parameters") or []
    if not isinstance(raw_parameters, list):
        raise ValidationError("'parameters' must be a list.")
    if len(raw_parameters) > MAX_PARAMETERS:
        raise ValidationError(f"A script may declare at most {MAX_PARAMETERS} parameters.")

    parameters = tuple(_parse_parameter(item) for item in raw_parameters)
    seen: set[str] = set()
    for parameter in parameters:
        if parameter.name in seen:
            raise ValidationError(f"Duplicate parameter '{parameter.name}'.")
        seen.add(parameter.name)

    timeout = int(entry.get("timeout_seconds", 300))
    if not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
        raise ValidationError(f"timeout_seconds must be between 1 and {MAX_TIMEOUT_SECONDS}.")

    try:
        risk = RiskLevel(str(entry.get("risk_level", RiskLevel.REVERSIBLE_WRITE.value)))
    except ValueError as exc:
        raise ValidationError(f"Unknown risk_level: {entry.get('risk_level')!r}.") from exc

    requires_approval = bool(entry.get("requires_approval", True))
    if risk is RiskLevel.EXTERNAL_WRITE and not requires_approval:
        # The registry is owner-authored, but an entry that marks itself
        # dangerous and unattended is more likely a mistake than an intention.
        raise ValidationError(
            "A script at external_write risk cannot set requires_approval to false."
        )

    environment = entry.get("environment") or {}
    if not isinstance(environment, dict):
        raise ValidationError("'environment' must be a mapping.")

    return ScriptDefinition(
        name=name,
        description=str(entry.get("description", "")).strip(),
        executable=executable,
        parameters=parameters,
        working_directory=(
            str(entry["working_directory"]) if entry.get("working_directory") else None
        ),
        risk_level=risk,
        requires_approval=requires_approval,
        timeout_seconds=timeout,
        enabled=bool(entry.get("enabled", True)),
        environment={str(key): str(value) for key, value in environment.items()},
    )


def _parse_parameter(item: Any) -> Parameter:
    if not isinstance(item, dict):
        raise ValidationError("Each parameter must be a mapping.")

    name = str(item.get("name", "")).strip()
    if not PARAM_NAME_PATTERN.fullmatch(name):
        raise ValidationError(f"Invalid parameter name: {name!r}.")

    param_type = str(item.get("type", "string"))
    if param_type not in PARAM_TYPES:
        raise ValidationError(f"Unknown parameter type: {param_type!r}.")

    choices = tuple(str(choice) for choice in (item.get("choices") or ()))
    if param_type == "choice" and not choices:
        raise ValidationError(f"Parameter '{name}' is a choice but lists no choices.")

    flag = item.get("flag")
    if flag is not None:
        flag = str(flag)
        if not flag.startswith("-"):
            raise ValidationError(f"Parameter '{name}' has a flag that is not a flag: {flag!r}.")
        if " " in flag:
            raise ValidationError(f"Parameter '{name}' has a flag containing a space.")

    pattern = item.get("pattern")
    if pattern is not None:
        try:
            re.compile(str(pattern))
        except re.error as exc:
            raise ValidationError(f"Parameter '{name}' has an invalid pattern: {exc}.") from exc
        pattern = str(pattern)

    return Parameter(
        name=name,
        type=param_type,
        required=bool(item.get("required", False)),
        default=item.get("default"),
        choices=choices,
        pattern=pattern,
        minimum=_optional_float(item.get("minimum")),
        maximum=_optional_float(item.get("maximum")),
        flag=flag,
        description=str(item.get("description", "")).strip(),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)

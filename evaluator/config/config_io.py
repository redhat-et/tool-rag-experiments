import os
import pathlib
from copy import deepcopy
from typing import Any, Dict, Mapping, MutableMapping

from evaluator.config.defaults import DEFAULT_CONFIG
from evaluator.config.schema import EvaluationConfig

try:
    import yaml
except ModuleNotFoundError as me:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install with `pip install pyyaml`.") from me

from dotenv import load_dotenv

load_dotenv()


class ConfigError(Exception):
    """Raised for any configuration load/validation problems."""


_ALLOWED_SCALARS = (type(None), bool, int, float, str)


def load_config_file(path: str | os.PathLike) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    if p.suffix.lower() not in (".yaml", ".yml"):
        raise ConfigError(f"Unsupported config extension '{p.suffix}'. Use .yaml or .yml")

    data = None
    with p.open("rb") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML parse error in {p}: {e}") from e

    if data is None:
        # Empty file -> treat as empty dict for convenience
        data = {}

    if not isinstance(data, dict):
        raise ConfigError("Top-level config must be a mapping/object.")

    validate_types(data)
    return data


def validate_types(node: Any, *, path: str = "$") -> None:
    """
    Enforce: only None/bool/int/float/str, lists, dicts-with-string-keys.
    Rejects YAML timestamps, sets, tuples, custom tags, etc.
    """
    if isinstance(node, _ALLOWED_SCALARS):
        return
    if isinstance(node, list):
        for i, v in enumerate(node):
            validate_types(v, path=f"{path}[{i}]")
        return
    if isinstance(node, dict):
        for k, v in node.items():
            if not isinstance(k, str):
                raise ConfigError(f"Non-string dict key at {path}: {k!r} (type={type(k).__name__})")
            validate_types(v, path=f"{path}.{k}")
        return
    # Anything else (e.g., datetime, set) is rejected.
    raise ConfigError(f"Disallowed type at {path}: {type(node).__name__}")


def deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """
    Recursively merge override into base (mutates base).
    Lists are replaced (not concatenated) to avoid surprises.
    """
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, Mapping):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def expand_env_vars(node: Any) -> Any:
    """
    Recursively expand ${VAR} in string values using the current environment.
    Leaves non-strings untouched.
    """
    if isinstance(node, dict):
        return {k: expand_env_vars(v) for k, v in node.items()}
    if isinstance(node, list):
        return [expand_env_vars(v) for v in node]
    if isinstance(node, str):
        return os.path.expandvars(node)
    return node


def to_typed_config(raw_cfg: Dict[str, Any]) -> EvaluationConfig:
    """
    Convert the already-sanitized dict (from YAML) into a typed EvaluationConfig instance.
    This is where all Pydantic validation/parsing happens.
    """
    try:
        return EvaluationConfig.model_validate(raw_cfg)
    except Exception as e:
        # Re-wrap as ConfigError to keep a single error surface for callers
        raise ConfigError(f"Schema validation error: {e}") from e


def load_config(path: str | None, use_defaults: bool = True) -> EvaluationConfig:
    cfg: Dict[str, Any] = {}

    if use_defaults:
        cfg = deepcopy(DEFAULT_CONFIG)

    if path:
        file_cfg = load_config_file(path)
        file_cfg = expand_env_vars(file_cfg)
        if use_defaults:
            deep_merge(cfg, file_cfg)
        else:
            cfg = file_cfg

    return to_typed_config(cfg)

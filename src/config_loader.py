"""Configuration file loader supporting JSON and YAML formats without hard external dependencies."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _parse_yaml_value(val_str: str) -> Any:
    """Parse a single YAML scalar value into Python native types."""
    val = val_str.strip()
    if not val:
        return None
    val_lower = val.lower()
    if val_lower in ("true", "yes", "on"):
        return True
    if val_lower in ("false", "no", "off"):
        return False
    if val_lower in ("null", "none", "~"):
        return None

    # Strip quotes
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]

    # Try numeric parsing
    try:
        if "." in val or "e" in val_lower:
            return float(val)
        return int(val)
    except ValueError:
        return val


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Lightweight pure-Python YAML parser for key-value configurations.

    Supports:
      - Comments starting with #
      - Top-level and nested key-value pairs (dict)
      - Lists (- item)
      - Primitives: strings, ints, floats, booleans, null
    """
    result: dict[str, Any] = {}
    lines = text.splitlines()

    current_section: str | None = None
    current_dict: dict[str, Any] = result

    for raw_line in lines:
        # Strip trailing comments
        line_no_comment = re.sub(r"(?<!\\)#.*$", "", raw_line).rstrip()
        if not line_no_comment.strip():
            continue

        indent = len(line_no_comment) - len(line_no_comment.lstrip())
        stripped = line_no_comment.strip()

        # Handle list item under current key
        if stripped.startswith("- "):
            item_val = _parse_yaml_value(stripped[2:])
            if current_section and isinstance(result.get(current_section), list):
                result[current_section].append(item_val)
            continue

        if ":" in stripped:
            key_part, val_part = stripped.split(":", 1)
            key = key_part.strip()
            val_raw = val_part.strip()

            if indent == 0:
                current_section = key
                if not val_raw:
                    # Could be start of a dict or list
                    result[key] = {}
                    current_dict = result[key]
                else:
                    parsed = _parse_yaml_value(val_raw)
                    result[key] = parsed
                    current_dict = result
            else:
                # Nested key under current_section
                if current_section and isinstance(result.get(current_section), dict):
                    result[current_section][key] = _parse_yaml_value(val_raw)
                else:
                    result[key] = _parse_yaml_value(val_raw)

    return result


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from a JSON or YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to configuration file (.json, .yaml, or .yml).

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    content = path.read_text(encoding="utf-8")
    ext = path.suffix.lower()

    if ext == ".json":
        return json.loads(content)

    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            parsed = yaml.safe_load(content)
            if isinstance(parsed, dict):
                return parsed
            return {}
        except ImportError:
            return parse_simple_yaml(content)

    # Fallback: attempt JSON first, then YAML
    try:
        return json.loads(content)
    except Exception:
        return parse_simple_yaml(content)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save configuration dictionary to a JSON file."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

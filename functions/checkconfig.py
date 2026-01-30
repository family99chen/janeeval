import os
import sys
from typing import Any, Dict, List, Set

import yaml


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping (dict).")
    return data


def _find_config_path() -> str:
    env_path = os.getenv("RAGSEARCH_CONFIG")
    if env_path:
        return os.path.abspath(env_path)

    current = os.path.abspath(os.path.dirname(__file__))
    while True:
        candidate = os.path.join(current, "config.yaml")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise FileNotFoundError("config.yaml not found. Set RAGSEARCH_CONFIG to override.")


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _validate_allowed(
    value: Any, allowed: List[Any], path: str, errors: List[str]
) -> None:
    if isinstance(value, list):
        errors.append(f"{path} must be a single value, not a list.")
        return
    if "..." in allowed:
        if not _is_scalar(value):
            errors.append(f"{path} must be a scalar value.")
        return
    if value not in allowed:
        errors.append(f"{path} must be one of: {allowed}")


def _validate_node(
    selection: Any,
    schema: Any,
    path: str,
    errors: List[str],
    required_paths: Set[str],
) -> None:
    if isinstance(schema, dict):
        if "allowed" in schema:
            allowed = schema.get("allowed", [])
            if selection is None:
                if path in required_paths:
                    errors.append(f"{path} is required.")
                return
            if not isinstance(allowed, list):
                errors.append(f"{path} has invalid allowed list in config.")
                return
            _validate_allowed(selection, allowed, path, errors)
            return

        if not isinstance(selection, dict):
            errors.append(f"{path} must be a mapping.")
            return

        for key in selection.keys():
            if key not in schema:
                errors.append(f"{path}.{key} is not allowed.")

        for key, sub_schema in schema.items():
            if key not in selection:
                if f"{path}.{key}" in required_paths:
                    errors.append(f"{path}.{key} is required.")
                continue
            _validate_node(
                selection[key],
                sub_schema,
                f"{path}.{key}",
                errors,
                required_paths,
            )
        return

    if selection is None:
        if path in required_paths:
            errors.append(f"{path} is required.")
        return
    if not _is_scalar(selection):
        errors.append(f"{path} must be a scalar value.")


def check_config(selection_path: str, config_path: str | None = None) -> Dict[str, Any]:
    config_path = config_path or _find_config_path()
    config = _load_yaml(config_path)
    search_space = config.get("rag_search_space", {})
    eval_metrics = config.get("eval_metrics", {})

    selection = _load_yaml(selection_path)

    errors: List[str] = []
    required_paths: Set[str] = {
        "selection.retrieve",
        "selection.retrieve.model_url",
        "selection.retrieve.topk",
        "selection.retrieve.bm25_weight",
        "selection.chunking",
        "selection.chunking.chunk_size",
        "selection.generator",
        "selection.generator.model_url",
    }

    selection_for_search = dict(selection)
    selection_for_search.pop("eval_metrics", None)
    _validate_node(selection_for_search, search_space, "selection", errors, required_paths)
    if "eval_metrics" in selection:
        _validate_node(
            selection.get("eval_metrics"),
            eval_metrics,
            "selection.eval_metrics",
            errors,
            set(),
        )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
    }


def main() -> None:
    if len(sys.argv) < 2:
        selection_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "configs", "demo2.yaml")
        )
    else:
        selection_path = sys.argv[1]
    result = check_config(selection_path)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()

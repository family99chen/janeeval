"""
Thin wrappers around core evaluation and search-space utilities.
"""

from typing import Any, Dict, List, Optional

from functions.checkconfig import check_config, check_config_multimodal
from functions.findsearchspace import (
    get_search_space,
    get_search_space_multimodal,
)
from mainfunction import (
    evaluate_rag,
    evaluate_rag_multimodal,
    run_algorithms,
)


def check_config_valid(config_path: str, multimodal: bool = False) -> Dict[str, Any]:
    """
    Validate config against allowed search space.
    """
    if multimodal:
        return check_config_multimodal(config_path)
    return check_config(config_path)


def find_search_space(config_path: str, multimodal: bool = False) -> Dict[str, Any]:
    """
    Return search space and template guidance.
    """
    if multimodal:
        return get_search_space_multimodal(config_path)
    return get_search_space(config_path)


__all__ = [
    "evaluate_rag",
    "evaluate_rag_multimodal",
    "check_config_valid",
    "find_search_space",
    "run_algorithms",
]

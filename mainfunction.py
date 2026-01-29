import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Tuple

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from functions.checkconfig import check_config
from rag.normal.pipeline import run_batch_async


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if path.endswith(".jsonl"):
        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("QA JSON must be a list of objects.")
    return data


def _extract_qa(qa_items: List[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
    queries: List[str] = []
    references: List[List[str]] = []
    for item in qa_items:
        query = item.get("query") or item.get("question")
        if not query:
            raise ValueError("Each QA item must include 'query' or 'question'.")
        queries.append(str(query))
        refs = item.get("references") or item.get("answers") or item.get("reference")
        if refs is None:
            references.append([])
        elif isinstance(refs, list):
            references.append([str(r) for r in refs])
        else:
            references.append([str(refs)])
    return queries, references


def evaluate_rag(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str = "both",
) -> Dict[str, Any]:
    check = check_config(config_path)
    if not check.get("is_valid", False):
        return {"error": "invalid_config", "errors": check.get("errors", [])}

    qa_items = _load_json_or_jsonl(qa_json_path)
    queries, references_list = _extract_qa(qa_items)

    result = asyncio.run(
        run_batch_async(
            queries=queries,
            selection_path=config_path,
            data_json_path=corpus_json_path,
            references_list=references_list,
            answers_list=None,
            eval_mode=eval_mode,
            debug_dump=False,
        )
    )
    return {
        "eval_report": result.get("report"),
        "outputs": result.get("outputs"),
    }


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage: python mainfunction.py <qa_json> <corpus_json> <config_yaml> [eval_mode]"
        )
        sys.exit(1)
    qa_json_path = sys.argv[1]
    corpus_json_path = sys.argv[2]
    config_path = sys.argv[3]
    eval_mode = sys.argv[4] if len(sys.argv) > 4 else "both"
    result = evaluate_rag(qa_json_path, corpus_json_path, config_path, eval_mode=eval_mode)
    metrics = (result.get("eval_report") or {}).get("metrics") or {}
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Tuple

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from functions.checkconfig import check_config, check_config_multimodal
from rag.normal.pipeline import getupperbound_external as getupperbound_external_pipeline
from rag.multimodal.pipeline import (
    getupperbound_external as getupperbound_external_multimodal,
)
from rag.normal.pipeline import run_batch_async
from rag.multimodal.pipeline import run_batch_async as run_batch_async_multimodal


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


def evaluate_rag_multimodal(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str = "both",
) -> Dict[str, Any]:
    check = check_config_multimodal(config_path)
    if not check.get("is_valid", False):
        return {"error": "invalid_config", "errors": check.get("errors", [])}

    qa_items = _load_json_or_jsonl(qa_json_path)
    queries, references_list = _extract_qa(qa_items)

    result = asyncio.run(
        run_batch_async_multimodal(
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

# you can use if you want, but may exceed the actual upperbound that config really can access
def theoretical_getupperbound(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str = "both",
) -> Dict[str, Any]:
    result = getupperbound_external_pipeline(
        qa_json_path=qa_json_path,
        corpus_json_path=corpus_json_path,
        config_path=config_path,
        eval_mode=eval_mode,
    )
    return {
        "eval_report": result.get("report"),
        "outputs": result.get("outputs"),
    }


def theoretical_getupperbound_multimodal(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str = "both",
) -> Dict[str, Any]:
    result = getupperbound_external_multimodal(
        qa_json_path=qa_json_path,
        corpus_json_path=corpus_json_path,
        config_path=config_path,
        eval_mode=eval_mode,
    )
    return {
        "eval_report": result.get("report"),
        "outputs": result.get("outputs"),
    }


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage:\n"
            "  python mainfunction.py <qa_json> <corpus_json> <config_yaml> [eval_mode]\n"
            "  python mainfunction.py multimodal <qa_json> <corpus_json> <config_yaml> [eval_mode]"
        )
        sys.exit(1)
    if sys.argv[1] == "multimodal":
        if len(sys.argv) < 5:
            print(
                "Usage: python mainfunction.py multimodal <qa_json> <corpus_json> <config_yaml> [eval_mode]"
            )
            sys.exit(1)
        qa_json_path = sys.argv[2]
        corpus_json_path = sys.argv[3]
        config_path = sys.argv[4]
        eval_mode = sys.argv[5] if len(sys.argv) > 5 else "both"
        result = evaluate_rag_multimodal(
            qa_json_path, corpus_json_path, config_path, eval_mode=eval_mode
        )
        outputs = result.get("outputs") or []
        report = result.get("eval_report") or {}
        per_item = report.get("per_item") or []
        item_count = min(len(outputs), len(per_item)) if per_item else len(outputs)
        item_summaries: List[Dict[str, Any]] = []
        for idx in range(item_count):
            output = outputs[idx] if idx < len(outputs) else {}
            scores = per_item[idx] if idx < len(per_item) else {}
            image_count = 0
            image_retrieval = output.get("image_retrieval")
            if isinstance(image_retrieval, list):
                image_count = len(image_retrieval)
            item_summaries.append(
                {
                    "index": idx,
                    "image_count": image_count,
                    "answer": output.get("answer", ""),
                    "references": scores.get("references") or [],
                    "llmaaj_reason": scores.get("LLMAAJ_reason") or "",
                    "score": scores,
                }
            )
        print(json.dumps(item_summaries, ensure_ascii=False, indent=2))
    else:
        qa_json_path = sys.argv[1]
        corpus_json_path = sys.argv[2]
        config_path = sys.argv[3]
        eval_mode = sys.argv[4] if len(sys.argv) > 4 else "both"
        result = evaluate_rag(
            qa_json_path, corpus_json_path, config_path, eval_mode=eval_mode
        )
        outputs = result.get("outputs") or []
        report = result.get("eval_report") or {}
        per_item = report.get("per_item") or []
        item_count = min(len(outputs), len(per_item)) if per_item else len(outputs)
        item_summaries: List[Dict[str, Any]] = []
        for idx in range(item_count):
            output = outputs[idx] if idx < len(outputs) else {}
            scores = per_item[idx] if idx < len(per_item) else {}
            item_summaries.append(
                {
                    "index": idx,
                    "answer": output.get("answer", ""),
                    "references": scores.get("references") or [],
                    "llmaaj_reason": scores.get("LLMAAJ_reason") or "",
                    "score": scores,
                }
            )
        if item_summaries:
            print(json.dumps(item_summaries, ensure_ascii=False, indent=2))
    metrics = (result.get("eval_report") or {}).get("metrics") or {}
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

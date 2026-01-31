import asyncio
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rag.normal.pipeline import run_batch_async
from rag.multimodal.pipeline import run_batch_async as run_batch_async_multimodal

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = handle.read().strip()
    if not data:
        return {}
    import yaml

    parsed = yaml.safe_load(data) or {}
    return parsed if isinstance(parsed, dict) else {}


def _dump_yaml(data: Dict[str, Any], path: str) -> None:
    import yaml

    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def _load_json(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of records.")
    return data


def _allowed_values(node: Any) -> List[Any]:
    if not isinstance(node, dict):
        return []
    allowed = node.get("allowed")
    if not isinstance(allowed, list):
        return []
    return [v for v in allowed if v != "..."]


def _max_allowed(allowed: List[Any]) -> Optional[Any]:
    if not allowed:
        return None
    numeric = [v for v in allowed if isinstance(v, (int, float))]
    if numeric:
        return max(numeric)
    return allowed[-1]


def _parse_score_weights(text: str) -> Optional[Dict[str, float]]:
    if not text:
        return None
    name_map = {
        "llmaaj": "LLMAAJ",
        "bertf1": "BERTScore-F1",
        "bert": "BERTScore-F1",
        "rougel": "ROUGE-L",
        "f1": "F1",
        "bleu": "BLEU",
        "exactmatch": "ExactMatch",
        "em": "ExactMatch",
    }
    weights: Dict[str, float] = {}
    for raw in text.split(","):
        part = raw.strip().lower()
        if not part:
            continue
        idx = len(part)
        while idx > 0 and (part[idx - 1].isdigit() or part[idx - 1] == "."):
            idx -= 1
        if idx == len(part):
            continue
        name = part[:idx]
        weight_str = part[idx:]
        metric_key = name_map.get(name)
        if not metric_key:
            continue
        try:
            weight = float(weight_str)
        except Exception:
            continue
        weights[metric_key] = weight
    return weights or None


def _score_from_report(
    report: Dict[str, Any],
    preferred: Optional[str],
    weights: Optional[Dict[str, float]],
) -> Tuple[str, float]:
    metrics = report.get("metrics") or {}
    if weights:
        total = 0.0
        denom = 0.0
        for key, weight in weights.items():
            if key not in metrics:
                continue
            try:
                total += float(metrics[key]) * weight
                denom += weight
            except Exception:
                continue
        return "weighted", (total / denom) if denom > 0 else 0.0
    if preferred and preferred in metrics:
        try:
            return preferred, float(metrics[preferred])
        except Exception:
            return preferred, 0.0
    for name in ("LLMAAJ", "BERTScore-F1", "ROUGE-L", "F1", "BLEU"):
        if name in metrics:
            try:
                return name, float(metrics[name])
            except Exception:
                return name, 0.0
    return "LLMAAJ", 0.0


def _write_temp_selection(selection: Dict[str, Any]) -> str:
    fd, path = tempfile.mkstemp(prefix="upperbound_selection_", suffix=".yaml")
    os.close(fd)
    _dump_yaml(selection, path)
    return path


def _build_upperbound_selection(
    search_space: Dict[str, Any],
    algo_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    selection: Dict[str, Any] = {}

    chunking_cfg = search_space.get("chunking", {})
    if isinstance(chunking_cfg, dict):
        chunk_size = _max_allowed(_allowed_values(chunking_cfg.get("chunk_size", {})))
        if chunk_size is not None:
            selection["chunking"] = {
                "chunk_size": int(chunk_size),
                "chunk_overlap": 0,
            }

    retrieve_cfg = search_space.get("retrieve", {})
    retrieve_model_url = ((algo_cfg or {}).get("retrieve") or {}).get("model_url")
    if isinstance(retrieve_model_url, list):
        retrieve_model_url = retrieve_model_url[0] if retrieve_model_url else None
    if isinstance(retrieve_cfg, dict):
        topk = _max_allowed(_allowed_values(retrieve_cfg.get("topk", {})))
        bm25 = _max_allowed(_allowed_values(retrieve_cfg.get("bm25_weight", {})))
        if retrieve_model_url or topk is not None or bm25 is not None:
            selection["retrieve"] = {}
            if retrieve_model_url:
                selection["retrieve"]["model_url"] = retrieve_model_url
            if topk is not None:
                selection["retrieve"]["topk"] = int(topk)
            if bm25 is not None:
                selection["retrieve"]["bm25_weight"] = float(bm25)

    generator_cfg = (algo_cfg or {}).get("generator")
    if isinstance(generator_cfg, dict):
        selection["generator"] = dict(generator_cfg)

    eval_metrics = (algo_cfg or {}).get("eval_metrics")
    if isinstance(eval_metrics, dict):
        selection["eval_metrics"] = dict(eval_metrics)

    return selection


def _maybe_pick_first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _is_multimodal(algo_cfg: Dict[str, Any]) -> bool:
    return isinstance(algo_cfg, dict) and "clip" in algo_cfg


def _add_clip_selection(
    selection: Dict[str, Any],
    search_space: Dict[str, Any],
    algo_cfg: Dict[str, Any],
) -> None:
    clip_cfg = (algo_cfg or {}).get("clip")
    if not isinstance(clip_cfg, dict):
        return
    clip_search = search_space.get("clip", {}) if isinstance(search_space, dict) else {}
    clip_topk = _max_allowed(_allowed_values(clip_search.get("topk", {})))
    model_url = _maybe_pick_first(clip_cfg.get("model_url"))
    if not model_url and clip_topk is None:
        return
    selection["clip"] = {}
    if model_url:
        selection["clip"]["model_url"] = model_url
    model_name = _maybe_pick_first(clip_cfg.get("model_name"))
    if model_name:
        selection["clip"]["model_name"] = model_name
    api_key = _maybe_pick_first(clip_cfg.get("api_key"))
    if api_key:
        selection["clip"]["api_key"] = api_key
    if clip_topk is not None:
        selection["clip"]["topk"] = int(clip_topk)


def _extract_context_for_qa(
    qa: Dict[str, Any],
    idx: int,
    corpus_items: List[Dict[str, Any]],
    corpus_by_id: Dict[str, str],
    corpus_ids: List[str],
) -> str:
    qa_id = qa.get("id") or qa.get("qid") or qa.get("doc_id")
    if qa_id is not None:
        qa_id_str = str(qa_id)
        exact = corpus_by_id.get(qa_id_str)
        if exact is not None:
            return exact
        matched = [
            corpus_by_id[cid]
            for cid in corpus_ids
            if cid == qa_id_str or cid.startswith(f"{qa_id_str}_")
        ]
        if matched:
            return "\n".join(matched)
    if idx < len(corpus_items):
        return str(corpus_items[idx].get("content", ""))
    return ""


def upperbound(
    qa_json_path: str,
    corpus_json_path: str,
    template_path: str,
    algo_config_path: str,
    eval_mode: str,
    report_path: str,
    score_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    algo_cfg = _load_yaml(algo_config_path)
    use_multimodal = _is_multimodal(algo_cfg)
    if use_multimodal and template_path.endswith("config.yaml"):
        template_path = os.path.join(os.path.dirname(template_path), "config_multimodal.yaml")
    template = _load_yaml(template_path)
    search_space = template.get("rag_search_space") or {}

    preferred_metric = None
    if isinstance(algo_cfg, dict):
        preferred_metric = algo_cfg.get("score_metric") or algo_cfg.get("metric")

    selection = _build_upperbound_selection(search_space, algo_cfg)
    if use_multimodal:
        _add_clip_selection(selection, search_space, algo_cfg)
    selection_path = _write_temp_selection(selection)

    qa_items = _load_json(qa_json_path)
    corpus_items = _load_json(corpus_json_path)

    corpus_by_id: Dict[str, str] = {}
    corpus_ids: List[str] = []
    for item in corpus_items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id") or item.get("qid") or item.get("doc_id")
        content = item.get("content")
        if item_id is None or content is None:
            continue
        item_id_str = str(item_id)
        corpus_by_id[item_id_str] = str(content)
        corpus_ids.append(item_id_str)

    outputs: List[Dict[str, Any]] = []
    scores: List[float] = []
    metric_name = None
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    def _write_report_snapshot() -> None:
        avg_score = sum(scores) / len(scores) if scores else 0.0
        snapshot = {
            "upperbound_score": avg_score,
            "metric": metric_name or preferred_metric or "LLMAAJ",
            "selection": selection,
            "outputs": outputs,
        }
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2)
    try:
        bar = tqdm(total=len(qa_items), desc="upperbound", unit="qa") if tqdm else None
        for idx, item in enumerate(qa_items):
            if not isinstance(item, dict):
                if bar:
                    bar.update(1)
                continue
            query = item.get("query") or item.get("question")
            if query is None:
                query = ""
            refs = item.get("references") or item.get("answers") or item.get("reference")
            if refs is None:
                refs_list = []
            elif isinstance(refs, list):
                refs_list = [str(r) for r in refs]
            else:
                refs_list = [str(refs)]

            run_fn = run_batch_async_multimodal if use_multimodal else run_batch_async
            result = asyncio.run(
                run_fn(
                    queries=[str(query)],
                    selection_path=selection_path,
                    data_json_path=corpus_json_path,
                    references_list=[refs_list],
                    answers_list=None,
                    eval_mode=eval_mode,
                    debug_dump=False,
                )
            )
            report = result.get("report") or {}
            outputs_list = result.get("outputs") or []
            output_item = outputs_list[0] if outputs_list else {}
            metric_name, score = _score_from_report(
                report, preferred_metric, score_weights
            )
            scores.append(score)
            outputs.append(
                {
                    "index": idx + 1,
                    "query": str(query),
                    "output": output_item,
                    "metric": metric_name,
                    "score": score,
                    "report": report,
                }
            )
            _write_report_snapshot()
            if bar:
                bar.update(1)
        if bar:
            bar.close()
    finally:
        os.remove(selection_path)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    result = {
        "upperbound_score": avg_score,
        "metric": metric_name or preferred_metric or "LLMAAJ",
        "selection": selection,
        "outputs": outputs,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    import argparse

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_template = os.path.join(base_dir, "config.yaml")
    default_algo_config = os.path.join(os.path.dirname(__file__), "configforalgo.yaml")
    default_report = os.path.join(base_dir, "outputs", "upperbound_report.json")

    parser = argparse.ArgumentParser(description="Upperbound evaluation for RAG.")
    parser.add_argument("--qa_json", required=True, help="Path to QA JSON/JSONL.")
    parser.add_argument("--corpus_json", required=True, help="Path to corpus JSON.")
    parser.add_argument(
        "--config_yaml",
        default=default_template,
        help="Path to config search space template.",
    )
    parser.add_argument(
        "--algo_config_yaml",
        default=default_algo_config,
        help="Path to algo config with real model URLs/keys.",
    )
    parser.add_argument(
        "--eval_mode",
        default="both",
        choices=["avg", "per_item", "both"],
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--report_path",
        default=default_report,
        help="Path to write upperbound report JSON.",
    )
    parser.add_argument(
        "--score_weights",
        default="",
        help="Weighted metrics, e.g. 'bertf11,llmaaj2'.",
    )
    args = parser.parse_args()

    score_weights = _parse_score_weights(args.score_weights)
    upperbound(
        qa_json_path=args.qa_json,
        corpus_json_path=args.corpus_json,
        template_path=args.config_yaml,
        algo_config_path=args.algo_config_yaml,
        eval_mode=args.eval_mode,
        report_path=args.report_path,
        score_weights=score_weights,
    )


if __name__ == "__main__":
    main()
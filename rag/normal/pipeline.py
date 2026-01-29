import asyncio
import json
import os
import sys
import uuid
import gc
from typing import Any, Dict, List

import yaml

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from rag.normal.models.rewriter import rewrite_query, rewrite_query_async
    from rag.normal.models.chunking import build_chroma_db, build_chroma_db_async
    from rag.normal.models.retriever import retrieve, retrieve_async
    from rag.normal.models.pruner import prune_chunks, prune_chunks_async
    from rag.normal.models.reranker import rerank, rerank_async
    from rag.normal.models.generator import generate_answer, generate_answer_async
    from rag.normal.models.eval import evaluate_report
except ModuleNotFoundError:
    from models.rewriter import rewrite_query, rewrite_query_async
    from models.chunking import build_chroma_db, build_chroma_db_async
    from models.retriever import retrieve, retrieve_async
    from models.pruner import prune_chunks, prune_chunks_async
    from models.reranker import rerank, rerank_async
    from models.generator import generate_answer, generate_answer_async
    from models.eval import evaluate_report

try:
    import torch
except Exception:
    torch = None


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping (dict).")
    return data


def _load_pipeline_config() -> Dict[str, Any]:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(base_dir, "normal", "pipelineconfig.yaml")
    if not os.path.isfile(config_path):
        return {}
    return _load_yaml(config_path)


def _load_json(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of records.")
    return data


def run_chunking_stage(
    data_json_path: str, selection_path: str, debug_dump: bool = False
) -> Dict[str, Any]:
    try:
        selection = _load_yaml(selection_path)
        chunking_cfg = selection.get("chunking")
        retrieve_cfg = selection.get("retrieve", {})

        model_url = retrieve_cfg.get("model_url")
        if not model_url:
            raise ValueError("retrieve.model_url is required for embedding model.")

        if chunking_cfg is None:
            chunk_size = None
            chunk_overlap = 0
        else:
            chunk_size = int(chunking_cfg.get("chunk_size", 512))
            chunk_overlap = int(chunking_cfg.get("chunk_overlap", 64))

        records = _load_json(data_json_path)
        collection_name = f"ragsearch_{uuid.uuid4().hex}"
        return build_chroma_db(
            records=records,
            embedding_model_path=model_url,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            collection_name=collection_name,
            debug_dump=debug_dump,
        )
    except Exception:
        return {"client": None, "collection": None, "num_chunks": 0, "debug_dump": None}


async def run_chunking_stage_async(
    data_json_path: str, selection_path: str, debug_dump: bool = False
) -> Dict[str, Any]:
    try:
        selection = _load_yaml(selection_path)
        chunking_cfg = selection.get("chunking")
        retrieve_cfg = selection.get("retrieve", {})

        model_url = retrieve_cfg.get("model_url")
        if not model_url:
            raise ValueError("retrieve.model_url is required for embedding model.")

        if chunking_cfg is None:
            chunk_size = None
            chunk_overlap = 0
        else:
            chunk_size = int(chunking_cfg.get("chunk_size", 512))
            chunk_overlap = int(chunking_cfg.get("chunk_overlap", 64))

        records = _load_json(data_json_path)
        collection_name = f"ragsearch_{uuid.uuid4().hex}"
        return await build_chroma_db_async(
            records=records,
            embedding_model_path=model_url,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            collection_name=collection_name,
            debug_dump=debug_dump,
        )
    except Exception:
        return {"client": None, "collection": None, "num_chunks": 0, "debug_dump": None}


def run_rewriter_stage(query: str, selection_path: str) -> str:
    try:
        selection = _load_yaml(selection_path)
        if "rewriter" not in selection:
            return query
        rewriter_cfg = selection.get("rewriter", {})
        return rewrite_query(query, rewriter_cfg)
    except Exception:
        return ""


async def run_rewriter_stage_async(query: str, selection_path: str) -> str:
    try:
        selection = _load_yaml(selection_path)
        if "rewriter" not in selection:
            return query
        rewriter_cfg = selection.get("rewriter", {})
        return await rewrite_query_async(query, rewriter_cfg)
    except Exception:
        return ""


def run_retriever_stage(query: str, selection_path: str, collection: Any) -> list[Dict[str, Any]]:
    try:
        selection = _load_yaml(selection_path)
        retrieve_cfg = selection.get("retrieve", {})
        topk = int(retrieve_cfg.get("topk", 5))
        bm25_weight = float(retrieve_cfg.get("bm25_weight", 0.0))
        return retrieve(query=query, collection=collection, topk=topk, bm25_weight=bm25_weight)
    except Exception:
        return []


async def run_retriever_stage_async(
    query: str, selection_path: str, collection: Any
) -> list[Dict[str, Any]]:
    try:
        selection = _load_yaml(selection_path)
        retrieve_cfg = selection.get("retrieve", {})
        topk = int(retrieve_cfg.get("topk", 5))
        bm25_weight = float(retrieve_cfg.get("bm25_weight", 0.0))
        return await retrieve_async(
            query=query, collection=collection, topk=topk, bm25_weight=bm25_weight
        )
    except Exception:
        return []


def run_pruner_stage(
    query: str,
    selection_path: str,
    candidates: list[Dict[str, Any]],
) -> str:
    try:
        selection = _load_yaml(selection_path)
        if "pruner" not in selection:
            return "\n".join([c["document"] for c in candidates])
        pruner_cfg = selection.get("pruner", {})
        return prune_chunks(query=query, candidates=candidates, pruner_cfg=pruner_cfg)
    except Exception:
        return ""


async def run_pruner_stage_async(
    query: str,
    selection_path: str,
    candidates: list[Dict[str, Any]],
) -> str:
    try:
        selection = _load_yaml(selection_path)
        if "pruner" not in selection:
            return "\n".join([c["document"] for c in candidates])
        pruner_cfg = selection.get("pruner", {})
        return await prune_chunks_async(
            query=query, candidates=candidates, pruner_cfg=pruner_cfg
        )
    except Exception:
        return ""


def run_reranker_stage(
    query: str,
    selection_path: str,
    candidates: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    try:
        selection = _load_yaml(selection_path)
        if "reranker" not in selection:
            return candidates
        reranker_cfg = selection.get("reranker", {})
        model_url = reranker_cfg.get("model_url")
        topk = int(reranker_cfg.get("topk", len(candidates)))
        if not model_url:
            return candidates
        return rerank(query=query, candidates=candidates, model_url=model_url, topk=topk)
    except Exception:
        return []


async def run_reranker_stage_async(
    query: str,
    selection_path: str,
    candidates: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    try:
        selection = _load_yaml(selection_path)
        if "reranker" not in selection:
            return candidates
        reranker_cfg = selection.get("reranker", {})
        model_url = reranker_cfg.get("model_url")
        topk = int(reranker_cfg.get("topk", len(candidates)))
        if not model_url:
            return candidates
        return await rerank_async(
            query=query, candidates=candidates, model_url=model_url, topk=topk
        )
    except Exception:
        return []


def run_generator_stage(query: str, selection_path: str, context: str) -> str:
    try:
        selection = _load_yaml(selection_path)
        generator_cfg = selection.get("generator", {})
        model_url = generator_cfg.get("model_url")
        if not model_url:
            return ""
        model_name = generator_cfg.get("model_name")
        api_key = generator_cfg.get("api_key")
        return generate_answer(
            query=query,
            context=context,
            model_url=model_url,
            api_key=api_key,
            model_name=model_name,
        )
    except Exception:
        return ""


async def run_generator_stage_async(
    query: str, selection_path: str, context: str
) -> str:
    try:
        selection = _load_yaml(selection_path)
        generator_cfg = selection.get("generator", {})
        model_url = generator_cfg.get("model_url")
        if not model_url:
            return ""
        model_name = generator_cfg.get("model_name")
        api_key = generator_cfg.get("api_key")
        return await generate_answer_async(
            query=query,
            context=context,
            model_url=model_url,
            api_key=api_key,
            model_name=model_name,
        )
    except Exception:
        return ""


async def run_pipeline_async(
    query: str,
    selection_path: str,
    collection: Any,
) -> Dict[str, Any]:
    rewritten = await run_rewriter_stage_async(query, selection_path)
    retrieval = await run_retriever_stage_async(
        rewritten, selection_path, collection
    )
    reranked = await run_reranker_stage_async(rewritten, selection_path, retrieval)
    pruned_text = await run_pruner_stage_async(rewritten, selection_path, reranked)
    answer = await run_generator_stage_async(query, selection_path, pruned_text)
    return {
        "query": query,
        "rewritten": rewritten,
        "retrieval": retrieval,
        "reranked": reranked,
        "pruned_text": pruned_text,
        "answer": answer,
    }


async def run_batch_async(
    queries: List[str],
    selection_path: str,
    data_json_path: str,
    references_list: List[List[str]] | None = None,
    answers_list: List[str] | None = None,
    eval_mode: str = "both",
    debug_dump: bool = False,
) -> Dict[str, Any]:
    result = await run_chunking_stage_async(
        data_json_path, selection_path, debug_dump=debug_dump
    )
    collection = result.get("collection")
    client = result.get("client")
    collection_name = result.get("collection_name")

    config = _load_pipeline_config()
    max_tasks = int(config.get("concurrency", {}).get("max_tasks", 8))
    semaphore = asyncio.Semaphore(max_tasks)

    async def _run_one(q: str) -> Dict[str, Any]:
        async with semaphore:
            return await run_pipeline_async(q, selection_path, collection)

    selection = _load_yaml(selection_path)
    eval_cfg = selection.get("eval_metrics")
    total = len(queries)
    tasks = [asyncio.create_task(_run_one(q)) for q in queries]
    outputs: List[Dict[str, Any]] = []
    if tqdm is not None:
        with tqdm(total=total, desc="progress", unit="qa") as bar:
            for coro in asyncio.as_completed(tasks):
                result_item = await coro
                outputs.append(result_item)
                bar.update(1)
    else:
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result_item = await coro
            outputs.append(result_item)
            completed += 1
            print(f"\rprogress: {completed}/{total}", end="", flush=True)
        if total > 0:
            print()
    try:
        preds = (
            answers_list if answers_list is not None else [o.get("answer", "") for o in outputs]
        )
        report = evaluate_report(
            preds=preds,
            refs_list=references_list or [],
            queries=queries,
            mode=eval_mode,
            eval_cfg=eval_cfg,
        )
        return {
            "debug_dump": result.get("debug_dump"),
            "outputs": outputs,
            "report": report,
        }
    finally:
        if client is not None and collection_name:
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass
        if torch and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    selection_path = os.path.join(base_dir, "configs", "demo2.yaml")
    data_json_path = os.path.join(base_dir, "rag", "normal", "testjson.json")
    queries = [
        "I am a businessperson. What are OpenAI's main products?",
        "Who is Justin's mother?",
        "What is David's profession?",
    ]
    references_list = [
        ["OpenAI's main products include ChatGPT, DALL-E, Codex, and the API."],
        ["Justin's mother is David."],
        ["David is a doctor."],
    ]
    result = asyncio.run(
        run_batch_async(
            queries=queries,
            selection_path=selection_path,
            data_json_path=data_json_path,
            answers_list=None,
            references_list=references_list,
            eval_mode="both",
            debug_dump=False,
        )
    )
    if result.get("debug_dump") is not None:
        print("chroma_debug_dump:")
        print(result["debug_dump"])
    print("batch_outputs:")
    print(result["outputs"])
    print("eval_report:")
    print(result["report"])


if __name__ == "__main__":
    main()

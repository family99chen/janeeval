import json
import math
import os
import random
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if os.getenv("PIPELINE_DEBUG") == "1" or os.getenv("EVAL_DEBUG") == "1":
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

from mainfunction import evaluate_rag, evaluate_rag_multimodal

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


def _allowed_values(node: Any) -> List[Any]:
    if node is None:
        return []
    if isinstance(node, list):
        return node
    if not isinstance(node, dict):
        return [node]
    allowed = node.get("allowed")
    if not isinstance(allowed, list):
        return []
    return [v for v in allowed if v != "..."]


def _split_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    search_space = config.get("rag_search_space") or {}
    eval_metrics = config.get("eval_metrics")
    algo_cfg = {
        key: value
        for key, value in config.items()
        if key not in {"rag_search_space", "eval_metrics"}
    }
    return search_space, algo_cfg, eval_metrics


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict):
            current = merged.get(key)
            if not isinstance(current, dict):
                current = {}
            merged_child = _deep_update(current, value)
            if merged_child:
                merged[key] = merged_child
            else:
                merged.pop(key, None)
            continue
        if isinstance(value, list):
            continue
        merged[key] = value
    return merged


def _override_choices(
    module: str, key: str, algo_cfg: Dict[str, Any]
) -> Optional[List[Any]]:
    if not isinstance(algo_cfg, dict):
        return None
    section = algo_cfg.get(module)
    if not isinstance(section, dict) or key not in section:
        return None
    value = section.get(key)
    if isinstance(value, list):
        return value
    if value is None:
        return None
    return [value]


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


def _sanitize_selection(selection: Dict[str, Any]) -> None:
    chunking = selection.get("chunking")
    if isinstance(chunking, dict):
        chunking.pop("model_url", None)
        chunking.pop("model_name", None)


def _write_temp_selection(selection: Dict[str, Any]) -> str:
    fd, path = tempfile.mkstemp(prefix="rl_selection_", suffix=".yaml")
    os.close(fd)
    _dump_yaml(selection, path)
    return path


def _is_multimodal(search_space: Dict[str, Any], algo_cfg: Dict[str, Any]) -> bool:
    if isinstance(search_space, dict) and "clip" in search_space:
        return True
    return isinstance(algo_cfg, dict) and "clip" in algo_cfg


def _set_eval_schema_env(config_path: str, use_multimodal: bool) -> None:
    if use_multimodal:
        os.environ["RAGSEARCH_CONFIG_MULTIMODAL"] = config_path
    else:
        os.environ["RAGSEARCH_CONFIG"] = config_path


def _module_forced_on(algo_cfg: Dict[str, Any], module: str) -> bool:
    if not isinstance(algo_cfg, dict):
        return False
    section = algo_cfg.get(module)
    return isinstance(section, dict) and len(section) > 0


def _paired_model_choices(
    params: Dict[str, Any], algo_cfg: Dict[str, Any], module: str
) -> Optional[List[Tuple[Any, Any]]]:
    if not isinstance(params, dict):
        return None
    url_override = _override_choices(module, "model_url", algo_cfg)
    name_override = _override_choices(module, "model_name", algo_cfg)
    url_choices = _allowed_values(params.get("model_url"))
    if url_override:
        url_choices = url_override
    name_choices = _allowed_values(params.get("model_name"))
    if name_override:
        name_choices = name_override
    if not url_choices or not name_choices:
        return None
    if len(url_choices) != len(name_choices):
        return None
    return list(zip(url_choices, name_choices))


class PolicyNetwork:
    def __init__(self, search_space: Dict[str, Any], algo_cfg: Dict[str, Any]):
        self.params: List[Dict[str, Any]] = []
        # Each param: { "name": str, "module": str, "key": str, "choices": list, "logits": list[float] }
        
        module_order = ["rewriter", "chunking", "retrieve", "clip", "reranker", "pruner", "generator"]
        
        for module in module_order:
            params = search_space.get(module)
            if not isinstance(params, dict):
                continue
            
            is_optional = module in {"rewriter", "reranker", "pruner"}
            forced_on = _module_forced_on(algo_cfg, module)
            
            # 1. Enable/Disable decision for optional modules
            if is_optional and not forced_on:
                self.params.append({
                    "name": f"{module}.__enabled__",
                    "module": module,
                    "key": "__enabled__",
                    "choices": [True, False],
                    "logits": [0.0, 0.0]
                })

            # 2. Paired model choices
            pair_choices = _paired_model_choices(params, algo_cfg, module)
            if pair_choices:
                self.params.append({
                    "name": f"{module}.__model_pair__",
                    "module": module,
                    "key": "__model_pair__",
                    "choices": pair_choices,
                    "logits": [0.0] * len(pair_choices)
                })

            # 3. Individual parameters
            for key, value in params.items():
                if pair_choices and key in {"model_url", "model_name"}:
                    continue
                
                override = _override_choices(module, key, algo_cfg)
                choices = override if override else _allowed_values(value)
                if not choices:
                    continue
                
                # If only 1 choice, no need to learn
                if len(choices) > 1:
                    self.params.append({
                        "name": f"{module}.{key}",
                        "module": module,
                        "key": key,
                        "choices": choices,
                        "logits": [0.0] * len(choices)
                    })

    def softmax(self, logits: List[float]) -> List[float]:
        if not logits:
            return []
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps]

    def sample(self, rng: random.Random) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        selection: Dict[str, Any] = {}
        trajectory: List[Dict[str, Any]] = []  # { "param_idx": int, "choice_idx": int }
        
        # Modules enabled status
        enabled_modules = set()
        
        # First pass: Determine enabled modules
        for idx, param in enumerate(self.params):
            if param["key"] == "__enabled__":
                probs = self.softmax(param["logits"])
                choice_idx = self._choice_index(rng, probs)
                is_enabled = param["choices"][choice_idx]
                trajectory.append({"param_idx": idx, "choice_idx": choice_idx})
                if is_enabled:
                    enabled_modules.add(param["module"])
            elif param["module"] not in ["rewriter", "reranker", "pruner"]:
                 enabled_modules.add(param["module"])

        # Second pass: Select values for enabled modules
        for idx, param in enumerate(self.params):
            module = param["module"]
            if param["key"] == "__enabled__":
                continue
            
            if module not in enabled_modules:
                continue

            probs = self.softmax(param["logits"])
            choice_idx = self._choice_index(rng, probs)
            choice_val = param["choices"][choice_idx]
            trajectory.append({"param_idx": idx, "choice_idx": choice_idx})

            selection.setdefault(module, {})
            
            if param["key"] == "__model_pair__":
                selection[module]["model_url"] = choice_val[0]
                selection[module]["model_name"] = choice_val[1]
            else:
                selection[module][param["key"]] = choice_val

        # Ensure chunking exists
        if "chunking" not in selection:
             selection["chunking"] = {}
             
        return selection, trajectory

    def _choice_index(self, rng: random.Random, probs: List[float]) -> int:
        r = rng.random()
        upto = 0.0
        for i, p in enumerate(probs):
            upto += p
            if r <= upto:
                return i
        return len(probs) - 1

    def clone_logits(self) -> List[List[float]]:
        return [list(param["logits"]) for param in self.params]

    def update_grpo(
        self,
        trajectories: List[List[Dict[str, Any]]],
        rewards: List[float],
        ref_logits: List[List[float]],
        learning_rate: float,
        kl_coeff: float,
    ) -> None:
        if not rewards:
            return
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(variance) if variance > 0 else 0.0
        denom = std if std > 0 else 1.0
        advantages = [(r - mean) / denom for r in rewards]

        # Accumulate gradients to ensure batch update
        logits_updates = [[0.0] * len(p["logits"]) for p in self.params]

        for traj, adv in zip(trajectories, advantages):
            for step in traj:
                param_idx = step["param_idx"]
                choice_idx = step["choice_idx"]
                param = self.params[param_idx]
                
                # Compute probabilities using current logits (sampling policy)
                # Note: In true PPO/GRPO, we should use importance sampling ratio (pi_new / pi_old).
                # Since we do single-step update here, pi_new approx pi_old, so ratio approx 1.
                # We use current logits to compute gradients.
                probs = self.softmax(param["logits"])
                ref_probs = self.softmax(ref_logits[param_idx])
                
                for j in range(len(param["logits"])):
                    # PG gradient: grad_log_pi * adv
                    # grad_log_pi(j) = (1 if j==choice else 0) - probs[j]
                    pg_grad = (1.0 if j == choice_idx else 0.0) - probs[j]
                    
                    # KL gradient: grad ( - beta * KL(ref || pi) )
                    # Simplifying to Reverse-KL regularization: push pi towards ref
                    # direction ~ (ref - pi), so gradient term is (probs - ref_probs)
                    # We subtract kl_coeff * (probs - ref_probs) which equals adding kl_coeff * (ref - probs)
                    kl_grad = probs[j] - ref_probs[j]
                    
                    logits_updates[param_idx][j] += learning_rate * (adv * pg_grad - kl_coeff * kl_grad)

        # Apply updates
        for i, param in enumerate(self.params):
            for j in range(len(param["logits"])):
                param["logits"][j] += logits_updates[i][j]


def _evaluate_selection(
    qa_json_path: str,
    corpus_json_path: str,
    selection: Dict[str, Any],
    eval_mode: str,
    preferred_metric: Optional[str],
    score_weights: Optional[Dict[str, float]],
    eval_fn,
) -> Tuple[float, Dict[str, Any]]:
    _sanitize_selection(selection)
    selection_path = _write_temp_selection(selection)
    try:
        result = eval_fn(
            qa_json_path=qa_json_path,
            corpus_json_path=corpus_json_path,
            config_path=selection_path,
            eval_mode=eval_mode,
        )
    finally:
        os.remove(selection_path)
    report = result.get("eval_report") or {}
    metric_name, score = _score_from_report(report, preferred_metric, score_weights)
    return score, {
        "metric": metric_name,
        "score": score,
        "report": report,
        "outputs": result.get("outputs"),
        "error": result.get("error"),
        "errors": result.get("errors"),
    }


def rl_search(
    qa_json_path: str,
    corpus_json_path: str,
    config_path: str,
    eval_mode: str,
    report_path: str,
    episodes: int,
    seed: int,
    learning_rate: float = 0.1,
    group_size: int = 4,
    kl_coeff: float = 0.02,
    score_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    config = _load_yaml(config_path)
    search_space, algo_cfg, eval_metrics = _split_config(config)
    use_multimodal = _is_multimodal(search_space, algo_cfg)
    _set_eval_schema_env(config_path, use_multimodal)
    eval_fn = evaluate_rag_multimodal if use_multimodal else evaluate_rag
    preferred_metric = None
    if isinstance(algo_cfg, dict):
        preferred_metric = algo_cfg.get("score_metric") or algo_cfg.get("metric")

    rng = random.Random(seed)
    policy = PolicyNetwork(search_space, algo_cfg)
    
    trials: List[Dict[str, Any]] = []
    best_score: float = float("-inf")
    best_config: Dict[str, Any] = {}
    group_size = max(1, int(group_size))
    bar = tqdm(total=episodes, desc="rl-grpo", unit="ep") if tqdm else None
    
    # Initialize reference logits (fixed reference policy)
    ref_logits = policy.clone_logits()

    def _write_report_snapshot() -> None:
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        snapshot = {
            "best_score": best_score,
            "best_config": best_config,
            "trials": trials,
        }
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2)

    for ep in range(episodes):
        trajectories: List[List[Dict[str, Any]]] = []
        rewards: List[float] = []

        for idx in range(group_size):
            selection, trajectory = policy.sample(rng)
            if eval_metrics:
                selection["eval_metrics"] = eval_metrics
            if algo_cfg:
                selection = _deep_update(selection, algo_cfg)

            print(
                f"\n[grpo] episode={ep+1} group={idx+1}/{group_size} selection={json.dumps(selection, ensure_ascii=False)}"
            )

            score, payload = _evaluate_selection(
                qa_json_path,
                corpus_json_path,
                selection,
                eval_mode,
                preferred_metric,
                score_weights,
                eval_fn,
            )

            if payload.get("error"):
                score = -1.0

            trajectories.append(trajectory)
            rewards.append(score)

            record = {
                "episode": ep + 1,
                "group": idx + 1,
                "score": payload.get("score"),
                "metric": payload.get("metric"),
                "selection": selection,
                "report": payload.get("report"),
                "outputs": payload.get("outputs"),
                "error": payload.get("error"),
                "errors": payload.get("errors"),
            }
            trials.append(record)

            if score >= best_score:
                best_score = score
                best_config = json.loads(json.dumps(selection))

            _write_report_snapshot()

        policy.update_grpo(
            trajectories=trajectories,
            rewards=rewards,
            ref_logits=ref_logits,
            learning_rate=learning_rate,
            kl_coeff=kl_coeff,
        )
        if bar:
            bar.update(1)

    if bar:
        bar.close()

    result = {
        "best_score": best_score,
        "best_config": best_config,
        "trials": trials,
    }
    _write_report_snapshot()
    return result


def main() -> None:
    import argparse

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_algo_config = os.path.join(os.path.dirname(__file__), "configforalgo.yaml")
    default_report = os.path.join(base_dir, "outputs", "rl_report.json")

    parser = argparse.ArgumentParser(description="RL (GRPO) search for RAG.")
    parser.add_argument("--qa_json", required=True, help="Path to QA JSON/JSONL.")
    parser.add_argument("--corpus_json", required=True, help="Path to corpus JSON.")
    parser.add_argument(
        "--config_yaml",
        default=default_algo_config,
        help="Path to algo config with search space.",
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
        help="Path to write RL report JSON.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of RL episodes.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=4,
        help="Samples per GRPO update.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate.",
    )
    parser.add_argument(
        "--kl_coeff",
        type=float,
        default=0.02,
        help="KL penalty coefficient.",
    )
    parser.add_argument(
        "--score_weights",
        default="",
        help="Weighted metrics, e.g. 'bertf11,llmaaj2'.",
    )
    args = parser.parse_args()

    score_weights = _parse_score_weights(args.score_weights)
    rl_search(
        qa_json_path=args.qa_json,
        corpus_json_path=args.corpus_json,
        config_path=args.config_yaml,
        eval_mode=args.eval_mode,
        report_path=args.report_path,
        episodes=args.episodes,
        seed=args.seed,
        learning_rate=args.lr,
        group_size=args.group_size,
        kl_coeff=args.kl_coeff,
        score_weights=score_weights,
    )


if __name__ == "__main__":
    main()

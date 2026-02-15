#!/usr/bin/env bash
set -euo pipefail

QA_JSON="${QA_JSON:-/home/xwh/janeeval/datasets/longbench-multifield/qa.json}"
CORPUS_JSON="${CORPUS_JSON:-/home/xwh/janeeval/datasets/longbench-multifield/corpus.json}"
CONFIG_YAML="${CONFIG_YAML:-/home/xwh/janeeval/algorithms/configforalgo.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/xwh/janeeval/outputs}"
EVAL_MODE="${EVAL_MODE:-avg}"
SCORE_WEIGHTS="${SCORE_WEIGHTS:-llmaaj1,bertf11,rougel1,f11,bleu1,em1}"
SEEDS="${SEEDS:-42 43 44}"

mkdir -p "$OUTPUT_DIR"

python /home/xwh/janeeval/algorithms/greedy.py \
  --qa_json "$QA_JSON" \
  --corpus_json "$CORPUS_JSON" \
  --config_yaml "$CONFIG_YAML" \
  --eval_mode "$EVAL_MODE" \
  --report_path "$OUTPUT_DIR/greedy.json" \
  --score_weights "$SCORE_WEIGHTS"

for SEED in $SEEDS; do
  python /home/xwh/janeeval/algorithms/grpo.py \
    --qa_json "$QA_JSON" \
    --corpus_json "$CORPUS_JSON" \
    --config_yaml "$CONFIG_YAML" \
    --eval_mode "$EVAL_MODE" \
    --report_path "$OUTPUT_DIR/grpo_seed${SEED}.json" \
    --episodes 10 \
    --group_size 2 \
    --seed "$SEED" \
    --update_epochs 2 \
    --score_weights "$SCORE_WEIGHTS"

  python /home/xwh/janeeval/algorithms/mab_ts.py \
    --qa_json "$QA_JSON" \
    --corpus_json "$CORPUS_JSON" \
    --config_yaml "$CONFIG_YAML" \
    --eval_mode "$EVAL_MODE" \
    --report_path "$OUTPUT_DIR/mab_ts_seed${SEED}.json" \
    --budget 20 \
    --pool_size 10 \
    --seed "$SEED" \
    --score_weights "$SCORE_WEIGHTS"

  python /home/xwh/janeeval/algorithms/mab_ucb.py \
    --qa_json "$QA_JSON" \
    --corpus_json "$CORPUS_JSON" \
    --config_yaml "$CONFIG_YAML" \
    --eval_mode "$EVAL_MODE" \
    --report_path "$OUTPUT_DIR/mab_ucb_seed${SEED}.json" \
    --budget 20 \
    --pool_size 10 \
    --seed "$SEED" \
    --score_weights "$SCORE_WEIGHTS"

  python /home/xwh/janeeval/algorithms/randomalgo.py \
    --qa_json "$QA_JSON" \
    --corpus_json "$CORPUS_JSON" \
    --config_yaml "$CONFIG_YAML" \
    --eval_mode "$EVAL_MODE" \
    --report_path "$OUTPUT_DIR/random_seed${SEED}.json" \
    --samples 20 \
    --seed "$SEED" \
    --score_weights "$SCORE_WEIGHTS"

  python /home/xwh/janeeval/algorithms/tpe.py \
    --qa_json "$QA_JSON" \
    --corpus_json "$CORPUS_JSON" \
    --config_yaml "$CONFIG_YAML" \
    --eval_mode "$EVAL_MODE" \
    --report_path "$OUTPUT_DIR/tpe_seed${SEED}.json" \
    --samples 20 \
    --seed "$SEED" \
    --startup_trials 5 \
    --candidate_pool_size 10 \
    --gamma 0.2 \
    --alpha 1.0 \
    --score_weights "$SCORE_WEIGHTS"
done

python - <<'PY'
import json
import math
import os

output_dir = os.environ.get("OUTPUT_DIR", "/home/xwh/janeeval/outputs")
seeds = os.environ.get("SEEDS", "42 43 44").split()
score_weights = os.environ.get("SCORE_WEIGHTS", "llmaaj1,bertf11,rougel1,f11,bleu1,em1")

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

def parse_weights(text):
    weights = {}
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
    return weights

weights = parse_weights(score_weights)

def weighted_score(metrics):
    if not metrics or not weights:
        return 0.0
    total = 0.0
    denom = 0.0
    for key, weight in weights.items():
        if key in metrics:
            total += float(metrics[key]) * weight
            denom += weight
    return total / denom if denom > 0 else 0.0

def load_best_metrics(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    trials = data.get("trials") or []
    if not trials:
        metrics = (data.get("report") or {}).get("metrics") or {}
        return metrics, data.get("best_score") or weighted_score(metrics)
    def score_key(item):
        score = item.get("score")
        if score is None:
            return float("-inf")
        try:
            return float(score)
        except Exception:
            return float("-inf")
    best = max(trials, key=score_key)
    metrics = (best.get("report") or {}).get("metrics") or {}
    score = best.get("score")
    if score is None:
        score = weighted_score(metrics)
    return metrics, float(score)

metrics_order = ["LLMAAJ", "BERTScore-F1", "ROUGE-L", "F1", "BLEU", "ExactMatch"]
algos = {
    "grpo": [f"grpo_seed{s}.json" for s in seeds],
    "greedy": ["greedy.json"],
    "mab_ts": [f"mab_ts_seed{s}.json" for s in seeds],
    "mab_ucb": [f"mab_ucb_seed{s}.json" for s in seeds],
    "random": [f"random_seed{s}.json" for s in seeds],
    "tpe": [f"tpe_seed{s}.json" for s in seeds],
}

rows = []
for algo, files in algos.items():
    for filename in files:
        metrics_score = load_best_metrics(os.path.join(output_dir, filename))
        if not metrics_score:
            continue
        metrics, score = metrics_score
        seed = "NA"
        if "seed" in filename:
            seed = filename.split("seed")[-1].split(".")[0]
        row = {
            "algo": algo,
            "seed": seed,
            "score": score,
        }
        for key in metrics_order:
            row[key] = float(metrics.get(key, 0.0))
        rows.append(row)

def fmt(v):
    return f"{v:.4f}"

print("\n[summary] per-run metrics")
header = ["algo", "seed", "weighted"] + metrics_order
print("\t".join(header))
for row in rows:
    values = [row["algo"], str(row["seed"]), fmt(row["score"])]
    for key in metrics_order:
        values.append(fmt(row[key]))
    print("\t".join(values))

print("\n[summary] mean over seeds")
by_algo = {}
for row in rows:
    by_algo.setdefault(row["algo"], []).append(row)

print("\t".join(["algo", "runs", "weighted_mean"] + [f"{m}_mean" for m in metrics_order]))
for algo, items in sorted(by_algo.items()):
    n = len(items)
    def mean(key):
        return sum(item[key] for item in items) / n if n else 0.0
    values = [algo, str(n), fmt(mean("score"))]
    for key in metrics_order:
        values.append(fmt(mean(key)))
    print("\t".join(values))
PY

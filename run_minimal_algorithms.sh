#!/usr/bin/env bash
set -euo pipefail

QA_JSON="${QA_JSON:-/home/xwh/janeeval/datasets/longbench-multifield/qa.json}"
CORPUS_JSON="${CORPUS_JSON:-/home/xwh/janeeval/datasets/longbench-multifield/corpus.json}"
CONFIG_YAML="${CONFIG_YAML:-/home/xwh/janeeval/algorithms/configforalgo.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/xwh/janeeval/outputs}"
EVAL_MODE="${EVAL_MODE:-avg}"

# python /home/xwh/janeeval/algorithms/coordinate_descent.py --qa_json "$QA_JSON" --corpus_json "$CORPUS_JSON" --config_yaml "$CONFIG_YAML" --eval_mode "$EVAL_MODE" --report_path "$OUTPUT_DIR/coord_desc_min.json" --max_rounds 1 --seed 1
python /home/xwh/janeeval/algorithms/cross_entropy.py --qa_json "$QA_JSON" --corpus_json "$CORPUS_JSON" --config_yaml "$CONFIG_YAML" --eval_mode "$EVAL_MODE" --report_path "$OUTPUT_DIR/cem_min.json" --iterations 1 --samples_per_iter 1 --elite_fraction 0.25 --seed 1 --alpha 1.0
python /home/xwh/janeeval/algorithms/greedy.py --qa_json "$QA_JSON" --corpus_json "$CORPUS_JSON" --config_yaml "$CONFIG_YAML" --eval_mode "$EVAL_MODE" --report_path "$OUTPUT_DIR/greedy_min.json"
python /home/xwh/janeeval/algorithms/grpo.py --qa_json "$QA_JSON" --corpus_json "$CORPUS_JSON" --config_yaml "$CONFIG_YAML" --eval_mode "$EVAL_MODE" --report_path "$OUTPUT_DIR/grpo_min.json" --episodes 1 --group_size 1 --seed 1 --update_epochs 1
python /home/xwh/janeeval/algorithms/iterative_local_search.py --qa_json "$QA_JSON" --corpus_json "$CORPUS_JSON" --config_yaml "$CONFIG_YAML" --eval_mode "$EVAL_MODE" --report_path "$OUTPUT_DIR/ils_min.json" --restarts 1 --steps_per_restart 1 --seed 1 --ils_perturb_steps 1 --ils_local_steps 1 --ils_neighborhood_size 1
python /home/xwh/janeeval/algorithms/mab_ts.py --qa_json "$QA_JSON" --corpus_json "$CORPUS_JSON" --config_yaml "$CONFIG_YAML" --eval_mode "$EVAL_MODE" --report_path "$OUTPUT_DIR/mab_ts_min.json" --budget 1 --pool_size 1 --seed 1
python /home/xwh/janeeval/algorithms/mab_ucb.py --qa_json "$QA_JSON" --corpus_json "$CORPUS_JSON" --config_yaml "$CONFIG_YAML" --eval_mode "$EVAL_MODE" --report_path "$OUTPUT_DIR/mab_ucb_min.json" --budget 1 --pool_size 1 --seed 1
python /home/xwh/janeeval/algorithms/randomalgo.py --qa_json "$QA_JSON" --corpus_json "$CORPUS_JSON" --config_yaml "$CONFIG_YAML" --eval_mode "$EVAL_MODE" --report_path "$OUTPUT_DIR/random_min.json" --samples 1 --seed 1
python /home/xwh/janeeval/algorithms/regularized_evolution.py --qa_json "$QA_JSON" --corpus_json "$CORPUS_JSON" --config_yaml "$CONFIG_YAML" --eval_mode "$EVAL_MODE" --report_path "$OUTPUT_DIR/evolution_min.json" --budget 1 --population_size 1 --sample_size 1 --seed 1
python /home/xwh/janeeval/algorithms/simulated_annealing.py --qa_json "$QA_JSON" --corpus_json "$CORPUS_JSON" --config_yaml "$CONFIG_YAML" --eval_mode "$EVAL_MODE" --report_path "$OUTPUT_DIR/anneal_min.json" --steps 1 --seed 1 --start_temp 1.0 --end_temp 1.0
python /home/xwh/janeeval/algorithms/successive_halving.py --qa_json "$QA_JSON" --corpus_json "$CORPUS_JSON" --config_yaml "$CONFIG_YAML" --eval_mode "$EVAL_MODE" --report_path "$OUTPUT_DIR/successive_halving_min.json" --num_configs 1 --eta 2 --seed 1
python /home/xwh/janeeval/algorithms/tpe.py --qa_json "$QA_JSON" --corpus_json "$CORPUS_JSON" --config_yaml "$CONFIG_YAML" --eval_mode "$EVAL_MODE" --report_path "$OUTPUT_DIR/tpe_min.json" --samples 1 --seed 1 --startup_trials 1 --candidate_pool_size 1 --gamma 0.2 --alpha 1.0

RAG 超参数搜索实验环境

这是一个用于 RAG 超参数搜索的实验项目。你可以使用内置算法，也可以自定义算法，只要最终生成的配置符合搜索空间约束，就能在本项目中评估并返回分数与详细报告。

项目架构

- `config.yaml` / `config_multimodal.yaml`：定义允许的超参数空间（搜索空间）。
- `functions/findsearchspace.py`：读取配置，展示可用空间与可删模块信息。
- `functions/checkconfig.py`：校验配置是否符合搜索空间与格式要求。
- `rag/*`：RAG 流水线实现与评估逻辑。
- `mainfunction.py`：封装评估入口，调用流水线并返回报告。
- `algorithms/*`：搜索算法实现与示例。

核心接口（3 个）

1) 搜索空间查询
- `functions/findsearchspace.py`
- 用于查看当前支持的搜索空间与参数范围。
- 同时会告诉你如何生成可评估配置，以及哪些模块可删除（不启动）/哪些模块必须保留。

2) 文本 RAG 评估（核心）
- `mainfunction.py` 中的 `evaluate_rag`
- 输入：`config` 配置 + `qa` + `corpus`
- 功能：检查配置合法性（是否在允许搜索空间内、格式是否正确），然后在 RAG 流水线上评估并返回分数与报告。

3) 多模态 RAG 评估 (核心)
- `mainfunction.py` 中的 `evaluate_rag_multimodal`
- 与文本版一致，但多了 `clip`，少了 `pruner`。

配置说明

- `config.yaml`：文本 RAG 的允许超参数空间。
- `config_multimodal.yaml`：多模态 RAG 的允许超参数空间。
- 校验函数与 RAG 代码会自适应读取这些配置文件，所以你只需保证生成的 `config` 在允许范围内即可。
- 如用本地模型留意内存使用，内存不足可能导致某些环节返回空值

API 使用（根目录 `api.py`）

你可以直接从根目录的 `api.py` 引用这 5 个方法：
- `evaluate_rag`
- `evaluate_rag_multimodal`
- `check_config_valid`
- `find_search_space`
- `run_algorithms`

参数说明（5 个方法）

1) `evaluate_rag(qa_json_path, corpus_json_path, config_path, eval_mode="both")`
- `qa_json_path`：QA JSON/JSONL 路径（每项含 `query` 和 `references`）。
- `corpus_json_path`：语料 JSON 路径（每项含 `id` 和 `content`）。
- `config_path`：可评估的 RAG 配置。
- `eval_mode`：`both` / `gen` / `retrieval`。

2) `evaluate_rag_multimodal(qa_json_path, corpus_json_path, config_path, eval_mode="both")`
- 参数同 `evaluate_rag`，多模态配置需包含 `clip` 模块。

3) `check_config_valid(config_path, multimodal=False)`
- `config_path`：待校验配置路径。
- `multimodal`：`False` 表示文本 RAG；`True` 表示多模态 RAG。

4) `find_search_space(config_path, multimodal=False)`
- `config_path`：搜索空间配置路径（`config.yaml` / `config_multimodal.yaml`）。
- `multimodal`：同上。

5) `run_algorithms(qa_json_path, corpus_json_path, config_path, algorithms=None, eval_mode="both", score_weights="", extra_args=None, cwd=None)`
- `qa_json_path` / `corpus_json_path` / `config_path`：同上。
- `algorithms`：要执行的算法名列表（如 `["randomalgo", "tpe"]`），不传则跑默认全部。
- `eval_mode`：同上。
- `score_weights`：传给算法的打分权重（如 `bertf11,llmaaj2`）。
- `extra_args`：为指定算法追加 CLI 参数（形如 `{"randomalgo": ["--foo", "bar"]}`）。
- `cwd`：子进程工作目录（默认项目根目录）。

示例（直接脚本运行）：
```python
from api import (
    evaluate_rag,
    evaluate_rag_multimodal,
    check_config_valid,
    find_search_space,
    run_algorithms,
)

# 1) 检查配置合法性
print(check_config_valid("configs/demo2.yaml"))

# 2) 查看搜索空间与模板
space = find_search_space("config.yaml")
print(space["description"])

# 3) 文本 RAG 评估
result = evaluate_rag(
    qa_json_path="datasets/bioasq/qa.json",
    corpus_json_path="datasets/bioasq/corpus.json",
    config_path="configs/demo2.yaml",
    eval_mode="both",
)
print(result["eval_report"])

# 4) 多模态 RAG 评估
mm_result = evaluate_rag_multimodal(
    qa_json_path="...",
    corpus_json_path="...",
    config_path="configs/demo3.yaml",
)

# 5) 串行批量执行算法（全部）
run_algorithms(
    qa_json_path="datasets/bioasq/qa.json",
    corpus_json_path="datasets/bioasq/corpus.json",
    config_path="algorithms/configforalgo.yaml",
    score_weights="bertf11,llmaaj2",
)

# 显式打印运行结果（run_algorithms 只返回结果，不会自动打印）
import json
res = run_algorithms(
    qa_json_path="datasets/bioasq/qa.json",
    corpus_json_path="datasets/bioasq/corpus.json",
    config_path="algorithms/configforalgo.yaml",
    score_weights="bertf11,llmaaj2",
)
print(json.dumps(res, ensure_ascii=False, indent=2))

# 只执行某几个算法
run_algorithms(
    qa_json_path="datasets/bioasq/qa.json",
    corpus_json_path="datasets/bioasq/corpus.json",
    config_path="algorithms/configforalgo.yaml",
    algorithms=["randomalgo", "tpe", "upperbound"],
)
```

算法与配置

- 目录：`algorithms/`
- 内置多种搜索算法（示例实现）。
- 使用方式：直接运行对应算法脚本即可，例如 `python algorithms/randomalgo.py`。
- `algorithms/configforalgo.yaml` 是算法输入示例，你可以填入自己的 key 直接运行内置算法。
- 你自己的算法**可以完全不依赖这个 YAML**，只要你生成的 `config` 符合搜索空间约束，`evaluate_rag` / `evaluate_rag_multimodal` 就会返回分数与报告，帮助你优化超参数选择。

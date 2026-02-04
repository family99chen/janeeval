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

算法与配置

- 目录：`algorithms/`
- 内置多种搜索算法（示例实现）。
- 使用方式：直接运行对应算法脚本即可，例如 `python algorithms/randomalgo.py`。
- `algorithms/configforalgo.yaml` 是算法输入示例，你可以填入自己的 key 直接运行内置算法。
- 你自己的算法**可以完全不依赖这个 YAML**，只要你生成的 `config` 符合搜索空间约束，`evaluate_rag` / `evaluate_rag_multimodal` 就会返回分数与报告，帮助你优化超参数选择。

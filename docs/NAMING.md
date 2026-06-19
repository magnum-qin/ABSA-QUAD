# 命名规范 (Naming Convention)

统一所有 KD (V23) 相关产物的命名，使 **脚本 / 配置 / 数据 / checkpoint** 一一对应、可由名字直接推断内容。

## 核心模式

```
v23_<dataset>_<teacher>[_<variant>]
```

| 占位符 | 取值 |
|---|---|
| `<dataset>` | 完整数据集名 `<lang>_<domain>`：`eng_laptop` `eng_restaurant` `jpn_hotel` `rus_restaurant` `tat_restaurant` `ukr_restaurant` `zho_laptop` `zho_restaurant` |
| `<teacher>` | silver 标注的 LLM：`deepseek`（云端）/ `qwen`（本地 qwen3.6）/ `gemma`（本地 gemma-31B）|
| `<variant>` | 可选限定词：`native`(LLM原生生成池) / `trans`(机器翻译池) / `real`(真实公开语料池) / `full` / `capped`(silver用量) / `gen`(生成池) |

## 各产物命名

| 产物 | 规则 | 示例 |
|---|---|---|
| 脚本 | `scripts/run_v23_<dataset>_<teacher>.sh` | `run_v23_rus_restaurant_deepseek.sh` |
| 多数据集脚本 | `scripts/run_v23_multi_<teacher>.sh` | `run_v23_multi_qwen.sh` |
| distill 构建脚本 | `scripts/build_v23_<dataset>_<teacher>.py` | `build_v23_eng_laptop_qwen.py` |
| 配置 | `configs/pipeline_<dataset>_v23_<teacher>.yaml` | `pipeline_eng_laptop_v23_deepseek.yaml` |
| 无标注池 | `data/.../<dataset>_pool_v23[_<src>].jsonl` | `tat_restaurant_pool_v23_native.jsonl` |
| silver 标签 | `data/.../<dataset>_silver_v23_<teacher>.jsonl` | `eng_laptop_silver_v23_deepseek.jsonl` |
| distill 训练集 | `data/.../<dataset>_train_v23_<teacher>[_<variant>].jsonl` | `zho_laptop_train_v23_qwen_full.jsonl` |
| checkpoint | `checkpoints/v23_<dataset>_<teacher>_seed<N>` | `v23_rus_restaurant_deepseek_seed42` |

## 关键约定

- **DeepSeek 也显式标注**为 `deepseek`。历史上 DeepSeek silver 曾简称 `_silver`（无 teacher 名），现已废弃——`silver` 仅表示"LLM 弱标签"这一通用概念，不再特指某 teacher。
- **dataset 永远用完整名**。历史上 config/checkpoint 曾用短名（`rus`/`tat`/`ukr`），现统一为 `rus_restaurant` 等。
- **池来源用 variant 区分**：`real`(真实公开语料 Yelp/amazon/geo)、`native`(LLM 原生生成)、`trans`(机器翻译)。
- baseline（非 KD）配置沿用 `pipeline_<dataset>_v<N>.yaml`（如 `pipeline_eng_restaurant_v16.yaml`），不带 teacher。

## 提交对应（当前 8 数据集）

| 数据集 | 提交配置 | teacher |
|---|---|---|
| eng_laptop | `pipeline_eng_laptop_v23_deepseek.yaml` | deepseek |
| rus_restaurant | `pipeline_rus_restaurant_v23_deepseek.yaml` | deepseek |
| ukr_restaurant | `pipeline_ukr_restaurant_v23_deepseek.yaml` | deepseek |
| tat_restaurant | `pipeline_tat_restaurant_v23_deepseek.yaml` | deepseek |
| zho_restaurant | `pipeline_zho_restaurant_v23_qwen.yaml` | qwen |
| eng_restaurant / jpn_hotel / zho_laptop | `pipeline_<dataset>_v16.yaml` | — (baseline) |

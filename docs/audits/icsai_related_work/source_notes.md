# 第二稿相关工作核验（2026-09-10）

只采用作者论文、ACL Anthology和arXiv原始来源。下面的“本稿定位”是对研究范围的比较，不是未经复现的性能排名。

| 文献与来源 | 核实的方法/任务范围 | 本稿中的用途及可比性限制 |
|---|---|---|
| [One-ASQP, Findings ACL 2023](https://aclanthology.org/2023.findings-acl.777/) | 同时识别类别和aspect–opinion–sentiment triplets，使用token-pair二维标注及NULL token；作者Zhou、Yang、He、Mou、Yang，页12249–12265 | 对比结构表示及隐式元素处理；离散极性结果不与维度cF1直接比较 |
| [SCRAP, Findings ACL 2024](https://aclanthology.org/2024.findings-acl.435/) | extract-then-assign文本推理与四元组生成，采用consistency voting；页7295–7303 | 区分生成式推理/投票与本稿数值BIO嵌入反馈；不把两者称为相同干预 |
| [AILS-NTUA, arXiv:2603.04933](https://arxiv.org/abs/2603.04933) | 语言适配encoder用于回归；LoRA指令调优LLM用于triplet/quadruplet结构抽取 | 引用任务适配与语言专门化设计，不摘取未对齐配置的分数作为基线 |
| [QuadAI, arXiv:2603.07766](https://arxiv.org/html/2603.07766v1) | Hybrid RoBERTa回归和离散头，encoder/LLM prediction-level组合。方法3.3节、实验4.3节为ensemble；限制节说明LLM/ensemble只在dev评估 | 本稿准确标为回归和开发集ensemble工作；不能作为本稿Subtask3 test的直接比较 |
| [LogSigma, arXiv:2603.24896](https://arxiv.org/html/2603.24896v1) | 第2节明确参加两track的Subtask1；给定text+aspect，学习V/A uncertainty权重；有语言专用encoder与多seed平均 | 仅支撑损失平衡设计背景，不能把回归RMSE或排名放进本稿quad extraction cF1表 |

新增正文均为简短方法概述，不复制论文段落。未声称穷尽所有2026相关研究，也未根据跨任务/跨split分数构造排行榜。原稿已有的ACOS、ASQP、DimABSA、共享任务概述、DeBERTaV3、Hungarian文献保留。

结论：本稿的定位是一个已实现BIO反馈路径的受控评估与可追溯负结果；不是提出另一种LLM推理框架，也不是外部SOTA对照。外部系统同条件复现仍不在当前证据范围内。

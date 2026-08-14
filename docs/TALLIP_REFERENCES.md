# TALLIP reference ledger

Updated: 2026-08-15.

This ledger resolves the temporary `[CITE: key]` markers in
`TALLIP_MANUSCRIPT_DRAFT.md`. It is deliberately not a BibTeX file: the final
bibliography style and the publication metadata of the accepted ICONIP extended
abstract are not yet fixed. Entries marked **verified** were checked against a
primary paper, publisher page, DOI record, or ACL Anthology record. Convert this
ledger to BibTeX only when the TALLIP author template is adopted.

## Structured and dimensional ABSA

### `cai2021acos` — verified

- Citation: Hongjie Cai, Rui Xia, and Jianfei Yu. 2021. “Aspect-Category-Opinion-Sentiment Quadruple Extraction with Implicit Aspects and Opinions.” In *Proceedings of ACL-IJCNLP 2021 (Volume 1: Long Papers)*, 340–350. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/2021.acl-long.29/
- DOI: https://doi.org/10.18653/v1/2021.acl-long.29
- Manuscript role: ACOS quadruple extraction and implicit spans.

### `zhang2021asqp` — verified

- Citation: Wenxuan Zhang, Yang Deng, Xin Li, Yifei Yuan, Lidong Bing, and Wai Lam. 2021. “Aspect Sentiment Quad Prediction as Paraphrase Generation.” In *Proceedings of EMNLP 2021*, 9209–9219. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/2021.emnlp-main.726/
- DOI: https://doi.org/10.18653/v1/2021.emnlp-main.726
- Manuscript role: definition and generative formulation of categorical ASQP.

### `lee2026dimabsa` — verified preprint

- Citation: Lung-Hao Lee, Liang-Chih Yu, Natalia Loukashevich, Ilseyar Alimova, Alexander Panchenko, Tzu-Mi Lin, Zhe-Yu Xu, Jian-Yu Zhou, Guangmin Zheng, Jin Wang, Sharanya Awasthi, Jonas Becker, Jan Philip Wahle, Terry Ruas, Shamsuddeen Hassan Muhammad, and Saif M. Mohammed. 2026. “DimABSA: Building Multilingual and Multidomain Datasets for Dimensional Aspect-Based Sentiment Analysis.” arXiv:2601.23022.
- Primary record: https://arxiv.org/abs/2601.23022
- Manuscript role: multilingual DimABSA resource, annotation design, VA formulation, and cF1 background.

### `yu2026semeval` — verified

- Citation: Liang-Chih Yu, Jonas Becker, Shamsuddeen Hassan Muhammad, Idris Abdulmumin, Lung-Hao Lee, Ying-Lung Lin, Jin Wang, Jan Philip Wahle, Terry Lima Ruas, Natalia Loukachevitch, Alexander Panchenko, Ilseyar Alimova, Lilian Diana Awuor Wanzare, Nelson Odhiambo, Bela Gipp, Kai-Wei Chang, and Saif Mohammad. 2026. “SemEval-2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (DimABSA).” In *Proceedings of the 20th International Workshop on Semantic Evaluation*, 3753–3778. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/2026.semeval-1.452/
- DOI: https://doi.org/10.18653/v1/2026.semeval-1.452
- Manuscript role: official task, datasets, subtasks, metric, and evaluation context.

### `alshawi2026classics` — verified

- Citation: Rafif Alshawi, Amit Raj, Aleksey Kudelya, and Alexander Shirnin. 2026. “The Classics at SemEval-2026 Task 3: Combining Transformer Models and LLM-Generated Annotations for Dimensional Aspect-Based Sentiment Analysis.” In *Proceedings of the 20th International Workshop on Semantic Evaluation*, 2648–2656. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/2026.semeval-1.334/
- DOI: https://doi.org/10.18653/v1/2026.semeval-1.334
- Manuscript role: closest task-specific precedent using LLM-generated supervision.

### `qin2026leader` — accepted; publication metadata pending

- Working citation: Hongzhen Qin, Yoshimi Suzuki, Fumiyo Fukumoto, and Renjie Zhou. 2026. “LEADER-DimASQP: Language-Specific Encoder and Data-Aware Ensemble Refinement for Dimensional ASQP.” Accepted for abstract presentation at ICONIP 2026; open-access extended abstract forthcoming.
- Local source of record: `docs/paper_iconip_current.tex` and the ICONIP 2026 acceptance email for submission 848.
- Pending before submission: final proceedings title, page range, publisher, URL, and DOI.
- Manuscript role: prior student architecture, dataset-specific recipes, and five-seed inference; these are not new TALLIP contributions.

## Distillation, pseudo-labeling, and LLM supervision

### `hinton2015distilling` — verified preprint

- Citation: Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. “Distilling the Knowledge in a Neural Network.” arXiv:1503.02531.
- Primary records: https://research.google/pubs/distilling-the-knowledge-in-a-neural-network/ and https://arxiv.org/abs/1503.02531
- Manuscript role: classical knowledge distillation and deployment motivation.

### `lee2013pseudolabel` — verified workshop paper

- Citation: Dong-Hyun Lee. 2013. “Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks.” In *ICML 2013 Workshop on Challenges in Representation Learning*.
- Primary manuscript record: https://citeseerx.ist.psu.edu/document?doi=798d9840d2439a0e5d47bcf5d164aa46d5e7dc26
- Manuscript role: hard-label self-training baseline.

### `hsieh2023distilling` — verified

- Citation: Cheng-Yu Hsieh, Chun-Liang Li, Chih-kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. 2023. “Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes.” In *Findings of ACL 2023*, 8003–8017. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/2023.findings-acl.507/
- DOI: https://doi.org/10.18653/v1/2023.findings-acl.507
- Manuscript role: LLM-generated response/rationale supervision for smaller students.

### `dasgupta2023cost` — verified

- Citation: Sayantan Dasgupta, Trevor Cohn, and Timothy Baldwin. 2023. “Cost-effective Distillation of Large Language Models.” In *Findings of ACL 2023*, 7346–7354. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/2023.findings-acl.463/
- DOI: https://doi.org/10.18653/v1/2023.findings-acl.463
- Manuscript role: efficient task-specific LLM distillation and deployment framing.

### `golde2023fabricator` — verified

- Citation: Jonas Golde, Patrick Haller, Felix Hamborg, Julian Risch, and Alan Akbik. 2023. “Fabricator: An Open Source Toolkit for Generating Labeled Training Data with Teacher LLMs.” In *Proceedings of EMNLP 2023: System Demonstrations*, 1–11. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/2023.emnlp-demo.1/
- DOI: https://doi.org/10.18653/v1/2023.emnlp-demo.1
- Manuscript role: reproducible LLM-generated labeled-data workflows.

### `ding2023annotator` — verified

- Citation: Bosheng Ding, Chengwei Qin, Linlin Liu, Yew Ken Chia, Boyang Li, Shafiq Joty, and Lidong Bing. 2023. “Is GPT-3 a Good Data Annotator?” In *Proceedings of ACL 2023 (Volume 1: Long Papers)*, 11173–11195. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/2023.acl-long.626/
- DOI: https://doi.org/10.18653/v1/2023.acl-long.626
- Manuscript role: direct evaluation of LLMs as task annotators.

### `zhang2023llmaaa` — verified

- Citation: Ruoyu Zhang, Yanzeng Li, Yongliang Ma, Ming Zhou, and Lei Zou. 2023. “LLMaAA: Making Large Language Models as Active Annotators.” In *Findings of EMNLP 2023*, 13088–13103. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/2023.findings-emnlp.872/
- DOI: https://doi.org/10.18653/v1/2023.findings-emnlp.872
- Manuscript role: LLM annotation, pseudo-label quality, and downstream student training.

### `li2023coannotating` — verified

- Citation: Minzhi Li, Taiwei Shi, Caleb Ziems, Min-Yen Kan, Nancy Chen, Zhengyuan Liu, and Diyi Yang. 2023. “CoAnnotating: Uncertainty-Guided Work Allocation between Human and Large Language Models for Data Annotation.” In *Proceedings of EMNLP 2023*, 1487–1505. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/2023.emnlp-main.92/
- DOI: https://doi.org/10.18653/v1/2023.emnlp-main.92
- Manuscript role: limits of fully automatic annotation and human--LLM work allocation.

## Multilingual annotation reliability and bias

### `jadhav2025limitations` — verified

- Citation: Suramya Jadhav, Abhay Shanbhag, Amogh Thakurdesai, Ridhima Sinare, and Raviraj Joshi. 2025. “On Limitations of LLM as Annotator for Low Resource Languages.” In *Proceedings of ICNLSP 2025*, 277–282. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/2025.icnlsp-1.27/
- Manuscript role: empirical caution about LLM annotation in a low-resource language.

### `cui2025bias` — verified

- Citation: Xia Cui, Ziyi Huang, and Naeemeh Adel. 2025. “Bias in, Bias out: Annotation Bias in Multilingual Large Language Models.” In *Proceedings of the Interdisciplinary Workshop on Observations of Misunderstood, Misguided and Malicious Use of Language Models*, 1–16. INCOMA Ltd.
- Primary record: https://aclanthology.org/2025.ommm-1.1/
- Manuscript role: annotation, instruction, contextual, and cultural bias in multilingual settings.

## Leakage, statistical inference, and multiplicity

### `lee2022deduplicating` — verified

- Citation: Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. 2022. “Deduplicating Training Data Makes Language Models Better.” In *Proceedings of ACL 2022 (Volume 1: Long Papers)*, 8424–8445. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/2022.acl-long.577/
- DOI: https://doi.org/10.18653/v1/2022.acl-long.577
- Manuscript role: train--evaluation overlap, deduplication, and selection bias.

### `sainz2024contamination` — verified

- Citation: Oscar Sainz, Iker García-Ferrero, Alon Jacovi, Jon Ander Campos, Yanai Elazar, Eneko Agirre, Yoav Goldberg, et al. 2024. “Data Contamination Report from the 2024 CONDA Shared Task.” In *Proceedings of the 1st Workshop on Data Contamination*, 41–56. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/2024.conda-1.4/
- DOI: https://doi.org/10.18653/v1/2024.conda-1.4
- Manuscript role: taxonomy and documented prevalence of benchmark contamination.
- Note: expand the full author list from the ACL Anthology BibTeX during final bibliography conversion.

### `yao2024crosslingual` — verified

- Citation: Feng Yao, Yufan Zhuang, Zihao Sun, Sunan Xu, Animesh Kumar, and Jingbo Shang. 2024. “Data Contamination Can Cross Language Barriers.” In *Proceedings of EMNLP 2024*, 17864–17875. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/2024.emnlp-main.990/
- DOI: https://doi.org/10.18653/v1/2024.emnlp-main.990
- Manuscript role: cross-lingual forms of contamination beyond simple same-language overlap.

### `koehn2004statistical` — verified

- Citation: Philipp Koehn. 2004. “Statistical Significance Tests for Machine Translation Evaluation.” In *Proceedings of EMNLP 2004*, 388–395. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/W04-3250/
- Manuscript role: paired bootstrap resampling for nonlinear corpus metrics.

### `dror2018hitchhiker` — verified

- Citation: Rotem Dror, Gili Baumer, Segev Shlomov, and Roi Reichart. 2018. “The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing.” In *Proceedings of ACL 2018 (Volume 1: Long Papers)*, 1383–1392. Association for Computational Linguistics.
- Primary record: https://aclanthology.org/P18-1128/
- DOI: https://doi.org/10.18653/v1/P18-1128
- Manuscript role: test selection and statistically sound NLP evaluation.

### `holm1979` — verified

- Citation: Sture Holm. 1979. “A Simple Sequentially Rejective Multiple Test Procedure.” *Scandinavian Journal of Statistics* 6(2): 65–70.
- DOI record: https://doi.org/10.2307/4615733
- Manuscript role: family-wise error correction across dataset-level comparisons.

### `benjamini1995` — verified

- Citation: Yoav Benjamini and Yosef Hochberg. 1995. “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.” *Journal of the Royal Statistical Society: Series B* 57(1): 289–300.
- DOI record: https://doi.org/10.2307/2346101
- Manuscript role: false-discovery-rate correction reported alongside Holm.

## Finalization checklist

- Replace every manuscript `[CITE: ...]` marker with the selected TALLIP/ACM
  citation command after the template is fixed.
- Export ACL records from their primary BibTeX blocks; do not retype author
  names or page ranges manually.
- Replace `qin2026leader` with its final open-access extended-abstract record
  once ICONIP publishes the proceedings metadata.
- Check whether the DimABSA preprint has acquired a venue or revised author
  metadata before submission.
- Keep this ledger as a provenance document even after generating the final
  `.bib` file.

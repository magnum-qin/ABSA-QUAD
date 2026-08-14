# Prior-publication and substantial-extension statement

Prepared for submission to *ACM Transactions on Asian and Low-Resource
Language Information Processing* (TALLIP), 2026-08-15.

## Related earlier work

Hongzhen Qin, Yoshimi Suzuki, Fumiyo Fukumoto, and Renjie Zhou.
“LEADER-DimASQP: Language-Specific Encoder and Data-Aware Ensemble Refinement
for Dimensional ASQP.” Accepted for abstract presentation at ICONIP 2026;
open-access extended abstract forthcoming.

The TALLIP manuscript cites this work as Qin et al. (2026). Its bibliographic
entry will be updated with the final proceedings metadata when those data are
available.

## Central distinction

The ICONIP paper asks how to construct and select a stable multilingual
DimASQP student system. It introduces LEADER-DimASQP, language- and
dataset-specific encoder/training recipes, and five-seed ensemble inference.

The TALLIP manuscript holds that student and its decoding procedure fixed and
asks when offline LLM-labeled silver data improves it. Its contribution is an
empirical and evaluation-methodology study of silver supervision, not a second
claim to the student architecture.

## Contribution-by-contribution comparison

| Dimension | ICONIP 2026 work | TALLIP manuscript |
|---|---|---|
| Primary question | How should a multilingual DimASQP student and per-dataset recipe be constructed? | When, why, and with what evidential confidence does offline LLM-labeled silver data improve the fixed student? |
| Claimed method | LEADER-DimASQP, including Rethink feedback, Hungarian matching, continuous VA prediction, training stabilization, encoder selection, and ensemble inference | An offline structured label-distillation protocol and an audit-first evaluation design; the inherited student is a controlled carrier |
| Main variables | Encoders, transfer, paraphrase/COMET augmentation, top-k, training components, and ensemble settings | Teacher identity, pool source, silver quantity/batch, contamination, exposure, teacher source, and complementarity descriptors |
| Teacher evidence | No 8-by-2 direct teacher-fidelity matrix | DeepSeek and Qwen evaluated on identical label-hidden gold development sets for all eight datasets, with paired intervals and multiplicity correction |
| Data-integrity evidence | Competition-era development protocol | Unified exact/normalized/source-ID audit over gold, silver, distill, development, and test inputs; three affected KD inputs cleaned and retrained |
| Downstream controls | Architecture, encoder, augmentation, and ensemble ablations | Five-domain repeated-gold exposure control, same-text student-self-training control, matched Tatar teacher control, Russian pool control, and a same-pool quantity intervention |
| Statistical design | Development-score comparisons and five-seed ensembles | Five fixed seeds, example-paired bootstrap intervals, equal-dataset macro inference, and family-specific Holm/BH diagnostics |
| Evaluation endpoint | Development results available during the shared-task workflow | Development-locked, one-shot local evaluation on released official test labels; no test-based model, threshold, teacher, pool, or augmentation selection |
| Main conclusion | No single student configuration is optimal across languages and domains | KD has a reliable average benefit on the fixed adopted set, but per-domain benefits are heterogeneous and no universal teacher/pool/quantity rule is supported |

## New material by manuscript section

- **Sections 1–2:** a new motivating problem and literature synthesis centered
  on LLM annotation, pseudo-labeling, contamination, and statistical evaluation.
- **Section 3.2:** a deliberately compressed description of the inherited
  LEADER-DimASQP student, cited to the ICONIP work. This is background, not a
  new contribution.
- **Sections 3.3–3.4:** the offline structured-label pipeline, experimental
  factors, and alternative explanations.
- **Sections 4.2–4.8:** contamination audit, teacher-fidelity protocol, frozen
  training/inference, matched controls, teacher/pool/quantity interventions,
  paired inference, multiplicity handling, and the official-test lock.
- **Section 5 in full:** all teacher-fidelity, clean-inductive, control,
  complementarity, error-transition, and official-test results are new.
- **Sections 6–7:** the interpretation, resource audit, limitations, ethics,
  and explicit relation to ICONIP are written around the new KD evidence.

This is a substantial extension in both research question and evidence, not an
unrevised republication. The new protocols, experiments, results, and
conclusions occupy well beyond the minimum 25% new-material threshold stated
in the available TALLIP author guidance and ACM's prior-publication guidance.
This assessment is based on contribution and section accounting, not on a
cosmetic page-count increase.

## Reused material and attribution

The following items are necessarily shared and are disclosed in the TALLIP
manuscript:

- the DimASQP task, eight official datasets, and official cF1 metric;
- the LEADER-DimASQP student, language-specific encoders and training recipes;
- five-seed ensemble inference and frozen vote/top-k settings; and
- the earlier non-KD systems used as matched or frozen baselines.

These elements are attributed to Qin et al. (2026) and are not listed as new
TALLIP contributions. No ICONIP table or figure is reproduced in the TALLIP
manuscript. A source audit found that the only long exact textual overlap is
task-definition notation; the TALLIP narrative, experimental protocols, tables,
figure, and conclusions are newly written for the KD study.

## Disclosure text used in the manuscript and cover letter

> This manuscript builds on our earlier ICONIP study of LEADER-DimASQP. The
> earlier work introduced the student architecture, language-specific recipe
> selection, and multi-seed inference, which are treated here as inherited
> components rather than new contributions. The present manuscript asks a
> different question: when LLM-labeled silver data improves that fixed student.
> It adds direct two-teacher fidelity evaluation across eight datasets,
> teacher/pool/quantity controls, exposure-matched and same-text self-training
> controls, direct complementarity and error measurements, leakage-audited
> clean-inductive retraining, multiplicity-aware paired inference, and a
> development-locked official-test evaluation. None of these post-ICONIP KD
> analyses or official-test results appeared in the earlier manuscript.

## Author attestation required before submission

The submitting author must confirm that:

1. all authors approve both manuscripts' relationship as described above;
2. the final ICONIP publication status and bibliographic metadata are accurate;
3. the TALLIP manuscript is not simultaneously under review elsewhere; and
4. the prior work and this statement are supplied wherever Manuscript Central
   requests related manuscripts or prior-publication disclosures.

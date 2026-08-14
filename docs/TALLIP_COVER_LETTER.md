# TALLIP cover-letter draft

Status: ready for author confirmation, 2026-08-15.

Do not upload this file until every item marked **AUTHOR CONFIRMATION** below has
been resolved. Delete this preparation note and the final checklist before
pasting the letter into Manuscript Central.

---

15 August 2026

Editor-in-Chief

*ACM Transactions on Asian and Low-Resource Language Information Processing*

Dear Editor-in-Chief,

On behalf of my co-authors, I am pleased to submit the Long Paper entitled
“When Does LLM-Labeled Silver Data Help? An Inductive-Safe Study of Offline
Structured Distillation for Multilingual Dimensional Aspect Sentiment Quad
Prediction” for consideration in *ACM Transactions on Asian and Low-Resource
Language Information Processing* (TALLIP).

The manuscript studies when labels produced offline by large language models
improve a fixed structured sentiment student. The evaluation covers eight
language--domain datasets in English, Japanese, Russian, Tatar, Ukrainian, and
Chinese. It combines direct label-hidden evaluation of two teachers, a unified
train/development/test contamination audit, clean-inductive five-seed training,
matched controls for training exposure and student self-training,
multiplicity-aware paired inference, and a development-locked one-shot
evaluation on the released official test set.

Across the five adopted distillation domains, the fixed-task development macro
effect is +0.0600 cF1 over the frozen non-KD baseline (95% CI [0.0291, 0.0911]).
The corresponding effects over exposure-matched repeated-gold training and
same-text student self-training are +0.0588 [0.0318, 0.0853] and +0.0621
[0.0351, 0.0888]. The official-test analysis supports four of five
pre-registered KD contrasts before multiplicity correction and three after
Holm correction. At the same time, the results do not support a universal
teacher, pool-source, or silver-quantity ranking. We believe this combination
of positive evidence and explicit boundary conditions is useful to researchers
working on Asian and low-resource language processing, especially where LLM
annotation cost and data scarcity make reliable silver-data selection
important.

We wish to disclose a closely related earlier work. “LEADER-DimASQP:
Language-Specific Encoder and Data-Aware Ensemble Refinement for Dimensional
ASQP,” by the same four authors, has been accepted for abstract presentation at
ICONIP 2026, with an open-access extended abstract forthcoming. That work
introduced the student architecture, language-specific recipe selection, and
five-seed ensemble inference. The present manuscript treats those components
as inherited controls and does not claim them as new.

The TALLIP manuscript asks a different research question and adds the complete
post-ICONIP distillation study: direct two-teacher fidelity on all eight
datasets; matched teacher, pool, and quantity comparisons; inductive cleaning
and retraining; repeated-gold and same-text self-training controls; direct
coverage and error-transition measurements; paired bootstrap inference with
multiplicity diagnostics; a reproducibility and resource audit; and a locked
official-test evaluation. None of these KD analyses or official-test results
appeared in the ICONIP manuscript. No ICONIP table or figure is reproduced.
Section 6.5 of the submitted manuscript cites the earlier work and explains the
relationship explicitly. A detailed contribution-by-contribution comparison is
available as an accompanying prior-publication statement.

**AUTHOR CONFIRMATION — retain the following sentence only after every author
has approved it:** All authors have approved this manuscript and its submission
to TALLIP; the manuscript is not under review elsewhere; and all related prior
work has been disclosed.

**AUTHOR CONFIRMATION — replace this sentence with the actual declaration:**
The authors declare [no competing interests / the following competing
interests: ...].

Thank you for considering our manuscript. We would be grateful for the
opportunity to have it reviewed for TALLIP.

Sincerely,

Hongzhen Qin

School of Computer Science and Technology, Hangzhou Dianzi University

University of Yamanashi

241050420@hdu.edu.cn; g24tka12@yamanashi.ac.jp

**[AUTHOR CONFIRMATION: corresponding author]**

---

## Author checks before use

- Confirm that Hongzhen Qin is the corresponding and submitting author.
- Confirm approval from Yoshimi Suzuki, Fumiyo Fukumoto, and Renjie Zhou.
- Confirm that no substantially similar manuscript is under review elsewhere.
- Replace the competing-interest placeholder with the authors' declaration.
- Replace the ICONIP status and citation if final proceedings metadata become
  available before TALLIP submission.
- Upload [`TALLIP_PRIOR_PUBLICATION_STATEMENT.md`](TALLIP_PRIOR_PUBLICATION_STATEMENT.md)
  if the portal provides a related-work or supplemental-document field; in all
  cases paste the disclosure into the cover letter.

# When Does LLM-Labeled Silver Data Help?

## An Inductive-Safe Study of Offline Structured Distillation for Multilingual Dimensional Aspect Sentiment Quad Prediction

**Draft status:** Sections 3--4, working manuscript, 2026-08-14.

> Editorial note: this Markdown file is the editable source draft and is not the
> ICONIP manuscript. Bracketed `[CITE: ...]` markers are unresolved bibliography
> tasks, not submission-ready citations. Numerical results must be imported from
> `docs/audits/tallip_kd/tallip_core_tables.md` after its verification command
> passes.

## 3 Task and Offline Structured Distillation

### 3.1 Dimensional Aspect Sentiment Quad Prediction

Given an input sentence (x=(x_1,\ldots,x_n)), the Dimensional Aspect
Sentiment Quad Prediction (DimASQP) task requires a system to predict an
unordered set

\[
Y=\{(a_i,c_i,o_i,V_i,A_i)\}_{i=1}^{m},
\]

where (a_i) is an aspect span or `NULL`, (o_i) is an opinion span or
`NULL`, and (c_i) is an entity--attribute category defined by the dataset.
The two continuous labels (V_i,A_i\in[1,9]) represent valence and arousal,
respectively. A sentence can contain no opinion, one opinion, or multiple
opinions; the cardinality (m) is therefore unknown at prediction time.
[CITE: DimABSA/SemEval-2026 task overview]

We use the official Task-3 continuous F1 (cF1) as the primary metric. A
predicted quad is eligible for matching when its categorical triple
((a,c,o)) matches a gold triple. For each matched pair, the continuous true
positive credit is

\[
\operatorname{cTP}
=1-\frac{\sqrt{(V_p-V_g)^2+(A_p-A_g)^2}}{\sqrt{2(9-1)^2}}.
\]

Continuous precision and recall replace the ordinary true-positive count with
the sum of this credit, and cF1 is their harmonic mean. The metric consequently
penalizes both structural extraction errors and inaccurate valence--arousal
estimates. We compute all primary scores with the official evaluator. For
teacher diagnosis, we additionally report exact categorical F1; aspect,
opinion, and category F1; valence--arousal mean absolute error and root mean
squared error on matched quads; and coverage for `NULL`, rare categories,
empty outputs, and invalid outputs.

### 3.2 Fixed Student Model

Our experiments build on the LEADER-DimASQP student introduced in our earlier
ICONIP study [CITE: ICONIP-V20]. We treat that model and its dataset-specific
training recipes as inherited experimental infrastructure rather than as a new
contribution of the present work. Fixing the student is essential here: the
object of study is the supervision supplied by LLM-labeled silver data, not a
new extraction architecture.

In brief, the student encodes the sentence and predicts aspect and opinion BIO
spans. It forms candidate aspect--opinion pairs and uses task-specific heads to
predict pair validity, entity and attribute labels, and continuous valence and
arousal. A lightweight Rethink stage embeds the draft span, category, and
valence--arousal predictions and fuses them back into the token
representations. Candidate quads are aligned with gold quads through Hungarian
set matching. The inherited training recipes also use mild fast-gradient
adversarial perturbations and an exponential moving average. At inference,
five independently trained students vote on the categorical triple; retained
quads carry the corresponding continuous predictions.

We preserve the encoder and optimization recipe selected for each dataset in
the prior study. The five adopted distillation settings use DeBERTa-v3-base for
English laptop, XLM-R-large for Russian and Tatar restaurant, mDeBERTa-v3-base
for Ukrainian restaurant, and Chinese RoBERTa-wwm-ext-large for Chinese
restaurant. No architecture component, encoder, or inference rule is selected
using the experiments reported in this paper.

### 3.3 Offline Structured Label Distillation

Let (G_d) denote the labeled training data for dataset (d), and let (U_d)
be an unlabeled text pool from the same task domain. An LLM teacher (t)
maps each (u\in U_d) to a set of structured DimASQP labels. The prompt
requires non-`NULL` aspects and opinions to be verbatim substrings of the
input, restricts categories to the inventory derived from (G_d), constrains
valence and arousal to ([1,9]), and permits an empty set when the sentence
contains no in-domain opinion. Responses are parsed into the task JSON schema,
and accepted examples form the silver set (S_{d,t}). The student is then
trained on

\[
T_{d,t}=G_d\cup S_{d,t}.
\]

This is response-based structured distillation: the teacher supplies hard
structured labels rather than logits, hidden states, or token-level
probabilities. The LLM is used only while constructing (S_{d,t}). Validation,
official-test evaluation, and deployment use the student checkpoints alone and
make no teacher calls.

The adopted systems cover five datasets and were fixed before official-test
labels were inspected. Table 1 records the supervision actually used after the
inductive audit. The “gold/base” column includes the inherited non-silver
training recipe, which can contain training-only augmentation from the prior
system; it is therefore not always equal to the raw gold-train size.

**Table 1. Adopted offline-distillation settings after inductive cleaning.**

| Dataset | Fixed encoder | LLM teacher | Silver pool | Gold/base rows | Silver rows | Total rows | Vote |
|---|---|---|---|---:|---:|---:|---:|
| English laptop | DeBERTa-v3-base | DeepSeek V4 Pro | existing in-domain pool | 15,048 | 7,547 | 22,595 | 4 |
| Russian restaurant | XLM-R-large | DeepSeek V4 Pro | native generation, gen4k batch | 1,240 | 3,000 | 4,240 | 3 |
| Tatar restaurant | XLM-R-large | Qwen3.6-35B | native generation | 1,240 | 3,915 | 5,155 | 3 |
| Ukrainian restaurant | mDeBERTa-v3-base | DeepSeek V4 Pro | translated construction | 2,966 | 5,534 | 8,500 | 3 |
| Chinese restaurant | Chinese RoBERTa-wwm-ext-large | DeepSeek V4 Pro | real-text pool | 6,050 | 3,882 | 9,932 | 3 |

The choices in Table 1 are not claims of a universally optimal teacher or pool.
They are the frozen adopted systems whose behavior we audit. Separate matched
experiments vary teacher identity, pool source, or silver quantity while
holding the remaining observable factors fixed when the available data permit
such a comparison.

### 3.4 Experimental Factors and Competing Explanations

We distinguish four factors that are often conflated in silver-data studies.
First, **teacher fidelity** measures whether an LLM can directly reproduce
hidden gold labels. Second, **pool source** describes where the unlabeled text
comes from: native generation, real public text, or machine translation.
Third, **silver quantity** is the number of accepted silver rows. A comparison
between independently generated batches changes both quantity and batch
composition, whereas a cap applied to one fixed pool changes quantity while
preserving the underlying source. Fourth, **information complementarity**
describes whether silver supervision supplies valid categories, spans, or
error corrections that are underrepresented in the base training data.

A higher downstream score alone cannot identify which factor is responsible.
It may also reflect more optimizer updates, a favorable random seed, train--test
text overlap, or labels that the student could have produced itself. We
therefore combine direct teacher evaluation with inductive cleaning,
five-seed training, exposure-matched repeated-gold controls, same-text student
self-training controls, and paired inference on frozen predictions.

## 4 Experimental Design

### 4.1 Datasets and Evaluation Roles

We study all eight datasets from SemEval-2026 Task 3 [CITE: DimABSA/SemEval-2026
task overview]. They span English, Japanese, Russian, Tatar, Ukrainian, and
Chinese, with laptop, restaurant, and hotel domains. Table 2 reports the raw
gold split sizes. All eight datasets participate in direct teacher-fidelity
evaluation. Five datasets---English laptop and the Russian, Tatar, Ukrainian,
and Chinese restaurant sets---have a frozen adopted KD system and enter the
matched downstream analyses. English restaurant, Japanese hotel, and Chinese
laptop retain their inherited non-KD main systems and are not silently counted
as KD successes.

**Table 2. Gold data and evaluation roles.**

| Dataset | Language | Domain | Train | Dev | Test | Direct fidelity | Adopted KD |
|---|---|---|---:|---:|---:|---|---|
| eng_laptop | English | laptop | 4,076 | 200 | 1,000 | yes | yes |
| eng_restaurant | English | restaurant | 2,284 | 200 | 1,000 | yes | no |
| jpn_hotel | Japanese | hotel | 1,600 | 200 | 800 | yes | no |
| rus_restaurant | Russian | restaurant | 1,240 | 48 | 630 | yes | yes |
| tat_restaurant | Tatar | restaurant | 1,240 | 48 | 630 | yes | yes |
| ukr_restaurant | Ukrainian | restaurant | 1,240 | 48 | 630 | yes | yes |
| zho_laptop | Chinese | laptop | 3,490 | 300 | 1,000 | yes | no |
| zho_restaurant | Chinese | restaurant | 6,050 | 300 | 1,000 | yes | yes |

The development sets serve two distinct roles. Their labels are hidden from the
LLMs when directly measuring teacher fidelity, while the student experiments
use development cF1 for the inherited validation-based checkpoint rule. The
released official test labels are reserved for a single post-competition local
evaluation after systems, thresholds, prediction files, and hashes have been
locked.

### 4.2 Contamination Audit and Inductive Cleaning

We audit the gold, pool, silver, combined distillation, development, and test
files registered for all eight datasets. Each training-side row is checked
against development and test by exact text, normalized text, and source ID.
The normalization key applies Unicode NFKC normalization, Unicode case folding,
and Unicode-whitespace collapse. The audit stores the source path, source
SHA-256, line number, matched split, match reason, and normalized text for each
detected row.

Three adopted KD inputs contain development- or test-overlapping text and are
filtered without reordering the retained rows. English laptop changes from
23,590 to 22,595 rows, Ukrainian restaurant from 9,105 to 8,500, and Chinese
restaurant from 10,911 to 9,932. The corresponding English and Ukrainian
baseline inputs each contain one normalized development overlap and are also
cleaned, from 15,049 to 15,048 rows and from 2,967 to 2,966 rows, respectively.
The Russian gen4k and Tatar Qwen adopted inputs have zero normalized overlap and
are retained unchanged. We then verify that every final adopted KD input and
every baseline input used for a clean paired comparison has zero normalized
text overlap with development and test. Original checkpoints are preserved;
clean runs use distinct prefixes.

This cleaning step defines the inductive protocol and is not conditioned on
whether removal raises or lowers cF1. We report comparisons with the earlier
transductive checkpoints only to quantify sensitivity, never as evidence that
overlap would have been acceptable.

### 4.3 Direct Teacher-Fidelity Evaluation

We evaluate DeepSeek V4 Pro and Qwen3.6-35B directly on every gold development
set. We remove the complete `Quadruplet` field before sending an example to a
teacher. The category inventory and six few-shot demonstrations are drawn only
from that dataset's training split. Both teachers receive prompt version
`dim-asqp-fidelity-v1`, temperature 0.3, and disabled reasoning output. A
10-example smoke run checks model identity, schema compatibility, and output
coverage before a full run is accepted.

Across the eight datasets, the development sets contain 1,344 examples. Each
teacher produces exactly one validated record for every example, giving 1,344
DeepSeek and 1,344 Qwen annotations with zero final error rows. We retain the
label-free inputs, raw responses, parsed outputs, requested and returned model
names, prompt version, temperature, timestamps, and SHA-256 hashes. The main
fidelity metric is official cF1. The auxiliary metrics defined in Section 3.1
diagnose whether a difference comes from structure, category, continuous
sentiment, empty output, or an infrequent training category. A category is
treated as rare for this diagnostic when it appears no more than five times in
the corresponding training split.

Direct fidelity does not by itself establish which teacher produces the best
training signal. To test whether a downstream teacher ranking can be reproduced
under matched conditions, we additionally train Tatar students on the same
1,240 gold rows and the same 3,915 silver texts in the same order, changing only
whether DeepSeek or Qwen supplies the silver labels. Both arms use all five
seeds and the same frozen inference threshold.

### 4.4 Student Training and Frozen Inference

Every five-seed experiment uses random seeds 42, 123, 456, 789, and 1024. We
inherit the dataset-specific model configuration, optimizer, training data
order, augmentation recipe, maximum of 30 epochs, validation-based checkpoint
selection, and early-stopping patience of eight from the corresponding fixed
student. The adopted configurations use AdamW, an encoder learning rate of
(1.5\times10^{-5}), a warmup ratio of 0.1, weight decay 0.01, gradient-norm
clipping at 1.0, FGM with \(\epsilon=0.1\), and EMA with decay 0.999. Batch size
is eight for the base-size English and Ukrainian encoders and four for the large
Russian, Tatar, and Chinese encoders. We retain and evaluate
`checkpoint_best.pt`; no last-epoch checkpoint is substituted.

For the clean-inductive reruns, the only intended experimental change is the
removal of overlapping rows. The five-seed ensemble thresholds are fixed at
vote 4 and top-k 0 for English laptop, and vote 3 and top-k 0 for Russian,
Tatar, Ukrainian, and Chinese restaurant. The matched baseline can retain its
own previously fixed threshold; thresholds are not re-swept to favor a
contrast. We report every single-seed cF1, the mean and sample standard
deviation across seeds, and the frozen ensemble cF1.

### 4.5 Controls for Exposure and Teacher Source

The **repeated-gold control** tests whether a gain can be explained by seeing
more training rows or performing more updates. For each adopted KD dataset, we
repeat the inductive-safe base training rows as evenly as possible until the
control has exactly the same number of rows as the corresponding gold-plus-LLM
training input. Complete copies preserve source order; a remainder is selected
after a deterministic shuffle with seed 20260713. Duplicate copies receive
unique IDs. Each constructed file must have zero normalized overlap with
development and test, and it is trained with the same student recipe and five
seeds as the adopted KD arm.

The **same-text self-training control** tests the value of LLM labels rather
than the value of the unlabeled text itself. We extract exactly the silver-text
tail used by an adopted LLM arm, preserve its IDs and order, and remove its
labels. A five-seed ensemble trained on the repeated-gold control then labels
those texts. We combine the inductive-safe base rows with the resulting student
pseudo-labels, retaining empty predictions as valid no-quad pseudo-labels. The
final self-training input therefore has the same base rows, the same silver
texts, the same silver row count, and the same total row count as the LLM arm;
the supervision source on the silver texts is the intended difference. It is
trained with the same five seeds and evaluated using the adopted system's
frozen threshold.

All exposure and self-training controls were registered after the official-test
systems had been locked. They use development data only, and their manifests
explicitly forbid official-test evaluation. Consequently, they refine the
mechanistic interpretation of the development results but cannot retrospectively
alter any main-system or test choice.

### 4.6 Teacher, Pool, and Quantity Ablations

We organize auxiliary ablations by the strength of their control. The direct
8-by-2 fidelity experiment compares teachers on identical label-hidden inputs.
The Tatar downstream experiment compares teachers on identical gold and silver
texts. For Russian pool source, real, translated, and generated text is labeled
by the same teacher and capped at 3,000 rows. The Chinese real-versus-generated
comparison is retained as observational evidence because other construction
details are not fully matched.

Quantity is evaluated separately from pool identity. In Russian, we take the
same gen8k silver pool and compare a cap of 3,000 with all 7,988 rows for paired
seeds 42 and 123. A preregistered gate required the two-seed mean to improve by
at least 1.5 percentage points before training the remaining seeds or adding a
5,000-row arm. The gate failed, so the remaining quantity curve was not run.
This stopping rule can test whether the earlier gen4k advantage is explained by
having fewer rows; it cannot identify a universal optimal silver dose.

### 4.7 Paired Inference, Macro Effects, and Multiplicity

For a comparison between systems (A) and (B), we resample evaluation
examples with replacement, recompute both cF1 scores from the corresponding
row-level sufficient statistics, and record \(\Delta=\operatorname{cF1}_A-
\operatorname{cF1}_B\). Every paired bootstrap uses 10,000 resamples and random
seed 20260713. We report the observed delta, percentile 95% confidence interval,
and two-sided bootstrap p-value. An interval whose lower bound is above zero is
classified as supported improvement; an interval that crosses zero is
inconclusive; and an interval whose upper bound is below zero supports harm.
An inconclusive interval is not interpreted as equivalence.

To summarize the five adopted datasets without allowing the larger English or
Chinese evaluation sets to dominate, we also use a fixed-task macro bootstrap.
Examples are resampled independently within each dataset, a paired delta is
recomputed, and the five dataset deltas receive equal weight. This estimand is
the macro-average effect over the fixed adopted-domain set, not an estimate for
unobserved languages or tasks.

We treat teacher fidelity, adopted KD versus baseline, exposure control,
same-text self-training, self-training versus repeated gold, and frozen
official-test comparisons as separate families. Within each family, we report
both Holm family-wise-error correction and Benjamini--Hochberg false-discovery
correction. The official-test correction is a post-hoc robustness analysis and
does not change the previously frozen systems. Dataset-level confidence
intervals remain visible so that aggregate support cannot be misread as support
for every domain.

### 4.8 Official-Test Lock and Reproducibility

The official-test evaluation is a one-shot, post-competition local analysis of
the released Task-3 labels. Before obtaining those labels, we fixed each main
system, comparison baseline, five-seed checkpoint set, vote and top-k threshold,
test input, prediction file, row count, and SHA-256 hash in a protocol manifest.
All selections were based on development evidence. Test labels were not used to
choose a checkpoint, teacher, pool, augmentation recipe, threshold, or stopping
rule. We therefore describe these scores as a frozen local evaluation, not as a
new shared-task submission.

The reproducibility package retains the contamination matches and deletion
records, clean-file hashes, teacher inputs and raw responses, validated outputs,
training and inference manifests, seed-level predictions, best-checkpoint
provenance, bootstrap seed, and exact-repeat audit. Automated checks verify that
teacher inputs contain no `Quadruplet`, each input ID has one output or an
explicit error record, final adopted training inputs have zero normalized
development/test overlap, inference uses `checkpoint_best.pt`, frozen ensemble
thresholds are respected, and the six core evidence tables match their locked
hashes.

Post-lock control training was run on a single 24-GB NVIDIA RTX 4090. LLMs are
offline annotators only, so student inference makes zero LLM calls. Historical
provider token counts and monetary billing were not retained; we report them as
unavailable rather than reconstructing a potentially misleading cost estimate.

<!--
Internal drafting provenance (remove only when the corresponding citations and
artifact references have been transferred to the submission source):

- Task/metric and inherited student: docs/paper_iconip_current.tex
- Dataset roles, seeds, thresholds, and clean reruns: scripts/kd/tallip.yaml
- Adopted KD inventory and frozen evidence: docs/audits/tallip_kd/tallip_core_tables.md
- Contamination method: scripts/kd/audit_contamination.py
- Cleaning counts and file hashes: docs/audits/tallip_kd/cleaning_manifest.json
- Teacher prompts and validation: scripts/kd/teacher_fidelity.py and
  data/audits/tallip_kd/teacher_fidelity/metadata/
- Exposure/self-training construction: scripts/kd/postlock_experiments.py and
  scripts/kd/postlock_v1.yaml / postlock_v2.yaml
- Statistical families: docs/audits/tallip_kd/statistical_robustness.json
- Official-test lock: scripts/kd/official_test.yaml and
  docs/audits/tallip_kd/official_test/protocol_lock.json
-->

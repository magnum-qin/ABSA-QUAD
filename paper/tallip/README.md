# TALLIP manuscript build

This directory is independent of the accepted ICONIP manuscript. The editable
narrative source remains `../../docs/TALLIP_MANUSCRIPT_DRAFT.md`; the LaTeX body
is a mechanically converted submission snapshot and should be regenerated when
the Markdown changes materially.

## Entry points

- `tallip_submission.tex`: one-column `manuscript,review` file for initial
  review.
- `tallip_submission_anonymous.tex`: double-anonymous fallback; use it only if
  the live Manuscript Central workflow explicitly requests anonymization.
- `tallip_acmsmall.tex`: TALLIP journal-layout audit using `acmsmall`.
- `tallip_common.tex`: title, authors, abstract, CCS concepts, and shared
  front/back matter.
- `tallip_body.tex`: converted article body.
- `tallip_refs.bib`: references, with ACL Anthology records used when
  available.

Build from this directory with either:

```bash
tectonic tallip_submission.tex
tectonic tallip_acmsmall.tex
```

A conventional TeX installation can instead use `latexmk -pdf`.

Use a TeX installation with the current `acmart` release. The final local audit
used CTAN `acmart` v2.19 (2026-06-27); Tectonic's default bundle may lag behind
CTAN, so its class search path must be overridden when necessary.

To generate isolated identified/anonymous review PDFs, minimal source archives,
and a SHA256 manifest, run from the repository root:

```bash
python scripts/paper/package_tallip_submission.py
```

The generated `artifacts/tallip_submission/` directory is ignored by Git. The
anonymous source replaces only the marked identifying-metadata block; the prior
ICONIP citation remains intact and is phrased in the third person.

## Submission-format audit

The following constraints were checked on 2026-08-15 against the ACM `acmart`
manual and the available TALLIP author-guideline mirror:

- TALLIP uses the `TALLIP` journal code and the `acmsmall` production format.
- ACM recommends `manuscript` for initial journal review.
- The abstract must contain 150--200 words; the current abstract has 198.
- A TALLIP long paper must occupy at least 15 journal-layout pages.
- Empirical papers must report statistical information.
- Citations use the author--year style.
- An extension of a conference/workshop publication must contain at least 25\%
  new material and cite the prior work.
- Submission is a PDF through the TALLIP Manuscript Central site.

The latest-class local build completed without LaTeX/BibTeX errors, undefined
citations or references, missing glyphs, duplicate labels, or overfull boxes.
The review PDF is 20 letter-size pages and the `acmsmall` audit PDF is 21 journal
pages, so the current long-paper draft clears the 15-page minimum. These counts
include references and remain provisional until the live submission template
and production metadata are applied.

Before submission, the authors must still:

1. verify whether the live Manuscript Central workflow requests an anonymized
   review file; this draft follows the guideline that title-page author names
   and affiliations are required;
2. replace the provisional accepted-ICONIP bibliography entry with its final
   DOI/venue metadata when available;
3. generate and paste the exact CCS XML from the ACM CCS tool;
4. add funding and acknowledgments supplied by the authors;
5. resolve the author-only confirmations in
   [`../../docs/TALLIP_COVER_LETTER.md`](../../docs/TALLIP_COVER_LETTER.md),
   including corresponding author, co-author approval, exclusive submission,
   and competing interests;
6. rerun all repository evidence checks and inspect every table and figure in
   both generated PDFs.

The submission-facing prior-publication comparison is in
[`../../docs/TALLIP_PRIOR_PUBLICATION_STATEMENT.md`](../../docs/TALLIP_PRIOR_PUBLICATION_STATEMENT.md),
and the full operational checklist is in
[`../../docs/TALLIP_SUBMISSION_CHECKLIST.md`](../../docs/TALLIP_SUBMISSION_CHECKLIST.md).

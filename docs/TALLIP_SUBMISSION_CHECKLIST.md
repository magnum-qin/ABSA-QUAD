# TALLIP submission checklist

Last audited: 2026-08-15.

This is the operational checklist for the first TALLIP submission. It separates
repository-verified items from declarations that only the authors can make.

## A. Already verified in the repository

- [x] Article category is **Long Paper**.
- [x] Review entry point uses ACM `manuscript,review`; the journal-layout audit
  uses `acmsmall` with journal code `TALLIP`.
- [x] Current CTAN `acmart` v2.19 builds both entry points without LaTeX/BibTeX
  errors, undefined references, missing glyphs, duplicate labels, or overfull
  boxes.
- [x] Review PDF is 20 pages and the journal-layout audit is 21 pages; the
  available TALLIP guidance requires at least 15 journal pages for a Long Paper.
- [x] Abstract is 198 words and within the stated 150–200-word range.
- [x] Author-year citations, numbered sections, CCS concepts, keywords, tables,
  statistical intervals, limitations, ethics, and references are present.
- [x] Frozen evidence tables reproduce from the registry and audit artifacts.
- [x] Adopted KD and paired-baseline training inputs have zero normalized-text
  overlap with development/test data after cleaning.
- [x] Teacher inputs omit the gold `Quadruplet` field and every input ID has an
  output or explicit error record.
- [x] Five-seed training, fixed ensemble thresholds, paired bootstrap seed, and
  `checkpoint_best.pt` inference are recorded.
- [x] Official-test predictions and hashes were frozen before gold retrieval;
  test labels were not used for model, threshold, teacher, pool, or augmentation
  selection.
- [x] The manuscript cites ICONIP V20 and contains an explicit relation-to-prior-
  work subsection.
- [x] No ICONIP table or figure is reproduced; the student is identified as an
  inherited component.
- [x] Cover-letter and detailed prior-publication drafts are prepared.
- [x] No manuscript LaTeX outside `paper/tallip/` has been modified by the
  TALLIP workflow.

## B. Author input required before upload

- [ ] **Corresponding author:** confirm whether Hongzhen Qin is both submitting
  and corresponding author.
- [ ] **Author approval:** obtain explicit approval of the final manuscript and
  submission from Yoshimi Suzuki, Fumiyo Fukumoto, and Renjie Zhou.
- [ ] **Exclusive submission:** confirm that no substantially similar manuscript
  is under review or submitted elsewhere.
- [ ] **Competing interests:** provide the exact declaration, including “none”
  if applicable.
- [ ] **Funding:** provide funder names and grant numbers exactly as they must
  appear in the ACM funding metadata and manuscript.
- [ ] **Acknowledgments:** provide any people, computing grants, API credits, or
  institutional support that should be acknowledged.
- [ ] **ORCID:** supply or confirm the ORCID for each author; do not invent a
  missing identifier.
- [ ] **Author affiliations:** confirm institution names, postal information,
  and the dual affiliation of Hongzhen Qin.
- [ ] **Author contributions:** agree on a CRediT statement if the live workflow
  requests one.
- [ ] **Artifact availability:** choose the public code/data URL and release
  timing. Ensure an anonymous URL if the live workflow uses anonymous review.
- [ ] **ICONIP metadata:** replace “accepted; forthcoming” with the final venue,
  pages, DOI, and URL once available.
- [ ] **CCS XML:** generate the exact concepts in ACM's CCS tool and paste its
  XML/LaTeX output instead of relying only on the provisional hand-written list.

## C. Live Manuscript Central checks

These fields can change and must be checked in the live portal immediately
before submission:

- [ ] Confirm whether review is single-anonymous, double-anonymous, or asks for
  both identified and anonymized files. Apply the portal's live instruction to
  the PDF, source archive, artifact links, acknowledgments, and metadata.
- [ ] Select **Long Paper** rather than Short Paper or Survey Paper.
- [ ] Enter the title, subtitle, abstract, keywords, authors, affiliations,
  emails, ORCIDs, funding, and conflicts exactly as in the final manuscript.
- [ ] Paste the ICONIP disclosure into the cover-letter field.
- [ ] If the portal has “related manuscript,” “prior publication,” or
  supplemental-document fields, upload the ICONIP manuscript and
  [`TALLIP_PRIOR_PUBLICATION_STATEMENT.md`](TALLIP_PRIOR_PUBLICATION_STATEMENT.md)
  as requested.
- [ ] Add suggested/opposed reviewers only if requested, checking conflicts and
  avoiding recent collaborators or same-institution researchers.
- [ ] Inspect the portal-generated PDF page by page before approving it.
- [ ] Save the manuscript ID and an exported/screenshot copy of all submitted
  declarations.

## D. Final local package and validation

- [ ] Remove all `AUTHOR CONFIRMATION` placeholders from
  [`TALLIP_COVER_LETTER.md`](TALLIP_COVER_LETTER.md).
- [ ] Rebuild `paper/tallip/tallip_submission.pdf` with the final metadata.
- [ ] Rebuild `paper/tallip/tallip_acmsmall.pdf` as a layout audit.
- [ ] Run:

  ```bash
  python scripts/kd/build_tallip_report.py --verify
  python scripts/kd/tallip_postlock_analysis.py --verify
  python scripts/kd/audit_contamination.py verify
  ```

- [ ] Confirm the final Git diff does not modify `docs/paper_iconip_current.tex`
  or any other ICONIP source.
- [ ] Package only required TeX/Bib/figure/class-support files; exclude secrets,
  `.env.local`, checkpoints, logs, raw provider credentials, and official gold
  labels unless the portal explicitly requests them.
- [ ] Open the packaged source in a clean directory and compile it once.
- [ ] Record SHA256 values for the uploaded manuscript PDF, cover letter,
  related-work statement, and source archive.

## E. Policy references checked

- ACM policy on prior publication and simultaneous submissions:
  <https://www.acm.org/publications/policies/simultaneous-submissions>
- ACM policy on pre-publication evaluation:
  <https://www.acm.org/publications/policies/pre-publication-evaluation>
- ACM primary article template:
  <https://www.acm.org/publications/proceedings-template>
- TALLIP Manuscript Central:
  <https://mc.manuscriptcentral.com/tallip>

The available TALLIP guidance says that a conference/workshop extension must
contain at least 25% new material and must cite the prior work. The manuscript
exceeds that boundary through a different research question and new KD
protocols, audits, controls, statistics, results, and conclusions. The
submitting author must nevertheless disclose the ICONIP work explicitly to the
editor; the repository audit cannot replace that declaration.

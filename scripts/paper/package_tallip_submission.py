#!/usr/bin/env python3
"""Build audited identified/anonymous TALLIP review packages.

The generated directory is intentionally ignored by Git. The script builds in
isolated staging directories, creates minimal source archives, audits the
anonymous PDF/source for identifying metadata, and records SHA256 hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper" / "tallip"
FIGURE = ROOT / "docs" / "figures" / "tallip_kd_effect_forest.pdf"
DEFAULT_OUTPUT = ROOT / "artifacts" / "tallip_submission"

IDENTITY_BEGIN = "% BEGIN IDENTIFYING METADATA"
IDENTITY_END = "% END IDENTIFYING METADATA"
ANONYMOUS_METADATA = r"""% BEGIN IDENTIFYING METADATA
\author{Anonymous Author(s)}
\affiliation{%
  \institution{Anonymous Institution}}
\renewcommand{\shortauthors}{Anonymous Author(s)}
% END IDENTIFYING METADATA"""

SOURCE_FILES = {
    "tallip_common.tex": PAPER / "tallip_common.tex",
    "tallip_body.tex": PAPER / "tallip_body.tex",
    "tallip_refs.bib": PAPER / "tallip_refs.bib",
    "tallip_kd_effect_forest.pdf": FIGURE,
}

FORBIDDEN_ANONYMOUS_METADATA = (
    "241050420@hdu.edu.cn",
    "g24tka12@yamanashi.ac.jp",
    "ysuzuki@yamanashi.ac.jp",
    "fukumoto@yamanashi.ac.jp",
    "rjzhou@hdu.edu.cn",
    "Hangzhou Dianzi University",
    "University of Yamanashi",
)
FORBIDDEN_FIRST_PAGE_NAMES = (
    "Hongzhen Qin",
    "Yoshimi Suzuki",
    "Fumiyo Fukumoto",
    "Renjie Zhou",
)


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    if result.returncode:
        joined = " ".join(command)
        raise RuntimeError(
            f"command failed ({result.returncode}): {joined}\n{result.stdout}"
        )
    return result.stdout


def locate_tectonic() -> Path:
    candidates = []
    if os.environ.get("TECTONIC_BIN"):
        candidates.append(Path(os.environ["TECTONIC_BIN"]))
    found = shutil.which("tectonic")
    if found:
        candidates.append(Path(found))
    candidates.append(Path("/tmp/tallip_tools/tectonic"))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "tectonic not found; set TECTONIC_BIN or install Tectonic 0.17+"
    )


def locate_acmart() -> tuple[Path, Path]:
    roots = []
    if os.environ.get("ACMART_ROOT"):
        roots.append(Path(os.environ["ACMART_ROOT"]))
    roots.append(Path("/tmp/acmart_tlnet_219"))
    for root in roots:
        cls = root / "tex" / "latex" / "acmart" / "acmart.cls"
        bst = root / "bibtex" / "bst" / "acmart" / "ACM-Reference-Format.bst"
        if cls.is_file() and bst.is_file():
            version = cls.read_text(errors="replace")
            if "2026/06/27 v2.19" not in version:
                raise RuntimeError(
                    f"expected acmart v2.19, found a different class at {cls}"
                )
            return cls, bst
    raise FileNotFoundError(
        "acmart v2.19 not found; set ACMART_ROOT to an extracted CTAN tree"
    )


def source_commit() -> tuple[str, bool]:
    commit = run(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
    dirty = bool(run(["git", "status", "--porcelain"], cwd=ROOT).strip())
    return commit, dirty


def anonymize_common(text: str) -> str:
    pattern = re.compile(
        re.escape(IDENTITY_BEGIN) + r".*?" + re.escape(IDENTITY_END), re.DOTALL
    )
    result, replacements = pattern.subn(lambda _: ANONYMOUS_METADATA, text)
    if replacements != 1:
        raise RuntimeError(
            f"expected one identifying-metadata block, found {replacements}"
        )
    return result


def prepare_source(variant: str, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    wrapper = PAPER / (
        "tallip_submission_anonymous.tex"
        if variant == "anonymous"
        else "tallip_submission.tex"
    )
    shutil.copy2(wrapper, destination / "main.tex")

    common = SOURCE_FILES["tallip_common.tex"].read_text()
    old_graphics = r"\graphicspath{{../../docs/figures/}}"
    if old_graphics not in common:
        raise RuntimeError("expected repository graphicspath was not found")
    common = common.replace(old_graphics, r"\graphicspath{{./}}", 1)
    if variant == "anonymous":
        common = anonymize_common(common)
    (destination / "tallip_common.tex").write_text(common)

    for name in (
        "tallip_body.tex",
        "tallip_refs.bib",
        "tallip_kd_effect_forest.pdf",
    ):
        shutil.copy2(SOURCE_FILES[name], destination / name)

    readme = (
        "TALLIP initial-review source package\n"
        f"Variant: {variant}\n"
        "Entry point: main.tex\n"
        "Template: ACM acmart v2.19 (2026-06-27), manuscript/review layout\n"
        "The official acmart class and bibliography style are not bundled.\n"
    )
    (destination / "README.txt").write_text(readme)
    return ["main.tex", *SOURCE_FILES.keys(), "README.txt"]


def compile_source(
    source_dir: Path, tectonic: Path, cls: Path, bst: Path
) -> tuple[Path, int]:
    shutil.copy2(cls, source_dir / "acmart.cls")
    shutil.copy2(bst, source_dir / "ACM-Reference-Format.bst")
    build_env = os.environ.copy()
    build_env["SOURCE_DATE_EPOCH"] = "1786752000"
    build_env["FORCE_SOURCE_DATE"] = "1"
    run(
        [
            str(tectonic),
            "--keep-logs",
            "--keep-intermediates",
            "--color",
            "never",
            "main.tex",
        ],
        cwd=source_dir,
        env=build_env,
    )
    pdf = source_dir / "main.pdf"
    log = source_dir / "main.log"
    if not pdf.is_file() or not log.is_file():
        raise RuntimeError("Tectonic completed without producing main.pdf/main.log")

    log_text = log.read_text(errors="replace")
    fatal_patterns = (
        "LaTeX Warning: There were undefined references",
        "LaTeX Warning: There were undefined citations",
        "Overfull \\hbox",
        "Missing character:",
        "destination with the same identifier",
    )
    problems = [pattern for pattern in fatal_patterns if pattern in log_text]
    if problems:
        raise RuntimeError(f"PDF log audit failed: {', '.join(problems)}")
    if "2026/06/27 v2.19" not in log_text:
        raise RuntimeError("PDF was not built with acmart v2.19")

    info = run(["pdfinfo", str(pdf)])
    match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    if not match:
        raise RuntimeError("could not read PDF page count")
    return pdf, int(match.group(1))


def audit_anonymous(source_dir: Path, pdf: Path) -> dict[str, str]:
    info = run(["pdfinfo", str(pdf)])
    first_page_path = source_dir / "first_page.txt"
    run(
        [
            "pdftotext",
            "-f",
            "1",
            "-l",
            "1",
            str(pdf),
            str(first_page_path),
        ]
    )
    first_page = first_page_path.read_text(errors="replace")
    source_text = "\n".join(
        (source_dir / name).read_text(errors="replace")
        for name in ("main.tex", "tallip_common.tex", "tallip_body.tex")
    )

    leaks = []
    for value in FORBIDDEN_ANONYMOUS_METADATA:
        if value.lower() in info.lower() or value.lower() in source_text.lower():
            leaks.append(value)
    for value in FORBIDDEN_FIRST_PAGE_NAMES:
        if value.lower() in first_page.lower():
            leaks.append(f"first-page byline: {value}")
    if leaks:
        raise RuntimeError(
            "anonymous audit found identifying metadata: " + ", ".join(leaks)
        )

    author_line = next(
        (
            line.split(":", 1)[1].strip()
            for line in info.splitlines()
            if line.startswith("Author:")
        ),
        "",
    )
    if author_line and "anonymous" not in author_line.lower():
        raise RuntimeError(
            f"anonymous PDF Author metadata is not anonymous: {author_line}"
        )
    return {
        "pdf_author": author_line or "(empty)",
        "first_page": "no author byline leak",
    }


def write_reproducible_zip(
    archive: Path, source_dir: Path, names: list[str]
) -> None:
    timestamp = (2026, 8, 15, 0, 0, 0)
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zf:
        for name in sorted(names):
            data = (source_dir / name).read_bytes()
            entry = zipfile.ZipInfo(name, date_time=timestamp)
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            zf.writestr(entry, data)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    build_root = output / "build"
    build_root.mkdir()

    tectonic = locate_tectonic()
    cls, bst = locate_acmart()
    commit, dirty = source_commit()
    records: dict[str, object] = {
        "source_commit": commit,
        "source_dirty": dirty,
        "acmart": "v2.19 (2026-06-27)",
        "variants": {},
    }
    deliverables: list[Path] = []

    for variant in ("identified", "anonymous"):
        source_dir = build_root / variant
        names = prepare_source(variant, source_dir)
        pdf, pages = compile_source(source_dir, tectonic, cls, bst)

        output_pdf = output / f"tallip_review_{variant}.pdf"
        shutil.copy2(pdf, output_pdf)
        archive = output / f"tallip_source_{variant}.zip"
        write_reproducible_zip(archive, source_dir, names)
        deliverables.extend((output_pdf, archive))

        variant_record: dict[str, object] = {"pages": pages}
        if variant == "anonymous":
            variant_record["anonymous_audit"] = audit_anonymous(source_dir, pdf)
        records["variants"][variant] = variant_record

    manifest = output / "MANIFEST.sha256"
    manifest.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n" for path in sorted(deliverables)
        )
    )
    info = output / "PACKAGE_INFO.json"
    info.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")

    # Staging contains copied template files and intermediates; only audited
    # deliverables and metadata remain in the handoff directory.
    shutil.rmtree(build_root)

    print(f"packaged {len(deliverables)} deliverables in {output}")
    print(f"source commit: {commit[:12]} ({'dirty' if dirty else 'clean'})")
    for variant, record in records["variants"].items():
        print(f"{variant}: {record['pages']} pages")
    print("anonymous metadata audit: PASS")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

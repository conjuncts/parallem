from __future__ import annotations

from pathlib import Path


def _assert_nonempty_file(path: Path, min_bytes: int = 16) -> None:
    assert path.exists(), f"Missing file: {path}"
    assert path.is_file(), f"Expected a file: {path}"
    assert path.stat().st_size >= min_bytes, f"File is too small or empty: {path}"


def _read_prefix(path: Path, n: int = 8) -> bytes:
    with path.open("rb") as f:
        return f.read(n)


def _assert_pdf(path: Path) -> None:
    _assert_nonempty_file(path)
    prefix = _read_prefix(path, 5)
    assert prefix.startswith(b"%PDF"), f"Not a valid PDF header: {path}"


def _assert_xlsx(path: Path) -> None:
    _assert_nonempty_file(path)
    prefix = _read_prefix(path, 4)
    assert prefix == b"PK\x03\x04", f"Not an XLSX/ZIP signature: {path}"


def _assert_csv_like(path: Path) -> None:
    _assert_nonempty_file(path)
    first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    assert "," in first_line or "\t" in first_line, f"CSV/TSV header not detected: {path}"
    num_fields = len(first_line.replace("\t", ",").split(","))
    assert num_fields >= 2, f"Expected at least 2 columns in CSV header: {path}"


def _assert_nonempty_dir(path: Path) -> None:
    assert path.exists(), f"Missing directory: {path}"
    assert path.is_dir(), f"Expected directory: {path}"
    children = list(path.iterdir())
    assert children, f"Directory is empty: {path}"


def test_core_dataset_files_exist(repo_root: Path, core_dataset_expectations: dict[str, list[str]]) -> None:
    missing = []
    for _, rel_paths in core_dataset_expectations.items():
        for rel in rel_paths:
            path = repo_root / rel
            if not path.exists():
                missing.append(rel)
    assert not missing, f"Missing required core dataset files: {missing}"


def test_nucb_download_looks_valid(repo_root: Path) -> None:
    csv_path = repo_root / "usecases/enzyme_design_arena/data/raw/nucb/landscape.csv"
    readme_path = repo_root / "usecases/enzyme_design_arena/data/raw/nucb/README_source.txt"
    _assert_csv_like(csv_path)
    _assert_nonempty_file(readme_path)
    text = readme_path.read_text(encoding="utf-8", errors="ignore").lower()
    assert "nuclease_design" in text or "nucb" in text, "NucB provenance note looks incomplete"


def test_modify_rma_download_looks_valid(repo_root: Path) -> None:
    pdf_path = repo_root / "usecases/enzyme_design_arena/data/raw/modify_rma/supplement.pdf"
    xlsx_path = repo_root / "usecases/enzyme_design_arena/data/raw/modify_rma/supplementary_data_1.xlsx"
    readme_path = repo_root / "usecases/enzyme_design_arena/data/raw/modify_rma/README_source.txt"
    _assert_pdf(pdf_path)
    _assert_xlsx(xlsx_path)
    _assert_nonempty_file(readme_path)
    text = readme_path.read_text(encoding="utf-8", errors="ignore").lower()
    assert "10.1038/s41467-024-50698-y" in text or "rma" in text, "MODIFY provenance note looks incomplete"


def test_mcba_download_looks_valid(repo_root: Path) -> None:
    pdf_path = repo_root / "usecases/enzyme_design_arena/data/raw/mcba/supplement.pdf"
    xlsx_path = repo_root / "usecases/enzyme_design_arena/data/raw/mcba/source_data.xlsx"
    repo_path = repo_root / "usecases/enzyme_design_arena/data/raw/mcba/accelerated_enzyme_engineering_repo"
    readme_path = repo_root / "usecases/enzyme_design_arena/data/raw/mcba/README_source.txt"
    _assert_pdf(pdf_path)
    _assert_xlsx(xlsx_path)
    _assert_nonempty_dir(repo_path)
    _assert_nonempty_file(readme_path)
    text = readme_path.read_text(encoding="utf-8", errors="ignore").lower()
    assert "10.1038/s41467-024-55399-0" in text or "mcba" in text, "McbA provenance note looks incomplete"


def test_optional_proteingym_download_if_present(repo_root: Path) -> None:
    csv_path = repo_root / "usecases/enzyme_design_arena/data/raw/proteingym_optional/DMS_substitutions.csv"
    readme_path = repo_root / "usecases/enzyme_design_arena/data/raw/proteingym_optional/README_source.txt"

    if not csv_path.exists() and not readme_path.exists():
        return

    _assert_csv_like(csv_path)
    _assert_nonempty_file(readme_path)
    text = readme_path.read_text(encoding="utf-8", errors="ignore").lower()
    assert "proteingym" in text, "ProteinGym provenance note looks incomplete"

import json
from pathlib import Path
import zipfile


def persist_to_zip(
    fpath: Path,
    stuff: str | list[dict],
    *,
    inner_fname: str | None = None,
    compression: int | None = None,
) -> None:
    """
    Persist data into a zip file as a single file or JSONL.

    :param fpath: Path to the zip file to create.
    :param stuff: If str, writes the string as a single file.
    :param inner_fname: Name of the file inside the zip.
        If not given, then `fpath` minus ".zip".
    :param compression: Optional compression method for write modes.
    :return: None.
    """
    if inner_fname is None:
        inner_fname = fpath.stem

    with zipfile.ZipFile(fpath, "w", compression=compression) as zf:
        if isinstance(stuff, str):
            zf.writestr(inner_fname, stuff)
            return

        inner_jsonl = inner_fname + ".jsonl"
        with zf.open(inner_jsonl, "w") as f:
            for item in stuff:
                line = json.dumps(item) + "\n"
                f.write(line.encode("utf-8"))


def read_zip_text(fpath: Path, inner_fname: str) -> str:
    """
    Read a text file from within a zip.

    :param fpath: Path to the zip file.
    :param inner_fname: Name of the file inside the zip.
    :return: Decoded text content.
    """
    with zipfile.ZipFile(fpath, "r") as zf:
        return zf.read(inner_fname).decode("utf-8")


def read_jsonl_items_from_zip(
    fpath: Path,
    *,
    inner_fname: str | None = None,
) -> list[dict]:
    """
    Read JSONL items from a zip file.

    :param fpath: Path to the zip file.
    :param inner_fname: Name of the JSONL file inside the zip.
        If not provided, prefers any .jsonl member then falls back to the first entry.
    :return: List of parsed JSON objects.
    """
    with zipfile.ZipFile(fpath, "r") as zf:
        if inner_fname is None:
            names = zf.namelist()
            if not names:
                raise ValueError("Zip file is empty")
            inner_fname = next((n for n in names if n.endswith(".jsonl")), names[0])

        with zf.open(inner_fname) as fh:
            return [json.loads(line.decode("utf-8")) for line in fh]


def compress_file_to_zip(
    fpath: Path,
    *,
    preserve_source_file: bool = False,
) -> Path:
    """
    Compress an OpenAI batch input JSONL file into a companion zip file.

    :param fpath: Path to source JSONL batch input file.
    :param preserve_source_file: Whether to preserve the source JSONL file.
    :return: Path to the generated zip file.
    """
    zip_fpath = fpath.with_suffix(".zip")
    with zipfile.ZipFile(zip_fpath, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(fpath, arcname=fpath.name)

    if not preserve_source_file:
        fpath.unlink(missing_ok=True)

    return zip_fpath

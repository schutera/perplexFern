"""
Parser module for extracting text from various file formats.

Supports: plain strings, .txt, .json, .csv files.
"""

import csv
import io
import json
import os
from typing import Union


def parse(source: Union[str, os.PathLike]) -> str:
    """
    Extract text content from a source.

    Parameters
    ----------
    source : str or path-like
        Either a raw text string **or** a path to a .txt, .json, or .csv file.

    Returns
    -------
    str
        The extracted text content, ready for metric computation.

    Raises
    ------
    ValueError
        If the file extension is unsupported.
    FileNotFoundError
        If *source* looks like a file path but doesn't exist.
    """
    # If source is a path to an existing file, read it
    if isinstance(source, os.PathLike) or (
        isinstance(source, str) and os.path.isfile(source)
    ):
        return _read_file(str(source))

    # Otherwise treat it as a raw text string
    if isinstance(source, str):
        return source

    raise TypeError(f"Unsupported source type: {type(source)}")


def _read_file(path: str) -> str:
    """Dispatch to the right reader based on file extension."""
    ext = os.path.splitext(path)[1].lower()

    readers = {
        ".txt": _read_txt,
        ".json": _read_json,
        ".csv": _read_csv,
    }

    reader = readers.get(ext)
    if reader is None:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {', '.join(readers.keys())}"
        )
    return reader(path)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_json(path: str) -> str:
    """
    Recursively extract all string values from a JSON structure
    and concatenate them into a single text block.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _extract_strings(data)


def _extract_strings(obj) -> str:
    """Walk a JSON-deserialized object and collect every string value."""
    parts: list[str] = []

    if isinstance(obj, str):
        parts.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            parts.append(_extract_strings(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            parts.append(_extract_strings(item))

    return " ".join(p for p in parts if p)


def _read_csv(path: str) -> str:
    """Read every cell of a CSV and join them into one text block."""
    parts: list[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            parts.extend(cell.strip() for cell in row if cell.strip())
    return " ".join(parts)

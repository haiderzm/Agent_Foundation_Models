# zip_tool.py
"""
GAIA-compatible ZipTool that:
- extracts a .zip into the fixed GAIA files directory:
  /share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/gaia/files
- returns ONLY filenames (relative to that directory), NOT absolute paths
- prevents Zip Slip (path traversal) attacks

Agent usage:
<zip_tool>{"path": "archive.zip"}</zip_tool>

Then agent can call:
<document_reader>{"path": "<returned_filename>"} </document_reader>
"""

from __future__ import annotations

import os
import re
import json
import shutil
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List


GAIA_FILES_DIR = "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/gaia/files"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_name(name: str, max_len: int = 120) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:max_len] if len(name) > max_len else name


def _is_within_directory(base_dir: str, target_path: str) -> bool:
    base_dir = os.path.abspath(base_dir)
    target_path = os.path.abspath(target_path)
    return os.path.commonpath([base_dir]) == os.path.commonpath([base_dir, target_path])


def _list_files_recursive(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            out.append(os.path.join(dirpath, fn))
    return out


def _rel_to_gaia_files(path: str) -> str:
    """
    Convert absolute path under GAIA_FILES_DIR to a filename/relative path.
    Example:
      /.../gaia/files/myzip__extracted/a/b.txt  ->  myzip__extracted/a/b.txt
    """
    base = os.path.abspath(GAIA_FILES_DIR)
    p = os.path.abspath(path)
    if not _is_within_directory(base, p):
        raise ValueError(f"Path is not within GAIA_FILES_DIR: {p}")
    rel = os.path.relpath(p, base)
    # Normalize to forward slashes for tool consumers (optional, but nice)
    return rel.replace(os.sep, "/")


@dataclass
class ZipToolResult:
    type: str
    zip_path: str                 # original input (kept as given / abs)
    extract_dir: str              # relative dir under GAIA_FILES_DIR
    extracted_files: List[str]    # relative file paths (filenames for DocumentReader)
    skipped: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            # "type": self.type,
            # "zip_path": self.zip_path,
            "extract_dir": self.extract_dir,
            "extracted_files": self.extracted_files,
            # "skipped": self.skipped,
            "errors": self.errors,
        }


class ZipTool:
    """
    Extract zip files into GAIA_FILES_DIR and return relative filenames only.
    Structure and call style mirrors DocumentReader:
      tool.extract(path) -> dict
    """

    def __init__(self, overwrite: bool = True):
        self.overwrite = overwrite
        _ensure_dir(GAIA_FILES_DIR)

    def extract(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")

        ext = os.path.splitext(file_path)[1].lower()
        if ext != ".zip":
            raise ValueError(f"Unsupported file type for ZipTool: {ext}")

        zip_path_abs = os.path.abspath(file_path)

        # Extract into a deterministic folder under GAIA_FILES_DIR
        extract_folder_name = f"{_safe_name(os.path.splitext(os.path.basename(zip_path_abs))[0])}__extracted"
        extract_dir_abs = os.path.join(GAIA_FILES_DIR, extract_folder_name)
        extract_dir_rel = extract_folder_name  # relative to GAIA_FILES_DIR

        if self.overwrite and os.path.isdir(extract_dir_abs):
            shutil.rmtree(extract_dir_abs, ignore_errors=True)
        _ensure_dir(extract_dir_abs)

        skipped: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        with zipfile.ZipFile(zip_path_abs, "r") as zf:
            for member in zf.infolist():
                member_name = member.filename

                # Skip explicit directory entries
                if member.is_dir():
                    continue

                dest_path = os.path.join(extract_dir_abs, member_name)

                # Zip Slip defense
                if not _is_within_directory(extract_dir_abs, dest_path):
                    skipped.append({"name": member_name, "reason": "zip_slip_blocked"})
                    continue

                try:
                    _ensure_dir(os.path.dirname(dest_path))
                    with zf.open(member, "r") as src, open(dest_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                except Exception as e:
                    errors.append({"name": member_name, "error": repr(e)})

        extracted_abs = sorted(_list_files_recursive(extract_dir_abs))
        extracted_rel = [_rel_to_gaia_files(p) for p in extracted_abs]

        result = ZipToolResult(
            type="zip",
            zip_path=zip_path_abs,
            extract_dir=extract_dir_rel,
            extracted_files=extracted_rel,
            skipped=skipped,
            errors=errors,
        )
        return result.to_dict()


def main():
    tool = ZipTool(overwrite=True)
    fp = "/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/gaia/files/bfcd99e1-0690-4b53-a85c-0174a8629083.zip"  # change this
    out = tool.extract(fp)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

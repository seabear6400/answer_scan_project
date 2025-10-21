"""Utility module to scan exam folders and build result directories.

The real analysis code is replaced with print statements so this module can be
integrated later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SERIES_CODE: Dict[str, str] = {"인문": "1", "자연": "2"}
SESSION_SUFFIX = "교시"


@dataclass
class ExamFolder:
    series: str
    session: str
    candidate: str
    source: Path

    @property
    def result_name(self) -> str:
        return f"{self.series}{self.session}{self.candidate}_결과"

    @property
    def result_path(self) -> Path:
        return self.source.parent / self.result_name


def _iter_dirs(root: Path) -> Iterable[Path]:
    for current_root, dirnames, _ in os.walk(root):
        for dirname in dirnames:
            yield Path(current_root) / dirname


def _is_candidate_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    name = path.name
    if not name.isdigit():
        return False
    return len(name) == 3 and name.isnumeric()


def _find_series_code(path: Path) -> Optional[str]:
    for part in path.parents:
        for key, code in SERIES_CODE.items():
            if key in part.name:
                return code
    return None


def _find_session_code(path: Path) -> Optional[str]:
    for part in path.parents:
        name = part.name
        if name.endswith(SESSION_SUFFIX):
            digits = ''.join(ch for ch in name if ch.isdigit())
            if digits:
                return digits
    return None


def _has_images(path: Path) -> bool:
    if not path.exists():
        return False
    for root, _dirs, files in os.walk(path):
        for fname in files:
            if fname.lower().endswith(IMAGE_EXTS):
                return True
    return False


def discover_exam_folders(root_path: str) -> List[ExamFolder]:
    root = Path(root_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"지정한 경로가 없습니다: {root}")

    folders: List[ExamFolder] = []
    for directory in _iter_dirs(root):
        if not _is_candidate_dir(directory):
            continue
        if not _has_images(directory):
            continue
        session = _find_session_code(directory)
        series = _find_series_code(directory)
        if not (session and series):
            continue
        folders.append(
            ExamFolder(
                series=series,
                session=session,
                candidate=directory.name,
                source=directory,
            )
        )
    folders.sort(key=lambda f: (f.series, f.session, f.candidate, str(f.source)))
    return folders


def perform_dummy_analysis(source_dir: Path, destination: Path) -> None:
    print(f"분석 실행: {source_dir} -> {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    # TODO: Replace print statement with real analysis code.
    print(f"(더미) {source_dir}의 데이터를 처리했습니다.")


def process_exam_root(root_path: str) -> None:
    print(f"루트 경로 탐색 시작: {root_path}")
    folders = discover_exam_folders(root_path)
    if not folders:
        print("응시 번호 폴더를 찾지 못했습니다.")
        return

    for idx, folder in enumerate(folders, start=1):
        print(f"[{idx}/{len(folders)}] {folder.source} 처리 중")
        perform_dummy_analysis(folder.source, folder.result_path)
    print("모든 분석을 완료했습니다.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exam folder result builder")
    parser.add_argument("root", help="분석을 시작할 루트 경로")
    args = parser.parse_args()
    process_exam_root(args.root)

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from .detector_pipeline import DetectorConfig, detect_pipeline
except ImportError:  # 실행 컨텍스트에 따라 상대 임포트가 실패할 수 있음
    from detector_pipeline import DetectorConfig, detect_pipeline

IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# 계열과 교시 코드를 추출할 때 사용할 간단한 패턴
_SERIES_KEYWORDS: Dict[str, str] = {
    "인문": "1",
    "자연": "2",
}


@dataclass
class ExamScope:
    """단일 응시자 데이터를 처리하기 위한 실행 단위 정보."""

    series_code: str
    session_code: str
    candidate_code: str
    source_dir: Path
    relative_path: Path
    requires_recursive: bool

    @property
    def result_dir_name(self) -> str:
        return f"{self.series_code}{self.session_code}{self.candidate_code}_결과"

    @property
    def result_path(self) -> Path:
        return self.source_dir.parent / self.result_dir_name


def _extract_series_code(parts: Sequence[Path]) -> Optional[str]:
    for part in parts:
        name = part.name
        for keyword, code in _SERIES_KEYWORDS.items():
            if keyword in name:
                return code
    return None


def _extract_session_code(parts: Sequence[Path]) -> Optional[str]:
    for part in parts:
        match = re.search(r"(\d+)\s*교시", part.name)
        if match:
            return match.group(1)
    return None


def _extract_candidate_code(name: str) -> Optional[str]:
    matches = re.findall(r"(\d{3,})", name)
    if not matches:
        return None
    candidate = matches[-1]
    return candidate[-3:]


def _has_images(path: Path) -> bool:
    for root, _dirs, files in os.walk(path):
        for fname in files:
            if fname.lower().endswith(IMAGE_EXTS):
                return True
    return False


def _has_images_top_level(path: Path) -> bool:
    try:
        for fname in os.listdir(path):
            if fname.lower().endswith(IMAGE_EXTS) and (path / fname).is_file():
                return True
    except FileNotFoundError:
        return False
    return False


def _iter_candidate_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    if _is_candidate_dir(root):
        yield root
        return
    for current_root, dirnames, _files in os.walk(root):
        current_path = Path(current_root)
        dirnames[:] = [d for d in dirnames if not d.endswith("_결과")]
        for dirname in dirnames:
            candidate = current_path / dirname
            if _is_candidate_dir(candidate):
                yield candidate


def _is_candidate_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    name = path.name
    if name.endswith("_결과"):
        return False
    return _extract_candidate_code(name) is not None


def discover_exam_scopes(root_path: str) -> Tuple[List[ExamScope], List[str]]:
    root = Path(root_path).resolve()
    issues: List[str] = []
    scopes: List[ExamScope] = []

    if not root.exists():
        raise FileNotFoundError(f"경로가 존재하지 않습니다: {root}")

    for candidate_dir in _iter_candidate_dirs(root):
        candidate_code = _extract_candidate_code(candidate_dir.name)
        if not candidate_code:
            continue

        parts: List[Path] = []
        cur = candidate_dir
        while True:
            parts.append(cur)
            if cur == cur.parent:
                break
            cur = cur.parent
        series_code = _extract_series_code(parts)
        session_code = _extract_session_code(parts)

        if not series_code:
            issues.append(f"계열 코드를 찾지 못했습니다: {candidate_dir}")
            continue
        if not session_code:
            issues.append(f"교시 코드를 찾지 못했습니다: {candidate_dir}")
            continue

        if not _has_images(candidate_dir):
            issues.append(f"이미지 파일이 없어 건너뜁니다: {candidate_dir}")
            continue

        needs_recursive = not _has_images_top_level(candidate_dir)
        rel_path = candidate_dir.relative_to(root)
        scopes.append(
            ExamScope(
                series_code=series_code,
                session_code=session_code,
                candidate_code=candidate_code,
                source_dir=candidate_dir,
                relative_path=rel_path,
                requires_recursive=needs_recursive,
            )
        )

    scopes.sort(key=lambda s: (s.series_code, s.session_code, s.candidate_code, str(s.relative_path)))
    return scopes, issues


def run_scoped_pipeline(
    root_path: str,
    output_base: str,
    config: Optional[DetectorConfig] = None,
    recursive: bool = False,
    progress_callback: Optional[Callable[[str, float, str], None]] = None,
) -> Dict[str, object]:
    scopes, issues = discover_exam_scopes(root_path)

    # single_mode는 사용자가 곧바로 이미지가 있는 폴더를 선택한 경우
    single_mode = not scopes
    if single_mode:
        os.makedirs(output_base, exist_ok=True)

    results: Dict[str, object] = {
        "scopes": scopes,
        "issues": issues,
        "mode": "scoped" if scopes else "single",
        "runs": 0,
        "output_base": str(output_base),
        "result_paths": [],
    }

    if single_mode:
        detect_pipeline(
            root_path,
            output_base,
            config=config,
            recursive=recursive,
            progress_callback=progress_callback,
        )
        results["runs"] = 1
        results["result_paths"] = [str(Path(output_base).resolve())]
        return results

    total = len(scopes)
    for idx, scope in enumerate(scopes, start=1):
        destination = scope.source_dir.parent / scope.result_dir_name
        stage_prefix = f"{scope.result_dir_name}"

        def _scoped_progress(stage: str, pct: float = 0.0, msg: str = "") -> None:
            if progress_callback is None:
                return
            pref_stage = f"{stage_prefix}:{stage}"
            progress_callback(stage=pref_stage, pct=pct, msg=msg)

        print(f"[{idx}/{total}] {scope.result_dir_name} 처리 중 ({scope.source_dir})")
        detect_pipeline(
            str(scope.source_dir),
            str(destination),
            config=config,
            recursive=recursive or scope.requires_recursive,
            progress_callback=_scoped_progress,
        )
        results["runs"] = idx

    results["result_paths"] = [str(scope.result_path.resolve()) for scope in scopes]

    return results

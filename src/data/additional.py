"""Fetch additional Vietnamese hate-speech datasets that share the ViHSD labels.

Five datasets were verified (October 2026) to use **the same** three labels as
ViHSD - ``CLEAN`` / ``OFFENSIVE`` / ``HATE``:

1. ``vominhmanh/vihsd-explainable`` (aka ``dongeov/vihsd-explainable``)
   The original ViHSD splits plus, for every non-clean row, an ``explanation``
   (short Vietnamese rationale) and ``evidence`` (verbatim substrings that
   justify the label).  4.193 toxic training rows carry evidence spans - a
   ready-made high-precision sensitive-word source.

2. ``leminhhung0101/R-ViHSD``
   A robustness extension of ViHSD: 48.092 training rows (originals plus
   synthetically noised variants, flagged in the ``noise_type`` column).

3. ``BonTori/vietnamese-hsd-cleaned``
   ViHSD merged with ~62.671 NEW comments from TDTU YouTube and voz forums,
   re-labelled with the same three classes and cleaned.  The ``source``
   column tells you where each row comes from.

4. ``ViCM`` (``github.com/tiennho2608/ViCM``) - NEW data, code-mixed
   Vietnamese/English Facebook comments with the same three label *names* but a
   different integer encoding (``0=Clean, 1=Hate, 2=Offensive``).  Downloaded
   straight from GitHub as ``.xlsx``; the mapping is applied when counting.

5. ``VLSP2019-HSD`` - NEW data, 25.431 human-annotated Facebook posts from the
   VLSP 2019 shared task with exactly ``HATE`` / ``OFFENSIVE`` / ``CLEAN``.
   There is no public mirror, so this one is **manual**: sign the HSD
   agreement form, drop the csv/xlsx files into
   ``data/raw/additional/vlsp2019-hsd/``, then re-run the ``verify`` stage.

Files are downloaded once into ``data/raw/additional/<name>/`` (kept
byte-for-byte identical to upstream).  After downloading, ``verify`` reads
every file, counts the label values (applying ``label_map`` where the source
uses a different integer encoding) and writes:

* ``NOTES.md`` (per dataset) - Vietnamese explanation of the dataset, its
  label mapping and how it relates to ViHSD
* ``additional_datasets_report.json`` (one file, ``data/raw/additional/``) -
  per-file label counts, ``labels_exact_match`` flag against ViHSD's
  CLEAN/OFFENSIVE/HATE, plus ``source`` / ``requires_remap`` /
  ``pending_manual_download``

Usage
-----
    python scripts/14_fetch_additional_datasets.py
    python scripts/14_fetch_additional_datasets.py --stage download
    python scripts/14_fetch_additional_datasets.py --stage verify
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from xml.etree import ElementTree

from ..common.constants import RAW_DIR
from ..common.logging_utils import get_logger

LOGGER = get_logger("sensitiveai.additional")

STAGES = ("download", "verify", "summary")

ADDITIONAL_RAW_DIR = RAW_DIR / "additional"

#: The three ViHSD labels - every dataset below must match this set exactly.
VIHSD_LABELS = ("CLEAN", "OFFENSIVE", "HATE")

REPORT_NAME = "additional_datasets_report.json"

#: Columns that may hold the label, tried in order for manual / glob datasets.
LABEL_COLUMN_CANDIDATES = ("label", "label_id", "labels", "label_name", "Label")

ADDITIONAL_DATASETS: Dict[str, Dict[str, Any]] = {
    "vominhmanh-vihsd-explainable": {
        "source": "hf",
        "hf_id": "vominhmanh/vihsd-explainable",
        "alt_hf_id": "dongeov/vihsd-explainable",
        "title": "ViHSD-Explainable",
        "files": {
            "train.jsonl": "train.jsonl",
            "validation.jsonl": "validation.jsonl",
            "test.jsonl": "test.jsonl",
            "dataset_info.json": "dataset_info.json",
            "README.md": "README.md",
        },
        "labels": VIHSD_LABELS,
        "label_column": "label",
        "task": "Phân loại 3 lớp (text, label_id, label) + explanation + evidence spans",
        "relationship": (
            "Cùng đúng 33.400 mẫu ViHSD (24.048 / 2.672 / 6.680), thêm cột `explanation` "
            "(lý giải ngắn) và `evidence` (các chuỗi từ trong văn bản chứng minh nhãn; "
            "mẫu CLEAN để rỗng). 4.193 mẫu HATE/OFFENSIVE của train có evidence."
        ),
        "value": "Evidence spans là nguồn từ nhạy cảm chất lượng cao, đã được người/LLM trỏ đích danh trong câu.",
        "caveat": "Không phải dữ liệu mới - là ViHSD + chú giải. Dùng để trích từ nhạy cảm, không dùng để tăng mẫu train.",
        "license": "CC BY 4.0 (theo README của dataset)",
        "citation": "Luu et al. (2021), ViHSD; phần explanation/evidence do tác giả sinh và duyệt.",
    },
    "leminhhung0101-r-vihsd": {
        "source": "hf",
        "hf_id": "leminhhung0101/R-ViHSD",
        "alt_hf_id": None,
        "title": "R-ViHSD (robustness-extended ViHSD)",
        "files": {
            "training_set.csv": "training_set.csv",
            "validation_set.csv": "validation_set.csv",
            "public_test.csv": "public_test.csv",
            "private_test.csv": "private_test.csv",
        },
        "labels": VIHSD_LABELS,
        "label_column": "label",
        "task": "Phân loại 3 lớp, thêm cột `id`, `noise_type` (7 kiểu nhiễu tổng hợp)",
        "relationship": (
            "Mở rộng ViHSD bằng biến thể nhiễu (teencode, typo, …): train 48.092 mẫu "
            "(CLEAN 39.770 / HATE 5.112 / OFFENSIVE 3.210). Cột `noise_type` cho biết mẫu "
            "nào là biến thể nhân tạo. 26.718 dòng ORIGINAL trùng 100% ViHSD train+dev."
        ),
        "value": "Phù hợp kiểm tra độ robust của bộ lọc từ nhạy cảm trước các biến thể che đậy.",
        "caveat": "Nhiều mẫu là biến thể nhân tạo của ViHSD - không dùng làm mẫu test độc lập.",
        "license": "Không ghi rõ trên dataset card",
        "citation": "Không có paper kèm theo (dataset cá nhân tải lên Hugging Face).",
    },
    "bontori-vietnamese-hsd-cleaned": {
        "source": "hf",
        "hf_id": "BonTori/vietnamese-hsd-cleaned",
        "alt_hf_id": None,
        "title": "Vietnamese HSD cleaned (ViHSD + TDTU YouTube/voz)",
        "files": {
            "tdtu_train.parquet": "data/tdtu_train-00000-of-00001.parquet",
            "tdtu_dev.parquet": "data/tdtu_dev-00000-of-00001.parquet",
            "tdtu_test.parquet": "data/tdtu_test-00000-of-00001.parquet",
            "README.md": "README.md",
        },
        "labels": VIHSD_LABELS,
        "label_column": "label",
        "task": "Phân loại 3 lớp, cột `source` (vihsd / tdtu_youtube / tdtu_voz) và `split`",
        "relationship": (
            "Gộp ViHSD (24.048 mẫu) với ~62.671 MẪU MỚI từ YouTube và diễn đàn voz của "
            "TDTU, tái gán nhãn CLEAN/OFFENSIVE/HATE và làm sạch. Train 86.719 mẫu "
            "(CLEAN 61.410 / OFFENSIVE 19.577 / HATE 5.732) + dev/test."
        ),
        "value": "Nguồn mẫu MỚI duy nhất trong ba dataset tải tự động - mở rộng thực sự kho dữ liệu cùng nhãn.",
        "caveat": "Chỉ dòng có `source` là vihsd là trùng ViHSD; lọc theo cột `source` khi cần dữ liệu mới.",
        "license": "Không ghi rõ trên dataset card",
        "citation": "Không có paper kèm theo; liên quan luận văn TDTU (nguồn YouTube/voz).",
    },
    "vicm-code-mixed": {
        "source": "github",
        "hf_id": None,
        "repo": "https://github.com/tiennho2608/ViCM",
        "title": "ViCM (Vietnamese Code-Mixed)",
        "files": {
            "train.xlsx": "https://raw.githubusercontent.com/tiennho2608/ViCM/main/data/train.xlsx",
            "dev.xlsx": "https://raw.githubusercontent.com/tiennho2608/ViCM/main/data/dev.xlsx",
            "test.xlsx": "https://raw.githubusercontent.com/tiennho2608/ViCM/main/data/test.xlsx",
            "forum_data.csv": "https://raw.githubusercontent.com/tiennho2608/ViCM/main/data/forum_data.csv",
            "README.md": "https://raw.githubusercontent.com/tiennho2608/ViCM/main/README.md",
        },
        "file_label_columns": {"forum_data.csv": "Real_Label"},
        "labels": VIHSD_LABELS,
        "label_column": "Label",
        "label_map": {"0": "CLEAN", "1": "HATE", "2": "OFFENSIVE"},
        "requires_remap": True,
        "task": "Phân loại 3 lớp trên bình luận CODE-MIXED (Việt + Anh): `id`, `text`, `Label`.",
        "relationship": (
            "5.415 mẫu MỚI (train 3.787 / dev 542 / test 1.086) do nhóm UIT-VNUHCM thu thập/"
            "chú giải cho bình luận trộn ngôn ngữ - KHÔNG lấy từ ViHSD. Tập `forum_data.csv` "
            "là 480 mẫu diễn đàn phụ; cột thật là `Real_Label` (3 nhãn dạng chuỗi), còn "
            "`Predicted_Label` là dự đoán của mô hình - KHÔNG dùng làm nhãn."
        ),
        "value": "Dữ liệu mới thuộc miền code-mixed - hữu ích để kiểm tra bộ lọc với câu pha tiếng Anh.",
        "caveat": (
            "Mã hoá nhãn KHÁC ViHSD: `0=Clean, 1=Hate, 2=Offensive`. Cũng khác thứ tự dự án "
            "(HATE=0, OFFENSIVE=1, CLEAN=2) nên PHẢI remap. Dataset nhỏ và một phần nội dung "
            "trông như sinh bởi mô hình - chỉ dùng như nguồn phụ."
        ),
        "license": "CC BY 4.0",
        "citation": (
            "Nguyen et al. (2026), An Effective Convolutional Cross-Lingual Language Model "
            "for Vietnamese Online Social Media Mining, IJMLC 17(69). "
            "https://doi.org/10.1007/s13042-025-02904-6"
        ),
    },
    "vlsp2019-hsd": {
        "source": "manual",
        "hf_id": None,
        "repo": "https://github.com/vietnlp/vlsp2019_hatespeech_task",
        "task_page": "https://vlsp.org.vn/vlsp2019/eval/hsd",
        "agreement_url": "https://drive.google.com/drive/folders/1HSOWdiMFQy0bjd09TBYaI7lsf7iYZaq5",
        "title": "VLSP 2019 - Hate Speech Detection (HSD)",
        "files": {},
        "data_globs": ("*.csv", "*.xlsx"),
        "label_columns": LABEL_COLUMN_CANDIDATES,
        "labels": VIHSD_LABELS,
        "label_column": "label",
        "label_map": {
            "0": "CLEAN",
            "1": "OFFENSIVE",
            "2": "HATE",
            "NORMAL": "CLEAN",
            "CLEAN": "CLEAN",
            "OFFENSIVE": "OFFENSIVE",
            "HATE": "HATE",
        },
        "requires_remap": False,
        "task": "Phân loại 3 lớp trên bài đăng/bình luận Facebook (VLSP shared task 2019).",
        "relationship": (
            "25.431 mẫu MỚI, con người gán nhãn (train 20.345: CLEAN 18.614 / OFFENSIVE 1.022 / "
            "HATE 709; còn lại là public-test). Đây là bộ HSD tiếng Việt sớm nhất, KHÁC nguồn, "
            "khác quy trình gán nhãn với ViHSD."
        ),
        "value": "Bổ sung bài đăng Facebook ngoài miền YouTube/Facebook-video của ViHSD - nguồn mới thật sự.",
        "caveat": (
            "Không tải tự động được: phải ký form agreement của ban tổ chức rồi bỏ file vào thư mục. "
            "Thứ tự nhãn gốc là `0=CLEAN, 1=OFFENSIVE, 2=HATE` (giống ViHSD), nhưng script vẫn "
            "chấp nhận cả nhãn dạng chữ để tự chuẩn hoá."
        ),
        "license": "Theo agreement form của VLSP (chỉ dùng cho nghiên cứu).",
        "citation": "Vu et al. (2020), HSD Shared Task in VLSP Campaign 2019, arXiv:2007.06493.",
    },
}


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _download_http(url: str, target: Path) -> None:
    """Download ``url`` to ``target`` with a browser-ish User-Agent."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (SensitiveAI data fetcher)"})
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())


def download_additional(
    output_dir: str | Path = ADDITIONAL_RAW_DIR,
    force: bool = False,
) -> Dict[str, Path]:
    """Fetch every registered dataset file into ``output_dir/<name>/``.

    Hugging Face datasets use ``hf_hub_download``; GitHub datasets are fetched
    over plain HTTP; manual datasets are only checked (files must be dropped in
    by hand after signing the data agreement).
    """
    out = Path(output_dir)
    paths: Dict[str, Path] = {}
    for name, spec in ADDITIONAL_DATASETS.items():
        target_dir = out / name
        target_dir.mkdir(parents=True, exist_ok=True)
        source = spec.get("source", "hf")
        if source == "manual":
            _warn_manual(name, spec, target_dir)
            continue
        for local_name, remote_name in spec["files"].items():
            target = target_dir / local_name
            if target.is_file() and not force:
                LOGGER.info("[additional] %s/%s already present", name, local_name)
            elif source == "github":
                LOGGER.info("[additional] fetching (github) %s ...", remote_name)
                _download_http(remote_name, target)
                LOGGER.info("[additional] saved %s (%d bytes)", target, target.stat().st_size)
            else:
                from huggingface_hub import hf_hub_download

                LOGGER.info("[additional] fetching %s :: %s ...", spec["hf_id"], remote_name)
                cached = hf_hub_download(repo_id=spec["hf_id"], filename=remote_name, repo_type="dataset")
                shutil.copyfile(cached, target)
                LOGGER.info("[additional] saved %s (%d bytes)", target, target.stat().st_size)
            paths[f"{name}/{local_name}"] = target
    return paths


def _warn_manual(name: str, spec: Dict[str, Any], target_dir: Path) -> None:
    present = sorted(p.name for p in target_dir.glob("*") if p.is_file() and p.suffix.lower() != ".md")
    if present:
        LOGGER.info("[additional] %s (manual source) has %d file(s): %s", name, len(present), ", ".join(present))
        return
    LOGGER.warning(
        "[additional] %s phải tải THỦ CÔNG: ký agreement form (%s) rồi bỏ file .csv/.xlsx vào %s, "
        "sau đó chạy lại `--stage verify`.",
        name,
        spec.get("agreement_url", spec.get("task_page", "")),
        target_dir,
    )


# --------------------------------------------------------------------------- #
# Reading helpers
# --------------------------------------------------------------------------- #
def _xlsx_rows(path: Path) -> List[Dict[str, str]]:
    """Read the first worksheet of an xlsx file using only the stdlib."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in root.findall(ns + "si"):
                shared.append("".join(t.text or "" for t in si.iter(ns + "t")))
        sheet_name = next(n for n in archive.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
        sheet = ElementTree.fromstring(archive.read(sheet_name))

    def _cell_value(cell: ElementTree.Element) -> str:
        if cell.get("t") == "inlineStr":
            return "".join(t.text or "" for t in cell.iter(ns + "t"))
        value = cell.find(ns + "v")
        if value is None or value.text is None:
            return ""
        return shared[int(value.text)] if cell.get("t") == "s" else value.text

    def _col_index(ref: str) -> int:
        letters = "".join(ch for ch in ref if ch.isalpha())
        idx = 0
        for ch in letters:
            idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
        return idx - 1

    rows: List[Dict[str, str]] = []
    header: Optional[List[str]] = None
    for row in sheet.find(ns + "sheetData").findall(ns + "row"):
        cells: Dict[int, str] = {}
        for cell in row.findall(ns + "c"):
            ref = cell.get("r") or ""
            cells[_col_index(ref)] = _cell_value(cell)
        if header is None:
            width = max(cells) + 1 if cells else 0
            header = [cells.get(i) or f"col{i + 1}" for i in range(width)]
            continue
        if not cells:
            continue
        rows.append({header[i]: cells.get(i, "") for i in range(len(header))})
    return rows


def _read_rows(path: Path) -> List[Dict[str, str]]:
    """Read a csv / jsonl / xlsx / parquet file into a list of string dicts."""
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return _xlsx_rows(path)
    if suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append({k: str(v) for k, v in json.loads(line).items()})
        return rows
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as fh:
            return [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(fh)]
    if suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        return [{k: str(v) for k, v in record.items()} for record in frame.to_dict(orient="records")]
    raise ValueError(f"Unsupported file type: {path}")


def _count_labels(
    path: Path,
    label_column: str,
    label_map: Optional[Dict[str, str]] = None,
    label_columns: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """Count label values in a csv / parquet / jsonl / xlsx file.

    ``label_map`` rewrites raw values (e.g. ``0``/``1``/``2``) to canonical
    names before counting, so the resulting keys can be compared with ViHSD.
    When ``label_columns`` is given, the first column that exists is used.
    """
    rows = _read_rows(path)
    if not rows:
        return {}
    candidates = list(label_columns) if label_columns else [label_column]
    available = set(candidates) & set(rows[0].keys())
    if not available:
        raise KeyError(f"{path.name}: none of {candidates} in {list(rows[0].keys())}")
    column = next(c for c in candidates if c in available)

    mapping = label_map or {}
    counts: Dict[str, int] = {}
    for row in rows:
        raw = str(row.get(column, "")).strip()
        canonical = mapping.get(raw, mapping.get(raw.upper(), raw))
        counts[canonical] = counts.get(canonical, 0) + 1
    return counts


def _resolve_data_files(dataset_dir: Path, spec: Dict[str, Any]) -> List[Path]:
    """List data files for a dataset: fixed filenames plus optional globs."""
    found: List[Path] = []
    for local_name in spec.get("files", {}):
        path = dataset_dir / local_name
        if path.is_file():
            found.append(path)
    for pattern in spec.get("data_globs", ()):
        for path in sorted(dataset_dir.glob(pattern)):
            if path.is_file() and path not in found:
                found.append(path)
    return found


def _table_files(spec: Dict[str, Any], files: Iterable[Path]) -> List[Path]:
    """Keep only files that can hold labels (drop READMEs and other docs)."""
    return [p for p in files if p.suffix.lower() in (".csv", ".parquet", ".jsonl", ".xlsx")]


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
_NOTES_TEMPLATE = """# {title}

> File giải thích tự sinh bởi `python scripts/14_fetch_additional_datasets.py --stage verify`.
> Nguồn dữ liệu: {source_url}

## 1. Đây là dataset gì?

{relationship}

- **Task:** {task}
- **File đã tải:** {files}
- **Giấy phép:** {license}

## 2. So sánh nhãn với ViHSD

| | ViHSD (dự án) | {title} |
|---|---|---|
| Nhãn | `CLEAN`, `OFFENSIVE`, `HATE` | {labels} |
| Khớp tuyệt đối? | - | **{exact_match}** |
| Ánh xạ nhãn | - | `{mapping}` |

{mapping_note}
{manual_block}
## 3. Thống kê nhãn đã đếm trực tiếp trên file

{label_stats}

## 4. Giá trị & lưu ý khi dùng

- **Giá trị:** {value}
- **Lưu ý:** {caveat}

## 5. Trích dẫn / nguồn

- Nguồn: {source_url}
{alt_line}
- Tài liệu: {citation}
- Dataset chính của dự án: `uitnlp/vihsd` (Luu et al., 2021).
"""


def _mapping_note(spec: Dict[str, Any], exact: bool) -> str:
    if spec.get("requires_remap"):
        return (
            "File gốc dùng mã hoá KHÁC ViHSD; bảng ở trên là ánh xạ đã áp khi đếm để quy về "
            "`CLEAN`/`OFFENSIVE`/`HATE`. Khi nạp dữ liệu cho hệ thống phải remap y như vậy."
        )
    if exact:
        return (
            "Bộ nhãn giống hệt ViHSD nên **không cần remap**: `CLEAN`→`CLEAN`, "
            "`OFFENSIVE`→`OFFENSIVE`, `HATE`→`HATE`."
        )
    return (
        "Nhãn KHÔNG giống hệt - cần remap thủ công trước khi hợp nhất. "
        "Bảng ánh xạ đề xuất nằm ngay bên trên, mọi quyết định khác phải được ghi chú lại."
    )


def _mapping_text(spec: Dict[str, Any], exact: bool) -> str:
    mapping = spec.get("label_map")
    if mapping:
        return ", ".join(f"{raw}→{canonical}" for raw, canonical in mapping.items())
    return "CLEAN→CLEAN, OFFENSIVE→OFFENSIVE, HATE→HATE" if exact else "xem mục 2"


def _manual_block(spec: Dict[str, Any], present: List[Path]) -> str:
    if spec.get("source") != "manual":
        return ""
    status = (
        f"Đã thấy {len(present)} file trong thư mục."
        if present
        else "**Chưa có file nào** - hãy tải thủ công:"
    )
    return (
        "\n## 1b. Cách tải dữ liệu (thủ công)\n\n"
        f"{status}\n\n"
        f"1. Trang task: {spec.get('task_page', '')}\n"
        f"2. Ký **HSD Agreement form**: {spec.get('agreement_url', '')}\n"
        "3. Chép file `.csv`/`.xlsx` nhận được vào `data/raw/additional/vlsp2019-hsd/`.\n"
        "4. Chạy lại: `python scripts/14_fetch_additional_datasets.py --stage verify`\n"
    )


def verify_additional(output_dir: str | Path = ADDITIONAL_RAW_DIR) -> Dict[str, Any]:
    """Count labels in every downloaded file, flag exact matches, write NOTES.md."""
    out = Path(output_dir)
    report: Dict[str, Any] = {
        "vihsd_labels": list(VIHSD_LABELS),
        "datasets": {},
    }
    for name, spec in ADDITIONAL_DATASETS.items():
        dataset_dir = out / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        resolved = _resolve_data_files(dataset_dir, spec)
        table_files = _table_files(spec, resolved)

        files_report: Dict[str, Any] = {}
        all_labels: Dict[str, int] = {}
        per_file_columns = spec.get("file_label_columns", {})
        for path in resolved:
            if path.suffix.lower() not in (".csv", ".parquet", ".jsonl", ".xlsx"):
                files_report[path.name] = {"skipped": "not a tabular data file"}
                continue
            column = per_file_columns.get(path.name, spec["label_column"])
            try:
                counts = _count_labels(
                    path,
                    column,
                    label_map=spec.get("label_map"),
                    label_columns=spec.get("label_columns"),
                )
            except KeyError:
                files_report[path.name] = {
                    "error": "no label column - held-out test set, labels not published upstream"
                }
                continue
            except ValueError as exc:
                LOGGER.warning("[additional] cannot count labels in %s: %s", path, exc)
                files_report[path.name] = {"error": str(exc)}
                continue
            files_report[path.name] = {"label_counts": counts}
            if set(counts) <= set(spec["labels"]) and counts:  # data files only
                for label, n in counts.items():
                    all_labels[label] = all_labels.get(label, 0) + n

        observed = set(all_labels)
        exact = observed == set(spec["labels"]) if observed else False
        pending = not table_files and spec.get("source") == "manual"
        report["datasets"][name] = {
            "source": spec.get("source", "hf"),
            "hf_id": spec.get("hf_id"),
            "repo": spec.get("repo"),
            "title": spec["title"],
            "expected_labels": list(spec["labels"]),
            "observed_labels": sorted(observed),
            "labels_exact_match": exact,
            "requires_remap": bool(spec.get("requires_remap", False)),
            "pending_manual_download": pending,
            "total_rows": sum(all_labels.values()),
            "label_totals": dict(sorted(all_labels.items())),
            "files": files_report,
        }
        _write_notes(dataset_dir, spec, files_report, all_labels, exact, resolved)
        LOGGER.info(
            "[additional] %s: labels=%s exact_match=%s rows=%d",
            name, sorted(observed), exact, sum(all_labels.values()),
        )

    report_path = out / REPORT_NAME
    with report_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    LOGGER.info("[additional] wrote %s", report_path)
    return report


def _source_url(spec: Dict[str, Any]) -> str:
    if spec.get("hf_id"):
        return f"https://huggingface.co/datasets/{spec['hf_id']}"
    if spec.get("repo"):
        return spec["repo"]
    return "(không rõ)"


def _write_notes(
    dataset_dir: Path,
    spec: Dict[str, Any],
    files_report: Dict[str, Any],
    label_totals: Dict[str, int],
    exact: bool,
    resolved: List[Path],
) -> None:
    stats_lines = ["| File | Phân bố nhãn |", "|---|---|"]
    for file_name, meta in files_report.items():
        counts = meta.get("label_counts")
        if counts:
            pretty = ", ".join(f"{k} {v:,}" for k, v in sorted(counts.items()))
            stats_lines.append(f"| `{file_name}` | {pretty} |")
        elif "error" in meta:
            stats_lines.append(f"| `{file_name}` | (không đếm được: {meta['error']}) |")
        elif "skipped" in meta:
            stats_lines.append(f"| `{file_name}` | (bỏ qua: {meta['skipped']}) |")
    if len(stats_lines) == 2:
        stats_lines.append("| (chưa có file) | - |")
    totals = ", ".join(f"{k} {v:,}" for k, v in sorted(label_totals.items())) or "-"

    labels = ", ".join(f"`{label}`" for label in spec["labels"])
    notes = _NOTES_TEMPLATE.format(
        title=spec["title"],
        source_url=_source_url(spec),
        relationship=spec["relationship"],
        task=spec["task"],
        files=", ".join(f"`{p.name}`" for p in resolved) or "(chưa có - xem mục 1b)",
        license=spec.get("license", "không ghi rõ"),
        labels=labels,
        exact_match="CÓ - giống hệt" if exact else "KHÔNG",
        mapping=_mapping_text(spec, exact),
        mapping_note=_mapping_note(spec, exact),
        manual_block=_manual_block(spec, resolved),
        label_stats="\n".join(stats_lines) + f"\n\n**Tổng hợp:** {totals}",
        value=spec["value"],
        caveat=spec["caveat"],
        alt_line=(
            f"- Bản mirror: https://huggingface.co/datasets/{spec['alt_hf_id']}"
            if spec.get("alt_hf_id")
            else "- Không có repo mirror."
        ),
        citation=spec.get("citation", "không rõ"),
    )
    (dataset_dir / "NOTES.md").write_text(notes, encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.additional",
        description="Download and verify additional Vietnamese datasets with ViHSD's labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", nargs="*", default=list(STAGES), choices=STAGES, help="Which stages to run.")
    p.add_argument("--raw-dir", default=None, help="Override the raw additional-datasets directory.")
    p.add_argument("--force-download", action="store_true", help="Re-fetch files even if present.")
    return p


def _print_summary(report: Dict[str, Any]) -> None:
    for name, meta in sorted(report.get("datasets", {}).items()):
        LOGGER.info(
            "[additional] %-32s exact_match=%-5s rows=%-6d labels=%s",
            name, meta["labels_exact_match"], meta["total_rows"], meta["observed_labels"],
        )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    raw_dir = Path(args.raw_dir) if args.raw_dir else ADDITIONAL_RAW_DIR
    LOGGER.info("[additional] raw -> %s", raw_dir)

    if "download" in args.stage:
        download_additional(raw_dir, force=args.force_download)

    report: Optional[Dict[str, Any]] = None
    if "verify" in args.stage:
        report = verify_additional(raw_dir)

    if "summary" in args.stage:
        if report is None:
            report_path = raw_dir / REPORT_NAME
            if not report_path.is_file():
                LOGGER.error("[additional] %s not found - run the verify stage first.", report_path)
                return 1
            report = json.loads(report_path.read_text(encoding="utf-8"))
        _print_summary(report)

    LOGGER.info("[additional] done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

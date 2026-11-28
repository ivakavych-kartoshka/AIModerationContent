"""Step 14 - download 5 extra datasets that share ViHSD's three labels.

    python scripts/14_fetch_additional_datasets.py
    python scripts/14_fetch_additional_datasets.py --stage download
    python scripts/14_fetch_additional_datasets.py --stage verify summary

Datasets (all use CLEAN / OFFENSIVE / HATE):

* ``vominhmanh/vihsd-explainable`` - ViHSD + explanation + evidence spans
* ``leminhhung0101/R-ViHSD``       - robustness-extended ViHSD (48K train)
* ``BonTori/vietnamese-hsd-cleaned`` - ViHSD + 62K NEW TDTU YouTube/voz rows
* ``ViCM`` (GitHub)                - 5.4K NEW code-mixed rows, nhãn cần remap
* ``VLSP2019-HSD`` (thủ công)      - 25.4K bài Facebook mới, phải ký form

Raw files land in ``data/raw/additional/<name>/`` (never edited). The verify
stage counts labels per file and writes ``additional_datasets_report.json``
plus a Vietnamese ``NOTES.md`` explaining each dataset and its label mapping.

VLSP2019-HSD không tải tự động được: ký agreement form, chép file vào
``data/raw/additional/vlsp2019-hsd/`` rồi chạy lại ``--stage verify``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.additional import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

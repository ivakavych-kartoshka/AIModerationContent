"""Regenerate the ``explain.md`` file of every folder in the project.

    python scripts/05_document_folders.py            # whole tree
    python scripts/05_document_folders.py --root reports
    python scripts/05_document_folders.py --check    # report, write nothing

Each folder gets an ``explain.md`` stating what it is for, which files it holds,
how those files were produced, whether it is safe to delete, and - for model
folders - a table of every run with its status.  Only ``explain.md`` files are
written; no log or artifact is touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.constants import PROJECT_ROOT  # noqa: E402
from src.common.folder_docs import (  # noqa: E402
    EXPLAIN_FILENAME,
    document_tree,
    index_markdown,
    iter_project_folders,
    relative_to_root,
    spec_for,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        default=None,
        help="Only document this folder and everything below it (default: project root).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="How deep to walk below --root (default: 6).",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="Also write a one-page index of all folders to this markdown file.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report what would be written; do not modify any file.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else Path(PROJECT_ROOT).resolve()
    if not root.is_dir():
        print(f"[docs] not a folder: {root}")
        return 1

    folders = list(iter_project_folders(root, max_depth=args.max_depth))
    undocumented = [f for f in folders if spec_for(relative_to_root(f)) is None]

    if args.check:
        print(f"[docs] {len(folders)} folders under {root}")
        print(f"[docs] {len(folders) - len(undocumented)} have a registry entry")
        for folder in undocumented:
            print(f"       - {relative_to_root(folder)}")
        return 0

    written = document_tree(root, max_depth=args.max_depth)
    print(f"[docs] wrote {len(written)} {EXPLAIN_FILENAME} files under {root}")

    if args.index:
        index_path = Path(args.index)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            "# Chỉ mục thư mục dự án\n\n"
            "Sinh tự động bởi `scripts/05_document_folders.py --index`. Mỗi dòng dẫn tới "
            "`explain.md` của thư mục đó.\n\n" + index_markdown(root),
            encoding="utf-8",
        )
        print(f"[docs] index -> {index_path}")

    if undocumented:
        print(f"[docs] {len(undocumented)} folders have no registry entry yet:")
        for folder in undocumented:
            print(f"       - {relative_to_root(folder)}")
        print("[docs] add them to _specs() in src/common/folder_docs.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

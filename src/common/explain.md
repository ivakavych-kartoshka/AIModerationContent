<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# common - hằng số, config, đường dẫn

> **Dùng để làm gì:** Tầng nền tảng: ánh xạ nhãn cố định, nạp/merge YAML, quy ước `run_XXX`, seed, logging và registry tài liệu folder.

> **Sinh ra bởi:** Không sinh ra - mã nguồn.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `constants.py` | Nhãn cố định `HATE=0, OFFENSIVE=1, CLEAN=2`, tên 4 model, đường dẫn gốc. |
| `config.py` | Nạp YAML, merge override `--set`, kiểm tra kiểu, ghi `config.yaml`/`dump_json`. |
| `paths.py` | Quy ước `reports/<model>/run_XXX`, `experiments/<model>/run_XXX`, con trỏ run mới nhất, `find_best_report_dir()`. |
| `seeding.py` | Seed cho torch/numpy/random/PYTHONHASHSEED + khởi tạo worker DataLoader. |
| `logging_utils.py` | Logger dùng chung, ghi ra console **và** file (`train.log`, `evaluate.log`). |
| `env_info.py` | Ghi lại phiên bản thư viện, GPU, hệ điều hành vào báo cáo để tái lập. |
| `folder_docs.py` | **Registry `explain.md`**: định nghĩa nội dung mọi folder và bộ sinh file giải thích. |
| `explain.md` | File này. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 8 file, 101.5 KB |
| `__init__.py` | file | 781 B · 2026-10-04 12:36:15 |
| `config.py` | file | 10.5 KB · 2026-10-04 10:34:15 |
| `constants.py` | file | 2.8 KB · 2026-10-04 10:33:47 |
| `env_info.py` | file | 3.6 KB · 2026-10-04 10:34:28 |
| `explain.md` | file | 2.4 KB · 2026-10-04 12:58:19 |
| `folder_docs.py` | file | 59.7 KB · 2026-10-04 12:59:52 |
| `logging_utils.py` | file | 1.2 KB · 2026-10-04 11:40:29 |
| `paths.py` | file | 6.5 KB · 2026-10-04 12:36:36 |
| `seeding.py` | file | 1.3 KB · 2026-10-04 10:33:52 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._

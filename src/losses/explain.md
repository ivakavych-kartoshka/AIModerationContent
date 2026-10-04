<!-- AUTO-GENERATED: src/common/folder_docs.py -->
# losses - hàm mất mát

> **Dùng để làm gì:** Một nơi duy nhất để dựng loss từ YAML, gồm cross-entropy, focal loss (alpha theo nhãn), class weight và concept-focused curriculum loss.

> **Sinh ra bởi:** Không sinh ra - mã nguồn.

> **Quy tắc giữ lại:** **KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được `--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ.

## File cố định của folder này

| File / thư mục | Ý nghĩa |
|---|---|
| `builder.py` | `build_loss()` - đọc `loss:` trong YAML rồi trả về module loss tương ứng (cross_entropy / focal / concept). |
| `focal.py` | Focal loss `FL(p_t) = -alpha_t (1-p_t)^gamma log(p_t)`, `alpha` có thể khác nhau theo từng nhãn. |
| `class_weights.py` | 5 chiến lược cân bằng lớp (none / inverse_frequency / inverse_sqrt_frequency / effective_number / manual), luôn ước lượng trên **train**. |
| `curriculum.py` | Concept-focused loss: một nhãn là dương, hai nhãn còn lại là âm; softmax giữ thứ tự giữa hai lớp âm. |
| `explain.md` | File này. |

## Nội dung hiện có trên đĩa

| Tên | Loại | Kích thước · sửa lần cuối |
|---|---|---|
| `__pycache__` | thư mục | 5 file, 34.7 KB |
| `__init__.py` | file | 783 B · 2026-10-04 10:38:22 |
| `builder.py` | file | 6.7 KB · 2026-10-04 11:39:41 |
| `class_weights.py` | file | 4.6 KB · 2026-10-04 10:55:45 |
| `curriculum.py` | file | 5.9 KB · 2026-10-04 11:39:41 |
| `explain.md` | file | 2.0 KB · 2026-10-04 12:58:19 |
| `focal.py` | file | 4.6 KB · 2026-10-04 10:37:48 |

---

_Sinh tự động bởi `scripts/05_document_folders.py` lúc 2026-10-04 13:00:00. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._

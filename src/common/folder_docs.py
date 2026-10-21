"""Self-documenting folders: one ``explain.md`` per directory.

Every folder in the project - source folders as well as folders produced by a
run - carries an ``explain.md`` that answers three questions:

1. what is this folder for,
2. which files live in it and what each one means,
3. is it safe to delete (the answer is "no" for every artifact folder).

The registry below is the single source of truth.  ``scripts/05_document_folders.py``
renders it for the whole tree, and the training / evaluation entry points render it
for the folders they create so a freshly produced run is documented immediately.

Regenerating never deletes user content: ``explain.md`` files are the only files
this module writes.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .constants import PROJECT_ROOT
from .paths import ensure_dir

EXPLAIN_FILENAME = "explain.md"
GENERATED_MARKER = "<!-- AUTO-GENERATED: src/common/folder_docs.py -->"

KEEP_POLICY = (
    "**KHÔNG XÓA.** Mọi lần chạy đều tạo một thư mục riêng (`run_XXX` hoặc thư mục được "
    "`--reports-dir`/`--experiments-dir` trỏ tới), nên không lần chạy nào ghi đè lần trước. "
    "Muốn dọn dung lượng thì copy ra nơi khác rồi mới xóa - không xóa tại chỗ."
)


@dataclass(frozen=True)
class FolderDoc:
    """Static documentation for one folder (or one folder pattern)."""

    title: str
    purpose: str
    produced_by: str
    contents: Sequence[Tuple[str, str]] = field(default_factory=tuple)
    children: Sequence[Tuple[str, str]] = field(default_factory=tuple)
    reading_guide: Sequence[str] = field(default_factory=tuple)
    keep_policy: str = KEEP_POLICY
    extra_notes: Sequence[str] = field(default_factory=tuple)
    #: free-form markdown blocks rendered verbatim under "## Chi tiết"
    details: Sequence[str] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
# Keys are POSIX-style paths relative to the project root.  A ``*`` segment
# matches exactly one directory level (``reports/*/run_*``).

_RUN_FILE_DOCS: Tuple[Tuple[str, str], ...] = (
    ("config.yaml", "Cấu hình **đã resolve** của riêng run này (đã áp mọi override). Snapshot chính xác để tái lập."),
    ("run_pointer.txt", "Tên `run_XXX` mới nhất trong folder model. Chỉ là con trỏ, dùng cho `find_best_report_dir()`."),
    ("explain.md", "File này: giải thích folder run và các file bên trong."),
    ("train.log", "Log console đầy đủ của **bước train** (mọi epoch, mọi stage, cảnh báo). Nguồn sự thật khi cần truy vết."),
    ("evaluate.log", "Log console đầy đủ của **bước chấm test** (nếu đã chạy `02_evaluate_all.py`)."),
    ("training_log.csv", "**Một dòng cho mỗi epoch.** Cột: model_name, stage, stage_index, epoch, epoch_in_stage, learning_rate, encoder_lr, head_lr, loss, accuracy, val_loss, val_accuracy, val_macro_f1, val_weighted_f1, val_{HATE,OFFENSIVE,CLEAN}_{f1,recall}, grad_norm, epoch_seconds, cumulative_seconds, is_best. Mọi biểu đồ validation trong paper đều đọc từ file này."),
    ("training_summary.json", "Tổng kết sau train: best epoch/stage theo validation macro F1, thời gian từng epoch, số tham số, learning rate theo nhóm, lịch sử loss/val_metric, cờ `is_smoke_test`."),
    ("class_weights.json", "Class weight ước lượng **chỉ trên tập train** + đếm nhãn train + giá trị theo từng stage. Có trường `estimated_on` để chứng minh không rò rỉ từ test."),
    ("stage_transitions.json", "Audit trail multi-stage: mỗi lần chuyển stage ghi stage trước, stage sau, file checkpoint nguồn, số epoch, loss. Dùng để chứng minh stage cuối cùng thực sự đã chạy."),
    ("best_model/", "Weight của epoch có **validation macro F1** cao nhất + tokenizer. Đây là checkpoint dùng để chấm test."),
    ("best_training_state.pt", "Trạng thái optimizer/scheduler tại đúng epoch đó (để có thể tiếp tục huấn luyện thay vì bắt đầu lại)."),
    ("last_model/", "Weight của epoch cuối cùng. Giữ lại để đối chiếu với `best_model/`."),
    ("metrics.json", "Toàn bộ chỉ số test (accuracy, macro/weighted F1, per-class P/R/F1, macro & per-class ROC-AUC và average precision), cùng quy tắc quyết định đã dùng."),
    ("thresholds.json", "Threshold tối ưu **tìm trên validation** + giá trị objective trước/sau. Áp đông cứng cho test."),
    ("thresholds.txt", "Bản đọc được của `thresholds.json`."),
    ("test_predictions.csv", "**Từng mẫu test**: text, nhãn thật, nhãn dự đoán, xác suất 3 lớp, confidence. File này dùng để kiểm tra lại kết quả bằng tay."),
    ("classification_report.txt", "Báo cáo `sklearn` (P/R/F1/support từng lớp) trên split đã chấm."),
    ("Best_model_classification_report.txt", "Bản sao của báo cáo trên, giữ tên file cũ để tương thích."),
    ("confusion_matrix.txt", "Ma trận nhầm lẫn dạng văn bản (số mẫu + tỉ lệ theo hàng)."),
    ("confusion_matrix.png", "Ma trận nhầm lẫn dạng hình (số mẫu)."),
    ("confusion_matrix_raw.png", "Ma trận nhầm lẫn dạng hình, **không chuẩn hoá** (số mẫu tuyệt đối)."),
    ("confusion_matrix_normalized.png", "Ma trận nhầm lẫn **chuẩn hoá theo hàng** (mỗi hàng = 100 %, dễ thấy lớp nào bị nhầm sang đâu)."),
    ("pr_curve.png", "Đường cong Precision-Recall cho cả 3 lớp trên split đã chấm."),
    ("roc_curve.png", "Đường cong ROC cho cả 3 lớp (tham khảo; PR đáng tin hơn khi dữ liệu mất cân bằng)."),
    ("pr_curve_data.json", "Điểm dữ liệu gốc của `pr_curve.png` (precision, recall, average precision) để vẽ lại."),
    ("roc_curve_data.json", "Điểm dữ liệu gốc của `roc_curve.png` (fpr, tpr, auc)."),
    ("per_class_f1.png", "Biểu đồ cột F1 của từng lớp."),
    ("val_accuracy_curve.png", "Validation accuracy theo epoch, đánh dấu chuyển stage và epoch tốt nhất."),
    ("val_loss_curve.png", "Validation loss theo epoch, đánh dấu chuyển stage."),
    ("train_loss_curve.png", "Training loss theo epoch, đánh dấu chuyển stage."),
    ("benchmark.json", "Tốc độ suy luận: latency ms/sample, throughput (samples/s), số batch đo, warmup, thiết bị."),
    ("model_summary.txt", "Bản tóm tắt dễ đọc cho một model: checkpoint dùng, quy tắc quyết định, chỉ số chính, môi trường."),
    ("history/", "Các lần chấm test **trước** của chính run này, được ghi lại thay vì xóa. Mỗi lần nằm trong thư mục `eval_<YYYYmmdd-HHMMSS>/`."),
)

_RUN_READING_GUIDE: Tuple[str, ...] = (
    "Muốn biết model này học được gì: mở `training_summary.json` (best epoch, best macro F1).",
    "Muốn xem diễn biến từng epoch: mở `training_log.csv`, cột `epoch`, `loss`, `val_macro_f1`, `is_best`.",
    "Muốn biết chính xác đã chạy những gì: đọc `train.log` từ đầu đến cuối.",
    "Muốn xem kết quả trên test: `metrics.json` (số liệu) + `test_predictions.csv` (kiểm chứng từng mẫu).",
    "Muốn biết tại sao chọn checkpoint này: cột `is_best` trong `training_log.csv` + `training_summary.json`.",
    "Muốn hiểu vì sao có nhiều stage: `stage_transitions.json`.",
    "Muốn biết chạy lại đúng như vậy: `config.yaml` là snapshot đã resolve, không cần đoán lại override.",
)


_CONFIDENCE_DETAIL: Tuple[str, ...] = (
    "### Cách tính `confidence` - độ chắc chắn của một entry từ điển",
    "",
    "Cột `confidence` trong `verified_lexicon.csv` trả lời câu hỏi *\"term này chắc chắn là từ độc hại đến đâu\"* "
    "- **không** phải tần suất xuất hiện thô. Nó do `lexical_labeler._confidence()` tính từ bằng chứng "
    "**chỉ trên train split** (plan §36.1).",
    "",
    "**Bước 1 - đếm bằng chứng (train-only).** Quét từng câu train của 3 bộ (ViHSD / ViCM / bontori), đếm "
    "document frequency theo nhãn (mỗi câu chỉ bầu một lần):",
    "",
    "- `n_hate`, `n_offensive`, `n_clean`: số câu mỗi lớp chứa term;",
    "- `r_hate = n_hate / (số câu HATE)`, tương tự `r_offensive`, `r_clean` - **chuẩn hoá theo kích thước lớp** "
    "nên lớp CLEAN đông không lấn át;",
    "- `share_hate = r_hate / (r_hate + r_offensive)`: term nghiêng về lớp độc hại nào;",
    "- `lift_hate = r_hate / r_clean` (và `lift_offensive`): term xuất hiện ở HATE nhiều hơn CLEAN bao nhiêu lần.",
    "",
    "**Bước 2 - công thức.** `confidence` là trung bình có trọng số của **3 tín hiệu**, mỗi tín hiệu đã được "
    "chuẩn hoá về `[0, 1]`:",
    "",
    "```",
    "value      = 0.4 * ev + 0.3 * margin + 0.3 * lift_score",
    "confidence = clip(value, 0.05, 0.99)",
    "```",
    "",
    "| Thành phần | Công thức | Ý nghĩa | Khoảng |",
    "|---|---|---|---|",
    "| `ev` (evidence) | `min(1, total_toxic / 10)` với `total_toxic = n_hate + n_offensive` | càng nhiều câu độc hại chứa term càng chắc; đủ 10 câu là bão hoà | 0-1 |",
    "| `margin` | `abs(share_hate - 0.5) * 2` | nghiêng hẳn về một lớp -> chắc; `share = 0.5` (lưỡng lự) -> 0 | 0-1 |",
    "| `lift_score` | `log1p(min(lift, 100)) / log1p(100)` | lift càng lớn càng chắc nhưng nén bằng log, trần 100 | 0-1 |",
    "",
    "Trọng số `0.4 / 0.3 / 0.3` (`confidence_weights`) và các mốc `10` / `100` đều đổi được trong `LabelingConfig`.",
    "",
    "**Ví dụ.** Term có `n_hate = 12`, `n_offensive = 2`, `n_clean = 1` (`total_toxic = 14`), `share_hate ~ 0.9`, "
    "`lift ~ 40`:",
    "",
    "- `ev = min(1, 14/10) = 1.000` -> `0.4 * 1.000 = 0.400`",
    "- `margin = |0.9 - 0.5| * 2 = 0.800` -> `0.3 * 0.800 = 0.240`",
    "- `lift_score = log1p(40) / log1p(100) = 0.805` -> `0.3 * 0.805 = 0.242`",
    "- `value = 0.400 + 0.240 + 0.242 = 0.881` -> **`confidence = 0.881`**",
    "",
    "**Bước 3 - dùng ở đâu.**",
    "",
    "- Vào `match_score` của rule engine: `severity x confidence x match_factor x context_factor` (plan §39);",
    "- Xếp loại entry (`review_status`): `confidence >= 0.7` **và** `total_toxic >= 10` -> `verified`; "
    "`>= 0.4` -> `auto`; còn lại -> `needs_review`.",
    "",
    "Lưu ý: `severity` **không** tính từ dữ liệu - đây là bảng cố định theo nhãn (`HATE = 0.90`, "
    "`OFFENSIVE = 0.60`); chỉ `confidence` mới suy ra từ bằng chứng train.",
)


def _specs() -> Dict[str, FolderDoc]:
    """Return the folder registry (built once, kept as a plain dict)."""
    return {
        # ---------------- project / source ---------------- #
        ".": FolderDoc(
            title="AIModerationContent - SensitiveAI",
            purpose=(
                "Thư mục gốc của dự án phân loại ngôn tữ độc hại tiếng Việt (HATE / OFFENSIVE / CLEAN) "
                "trên ViHSD, so sánh 4 hệ thống dùng chung backbone `microsoft/mdeberta-v3-base`.\n\n"
                "`README.md` là tài liệu chính để mentor đọc. File này (`explain.md`) chỉ giải thích "
                "cấu trúc thư mục."
            ),
            produced_by="Không sinh ra - đây là thư mục gốc của kho mã.",
            contents=(
                ("README.md", "Tài liệu chi tiết toàn diện: bài toán, 4 hệ thống, dữ liệu, chiến lược train, lệnh chạy, cấu trúc output, quy chuẩn công bằng, khắc phục sự cố."),
                ("plan.md", "Kế hoạch nghiên cứu gốc. Mọi quyết định thiết kế trong mã đều truy về đây."),
                ("requirements.txt", "Dependency và phiên bản tối thiểu."),
                ("explain.md", "File này."),
            ),
            children=(
                ("configs/", "YAML: data split + 4 cấu hình model + 2 cấu hình rule engine."),
                ("src/", "Mã nguồn Python (43 module)."),
                ("scripts/", "5 entry point chạy theo thứ tự 00 -> 05."),
                ("data/", "Dữ liệu ViHSD: raw, processed, splits."),
                ("outputs/", "Toàn bộ kết quả chính thức: reports, experiments, evaluation, paper."),
                ("outputs_smoke/", "Toàn bộ kết quả smoke test (nếu gom vào thư mục con)."),
                ("checkpoints/", "Kho checkpoint dự trữ."),
            ),
            reading_guide=(
                "Người mới bắt đầu: đọc `README.md` mục 1 và mục 9.",
                "Muốn hiểu code: `src/` (mỗi subfolder có `explain.md` riêng).",
                "Muốn xem log train: `outputs/reports/<model>/run_XXX/explain.md`.",
            ),
            keep_policy="Thư mục gốc không được xóa. Các thư mục con sinh ra khi chạy có quy tắc giữ riêng.",
        ),
        "configs": FolderDoc(
            title="configs - cấu hình YAML",
            purpose=(
                "Nguồn sự thật cho mọi tham số thực nghiệm: chia dữ liệu và 4 biến thể mô hình. "
                "Mỗi lần train đều lưu bản đã resolve vào `reports/<model>/run_XXX/config.yaml`."
            ),
            produced_by="Chỉnh tay. Đọc bằng `src/common/config.py`; ghi đè tạm thời bằng cờ `--set`.",
            contents=(
                ("data_split.yaml", "Nguồn dữ liệu, chiến lược split (official/stratified), tỉ lệ, seed, và **toàn bộ quy tắc làm sạch** văn bản."),
                ("sensitiveai-vi.yaml", "Model 1 - baseline: standard `[CLS]` head, cross-entropy, 1 learning rate, 6 epoch, quyết định bằng argmax."),
                ("sensitiveai-vi-custom.yaml", "Model 2 - giống baseline, **chỉ thay head** bằng multi-view pooling, 6 epoch."),
                ("sensitiveai-vi-customdlr2stage.yaml", "Model 3 - custom head + DLR 4 nhóm + 2 stage (CE -> focal) + class weight + threshold, 7 epoch."),
                ("sensitiveai-vi-custom-curriculum.yaml", "Model 4 - custom head + DLR + curriculum 3 stage (OFFENSIVE -> HATE -> CE) + class weight manual + threshold, 10 epoch."),
                ("rule_engine.yaml", "Cấu hình rule engine: chuẩn hoá, view chống ngụy trang, ngưỡng khớp và trọng số điểm (plan §37-39). Mọi trọng số là giả thuyết ban đầu, tune trên validation. Baseline hiện tại với lexicon 668 entry: validation macro-F1 0.4555 (thresholds [0.85, 0.0, 0.41]) -> test 0.4536 (đo 1 lần)."),
                ("rule_engine_tuned.yaml", "Bản tune của rule engine (plan §38-39, §44) do `scripts/27_tune_rule_weights.py` sinh: coordinate descent 155 trial trên **validation** với lexicon 668 entry, đổi 3 trọng số - `match_factors.accentless_alias` 0.90 -> 0.00 (view không dấu khớp nhầm âm tiết phổ biến, plan §38), `match_factors.obfuscated_combined` 0.60 -> 0.45, `negation_factor` 0.50 -> 1.00. Validation macro-F1 0.4555 -> 0.5481; test chấm một lần: 0.5335. Đánh đổi: recall `no_accent` 0.530 -> 0.3625. Không ghi đè bản gốc."),
                ("explain.md", "File này."),
            ),
            keep_policy="**KHÔNG SỬA** sau khi đã chạy thật. Muốn thử biến thể mới thì copy sang file mới.",
        ),
        "scripts": FolderDoc(
            title="scripts - các entry point",
            purpose=(
                "5 script chạy theo thứ tự tăng dần: chuẩn bị dữ liệu -> kiểm tra môi trường -> train -> "
                "đánh giá -> so sánh -> sinh tài liệu folder. Mỗi script chỉ là lớp vỏ mỏng gọi hàm trong `src/`."
            ),
            produced_by="Chỉnh tay.",
            contents=(
                ("00_prepare_data.py", "Tải + làm sạch + tách split ViHSD, kiểm tra SHA-256. Chạy 1 lần đầu tiên."),
                ("01_train_all.py", "Train lần lượt cả 4 model. Mỗi model một `run_XXX` riêng."),
                ("02_evaluate_all.py", "Chấm test cho các run đã train, tune threshold trên validation, đo tốc độ."),
                ("03_compare_models.py", "Gộp kết quả 4 model thành bảng CSV/Markdown, biểu đồ, và file LaTeX cho paper."),
                ("04_check_setup.py", "Kiểm tra môi trường, split, config, và forward pass. **Không** train."),
                ("05_document_folders.py", "Sinh lại `explain.md` cho toàn bộ cây thư mục + bảng liệt kê các run."),
                ("12_build_lexicon.py", "Tải 2 từ điển từ ngữ nhạy cảm từ GitHub về `data/raw/lexicon/` rồi trích xuất danh sách từ đã chuẩn hoá vào `data/processed/lexicon/`."),
                ("13_extract_vihsd_words.py", "Quét ViHSD, trích từ/cụm từ xuất hiện nhiều ở HATE+OFFENSIVE và gần như không ở CLEAN, gộp vào lexicon."),
                ("14_fetch_additional_datasets.py", "Tải 5 dataset tiếng Việt cùng nhãn CLEAN/OFFENSIVE/HATE về `data/raw/additional/` + file giải thích nhãn `NOTES.md` (ViCM tải từ GitHub, VLSP2019 ký form rồi tự chép file vào)."),
                ("15_extract_corpus_words.py", "Quét cả 3 dataset (ViHSD/ViCM/bontori), trích ứng viên từ/cụm từ nhạy cảm ra `data/processed/lexicon/sources/sensitive_words_from_datasets.txt`."),
                ("16_label_lexicon.py", "Gán nhãn HATE/OFFENSIVE cho từng ứng viên dựa bằng chứng trên **train split** và áp cổng lọc nhiễu TF-IDF, ghi `verified_lexicon.csv`, `label_mapping.yaml`, `provenance.json` (gốc) + `labeling/candidates_train.csv`, `labeling/review_queue.csv` (plan §36)."),
                ("17_evaluate_rule_engine.py", "Chấm rule engine trên validation (tune threshold) / test (threshold đóng băng), xuất metrics, confusion matrix, phân tích FP/FN, latency, robustness (plan §39, §44)."),
                ("18_smoke_test_rule_engine.py", "Smoke test nhanh rule engine (không cần GPU): nạp config/lexicon, kiểm tra offset map + giữ dấu, khớp exact, phân loại biến đổi, xử lý text rỗng/None/rất dài. `--config` (mặc định `configs/rule_engine.yaml`) để chọn từ điển, `--split` (validation/test/train/all) và `--limit 0` chạy end-to-end trên một phần hoặc toàn bộ ViHSD (argmax, không tune threshold); ghi kết quả vào `outputs_smoke/evaluation_smoke/rule_engine/`."),
                ("19_plot_rule_engine.py", "Trực quan hoá kết quả rule engine (plan §39, §44): đọc artifact của script 17, chấm lại bằng threshold đã đóng băng (không tune) và vẽ confusion matrix (thô + chuẩn hoá), per-class F1, PR/ROC, đường tune threshold, robustness (thêm baseline `clean` + mức giảm), histogram latency; thêm `rule_scores.csv` per-sample cho bước fusion. Ghi PNG vào `outputs/evaluation/rule_engine/` và `.../test/`."),
                ("20_prune_lexicon.py", "Chấm lại **toàn bộ** từ điển đã đóng băng bằng cổng TF-IDF trên train split và ghi bản đã lọc `pruning/verified_lexicon_pruned.csv` + audit từng entry `pruning/lexicon_prune_audit.csv`; tách thêm bản theo từng dataset (`pruning/verified_lexicon_pruned_<dataset>.csv`, hợp của chúng = file gộp); `--in-place` ghi đè `verified_lexicon.csv` (backup ở `pruning/verified_lexicon_preprune.csv`). Dùng để loại nốt nhiễu lọt qua lúc gán nhãn (plan §36)."),
                ("21_sweep_tfidf.py", "Quét lưới ngưỡng TF-IDF (`min_ratio`, `min_score`) trên **validation**: mỗi điểm dựng lexicon đã lọc rồi chấm rule engine và ghi macro-F1 vào `outputs/evaluation/rule_engine_tfidf_sweep/`; không bao giờ dùng test để tune (plan §36, §44)."),
                ("22_relabel_lexicon.py", "**Phân nhãn lại** tập đã lọc: đọc `pruning/verified_lexicon_pruned.csv`, chạy lại thuật toán gán nhãn (share/lift/confidence, bằng chứng gộp mọi train split) trên đúng các term đó, ghi từ điển mới `relabeling/verified_lexicon_relabeled.csv` + bản theo từng dataset `relabeling/verified_lexicon_relabeled_<dataset>.csv` + audit trước/sau `relabeling/lexicon_relabel_changes.csv`; cập nhật mục `relabeling` trong `lexicon_report.json` (plan §36)."),
                ("26_rule_significance.py", "**Kiểm định chênh lệch giữa 2 lần chạy rule engine** trên cùng test split: paired bootstrap (macro-F1/accuracy, CI 95%, p hai phía) + McNemar đúng/sai từng mẫu, so hàng theo `index` + `y_true` (không khớp là dừng). Input là `predictions.csv` đã chấm sẵn nên **không** tạo lần đo test mới; ghi JSON qua `--out` (mặc định `outputs/evaluation/rule_significance.json`) + bảng LaTeX `papers/tables/rule_significance.tex` (plan §43-44)."),
                ("27_tune_rule_weights.py", "**Tune trọng số scoring của rule engine** (`severity`, `match_factors`, `negation/quote/context_required_factor`, `aggregation_cap`) bằng coordinate descent trên **validation**; mỗi ứng viên tune lại threshold (macro-F1), mọi trial ghi vào `outputs/evaluation/rule_engine_tune/trials.csv`, cấu hình thắng ghi `configs/rule_engine_tuned.yaml`. Không bao giờ đọc test (plan §39, §44)."),
                ("28_compare_rule_engines.py", "**So sánh 2 rule engine đã đóng băng** (baseline vs tuned): đọc `metrics.json`/`robustness.json`/`latency.json`/`thresholds.json` của cả hai run ở validation + test, diff cấu hình từ YAML, ghép báo kiểm định của script 26; ghi `outputs/evaluation/comparison/rule_engine/rule_engine_baseline_vs_tuned.md` + 3 hình (macro-F1, per-class F1, robustness). Không chấm lại split nào (plan §44)."),
                ("29_plot_fusion_comparison.py", "**Hình so sánh fusion với AI-only**: đọc `metrics.json` từng model + `fusion_significance.json` đã có và sinh 4 hình vào `outputs/evaluation/fusion/` (macro-F1 + Δ, forest plot CI paired bootstrap, Δ của 4 biến thể fusion, F1 từng lớp). Chỉ đọc artifact đóng băng, không chấm lại split nào (plan §44)."),
                ("30_plot_system_architecture.py", "**Sơ đồ kiến trúc toàn hệ thống** cho bài báo: vẽ nhánh AI (mDeBERTa-v3-base + multi-view head → xác suất) + nhánh rule (normalise → lexicon 668 term → điểm rule) hợp nhất bằng fusion `p' = p + β·r` (AI giữ trọng số 1, rule hạ trọng số β) rồi ngưỡng per-channel đóng băng; ghi `papers/figures/system_architecture.png`. Chỉ vẽ, không đọc dữ liệu thực nghiệm nào."),
                ("31_ensemble_experiment.py", "**Thí nghiệm ensemble**: trung bình xác suất của 3 model đa dạng (`ens3`) hoặc cả 6 (`ens6`), có/không cộng `beta·rule_scores`; β và ngưỡng chọn trên **validation**, 4 ứng viên đóng băng rồi chấm test **một lần**; kiểm định paired bootstrap + McNemar của ứng viên chính (chọn theo val) và phần post-hoc cho ens6; ghi `outputs/evaluation/ensemble/` (plan §43-44)."),
                ("32_plot_ensemble_comparison.py", "**Hình so sánh ensemble**: đọc `ensemble_metrics.json` + `significance.json` + số liệu 2 hệ thống tham chiếu (curriculum fusion, dlr2stage) và sinh 4 hình vào `outputs/evaluation/ensemble/` (bar test macro-F1, val vs test đảo chiều, forest plot CI paired bootstrap, F1 từng lớp). Chỉ đọc artifact đóng băng, không chấm lại split nào (plan §44)."),
                ("README.md", "Hướng dẫn dùng từng script."),
                ("explain.md", "File này."),
            ),
        ),
        "src": FolderDoc(
            title="src - mã nguồn",
            purpose=(
                "Toàn bộ logic của dự án, chia theo vai trò. Không có file nào ở đây được sinh ra khi chạy, "
                "và không có script nào ghi vào thư mục này."
            ),
            produced_by="Không sinh ra - mã nguồn.",
            contents=(("explain.md", "File này."),),
            children=(
                ("common/", "Hằng số nhãn, nạp config, quy ước đường dẫn, seed, logging, thông tin môi trường, và registry `explain.md`."),
                ("preprocessing/", "Làm sạch văn bản (12 bước, không xóa thông tin)."),
                ("data/", "Tải, tiền xử lý, tách split có fingerprint, và dataset đã tokenize."),
                ("models/", "Bọc encoder + standard head / custom multi-view head, lưu & nạp checkpoint."),
                ("losses/", "Cross-entropy, focal loss, class weight, concept-focused curriculum loss."),
                ("callbacks/", "Ghi CSV từng epoch, theo dõi best checkpoint, early stopping, điều phối stage."),
                ("training/", "Optimizer theo nhóm DLR, kế hoạch stage, vòng lặp train, CLI `run_experiment`."),
                ("evaluation/", "Chỉ số, đường cong, tìm threshold, benchmark tốc độ, biểu đồ, CLI đánh giá."),
                ("rule_engine/", "Rule engine từ vựng đứng độc lập: chuẩn hoá giữ offset, view chống ngụy trang, khớp từ điển và chấm điểm có bằng chứng (plan §35-39)."),
                ("comparison/", "Gộp 4 model và sinh bảng + LaTeX cho paper."),
            ),
            keep_policy="**KHÔNG XÓA.** Sửa mã ở đây không ảnh hưởng tới log đã tạo.",
        ),
        "src/common": FolderDoc(
            title="common - hằng số, config, đường dẫn",
            purpose="Tầng nền tảng: ánh xạ nhãn cố định, nạp/merge YAML, quy ước `run_XXX`, seed, logging và registry tài liệu folder.",
            produced_by="Không sinh ra - mã nguồn.",
            contents=(
                ("constants.py", "Nhãn cố định `HATE=0, OFFENSIVE=1, CLEAN=2`, tên 4 model, đường dẫn gốc."),
                ("config.py", "Nạp YAML, merge override `--set`, kiểm tra kiểu, ghi `config.yaml`/`dump_json`."),
                ("paths.py", "Quy ước `reports/<model>/run_XXX`, `experiments/<model>/run_XXX`, con trỏ run mới nhất, `find_best_report_dir()`."),
                ("seeding.py", "Seed cho torch/numpy/random/PYTHONHASHSEED + khởi tạo worker DataLoader."),
                ("logging_utils.py", "Logger dùng chung, ghi ra console **và** file (`train.log`, `evaluate.log`)."),
                ("env_info.py", "Ghi lại phiên bản thư viện, GPU, hệ điều hành vào báo cáo để tái lập."),
                ("folder_docs.py", "**Registry `explain.md`**: định nghĩa nội dung mọi folder và bộ sinh file giải thích."),
                ("explain.md", "File này."),
            ),
        ),
        "src/preprocessing": FolderDoc(
            title="preprocessing - làm sạch văn bản",
            purpose=(
                "Chuẩn hoá văn bản tiếng Việt trước khi mã hoá, theo nguyên tắc **không thao tác nào xóa thông tin được bật mặc định**: "
                "mọi tuỳ chọn nguy hiểm (xoá URL, xoá mention, gộp dấu câu lặp) đều tắt mặc định."
            ),
            produced_by="Không sinh ra - mã nguồn.",
            contents=(
                ("text_cleaning.py", "12 bước: hạ chữ Unicode, NFC, giải thực thể HTML, bỏ ký tự vô hình, bỏ thẻ HTML, chuẩn hoá space, sửa khoảng trắng quanh dấu câu, gộp space, cắt đầu/cuối. Chỉ loại dòng mà **không còn ký tự nhìn thấy nào**."),
                ("explain.md", "File này."),
            ),
        ),
        "src/data": FolderDoc(
            title="data - dữ liệu",
            purpose="Tải ViHSD, tiền xử lý, tách split có fingerprint SHA-256, và tạo dataset đã tokenize cho PyTorch.",
            produced_by="Không sinh ra - mã nguồn. Kết quả của nó nằm ở `data/` (thư mục dữ liệu).",
            contents=(
                ("download.py", "Tải `uitnlp/vihsd` từ Hugging Face về `data/raw/vihsd/`."),
                ("lexicon.py", "Tải 2 lexicon GitHub về `data/raw/lexicon/`, chuẩn hoá (NFC, hạ chữ, `_`→space) và ghi danh sách từ nhạy cảm + `lexicon_report.json` (SHA-256) vào `data/processed/lexicon/`."),
                ("vihsd_lexicon.py", "Quét ViHSD đã làm sạch, đếm document frequency theo nhóm toxic/clean, trích từ/cụm từ mới vào lexicon (ghép `vihsd_extracted_words.txt` + cập nhật report)."),
                ("additional.py", "Tải 5 dataset cùng nhãn CLEAN/OFFENSIVE/HATE (3 từ Hugging Face, ViCM từ GitHub, VLSP2019 tải thủ công) về `data/raw/additional/`, đếm nhãn từng file (áp `label_map` nếu mã hoá khác), ghi `NOTES.md` từng dataset + `additional_datasets_report.json` tổng hợp."),
                ("preprocess.py", "Làm sạch + remap nhãn canonical + ghi `preprocessing_report.json`."),
                ("split.py", "Tạo split, tính SHA-256 từng file, kiểm tra toàn vẹn, ghi `split_manifest.json`."),
                ("dataset.py", "Tokenize `train/validation/test`, `TokenizedTextDataset`, `collate_fn` an toàn Windows (num_workers>0)."),
                ("corpus_lexicon.py", "Trích ứng viên từ vựng từ các split train (ViHSD/ViCM/bontori), đếm document frequency toxic/clean, ghi `sources/sensitive_words_from_datasets.txt` đầu vào cho bộ gán nhãn."),
                ("lexical_labeler.py", "Gán nhãn HATE/OFFENSIVE cho từng ứng viên dựa bằng chứng train-only (share/lift/confidence), áp cổng lọc nhiễu TF-IDF, ghi `verified_lexicon.csv`, `label_mapping.yaml`, `provenance.json` (gốc) + `labeling/candidates_train.csv`, `labeling/review_queue.csv` (plan §36)."),
                ("tfidf_filter.py", "Bộ lọc nhiễu TF-IDF thuần (không I/O) cho ứng viên lexicon: IDF làm mượt, trọng số TF-IDF theo lớp, tỉ lệ toxic và điểm tách toxic-vs-clean; quyết định giữ/loại (plan §36)."),
                ("lexicon_pruner.py", "Chấm lại **từ điển đã đóng băng** bằng cổng TF-IDF trên train split, ghi `pruning/verified_lexicon_pruned.csv` (gộp) + bản theo từng dataset `pruning/verified_lexicon_pruned_<dataset>.csv` + audit `pruning/lexicon_prune_audit.csv` và cập nhật `lexicon_report.json`; CLI cho `scripts/20_prune_lexicon.py` (plan §36)."),
                ("lexicon_relabeler.py", "**Phân nhãn lại** tập đã lọc: chạy lại thuật toán gán nhãn (share/lift/confidence, bằng chứng gộp train split) trên `pruning/verified_lexicon_pruned.csv`, ghi `relabeling/verified_lexicon_relabeled.csv` (gộp) + bản theo từng dataset `relabeling/verified_lexicon_relabeled_<dataset>.csv` + audit trước/sau `relabeling/lexicon_relabel_changes.csv`, cập nhật mục `relabeling` của `lexicon_report.json`; CLI cho `scripts/22_relabel_lexicon.py` (plan §36)."),
                ("prepare_data.py", "CLI cho `scripts/00_prepare_data.py`."),
                ("explain.md", "File này."),
            ),
            details=_CONFIDENCE_DETAIL,
        ),
        "src/models": FolderDoc(
            title="models - kiến trúc mô hình",
            purpose=(
                "Bọc `microsoft/mdeberta-v3-base` với hai loại head: standard `[CLS]` (model 1) và custom multi-view "
                "cls+mean+max (model 2/3/4). Cùng một đường nạp/lưu cho cả hai để so sánh công bằng."
            ),
            produced_by="Không sinh ra - mã nguồn.",
            contents=(
                ("model.py", "`StandardHeadModel` và `CustomHeadModel`; ép checkpoint về fp32; save/load đúng định dạng cho từng loại head."),
                ("custom_head.py", "Custom head: 3 view (cls / mean / max) -> LayerNorm -> fusion MLP 2304->512 -> Linear 512->3."),
                ("explain.md", "File này."),
            ),
        ),
        "src/losses": FolderDoc(
            title="losses - hàm mất mát",
            purpose="Một nơi duy nhất để dựng loss từ YAML, gồm cross-entropy, focal loss (alpha theo nhãn), class weight và concept-focused curriculum loss.",
            produced_by="Không sinh ra - mã nguồn.",
            contents=(
                ("builder.py", "`build_loss()` - đọc `loss:` trong YAML rồi trả về module loss tương ứng (cross_entropy / focal / concept)."),
                ("focal.py", "Focal loss `FL(p_t) = -alpha_t (1-p_t)^gamma log(p_t)`, `alpha` có thể khác nhau theo từng nhãn."),
                ("class_weights.py", "5 chiến lược cân bằng lớp (none / inverse_frequency / inverse_sqrt_frequency / effective_number / manual), luôn ước lượng trên **train**."),
                ("curriculum.py", "Concept-focused loss: một nhãn là dương, hai nhãn còn lại là âm; softmax giữ thứ tự giữa hai lớp âm."),
                ("explain.md", "File này."),
            ),
        ),
        "src/callbacks": FolderDoc(
            title="callbacks - callback huấn luyện",
            purpose="Các bước gắn vào vòng lặp train: ghi CSV từng epoch, lưu best checkpoint, early stopping, điều phối chuyển stage.",
            produced_by="Không sinh ra - mã nguồn.",
            contents=(
                ("csv_logger.py", "`CSVLoggerCallback` - ghi `training_log.csv`, **một dòng mỗi epoch**, có cột `is_best`. Không vẽ lại từ RAM."),
                ("monitoring.py", "`BestCheckpointCallback` + `EarlyStoppingCallback` - chọn epoch tốt nhất theo **validation** macro F1."),
                ("stage_controller.py", "Điều phối stage: nạp `init_from`, ghi checkpoint stage, ghi `stage_transitions.json`."),
                ("base.py", "Giao diện callback chung."),
                ("explain.md", "File này."),
            ),
        ),
        "src/training": FolderDoc(
            title="training - quy trình huấn luyện",
            purpose=(
                "Dựng optimizer theo nhóm DLR, lập kế hoạch stage, chạy vòng lặp train tùy chỉnh (không dùng Hugging Face `Trainer` vì API đổi ở transformers 5.x) và cung cấp CLI."
            ),
            produced_by="Không sinh ra - mã nguồn. Kết quả nằm ở `reports/` và `experiments/`.",
            contents=(
                ("optim.py", "Chia tham số thành 4 nhóm (embeddings / encoder LN+rel / block dưới / block trên / head), tách weight decay cho bias-LayerNorm, dựng scheduler."),
                ("stages.py", "`resolve_stages()` - merge cấu hình global với override theo từng stage."),
                ("trainer.py", "Vòng lặp train: gradient accumulation, clip grad norm, evaluate mỗi epoch, callback, lưu `last_model`, viết `training_summary.json`."),
                ("run_experiment.py", "CLI train một model. Tự cấp `run_XXX` kế tiếp, ghi `train.log`, và **từ chối ghi đè** một run đã có dữ liệu."),
                ("explain.md", "File này."),
            ),
        ),
        "src/evaluation": FolderDoc(
            title="evaluation - đánh giá",
            purpose=(
                "Chấm điểm model đã huấn luyện. Nguyên tắc bất di bất dịch: **chỉ chấm test đúng một lần**, "
                "tìm threshold **trên validation**, dùng class weight đã đóng băng từ lúc train."
            ),
            produced_by="Không sinh ra - mã nguồn. Kết quả nằm trong `reports/<model>/run_XXX/` và `evaluation/`.",
            contents=(
                ("metrics.py", "Accuracy, macro/weighted F1, per-class P/R/F1, macro & per-class ROC-AUC và average precision, ma trận nhầm lẫn."),
                ("thresholds.py", "Tìm threshold riêng từng nhãn bằng **coordinate ascent trên validation**, mục tiêu mặc định macro F1."),
                ("benchmark.py", "Đo latency (ms/sample) và throughput (samples/s) với warmup, không tính thời gian warmup vào kết quả."),
                ("curves.py", "Chuỗi dữ liệu cho PR/ROC curve."),
                ("plots.py", "Vẽ ma trận nhầm lẫn, PR/ROC, per-class F1, đường cong validation theo epoch."),
                ("tfidf_sweep.py", "Quét lưới ngưỡng TF-IDF trên validation: chấm lại lexicon cho từng `(min_ratio, min_score)`, tune threshold, ghi macro-F1 và chọn cấu hình tốt nhất (plan §36, §44); CLI cho `scripts/21_sweep_tfidf.py`."),
                ("report.py", "`model_summary.txt` và khối thông tin môi trường để tái lập."),
                ("run_evaluation.py", "CLI đánh giá một model; **lưu trữ** kết quả cũ vào `history/` thay vì xóa."),
                ("explain.md", "File này."),
            ),
        ),
        "src/rule_engine": FolderDoc(
            title="rule_engine - rule engine từ vựng",
            purpose=(
                "Bộ phát hiện từ vựng **đứng độc lập** với model AI (plan §41): chuẩn hoá văn bản nhưng giữ nguyên "
                "offset về câu gốc, dựng view đối chiếu chống ngụy trang, khớp từ điển đã gán nhãn và cho ra điểm "
                "`rule_hate_score` / `rule_offensive_score` kèm bằng chứng giải thích được (span, từ điển, loại khớp, "
                "hệ số tin cậy). Đây là đầu vào cho bước fusion AI + rule ở giai đoạn sau (plan §40)."
            ),
            produced_by="Không sinh ra - mã nguồn. Kết quả đánh giá nằm ở `evaluation/rule_engine/`.",
            contents=(
                ("config.py", "`NormalizeConfig` / `ObfuscationConfig` / `MatcherConfig` / `ScoringConfig` / `RuleEngineConfig`, nạp từ `configs/rule_engine.yaml` và kiểm tra hợp lệ."),
                ("normalize.py", "Chuẩn hoá NFC + hạ chữ + bỏ ký tự vô hình + gộp space (kèm giới hạn lặp ký tự), **giữ offset map** về câu gốc; không xóa dấu tiếng Việt (plan §37)."),
                ("obfuscation.py", "Dựng view `base` / `plain` (không dấu + leet + gộp lặp) / `compact` (bỏ dấu cách, dấu câu) và phân loại biến đổi thành `match_type` + `transformation_trace` (plan §38)."),
                ("lexical_classifier.py", "Nạp `verified_lexicon.csv` thành `Lexicon` (chỉ dùng entry có `review_status` được cho phép) với tra cứu từ nhanh."),
                ("matcher.py", "Khớp `exact` / `accentless_alias` / `leet` / `repeat_collapsed` / `separator_inserted` / `obfuscated_combined` / `fuzzy` (Levenshtein có chặn), khử trùng span và ghi đầy đủ bằng chứng."),
                ("scoring.py", "Điểm `severity x confidence x match_factor x context_factor` (phủ định, trích dẫn, `context_required`) rồi gộp mỗi nhãn bằng `1 - prod(1 - s_i)` (plan §39)."),
                ("engine.py", "`RuleEngine.analyze()` gom toàn bộ pipeline và trả về evidence dict; `from_config` / `from_lexicon` để dựng nhanh."),
                ("explain.md", "File này."),
            ),
        ),
        "src/fusion": FolderDoc(
            title="src/fusion - mã nguồn hybrid fusion (rule engine + AI)",
            purpose=(
                "Ghép điểm rule engine với xác suất AI (plan §40-41): quét `alpha` (trọng số AI), per-channel alpha, "
                "tune threshold và huấn luyện meta-classifier - **tất cả trên validation**, đóng băng rồi chấm test một "
                "lần (plan §44). Không tự chứa tham số: đọc `configs/hybrid_fusion.yaml`."
            ),
            produced_by="Không sinh ra - mã nguồn. Chạy qua `python scripts/23_evaluate_fusion.py`; kết quả ở `evaluation/fusion/`.",
            contents=(
                ("run_fusion.py", "Điều phối toàn bộ: tìm run/checkpoint AI, dùng lại `per_model/<model>/{validation,test}_predictions.csv`, ghép với `rule_scores.csv`, chọn alpha/threshold/meta trên validation, ghi báo cáo + `fusion_significance` bước sau."),
                ("weighted_fusion.py", "Hàm thuần: `score_alpha_grid`, per-channel alpha, `tune_thresholds_at_alpha(s)`, `MetaFusion` (logistic regression trên 6 điểm rule/AI)."),
                ("__init__.py", "Export công khai của package."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Muốn hiểu 'alpha' ảnh hưởng gì: đọc `score_alpha_grid` trong `weighted_fusion.py` (1.0 = chỉ AI, 0.0 = chỉ rule).",
                "Kết quả và kiểm định: `evaluation/fusion/significance_report.md`.",
            ),
        ),
        "src/comparison": FolderDoc(
            title="comparison - so sánh 4 model",
            purpose=(
                "Đọc kết quả của cả 4 run, dựng bảng tổng hợp, vẽ biểu đồ so sánh, và **sinh tự động** file LaTeX "
                "(`model_comparison.tex`, `classification_results.tex`, `ablation_results.tex`) cho bài báo."
            ),
            produced_by="Không sinh ra - mã nguồn. Kết quả nằm ở `evaluation/comparison/` và `paper/tables/`.",
            contents=(
                ("compare_models.py", "CLI so sánh. Bảng ablation sinh từ đúng 4 hệ thống nên trả lời trực tiếp: đổi head được bao nhiêu, thêm DLR/focal/2 stage được bao nhiêu, curriculum 3 stage hơn 2 stage bao nhiêu."),
                ("explain.md", "File này."),
            ),
        ),
        # ---------------- data ---------------- #
        "data": FolderDoc(
            title="data - dữ liệu ViHSD",
            purpose=(
                "Ba lớp dữ liệu: bản tải về nguyên bản, bản đã làm sạch, và bản split cố định dùng chung cho mọi model. "
                "Chỉ tầng `splits/` là thứ mô hình thực sự đọc."
            ),
            produced_by="`python scripts/00_prepare_data.py`",
            children=(
                ("raw/", "Dữ liệu tải về, không sửa."),
                ("processed/", "Sau làm sạch và remap nhãn."),
                ("splits/", "Split cố định + SHA-256. **Đây là nguồn sự thật cho mọi lần chạy.**"),
            ),
            reading_guide=(
                "Muốn biết dữ liệu đã qua xử lý gì: `processed/vihsd/preprocessing_report.json`.",
                "Muốn xác nhận 4 model dùng cùng dữ liệu: `splits/split_manifest.json` (SHA-256 từng file).",
            ),
        ),
        "data/raw": FolderDoc(
            title="data/raw - dữ liệu tải về",
            purpose="Chỗ tải ViHSD về lần đầu. Giữ nguyên để có thể chạy lại tiền xử lý từ đầu.",
            produced_by="`python scripts/00_prepare_data.py --stage download`",
            children=(
                ("vihsd/", "3 file CSV gốc của ViHSD: `train.csv`, `dev.csv`, `test.csv`."),
                ("lexicon/", "2 bộ từ điển từ ngữ nhạy cảm tải từ GitHub (không sửa)."),
                ("additional/", "5 dataset tiếng Việt cùng nhãn CLEAN/OFFENSIVE/HATE (3 từ Hugging Face, ViCM từ GitHub, VLSP2019 thủ công - không sửa)."),
            ),
            keep_policy="**BẤT BIẾN.** Được xem là dữ liệu đầu vào gốc. Không sửa tay, không ghi đè bằng bản đã làm sạch.",
        ),
        "data/raw/vihsd": FolderDoc(
            title="data/raw/vihsd - ViHSD nguyên bản",
            purpose="File CSV gốc do nhóm UIT cung cấp. Tổng 33.400 dòng (24.048 / 2.672 / 6.680).",
            produced_by="Tải từ Hugging Face `uitnlp/vihsd` hoặc GitHub `sonlam1102/vihsd`.",
            contents=(
                ("train.csv", "24.048 dòng, cột `free_text`, `label`, `label_ids`."),
                ("dev.csv", "2.672 dòng - dùng làm validation, giữ nguyên số lượng gốc."),
                ("test.csv", "6.680 dòng - giữ nguyên tuyệt đối, không loại mẫu nào vì tiền xử lý."),
                ("explain.md", "File này."),
            ),
            keep_policy="**BẤT BIẾN - KHÔNG BAO GIỜ SỬA.** Đây là mốc đối chiếu để chứng minh tiền xử lý không làm mất dữ liệu.",
        ),
        "data/raw/lexicon": FolderDoc(
            title="data/raw/lexicon - từ điển từ nhạy cảm nguyên bản",
            purpose=(
                "2 bộ lexicon tiếng Việt tải từ GitHub, giữ nguyên từng byte để có thể trích xuất lại từ đầu: "
                "profanity + teencode + từ nhạy cảm chính trị (behitek) và offensive words có biến thể "
                "không dấu / teencode (blue-eyes-vn)."
            ),
            produced_by="`python scripts/12_build_lexicon.py --stage download`",
            children=(
                ("behitek-vietnam-sensitive-words/", "`bad_words.json`, `bad_words_22082018.json`, `hide.txt` từ github.com/behitek/vietnam-sensitive-words."),
                ("blue-eyes-vn-vietnamese-offensive-words/", "`vn_offensive_words.txt` (+ bản copy định dạng array, README, LICENSE) từ github.com/blue-eyes-vn/vietnamese-offensive-words."),
            ),
            keep_policy="**BẤT BIẾN.** Dữ liệu đầu vào gốc - không sửa tay, không ghi đè.",
        ),
        "data/raw/additional": FolderDoc(
            title="data/raw/additional - 5 dataset cùng nhãn với ViHSD",
            purpose=(
                "Năm bộ dữ liệu tiếng Việt đã xác minh dùng **đúng** bộ nhãn CLEAN / OFFENSIVE / HATE như ViHSD: "
                "ViHSD-Explainable (thêm evidence spans), R-ViHSD (nhiễu robustness) và Vietnamese-HSD-cleaned "
                "(ViHSD + 62K mẫu mới từ TDTU YouTube/voz); thêm ViCM (5.415 mẫu code-mixed MỚI, mã hoá "
                "`0=Clean, 1=Hate, 2=Offensive` - PHẢI remap) và VLSP2019-HSD (25.431 bài Facebook MỚI, tải thủ "
                "công sau khi ký agreement form). Mỗi thư mục con có `NOTES.md` giải thích nhãn; số liệu đếm thật "
                "trên file nằm ở `additional_datasets_report.json`."
            ),
            produced_by="`python scripts/14_fetch_additional_datasets.py`",
            children=(
                ("vominhmanh-vihsd-explainable/", "jsonl train/validation/test + README. 33.400 mẫu ViHSD + explanation + evidence."),
                ("leminhhung0101-r-vihsd/", "csv 4 split. Train 48.092 mẫu gồm biến thể nhiễu (cột `noise_type`)."),
                ("bontori-vietnamese-hsd-cleaned/", "parquet train/dev/test. 86.719 mẫu train gộp ViHSD + TDTU YouTube/voz."),
                ("vicm-code-mixed/", "xlsx train/dev/test + csv forum + README. 5.415 mẫu code-mixed MỚI (+480 diễn đàn, nhãn phải remap)."),
                ("vlsp2019-hsd/", "chưa có file - ký form VLSP rồi chép csv/xlsx vào đây, chạy lại `--stage verify`."),
                ("additional_datasets_report.json", "Tổng hợp: nhãn quan sát được, số dòng, cờ `labels_exact_match`, `source`, `requires_remap`, `pending_manual_download` từng dataset."),
            ),
            reading_guide=(
                "Muốn biết nhãn có khớp ViHSD không: mở `NOTES.md` trong từng thư mục con, mục 2.",
                "Muốn biết dataset nào có mẫu mới thật: `bontori-.../NOTES.md` - lọc theo cột `source`.",
                "Chỉ dùng `Real_Label` của ViCM làm nhãn; `Predicted_Label` là kết quả mô hình, không phải ground truth.",
            ),
            keep_policy="**BẤT BIẾN.** Dữ liệu đầu vào gốc - không sửa tay, không ghi đè.",
        ),
        "data/raw/additional/*": FolderDoc(
            title="data/raw/additional/<tên dataset> - file dữ liệu gốc",
            purpose="File dữ liệu tải nguyên bản (Hugging Face hoặc GitHub), kèm `NOTES.md` giải thích nhãn và quan hệ với ViHSD.",
            produced_by="`python scripts/14_fetch_additional_datasets.py`",
            reading_guide=(
                "Đọc `NOTES.md` trước khi dùng dataset - giải thích mối quan hệ với ViHSD và cách ánh xạ nhãn.",
                "Không sửa file dữ liệu - mọi xử lý nằm ở script, không ghi tay vào đây.",
            ),
            keep_policy="**BẤT BIẾN.** Nguyên bản từ upstream.",
        ),
        "data/raw/lexicon/*": FolderDoc(
            title="data/raw/lexicon/<nguồn> - file lexicon gốc",
            purpose="Các file lexicon tải trực tiếp từ repo GitHub tương ứng, không qua xử lý.",
            produced_by="`python scripts/12_build_lexicon.py --stage download`",
            reading_guide=(
                "Không đọc file này để dùng từ điển - hãy dùng bản đã chuẩn hoá ở `data/processed/lexicon/`.",
                "Muốn xác minh nguồn: đối chiếu SHA-256 trong `data/processed/lexicon/lexicon_report.json`.",
            ),
            keep_policy="**BẤT BIẾN.** Nguyên bản từ upstream.",
        ),
        "data/processed": FolderDoc(
            title="data/processed - dữ liệu sau tiền xử lý",
            purpose="Kết quả làm sạch + remap nhãn canonical. Phục vụ kiểm tra, không phải nơi mô hình đọc.",
            produced_by="`python scripts/00_prepare_data.py --stage preprocess`",
            children=(
                ("vihsd/", "CSV đã làm sạch + báo cáo tiền xử lý."),
                ("lexicon/", "Danh sách từ nhạy cảm đã chuẩn hoá + báo cáo trích xuất."),
            ),
        ),
        "data/processed/vihsd": FolderDoc(
            title="data/processed/vihsd - CSV đã làm sạch",
            purpose=(
                "Cùng số dòng với dữ liệu gốc trừ đúng 2 dòng train có text rỗng (không có gì để mã hoá). "
                "Nhãn đã chuyển sang thứ tự dự án: HATE=0, OFFENSIVE=1, CLEAN=2."
            ),
            produced_by="`python scripts/00_prepare_data.py --stage preprocess`",
            contents=(
                ("train.csv", "24.046 dòng (HATE 2.556 / OFFENSIVE 1.605 / CLEAN 19.885)."),
                ("dev.csv", "2.672 dòng (270 / 212 / 2.190)."),
                ("test.csv", "6.680 dòng (688 / 444 / 5.548) - **không mất mẫu nào**."),
                ("preprocessing_report.json", "Cấu hình làm sạch đã dùng, số dòng bị loại và lý do, phân bố nhãn từng split."),
                ("explain.md", "File này."),
            ),
            keep_policy="**Sinh lại được** từ `data/raw/vihsd`, nhưng nên giữ lại để đối chiếu.",
        ),
        "data/processed/lexicon": FolderDoc(
            title="data/processed/lexicon - danh sách từ nhạy cảm đã chuẩn hoá",
            purpose=(
                "Kết quả trích xuất từ 2 lexicon nguồn (mỗi file một từ/cụm từ trên một dòng, UTF-8, đã NFC + hạ chữ "
                "+ `_`→space + gộp space, sắp xếp và khử trùng lặp), cùng **từ điển đã gán nhãn HATE/OFFENSIVE** mà rule "
                "engine dùng để khớp (plan §36)."
            ),
            produced_by="`python scripts/12_build_lexicon.py --stage extract`, `scripts/15_extract_corpus_words.py`, `scripts/16_label_lexicon.py`, `scripts/20_prune_lexicon.py`, `scripts/22_relabel_lexicon.py`",
            contents=(
                ("verified_lexicon.csv", "**Từ điển đang dùng** (plan §36.2): `entry_id, term, canonical_form, lexical_labels, severity, confidence, context_required, source_datasets, source_split, review_status, dictionary_version`. Đây là file `configs/rule_engine.yaml` trỏ tới - chỉ entry `verified`/`auto` được khớp."),
                ("label_mapping.yaml", "Ánh xạ nhãn từ khóa -> id chuẩn (HATE=0, OFFENSIVE=1, CLEAN=2) và mô tả quy tắc gán nhãn đã dùng."),
                ("provenance.json", "Dấu vết tái lập của từ điển: tham số gán nhãn, SHA-256 file nguồn/đầu ra, phân bố nhãn và các sai lệch so với kế hoạch (deviation)."),
                ("manual_review.csv", "**Tùy chọn** - quyết định duyệt tay, merge sau khi gán nhãn tự động nếu tồn tại."),
                ("lexicon_report.json", "Báo cáo tổng hợp **của cả pipeline**: số từ từng nguồn, kết quả trích xuất, mục `tfidf_pruning` và `relabeling`, SHA-256 từng file output."),
                ("explain.md", "File này."),
            ),
            children=(
                ("sources/", "Từ vựng thô trích ra từ 2 repo GitHub + 3 dataset (đầu vào của bước gán nhãn)."),
                ("labeling/", "Bằng chứng trung gian của bước gán nhãn: bảng đếm `candidates_train.csv` + hàng chờ duyệt tay `review_queue.csv`."),
                ("pruning/", "Kết quả lọc TF-IDF: từ điển đã lọc, bản theo từng dataset, audit từng entry, backup trước khi ghi đè."),
                ("relabeling/", "Kết quả phân nhãn lại: từ điển mới, bản theo từng dataset, bảng so sánh trước/sau."),
                ("sweeps/", "Thí nghiệm quét ngưỡng TF-IDF (`pruned_scoreXXX/`) - mỗi folder một mức ngưỡng, không phải từ điển chính."),
            ),
            reading_guide=(
                "Muốn biết có bao nhiêu từ: `lexicon_report.json`, khoá `counts`.",
                "Nghi ngờ file bị đổi: so SHA-256 trong `lexicon_report.json` / `provenance.json` với file trên đĩa.",
                "Muốn lọc văn bản: đọc `sources/sensitive_words_combined.txt`, mỗi dòng một cụm từ (lowercase).",
                "Muốn xem từ điển rule engine thực dùng: `verified_lexicon.csv` (file này nằm ở gốc, không nằm trong thư mục con).",
                "Muốn biết đã lọc/phân nhãn lại những gì: vào `pruning/lexicon_prune_audit.csv` và `relabeling/lexicon_relabel_changes.csv`.",
            ),
            extra_notes=(
                "Cột `confidence` của `verified_lexicon.csv` được tính thế nào (bằng chứng train, share/lift, công thức): "
                "xem mục **Chi tiết** trong `src/data/explain.md`.",
            ),
            keep_policy="**Sinh lại được** từ `data/raw/lexicon` + 3 dataset, nhưng nên giữ lại để đối chiếu SHA-256.",
        ),
        "data/processed/lexicon/sources": FolderDoc(
            title="data/processed/lexicon/sources - từ vựng thô (đầu vào pipeline)",
            purpose=(
                "Mọi danh sách từ **chưa gán nhãn**, mỗi dòng một cụm từ, UTF-8, đã NFC + hạ chữ + gộp space, "
                "sắp xếp và khử trùng lặp. Đây là đầu vào của bước gán nhãn (step 16) và của bước quét corpus "
                "(step 15). Không file nào ở đây chứa nhãn HATE/OFFENSIVE."
            ),
            produced_by="`python scripts/12_build_lexicon.py --stage extract`, `python scripts/15_extract_corpus_words.py`",
            contents=(
                ("behitek_sensitive_words.txt", "Từ của `behitek/vietnam-sensitive-words` (profanity, teencode, từ nhạy cảm chính trị). Chỉ có khi step 12 chạy extract."),
                ("blueeyes_offensive_words.txt", "Từ của `blue-eyes-vn/vietnamese-offensive-words` (gồm biến thể không dấu / teencode). Chỉ có khi step 12 chạy extract."),
                ("sensitive_words_combined.txt", "Gộp 2 repo nguồn, khử trùng lặp - file nên dùng để quét văn bản."),
                ("vihsd_extracted_words.txt", "Ứng viên unigram/bigram trích từ ViHSD (đếm df toxic/clean trên train)."),
                ("vicm_extracted_words.txt", "Ứng viên trích từ ViCM (code-mixed)."),
                ("bontori_extracted_words.txt", "Ứng viên trích từ bontori (TDTU)."),
                ("sensitive_words_from_datasets.txt", "Gộp ứng viên từ 3 dataset - **đầu vào trực tiếp** của `16_label_lexicon.py`."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Chỉ muốn lọc văn bản thô: đọc `sensitive_words_combined.txt`.",
                "Muốn xem vì sao một từ lọt vào từ điển: tìm nó trong `../labeling/candidates_train.csv`.",
            ),
            keep_policy="**Sinh lại được** từ `data/raw/` (2 repo lexicon + 3 dataset). Giữ lại để đối chiếu SHA-256 trong `lexicon_report.json`.",
        ),
        "data/processed/lexicon/labeling": FolderDoc(
            title="data/processed/lexicon/labeling - bằng chứng trung gian của bước gán nhãn",
            purpose=(
                "Lưu **bằng chứng** mà thuật toán gán nhãn dùng, để người đọc kiểm chứng được từng quyết định "
                "gán nhãn mà không phải chạy lại pipeline. Không file nào ở đây là từ điển chính - "
                "`verified_lexicon.csv` nằm ở thư mục cha."
            ),
            produced_by="`python scripts/16_label_lexicon.py`",
            contents=(
                ("candidates_train.csv", "Một dòng cho mỗi ứng viên: số câu train chứa từ theo từng nhãn (`n_hate/n_offensive/n_clean`) + cột TF-IDF (`tf_*`, `idf`, `tfidf_*`, `tfidf_ratio`, `tfidf_score`) - chính là dữ liệu đầu vào của cổng lọc nhiễu."),
                ("review_queue.csv", "Ứng viên yếu / mơ hồ / bị cổng TF-IDF loại, kèm cột `reason` - danh sách cần người duyệt tay."),
                ("manual_review.csv", "**Tùy chọn** - quyết định duyệt tay; nếu có thì được merge vào từ điển khi step 16 chạy."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Tại sao từ X bị loại? Tìm X trong `review_queue.csv`, đọc cột `reason`.",
                "Tại sao từ X được gán HATE? Đọc bằng chứng trong `candidates_train.csv` rồi so với quy tắc ở `src/data/explain.md`.",
            ),
            keep_policy="**Sinh lại được** khi chạy lại step 16, nhưng là hồ sơ chứng minh - giữ lại.",
        ),
        "data/processed/lexicon/pruning": FolderDoc(
            title="data/processed/lexicon/pruning - kết quả lọc TF-IDF",
            purpose=(
                "Chấm lại **toàn bộ** từ điển đã đóng băng bằng cổng TF-IDF trên train split (plan §36), "
                "loại từ do CLEAN dùng nhiều / quá yếu để tách độc hại khỏi sạch. Kèm audit từng entry và "
                "bản sao theo từng dataset."
            ),
            produced_by="`python scripts/20_prune_lexicon.py` (thêm `--in-place` để ghi đè từ điển chính)",
            contents=(
                ("verified_lexicon_pruned.csv", "Từ điển sau khi lọc - file gộp, cùng schema với `verified_lexicon.csv`. Đây là input của step 22."),
                ("verified_lexicon_pruned_vihsd.csv", "Bản lọc chỉ gồm entry có `source_datasets` chứa `vihsd`."),
                ("verified_lexicon_pruned_vicm.csv", "Bản lọc chỉ gồm entry có `source_datasets` chứa `vicm`."),
                ("verified_lexicon_pruned_bontori.csv", "Bản lọc chỉ gồm entry có `source_datasets` chứa `bontori`. Hợp 3 file per-dataset = file gộp."),
                ("lexicon_prune_audit.csv", "Audit từng entry: số đếm theo nhãn, `idf`, các cột `tfidf_*`, quyết định `kept`/`dropped` và `reason`."),
                ("verified_lexicon_preprune.csv", "Backup từ điển **trước** khi `--in-place` ghi đè `../verified_lexicon.csv` - dùng để hoàn nguyên."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Tại sao từ X bị lọc? Tìm X trong `lexicon_prune_audit.csv`, đọc `decision` + `reason`.",
                "Muốn hoàn nguyên từ điển: copy `verified_lexicon_preprune.csv` thành `../verified_lexicon.csv`.",
            ),
            keep_policy="**Không xóa** - audit là hồ sơ của quyết định lọc; backup là đường hoàn nguyên duy nhất.",
        ),
        "data/processed/lexicon/relabeling": FolderDoc(
            title="data/processed/lexicon/relabeling - kết quả phân nhãn lại",
            purpose=(
                "Chạy lại thuật toán gán nhãn (share / lift / confidence, bằng chứng gộp mọi train split) "
                "trên đúng tập đã lọc, rồi ghi bảng so sánh trước/sau cho từng term (plan §36)."
            ),
            produced_by="`python scripts/22_relabel_lexicon.py`",
            contents=(
                ("verified_lexicon_relabeled.csv", "Từ điển sau phân nhãn lại - file gộp, cùng schema với `../verified_lexicon.csv`."),
                ("verified_lexicon_relabeled_vihsd.csv", "Bản chỉ gồm entry có `source_datasets` chứa `vihsd`."),
                ("verified_lexicon_relabeled_vicm.csv", "Bản chỉ gồm entry có `source_datasets` chứa `vicm`."),
                ("verified_lexicon_relabeled_bontori.csv", "Bản chỉ gồm entry có `source_datasets` chứa `bontori`. Hợp 3 file = file gộp."),
                ("lexicon_relabel_changes.csv", "Mỗi dòng một term: nhãn / `review_status` / `confidence` / `severity` **cũ và mới** + cờ `changed` + `reason`. Muốn biết thuật toán đổi gì, đọc file này."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Thuật toán có đổi nhãn gì không? Đếm cột `changed` = `true` trong `lexicon_relabel_changes.csv`.",
                "Muốn đưa từ điển này vào dùng: copy `verified_lexicon_relabeled.csv` thành `../verified_lexicon.csv`.",
            ),
            keep_policy="**Không xóa** - bảng before/after là hồ sơ chứng minh của bước phân nhãn lại.",
        ),
        "data/processed/lexicon/sweeps": FolderDoc(
            title="data/processed/lexicon/sweeps - thí nghiệm quét ngưỡng TF-IDF",
            purpose=(
                "Mỗi thư mục con là **một mức ngưỡng `tfidf_min_score`** (0.008 / 0.010 / 0.015), chứa bản từ điển "
                "lọc ở mức đó. Dùng để so sánh kích thước từ điển và chỉ số rule engine theo ngưỡng. "
                "**Không phải từ điển chính** - `configs/rule_engine.yaml` không trỏ vào đây."
            ),
            produced_by="`python tmp_prune_sweep.py` (script thí nghiệm, tạo bộ `pruned_scoreXXX/`)",
            children=(
                ("pruned_score008/", "Ngưỡng 0.008 - giữ nhiều từ nhất (~691 entry)."),
                ("pruned_score010/", "Ngưỡng 0.010 - trung gian (~576 entry)."),
                ("pruned_score015/", "Ngưỡng 0.015 - chặt nhất (~324 entry)."),
            ),
            contents=(
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Muốn so ngưỡng: mở `lexicon_report.json` của từng thư mục con, hoặc chạy `configs/rule_engine_tuned_p0XX.yaml`.",
                "Muốn tái lập một mức ngưỡng: `python scripts/20_prune_lexicon.py --tfidf-min-score 0.010 --output-dir data/processed/lexicon/sweeps/pruned_score010`.",
                "Số liệu sweep của lần đo trước nằm trong `outputs/evaluation/rule_engine_tuned_pruning_ablation_val.json` + `docs/bao_cao_tfidf_prune_ablation.md` (folder `outputs/evaluation/rule_engine_tuned_p0XX` đã được dọn ngày 19/10).",
            ),
            keep_policy="**Tùy** - các folder kết quả evaluation tương ứng (`outputs/evaluation/rule_engine_tuned_p0XX/`) đã được dọn; giữ nếu muốn tái lập ablation, xóa được nếu chỉ cần bảng số trong JSON + báo cáo.",
        ),
        "data/processed/lexicon/sweeps/pruned_score*": FolderDoc(
            title="data/processed/lexicon/sweeps/pruned_scoreXXX - một mức ngưỡng TF-IDF",
            purpose=(
                "Bản từ điển lọc ở **một mức `tfidf_min_score` cụ thể** (số trong tên folder = ngưỡng × 1000), "
                "cùng audit và báo cáo của riêng mức đó. Dùng để so sánh kích thước từ điển / chỉ số rule engine "
                "theo ngưỡng. **Không phải từ điển chính.**"
            ),
            produced_by="`python tmp_prune_sweep.py` (script thí nghiệm) hoặc `scripts/20_prune_lexicon.py --tfidf-min-score <ngưỡng> --output-dir <folder này>`",
            contents=(
                ("verified_lexicon_pruned.csv", "Từ điển lọc ở mức ngưỡng của folder này (cùng schema với `../../verified_lexicon.csv`)."),
                ("lexicon_prune_audit.csv", "Audit từng entry của lần lọc này: số đếm theo nhãn, `tfidf_score`, `tfidf_ratio`, `decision`, `reason`."),
                ("lexicon_report.json", "Báo cáo thu nhỏ của lần lọc này (tham số + số entry giữ/bỏ)."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "So các ngưỡng: mở `lexicon_report.json` của từng folder con trong `sweeps/` rồi so `kept`.",
                "Muốn chấm một biến thể: dùng config `configs/rule_engine_tuned_p0XX.yaml` (đã trỏ sẵn `lexicon_path` vào folder này).",
            ),
            keep_policy="**Không xóa** khi còn file kết quả `outputs/evaluation/rule_engine_tuned_p0XX/` tương ứng - xóa thì không tái lập được số liệu đó.",
        ),
        "data/splits": FolderDoc(
            title="data/splits - split cố định dùng chung",
            purpose=(
                "**Quan trọng nhất về tính công bằng.** Đây là 3 file CSV mà mọi model đều đọc, kèm SHA-256 của từng file. "
                "Mọi lần train/evaluate đều kiểm tra fingerprint trước khi nạp dữ liệu; sai hash là dừng ngay."
            ),
            produced_by="`python scripts/00_prepare_data.py --stage split`",
            contents=(
                ("train.csv", "24.046 mẫu dùng để huấn luyện và ước lượng class weight."),
                ("validation.csv", "2.672 mẫu dùng cho early stopping, chọn best checkpoint và tìm threshold."),
                ("test.csv", "6.680 mẫu dùng để chấm **đúng một lần** ở cuối cùng."),
                ("split_manifest.json", "Số mẫu, phân bố nhãn và SHA-256 của từng split + metadata (chiến lược, seed, tỉ lệ)."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Nghi ngờ ai đó đổi dữ liệu: so SHA-256 trong `split_manifest.json` với file trên đĩa.",
                "Muốn đổi chiến lược chia: `--set strategy=stratified` rồi chạy lại `00_prepare_data.py --stage split --overwrite` - **lưu `manifest` cũ lại trước**.",
            ),
            keep_policy=(
                "**KHÔNG XÓA, KHÔNG SỬA TAY.** Nếu vô tình đổi, phải tạo lại toàn bộ fingerprint bằng "
                "`00_prepare_data.py --stage split --overwrite` - nghĩa là mọi kết quả cũ không còn so sánh được."
            ),
        ),
        # ---------------- reports ---------------- #
        "reports": FolderDoc(
            title="reports - log và kết quả mỗi lần chạy",
            purpose=(
                "**Nơi đầu tiên cần tìm khi muốn xem log train.** Mỗi lần chạy một model tạo đúng một thư mục "
                "`reports/<model>/run_XXX/`, chứa toàn bộ log, artifact và kết quả của riêng lần chạy đó. "
                "Các lần chạy không bao giờ ghi đè lẫn nhau."
            ),
            produced_by="`python -m src.training.run_experiment --config <model>` rồi `python -m src.evaluation.run_evaluation --model <model>`",
            children=(
                ("sensitiveai-vi/", "Model 1 - baseline."),
                ("sensitiveai-vi-custom/", "Model 2 - custom head."),
                ("sensitiveai-vi-customdlr2stage/", "Model 3 - DLR + 2 stage + focal."),
                ("sensitiveai-vi-custom-curriculum/", "Model 4 - curriculum 3 stage."),
            ),
            reading_guide=(
                "Muốn xem log của một lần chạy: `reports/<model>/run_XXX/explain.md`.",
                "Muốn biết hiện có bao nhiêu lần chạy và lần nào là mới nhất: bảng **Danh sách các lần chạy** trong chính file này (do `05_document_folders.py` làm mới).",
                "Muốn tìm nhanh: dùng `run_pointer.txt` trong `reports/<model>/` (trỏ tới run mới nhất).",
            ),
            keep_policy=KEEP_POLICY,
        ),
        "reports/*": FolderDoc(
            title="reports/<model> - tất cả lần chạy của một model",
            purpose="Chứa `run_001`, `run_002`, ... theo thứ tự thời gian. Mỗi `run_XXX` là một lần train độc lập.",
            produced_by="`run_experiment` tự cấp số thứ tự tự do kế tiếp.",
            contents=(
                ("run_pointer.txt", "Tên thư mục của lần chạy mới nhất."),
                ("explain.md", "File này, kèm bảng liệt kê các lần chạy."),
            ),
            reading_guide=("So sánh hai lần chạy: mở `training_log.csv` của cả hai `run_XXX`.",),
            keep_policy=KEEP_POLICY,
        ),
        "reports/*/run_*": FolderDoc(
            title="reports/<model>/run_XXX - một lần chạy",
            purpose=(
                "Toàn bộ dấu vết của **một** lần train (+ đánh giá nếu đã chạy): log console, log từng epoch, "
                "checkpoint, class weight, chuyển stage, và kết quả test. Đây là đơn vị lưu trữ nhỏ nhất - "
                "muốn ghi lại một thí nghiệm thì chép nguyên thư mục này."
            ),
            produced_by="`python -m src.training.run_experiment --config <model>` (+ `run_evaluation` cho các file `metrics.*`)",
            contents=_RUN_FILE_DOCS,
            reading_guide=_RUN_READING_GUIDE,
            keep_policy=(
                KEEP_POLICY + " Nếu chấm test lại cùng một run, kết quả cũ được chuyển vào `history/` chứ không bị xóa."
            ),
            extra_notes=(
                "`config.yaml` là bản **đã resolve**: nếu bạn từng chạy kèm `--set`, giá trị đó nằm trong file này.",
                "Cột `is_best` trong `training_log.csv` đánh dấu đúng epoch có validation macro F1 cao nhất.",
            ),
        ),
        "reports/*/run_*/history": FolderDoc(
            title="history - kết quả đánh giá cũ",
            purpose=(
                "Mỗi lần chấm test lại sẽ tạo một thư mục `eval_<YYYYmmdd-HHMMSS>/` chứa toàn bộ artifact "
                "của lần chấm trước. Nhờ vậy không có số liệu nào bị ghi đè mà không còn dấu vết."
            ),
            produced_by="`python -m src.evaluation.run_evaluation --model <model> --run-id run_XXX` (chạy lại lần hai trở đi)",
            keep_policy="**KHÔNG XÓA** - đây chính là bản sao lịch sử của các lần đo khác nhau.",
        ),
        # ---------------- experiments ---------------- #
        "experiments": FolderDoc(
            title="experiments - checkpoint từng stage",
            purpose=(
                "Tách khỏi `reports/` để phân biệt **log/kết quả** với **trọng số mô hình**. "
                "Mỗi lần chạy có `experiments/<model>/run_XXX/stage_checkpoints/<stage>/` - "
                "cần cho multi-stage (model 3, 4) vì stage sau nạp checkpoint của stage trước."
            ),
            produced_by="`python -m src.training.run_experiment --config <model>`",
            children=(
                ("sensitiveai-vi/", "Model 1 (1 stage)."),
                ("sensitiveai-vi-custom/", "Model 2 (1 stage)."),
                ("sensitiveai-vi-customdlr2stage/", "Model 3 (2 stage)."),
                ("sensitiveai-vi-custom-curriculum/", "Model 4 (3 stage)."),
            ),
            reading_guide=(
                "Muốn nạp lại một stage giữa chừng để debug: `--checkpoint experiments/<model>/run_XXX/stage_checkpoints/stage_1`.",
                "Muốn biết stage nào tồn tại: xem `stage_transitions.json` trong `reports/<model>/run_XXX/`.",
            ),
            keep_policy=KEEP_POLICY,
        ),
        "experiments/*": FolderDoc(
            title="experiments/<model> - checkpoint của một model",
            purpose="Một thư mục con `run_XXX` cho mỗi lần chạy.",
            produced_by="`run_experiment`.",
            keep_policy=KEEP_POLICY,
        ),
        "experiments/*/run_*": FolderDoc(
            title="experiments/<model>/run_XXX - trọng số của một lần chạy",
            purpose="Chỉ chứa trọng số, không chứa log. Tách riêng để `reports/` luôn gọn và dễ đọc.",
            produced_by="`run_experiment` (`--no-save-checkpoints` sẽ bỏ qua việc lưu).",
            children=(
                ("stage_checkpoints/", "Một thư mục con cho mỗi stage đã lưu."),
            ),
            keep_policy=(
                "**Cần cho các lần chạy multi-stage.** Xóa checkpoint stage giữa chừng sẽ làm stage sau "
                "không nạp được (pipeline dừng bằng thông báo lỗi, không hỏng dữ liệu)."
            ),
        ),
        "experiments/*/run_*/stage_checkpoints": FolderDoc(
            title="stage_checkpoints - checkpoint của từng stage",
            purpose="Một subfolder cho mỗi stage: `stage_1`, `stage_2`, `stage_3`. Là nơi stage sau tìm checkpoint để nạp.",
            produced_by="`StageController` trong `src/callbacks/stage_controller.py`.",
            children=(("stage_N/", "Trọng số + tokenizer của stage đó."),),
            keep_policy=KEEP_POLICY,
        ),
        "experiments/*/run_*/stage_checkpoints/stage_*": FolderDoc(
            title="stage_checkpoints/stage_N - trọng số một stage",
            purpose="Trọng số mô hình tại cuối stage, kèm tokenizer và `sensitiveai_config.json` nếu là custom head.",
            produced_by="`StageController` khi kết thúc một stage có `save_checkpoint: true`.",
            contents=(
                ("model.safetensors", "Trọng số encoder (và head nếu lưu chung)."),
                ("config.json", "Cấu hình kiến trúc."),
                ("tokenizer_config.json", "Cấu hình tokenizer để nạp lại không cần mạng."),
                ("sensitiveai_config.json", "Chỉ có với custom head: hidden size, các view pooling đang bật."),
                ("explain.md", "File này."),
            ),
            keep_policy=KEEP_POLICY,
        ),
        "outputs": FolderDoc(
            title="outputs - kết quả huấn luyện & đánh giá chính thức",
            purpose=(
                "Thư mục chứa toàn bộ kết quả chính thức: `reports/` (chi tiết từng run của từng model), "
                "`experiments/` (trạng thái huấn luyện và checkpoints trung gian), `evaluation/` (đánh giá tổng hợp "
                "và so sánh 4 model), và `paper/` (bảng số liệu, biểu đồ cho bài báo)."
            ),
            produced_by="Các pipeline huấn luyện chính thức (scripts 01 -> 03).",
            children=(
                ("reports/", "Báo cáo chi tiết, log và checkpoint `best_model` của từng model theo run."),
                ("experiments/", "Checkpoints trung gian giữa các stage của mô hình đa giai đoạn."),
                ("evaluation/", "Log chấm điểm tổng hợp và bảng/biểu đồ so sánh 4 model."),
                ("paper/", "Bảng LaTeX và hình vẽ dùng cho bài báo."),
            ),
            keep_policy=KEEP_POLICY,
        ),
        "outputs_smoke": FolderDoc(
            title="outputs_smoke - kết quả chạy kiểm thử nhanh (smoke test)",
            purpose=(
                "Thư mục chứa toàn bộ kết quả của các lần chạy smoke test kiểm thử pipeline: "
                "`reports/`, `experiments/`, `evaluation/`, và `paper/`."
            ),
            produced_by="Các lệnh chạy với cờ `--smoke-test` hoặc trỏ `--reports-dir outputs_smoke/reports`.",
            children=(
                ("reports/", "Báo cáo chi tiết các lần chạy smoke test."),
                ("experiments/", "Trạng thái checkpoint smoke test."),
                ("evaluation/", "Đánh giá tổng hợp smoke test."),
                ("paper/", "Bảng số liệu smoke test."),
            ),
            keep_policy=KEEP_POLICY,
        ),
        "checkpoints": FolderDoc(
            title="checkpoints - kho trọng số dự trữ",
            purpose=(
                "Kho tập trung cho các checkpoint muốn giữ lâu dài (không nằm trong một lần chạy cụ thể). "
                "Mặc định pipeline không ghi vào đây; nếu bạn sao chép thủ công thì kho này giữ cho bạn."
            ),
            produced_by="Sao chép thủ công, hoặc truyền `--checkpoint` khi muốn dùng lại.",
            children=(),
            reading_guide=("Muốn dùng lại checkpoint cũ: `python -m src.evaluation.run_evaluation --model <model> --checkpoint <đường dẫn>`.",),
            keep_policy=KEEP_POLICY,
        ),
        # ---------------- evaluation ---------------- #
        "evaluation": FolderDoc(
            title="evaluation - kết quả đánh giá tổng hợp",
            purpose=(
                "Tầng tổng hợp cho 4 model: nhật ký mọi lần chấm điểm, kết quả theo từng model, và bảng/biểu đồ so sánh "
                "dùng để viết bài báo. Chi tiết theo từng run vẫn nằm trong `reports/<model>/run_XXX/`."
            ),
            produced_by="`python -m src.evaluation.run_evaluation ...` và `python scripts/03_compare_models.py`",
            children=(
                ("logs/", "Nhật ký CSV của **mọi** lần chấm điểm - nối các lần chạy lại thành một dòng thời gian."),
                ("per_model/", "Bản sao số liệu chính của từng model, gom lại một chỗ."),
                ("comparison/", "Bảng và biểu đồ so sánh 4 model + so sánh 2 rule engine."),
                ("analysis/", "Ba báo cáo phân tích hậu nghiệm: chống trùng lặp giữa các split, kiểm định thống kê 4 model, supplementary (calibration/MCC)."),
                ("rule_engine/", "Kết quả đánh giá rule engine từ vựng độc lập (plan §39, §44)."),
                ("rule_engine_tuned/", "Rule engine với cấu hình đã tune (`configs/rule_engine_tuned.yaml`) - bản tốt nhất, kèm hồ sơ tune (`tuning_trials.csv`, `tuning_report.json`)."),
                ("fusion/", "Ghép điểm rule engine với xác suất của từng model AI (plan §40-41)."),
            ),
            reading_guide=(
                "Muốn xem toàn bộ lịch sử chấm điểm: `evaluation/logs/evaluation_log.csv`.",
                "Muốn có bảng để dán vào bài: `evaluation/comparison/model_comparison.md`.",
            ),
            keep_policy=KEEP_POLICY,
        ),
        "evaluation/logs": FolderDoc(
            title="evaluation/logs - nhật ký chấm điểm",
            purpose="Mỗi lần chấm test thêm **một dòng** vào CSV (chế độ append, không ghi đè), nên lịch sử đầy đủ được giữ lại.",
            produced_by="`append_evaluation_log()` trong `src/evaluation/run_evaluation.py`",
            contents=(
                ("evaluation_log.csv", "Một dòng cho mỗi lần đánh giá: model, run_id, split, quy tắc quyết định, chỉ số, thời gian."),
                ("explain.md", "File này."),
            ),
            keep_policy=KEEP_POLICY,
        ),
        "evaluation/per_model": FolderDoc(
            title="evaluation/per_model - số liệu theo từng model",
            purpose="Gom chỉ số chính của từng model ra một chỗ, tiện để so sánh nhanh mà không phải mở từng `run_XXX`.",
            produced_by="`run_evaluation` (`--output-dir` hoặc mặc định ghi vào đây).",
            children=(("<model>/", "Số liệu của một model."),),
            keep_policy=KEEP_POLICY,
        ),
        "evaluation/per_model/*": FolderDoc(
            title="evaluation/per_model/<model>",
            purpose="Số liệu đánh giá của một model, sao chép ra từ `reports/<model>/run_XXX/`.",
            produced_by="`run_evaluation`.",
            keep_policy=KEEP_POLICY,
        ),
        "evaluation/comparison": FolderDoc(
            title="evaluation/comparison - bảng và biểu đồ so sánh",
            purpose=(
                "Nơi dễ dễ nhất để đọc kết quả so sánh: một bảng tất cả chỉ số của 4 model (CSV và Markdown), "
                "cùng các biểu đồ PR/ROC/validation dùng cho bài báo. Hai thư mục con: `rule_engine/` (so sánh "
                "baseline vs tuned) và `vi_vs_stage2/` (so 2 run cùng model)."
            ),
            produced_by=(
                "`python scripts/03_compare_models.py` (4 model); "
                "bộ so sánh 2 rule engine do scripts 26 + 28 ghi vào thư mục con `rule_engine/`"
            ),
            contents=(
                ("model_comparison.csv", "Bảng tổng hợp tất cả chỉ số của 4 model."),
                ("model_comparison.md", "Cùng bảng dạng Markdown, dễ đọc và dán vào bài báo."),
                ("comparison_summary.json", "Tóm tắt dạng máy-đọc."),
                ("pr_curve_all_models.png", "PR curve macro của 4 model trên cùng một trục."),
                ("roc_curve_all_models.png", "ROC curve macro của 4 model."),
                ("pr_hate_all_models.png", "PR curve riêng cho lớp HATE (lớp hiếm, khó nhất)."),
                ("pr_offensive_all_models.png", "PR curve riêng cho lớp OFFENSIVE."),
                ("pr_clean_all_models.png", "PR curve riêng cho lớp CLEAN."),
                ("roc_hate_all_models.png", "ROC curve riêng cho lớp HATE."),
                ("roc_offensive_all_models.png", "ROC curve riêng cho lớp OFFENSIVE."),
                ("roc_clean_all_models.png", "ROC curve riêng cho lớp CLEAN."),
                ("validation_accuracy_all_models.png", "Đường cong validation accuracy của 4 model theo epoch."),
                ("validation_loss_all_models.png", "Đường cong validation loss của 4 model theo epoch."),
                ("macro_f1_all_models.png", "So sánh macro F1 dạng thanh."),
                ("accuracy_all_models.png", "So sánh accuracy dạng thanh (tham khảo, vì dữ liệu mất cân bằng)."),
                ("per_class_f1_all_models.png", "F1 theo từng lớp của 4 model."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Muốn so 2 rule engine (baseline vs tuned): đọc `rule_engine/rule_engine_baseline_vs_tuned.md` - đủ bảng số, hình kèm và kết luận kiểm định.",
                "Bảng quan trọng nhất của 4 model: `model_comparison.md` (macro F1, weighted F1, per-class F1, latency).",
                "Giải thích vì sao accuracy không phải số liệu chính: xem mục 2.2 của `README.md`.",
            ),
            keep_policy=KEEP_POLICY,
        ),
        "evaluation/analysis": FolderDoc(
            title="evaluation/analysis - báo cáo phân tích hậu nghiệm",
            purpose=(
                "Ba báo cáo phân tích đọc từ **test predictions đã đóng băng** của 4 model (không huấn luyện, "
                "không tune gì thêm): (1) chống trùng lặp giữa các split - `leakage_report`; (2) kiểm định thống "
                "kê post-hoc - CI bootstrap 95% từng model, so cặp trên test - `significance_report`; "
                "(3) supplementary (calibration ECE/Brier/NLL, MCC...) - `supplementary_report`, được "
                "`papers/supplementary/main.tex` trỏ tới."
            ),
            produced_by=(
                "`python scripts/08_check_leakage.py` (leakage), `python scripts/06_analyze_significance.py` "
                "(significance), `python scripts/11_supplementary_analyses.py` (supplementary)"
            ),
            contents=(
                ("leakage_report.md / .json", "Trùng lặp giữa train/validation/test (895/6.680 comment test trùng train/val = 13.4%) + tập de-leaked 5.785 mẫu."),
                ("significance_report.md", "CI bootstrap accuracy/macro-F1 từng model + so cặp - nguồn cho mọi tuyên bố 'model nào hơn' trên test."),
                ("significance_summary.json", "Cùng nội dung ở dạng máy-đọc (CI, pairwise delta, p-value)."),
                ("supplementary_report.md", "Calibration (ECE/Brier/NLL), MCC... dạng Markdown cho bài báo."),
                ("supplementary_summary.json", "Dữ liệu của supplementary ở dạng máy-đọc."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Nghi ngờ leakage: đọc `leakage_report.md` trước - số trùng test/train quyết định mức độ tin số test.",
                "Muốn tuyên bố model A hơn model B: mở `significance_report.md`, xem CI của Δ và p - không đọc p từ bảng metric.",
                "Bài supplementary trỏ thẳng vào `supplementary_report.md` - đừng đổi tên file này.",
            ),
            keep_policy=KEEP_POLICY,
        ),
        "evaluation/rule_engine": FolderDoc(
            title="evaluation/rule_engine - đánh giá rule engine",
            purpose=(
                "Kết quả chấm riêng cho rule engine từ vựng (không dùng model AI). Threshold quyết định được tune "
                "**trên validation** rồi đóng băng vào `thresholds.json`; test chỉ được chấm một lần với threshold đã "
                "đóng băng (plan §44)."
            ),
            produced_by="`python scripts/17_evaluate_rule_engine.py` và `python scripts/19_plot_rule_engine.py`",
            contents=(
                ("thresholds.json", "Ngưỡng riêng từng nhãn tối ưu trên validation + vết tìm kiếm; đóng băng trước khi chấm test."),
                ("metrics.json", "Bộ chỉ số đầy đủ: accuracy, macro/weighted F1, per-class P/R/F1, ROC-AUC/AP, confusion matrix."),
                ("per_class_metrics.csv", "Precision/recall/F1/support/AUC/AP theo từng nhãn và dòng MACRO."),
                ("confusion_matrix.csv", "Ma trận nhầm lẫn dạng CSV (hàng = nhãn thật)."),
                ("confusion_matrix.txt", "Ma trận nhầm lẫn và bản chuẩn hoá theo hàng, in tên nhãn."),
                ("classification_report.txt", "Báo cáo phân loại đầy đủ của sklearn (tên nhãn, không dùng id)."),
                ("false_positive_analysis.csv", "Các câu CLEAN bị báo nhầm kèm bằng chứng khớp mạnh nhất."),
                ("false_negative_analysis.csv", "Các câu HATE/OFFENSIVE bị bỏ sót."),
                ("latency.json", "Latency (mean/p50/p95/max) và throughput mỗi câu."),
                ("robustness.json", "Tỉ lệ phát hiện trên mẫu bị làm nhiễu (clean/leet/spaced/no_accent/repeat) kèm `detection_drop_vs_clean` (do script 19 bổ sung dòng baseline `clean`)."),
                ("rule_scores.csv", "Điểm `rule_hate_score`/`rule_offensive_score` và nhãn dự đoán per-sample; đầu vào cho bước fusion (script 19 sinh)."),
                ("confusion_matrix.png", "Hình ma trận nhầm lẫn thô (script 19)."),
                ("confusion_matrix_normalized.png", "Hình ma trận nhầm lẫn chuẩn hoá theo hàng (script 19)."),
                ("per_class_f1.png", "Cột F1 theo từng lớp (script 19)."),
                ("pr_curve.png", "Đường Precision-Recall one-vs-rest + macro (script 19)."),
                ("roc_curve.png", "Đường ROC one-vs-rest + macro (script 19)."),
                ("threshold_sweep.png", "macro-F1 theo từng bước tune threshold trên validation (script 19)."),
                ("robustness_detection.png", "Cột detection rate theo biến đổi, gồm baseline `clean` (script 19)."),
                ("latency_distribution.png", "Histogram latency mỗi câu + p50/p95/mean (script 19)."),
                ("test/", "Kết quả chấm **test một lần** với threshold đã đóng băng, gồm cả PNG và `rule_scores.csv` (nằm riêng để không ghi đè kết quả validation)."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Số liệu chính (lexicon 668 entry): validation macro-F1 **0.4555** (HATE 0.3677, OFFENSIVE 0.1686, acc 0.7021), thresholds [0.85, 0.0, 0.41].",
                "`test/metrics.json` = **0.4536** (acc 0.7070, HATE 0.3717, OFFENSIVE 0.1546) - đây là baseline để so với bản tuned 0.5335 (**+0.0799 macro-F1**).",
                "Số liệu chính: `metrics.json` (macro F1) và `per_class_metrics.csv`.",
                "Xem vì sao báo nhầm: `false_positive_analysis.csv` (cột `top_matches`).",
                "So sánh AUC/AP giữa các lớp: cột `roc_auc`, `average_precision` trong `per_class_metrics.csv`.",
                "Xem hình: `confusion_matrix*.png`, `per_class_f1.png`, `pr_curve.png`, `roc_curve.png`, `robustness_detection.png`, `latency_distribution.png`.",
            ),
            keep_policy=KEEP_POLICY,
        ),
        "evaluation/rule_engine/test": FolderDoc(
            title="evaluation/rule_engine/test - chấm test một lần cho rule engine",
            purpose=(
                "Cùng bộ artifact như `evaluation/rule_engine/` nhưng cho **test split**, chấm đúng một lần với "
                "threshold đã đóng băng từ validation (plan §44). Nằm riêng để không bao giờ ghi đè kết quả validation "
                "đã dùng để tune."
            ),
            produced_by="`python scripts/17_evaluate_rule_engine.py --split test` và `python scripts/19_plot_rule_engine.py --split test`",
            contents=(
                ("metrics.json", "Chỉ số test với threshold đóng băng (dùng cho ablation §43)."),
                ("per_class_metrics.csv", "Precision/recall/F1/support/AUC/AP theo lớp trên test."),
                ("confusion_matrix.csv / .txt", "Ma trận nhầm lẫn thô và chuẩn hoá theo hàng."),
                ("classification_report.txt", "Báo cáo phân loại sklearn trên test."),
                ("false_positive_analysis.csv / false_negative_analysis.csv", "Phân tích lỗi kèm bằng chứng khớp mạnh nhất."),
                ("latency.json", "Latency/throughput trên test."),
                ("robustness.json", "Detection rate theo biến đổi + baseline `clean` và `detection_drop_vs_clean`."),
                ("rule_scores.csv", "Điểm rule per-sample trên test; đầu vào cho bước fusion."),
                ("*.png", "Confusion matrix, per-class F1, PR/ROC, threshold sweep, robustness, latency."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Đây là con số **test cuối cùng** của rule engine baseline: macro-F1 **0.4536**, acc 0.7070, HATE 0.3717, OFFENSIVE 0.1546 (lexicon 668 entry, thresholds đóng băng [0.85, 0.0, 0.41]); không tune lại sau khi có kết quả này.",
                "So với `evaluation/rule_engine_tuned/test/metrics.json` (0.5335): chênh lệch **+0.0799 macro-F1** đến từ tune trọng số, không phải từ điển hay ngưỡng khác.",
                "Bản test cũ chạy trên lexicon 922 entry (macro-F1 0.4667) được lưu ở `test_legacy922/` để không ghi đè lịch sử.",
                "Robustness test (400 mẫu nhiễu): clean 0.555, leet 0.440, spaced 0.4725, no_accent 0.540, repeat 0.3775.",
            ),
            keep_policy=KEEP_POLICY,
        ),
        "evaluation/rule_engine/test_legacy922": FolderDoc(
            title="evaluation/rule_engine/test_legacy922 - số test của phiên bản lexicon 922 entry (lưu trữ)",
            purpose=(
                "Kết quả chấm test **bản cũ** của rule engine baseline khi `verified_lexicon.csv` còn 922 entry "
                "(macro-F1 0.4667). Khi hệ thống được dựng lại với lexicon 668 entry, thư mục `test/` phải chấm lại "
                "một lần nên bản cũ được chuyển vào đây để không mất lịch sử (plan §44)."
            ),
            produced_by="Kết quả của lần chạy cũ, chỉ được **di chuyển** (không sinh lại được).",
            contents=(
                ("metrics.json", "Chỉ số test bản lexicon 922: macro-F1 0.4667, acc 0.7201."),
                ("thresholds.json", "Ngưỡng đóng băng thời điểm đó (khác với ngưỡng hiện tại [0.85, 0.0, 0.41])."),
                ("*.csv / *.txt", "Confusion matrix, per-class, FP/FN, latency bản cũ."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "**Chỉ dùng để đối chiếu lịch sử**, không dùng làm baseline cho kết quả hiện tại - baseline hiện tại là `evaluation/rule_engine/test/` (lexicon 668, macro-F1 0.4536).",
            ),
            keep_policy="**KHÔNG XÓA, KHÔNG CHẠY LẠI** - đây là lịch sử đo lường (mỗi lần đo test chỉ hợp lệ một lần).",
        ),
        "evaluation/rule_engine_tuned": FolderDoc(
            title="evaluation/rule_engine_tuned - rule engine đã tune trọng số",
            purpose=(
                "Cùng bộ artifact như `evaluation/rule_engine/` nhưng dùng **`configs/rule_engine_tuned.yaml`**: các trọng số "
                "`scoring.*` được tune bằng coordinate descent trên **validation** (`scripts/27_tune_rule_weights.py`, "
                "hồ sơ từng trial ở `tuning_trials.csv` của chính folder này). Ba tham số đổi so với bản gốc: "
                "`match_factors.accentless_alias` 0.90 -> 0.00 (view không dấu là tín hiệu phụ nhưng lại khớp nhầm âm tiết phổ biến, plan §38), "
                "`match_factors.obfuscated_combined` 0.60 -> 0.45, `negation_factor` 0.50 -> 1.00. "
                "Threshold tune trên validation rồi đóng băng cho test (plan §44). Đây là cấu hình rule engine tốt nhất, không ghi đè baseline."
            ),
            produced_by="`python scripts/17_evaluate_rule_engine.py --config configs/rule_engine_tuned.yaml --out outputs/evaluation/rule_engine_tuned` (và `--split test` cho bản test) + `python scripts/19_plot_rule_engine.py --config configs/rule_engine_tuned.yaml --out outputs/evaluation/rule_engine_tuned`",
            contents=(
                ("thresholds.json", "Ngưỡng tune trên validation với trọng số đã tune; đóng băng trước khi chấm test."),
                ("metrics.json", "Chỉ số validation của cấu hình tuned (macro-F1 0.5481)."),
                ("rule_scores.csv", "Điểm rule per-sample (script 19) - đầu vào của bước fusion."),
                ("tuning_trials.csv", "155 trial của coordinate descent (script 27) - bằng chứng trọng số được tune trên **validation** (plan §44). Đây là bản sao lưu từ nhật ký tune gốc; sau khi dọn cây (chỉ giữ baseline + tuned) đây là nơi duy nhất còn `trials.csv`."),
                ("tuning_report.json", "Tóm tắt phiên tune: baseline 0.4555 -> 0.5481, các trọng số đã đổi (`changed`), SHA-256 từ điển."),
                ("test/", "Chấm **test một lần** với threshold đóng băng (macro-F1 0.5335), gồm metrics, confusion, per-class, FP/FN, latency, robustness."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "So `metrics.json` (validation) với `evaluation/rule_engine/metrics.json` (baseline accentless=0.90): macro-F1 0.5481 vs 0.4555, HATE 0.4967 vs 0.3677, OFFENSIVE 0.2310 vs 0.1686.",
                "`test/metrics.json` là con số test cuối: macro-F1 **0.5335**, accuracy 0.8398, HATE 0.4323, OFFENSIVE 0.2418 (val 0.5481 -> test 0.5335, gap lành mạnh) - so baseline cùng lexicon 668 `evaluation/rule_engine/test/metrics.json` = 0.4536 (**+0.0799**).",
                "Đánh đổi phải nhớ (`robustness.json`, 400 mẫu nhiễu): `no_accent` giảm - validation 0.3625 (baseline 0.530), test 0.325 (baseline 0.540) - vì `accentless_alias = 0`; bù lại leet/spaced/repeat **tốt hơn** baseline (test: 0.460/0.605/0.5425 vs 0.440/0.4725/0.3775).",
                "Bộ artifact chi tiết giống hệt `evaluation/rule_engine/` (confusion matrix, per-class, PR/ROC, robustness...).",
            ),
            keep_policy=KEEP_POLICY,
        ),
        "evaluation/rule_engine_tuned/test": FolderDoc(
            title="evaluation/rule_engine_tuned/test - test một lần cho rule engine đã tune",
            purpose=(
                "Test split của rule engine dùng cấu hình tuned (`scratch 27`: accentless 0.0, obfuscated_combined 0.45, "
                "negation 1.0), chấm đúng một lần với threshold đã đóng băng từ validation (plan §44). "
                "Lần đo này thuộc **phiên bản lexicon 668 entry** (bản cũ 1828 entry từng cho macro-F1 0.4927)."
            ),
            produced_by="`python scripts/17_evaluate_rule_engine.py --config configs/rule_engine_tuned.yaml --split test --out outputs/evaluation/rule_engine_tuned`",
            contents=(
                ("metrics.json", "Chỉ số test với cấu hình tuned và threshold đóng băng (macro-F1 0.5335)."),
                ("*.csv / *.txt / *.json", "Confusion matrix, per-class, FP/FN, latency, robustness như baseline."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "So sánh với `evaluation/rule_engine_tuned/metrics.json` (validation 0.5481) để thấy gap validation -> test.",
                "Đánh đổi robustness (`robustness.json`): no_accent giảm vì accentless_alias = 0; leet/spaced/repeat tăng nhờ threshold HATE thấp hơn (0.64).",
            ),
            keep_policy=KEEP_POLICY,
        ),
        "evaluation/rule_engine_tune": FolderDoc(
            title="evaluation/rule_engine_tune - nhật ký tune trọng số rule engine",
            purpose=(
                "Hồ sơ của **mọi ứng viên trọng số** mà `scripts/27_tune_rule_weights.py` thử trên validation: "
                "mỗi dòng trong `trials.csv` là một trial (tham số, giá trị, macro-F1, per-class F1, threshold, "
                "có được chấp nhận hay không), `tuning_report.json` chốt lại cấu hình thắng. Đây là **hồ sơ "
                "quyết định**, không phải artifact chấm chính thức - số liệu chính thức nằm ở "
                "`evaluation/rule_engine_tuned/`. Cây đã được dọn ngày 19/10 (chỉ giữ baseline + tuned) nên folder "
                "này hiện **không tồn tại**: chạy lại `scripts/27_tune_rule_weights.py` sẽ tái tạo nó, còn bản sao "
                "của lần tune thắng đang giữ ở `evaluation/rule_engine_tuned/tuning_trials.csv`."
            ),
            produced_by="`python scripts/27_tune_rule_weights.py` (plan §39, §44)",
            contents=(
                ("trials.csv", "Một dòng / trial: `round, trial, param, value, macro_f1, accuracy, HATE_f1, OFFENSIVE_f1, CLEAN_f1, th_*, accepted, seconds` - dùng để vẽ đường tìm kiếm hoặc kiểm chứng không có gian lận test."),
                ("tuning_report.json", "Chốt kết quả: baseline vs best, các tham số đã đổi, SHA-256 từ điển, số dòng, giao thức (validation-only, threshold tune lại từng trial)."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Muốn biết vì sao trọng số X ra giá trị Y: lọc `trials.csv` theo cột `param = X`, so các dòng `accepted = True`.",
                "Muốn biết cấu hình thắng thay đổi gì so với config gốc: `tuning_report.json` -> khoá `changed`.",
                "Muốn tái lập: chạy lại `python scripts/27_tune_rule_weights.py` (cùng from điển + cùng split thì ra cùng kết quả).",
            ),
            keep_policy="**Không xóa** - đây là hồ sơ chứng minh trọng số được tune trên validation chứ không phải trên test.",
        ),
        "evaluation/fusion": FolderDoc(
            title="evaluation/fusion - hybrid fusion (rule engine + AI)",
            purpose=(
                "Kết quả ghép điểm rule engine với xác suất AI (plan §40-41): `alpha` (trọng số AI), threshold và "
                "meta-classifier đều tune trên **validation**, đóng băng rồi chấm test **một lần** (plan §44). "
                "Mỗi model AI một thư mục con, `comparison.md` là bảng tổng hợp. Kết luận hiện tại (lexicon 668 entry, "
                "rule engine test macro-F1 0.5335 so với AI-only 0.65+): chênh lệch fusion - AI nằm trong dải nhiễu "
                "với 5/6 model (Δ macro-F1 từ -0.0085 đến +0.0034); riêng `sensitiveai-vi-custom-curriculum` fusion "
                "**tốt hơn có ý nghĩa** (+0.0057, p=0.033, McNemar p=0.0005) còn `sensitiveai-vi-stage2` kém hơn "
                "(-0.0085) nhưng không có ý nghĩa. Chưa đủ bằng chứng khẳng định fusion cải thiện nhất quán - "
                "xem `significance_report.md`."
            ),
            produced_by=(
                "`python scripts/23_evaluate_fusion.py` rồi `python scripts/25_fusion_significance.py` (plan §43-44); "
                "hình so sánh: `python scripts/29_plot_fusion_comparison.py`"
            ),
            contents=(
                ("comparison.md", "Bảng tổng hợp: alpha tốt nhất trên validation, macro-F1 test của AI-only vs các biến thể fusion, biến thể `primary` và Δ so với AI-only cho từng model."),
                ("fusion_ai_vs_fusion.png", "Hình 1: thanh so macro-F1 test của AI-only vs fusion primary từng model (kèm đường đứt nét rule engine 0.5335) + panel phải là Δ zoom quanh 0."),
                ("fusion_delta_ci.png", "Hình 2 (forest plot): Δ = fusion − AI kèm CI 95% paired bootstrap - thanh nào cắt trục 0 là không kết luận được; ngôi sao = có ý nghĩa thống kê."),
                ("fusion_variants_delta.png", "Hình 3: Δ so với AI-only của cả 4 biến thể fusion (argmax/thresholds/per-channel/meta) cho từng model."),
                ("fusion_per_class_f1.png", "Hình 4: F1 từng lớp (HATE/OFFENSIVE/CLEAN) của AI-only vs fusion primary."),
                ("fusion_significance.json", "Kiểm định đầy đủ: paired bootstrap (macro-F1) + McNemar (độ đúng) giữa AI-only và fusion `primary`."),
                ("significance_report.md", "Bản Markdown của cùng kết quả - đọc file này để biết chênh lệch có ý nghĩa hay không."),
                ("<model>/fusion_report.md", "Báo cáo từng model: alpha, per-channel alpha, threshold đóng băng + bảng metric test của 6 hệ thống (AI-only, rule-only, fusion argmax/thresholds/per-channel/meta)."),
                ("<model>/metrics.json", "Chỉ số đầy đủ của từng biến thể fusion (validation + test) - nguồn dữ liệu của script 29."),
                ("<model>/fusion_predictions.csv", "Dự đoán per-sample của biến thể `primary` (nhãn ghi bằng **tên** `HATE/OFFENSIVE/CLEAN`) - đầu vào của script 25."),
                ("fusion.log", "Log chạy fusion: device, checkpoint, alpha và threshold đã chọn."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Hình trực quan nhất: `fusion_delta_ci.png` (model nào khác AI thật sự?) rồi `fusion_ai_vs_fusion.png` (số tuyệt đối).",
                "Đọc nhanh: `significance_report.md` (chênh lệch có ý nghĩa không?) rồi `comparison.md` (bảng theo model).",
                "Δ = AI-only - fusion: Δ dương nghĩa là fusion **kém hơn**; CI chứa 0 = không kết luận được. (Trên hình script 29 đổi chiều: Δ = fusion − AI.)",
                "`fusion.log` có dòng `best alpha on validation` và `frozen thresholds` - bằng chứng tham số không lấy từ test.",
            ),
            keep_policy=KEEP_POLICY,
        ),
        "evaluation/fusion/*": FolderDoc(
            title="evaluation/fusion/<model> - fusion cho một model AI",
            purpose=(
                "Một thư mục cho mỗi model AI: alpha/threshold đã đóng băng trên validation + metric test của 6 hệ "
                "thống (AI-only argmax, rule-only, fusion argmax/thresholds/per-channel/meta)."
            ),
            produced_by="`python scripts/23_evaluate_fusion.py --model <model>`",
            contents=(
                ("fusion_report.md", "Bảng so sánh test của các hệ thống + tham số đã đóng băng."),
                ("metrics.json", "Chỉ số đầy đủ từng biến thể (validation + test)."),
                ("fusion_predictions.csv", "Dự đoán per-sample của biến thể `primary`."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Cột `macro-F1` trong `fusion_report.md`: so `ai_only_argmax` với biến thể `primary`.",
                "`primary` chọn theo macro-F1 **validation**, nên đôi khi kém hơn một biến thể khác trên test - hệ quả của việc chọn tham số trên validation, không phải lỗi.",
            ),
            keep_policy=KEEP_POLICY,
        ),
        "evaluation/comparison/rule_engine": FolderDoc(
            title="evaluation/comparison/rule_engine - so sánh 2 rule engine",
            purpose=(
                "Bộ so sánh baseline (`rule_engine`) với bản thắng tune (`rule_engine_tuned`): báo Markdown 8 mục "
                "(khác biệt cấu hình + ngưỡng đóng băng, metric validation/test, chi tiết per-class, robustness "
                "theo mode nhiễu, độ trễ, kiểm định paired bootstrap + McNemar, kết luận) kèm 3 hình và số kiểm định. "
                "Tất cả đều đọc artifact đã đóng băng - không chấm lại split nào (plan §44)."
            ),
            produced_by=(
                "`python scripts/26_rule_significance.py --a outputs/evaluation/rule_engine_tuned/test/predictions.csv "
                "--b outputs/evaluation/rule_engine/test/predictions.csv --out .../rule_engine_significance.json` "
                "rồi `python scripts/28_compare_rule_engines.py` (plan §43-44)"
            ),
            contents=(
                ("rule_engine_baseline_vs_tuned.md", "Báo chính 8 mục - từ khác biệt cấu hình đến kết luận có kiểm định; nhúng 3 hình bên dưới."),
                ("rule_engine_significance.json", "Số kiểm định Δ trên test (CI 95%, p, McNemar) do `scripts/26` sinh - nguồn cho mục 7 của báo cáo."),
                ("rule_engine_macro_f1.png", "Thanh so macro-F1 và accuracy của 2 hệ thống, validation và test."),
                ("rule_engine_per_class_f1.png", "F1 từng lớp (HATE/OFFENSIVE/CLEAN) của 2 hệ thống, validation và test."),
                ("rule_engine_robustness.png", "Tỉ lệ phát hiện theo 5 chế độ nhiễu của 2 hệ thống, validation và test."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Đọc `rule_engine_baseline_vs_tuned.md` từ đầu đến cuối là đủ cho báo cáo; hình đã nhúng theo từng mục.",
                "Mục 7-8 (kiểm định + kết luận) là phần cần trích khi mentor hỏi 'cải thiện có thật không'.",
            ),
            keep_policy=KEEP_POLICY,
        ),
        "evaluation/comparison/vi_vs_stage2": FolderDoc(
            title="evaluation/comparison/vi_vs_stage2 - so hai run cùng model, khác cấu hình",
            purpose=(
                "So sánh riêng cặp `sensitiveai-vi` (run 1 tầng) với `sensitiveai-vi-stage2` (2 tầng): "
                "bảng metric trên validation + các hình PR/ROC/learning curve của **mọi** model để làm phụ lục."
            ),
            produced_by="`python scripts/03_compare_models.py`",
            contents=(
                ("model_comparison.csv / .md", "Bảng metric validation của hai run được so (accuracy, macro-F1, per-class F1)."),
                ("comparison_summary.json", "Cùng số liệu ở dạng máy đọc."),
                ("*_all_models.png", "Hình so mọi model: accuracy, macro-F1, per-class F1, PR/ROC theo từng nhãn, validation loss/accuracy."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Dùng khi cần trả lời '2 tầng có hơn 1 tầng không?': so cột macro-F1 giữa hai run trong `model_comparison.md`.",
                "Đây là số **validation**, không phải test.",
            ),
            keep_policy=KEEP_POLICY,
        ),
        "outputs_smoke/evaluation_smoke/rule_engine": FolderDoc(
            title="outputs_smoke/evaluation_smoke/rule_engine - smoke test rule engine",
            purpose=(
                "Kết quả **smoke test** nhanh cho rule engine từ vựng (không cần GPU, chạy vài giây). "
                "Chỉ kiểm tra code chạy đúng (offset map, giữ dấu, khớp exact, phân loại biến đổi, xử lý text rỗng/rất dài) "
                "và chạy end-to-end trên một phần hoặc **toàn bộ ViHSD** bằng **argmax trực tiếp, KHÔNG tune threshold** "
                "nên số liệu thấp hơn `evaluation/rule_engine/` (nơi có tune threshold)."
            ),
            produced_by="`python scripts/18_smoke_test_rule_engine.py --split all --limit 0`",
            contents=(
                ("smoke_test.json", "Kết quả đầy đủ dạng JSON: danh sách check (label/ok/detail), metrics end-to-end theo từng split + combined, meta (config, lexicon, split, limit) và cờ `passed`."),
                ("smoke_test.md", "Bản báo cáo Markdown của cùng nội dung (bảng check + bảng metrics) để đọc nhanh."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Xem nhanh: `smoke_test.md` (bảng check và bảng macro-F1/accuracy theo split).",
                "Muốn trích xuất: đọc `smoke_test.json` (khóa `passed`, `eval`, `checks`).",
                "Số liệu chính thức (có tune threshold): dùng `evaluation/rule_engine/`, không dùng folder này.",
            ),
            keep_policy=KEEP_POLICY,
        ),
        "evaluation/ensemble": FolderDoc(
            title="evaluation/ensemble - thí nghiệm ensemble 6 model AI (+ cộng điểm rule)",
            purpose=(
                "Bốn ứng viên ensemble **đóng băng trên validation** rồi chấm test **một lần** (plan §44): "
                "`ens3` (trung bình đều 3 model đầu ra/recipe đa dạng: base + customdlr2stage-ext + stage2), "
                "`ens6` (cả 6 model), `ens3_rule`/`ens6_rule` (cộng `beta·rule_scores`, AI giữ trọng số 1). "
                "β chọn trên validation từ {0.05, 0.11, 0.25, 0.5} cùng với ngưỡng per-class (101 điểm, 3 vòng "
                "coordinate ascent). Ứng viên chính = có val macro-F1 cao nhất (`ens3_rule`, val 0.7280). "
                "Kết quả test: ens3 0.6609, ens6 0.6685, ens3_rule 0.6614, ens6_rule 0.6669 so với SOTA cũ "
                "(curriculum + fusion) 0.6647 và model đơn tốt nhất 0.6638. Ứng viên chính KHÔNG cải thiện "
                "(-0.0033, CI [-0.0172, +0.0107], p=0.65); ens6 cao nhất (+0.0038) nhưng CI cũng chứa 0 "
                "([-0.0066, +0.0143], p=0.48) và là lựa chọn post-hoc theo test."
            ),
            produced_by="`python scripts/31_ensemble_experiment.py` (plan §43-44); hình so sánh: `python scripts/32_plot_ensemble_comparison.py`",
            contents=(
                ("ensemble_report.md", "Báo chính: protocol đóng băng, bảng val/test của 4 ứng viên + 2 tham chiếu, kiểm định paired bootstrap + McNemar, kết luận."),
                ("ensemble_metrics.json", "Toàn bộ số liệu val/test của từng ứng viên + tham chiếu ở dạng máy đọc."),
                ("thresholds.json", "β và ngưỡng per-class đã đóng băng cho từng ứng viên (chỉ ghi nguồn từ validation)."),
                ("ensemble_predictions.csv", "Dự đoán per-sample trên test của ứng viên chính (index, text, xác suất 3 lớp)."),
                ("significance.json", "Kiểm định paired bootstrap (10.000 lần) + McNemar: ứng viên chính vs SOTA cũ và vs model đơn tốt nhất; kèm phần post-hoc cho ens6."),
                ("ensemble_test_macro_f1.png", "Hình 1: thanh so macro-F1 test của 4 ứng viên + SOTA cũ + model đơn tốt nhất (kèm đường đứt nét rule engine) + panel phải Δ zoom so SOTA."),
                ("ensemble_val_vs_test.png", "Hình 2: val vs test từng ứng viên - minh họa thứ hạng đảo chiều (val chọn ens3_rule, test thích ens6)."),
                ("ensemble_delta_ci.png", "Hình 3 (forest plot): Δ = ứng viên − SOTA kèm CI 95% paired bootstrap; ứng viên chính và ens6 có CI, ens3/ens6_rule chỉ điểm ước lượng."),
                ("ensemble_per_class_f1.png", "Hình 4: F1 từng lớp (HATE/OFFENSIVE/CLEAN) của 4 ứng viên + 2 tham chiếu."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Đọc `ensemble_report.md` để trả lời 'ensemble có cải thiện không?'.",
                "Hình trực quan nhất: `ensemble_delta_ci.png` (CI có chứa 0 không?) rồi `ensemble_test_macro_f1.png` (số tuyệt đối).",
                "Số liệu chính: ứng viên chính (chọn theo val) không cải thiện có ý nghĩa; ens6 (+0.0038) chỉ là dấu hiệu post-hoc.",
                "Muốn trích vào báo cáo: dùng `ensemble_metrics.json` + `significance.json` (không cần đọc lại CSV).",
            ),
            keep_policy=KEEP_POLICY,
        ),
        # ---------------- paper ---------------- #
        "papers/figures": FolderDoc(
            title="papers/figures - hình chính của bài báo",
            purpose="Hình PNG của bài báo: phần lớn **copy tự động** từ `outputs/evaluation/comparison/` (script 03) để `main.tex` chèn được; riêng `system_architecture.png` do script 30 vẽ trực tiếp. Không vẽ tay.",
            produced_by="`python scripts/03_compare_models.py` (hình so sánh model); `python scripts/30_plot_system_architecture.py` (sơ đồ kiến trúc toàn hệ thống)",
            contents=(
                ("macro_f1_all_models.png / accuracy_all_models.png", "So sánh tổng hợp giữa các model."),
                ("per_class_f1_all_models.png / f1_summary_all_models.png", "F1 theo từng lớp."),
                ("confusion_matrices.png", "Lưới confusion matrix của các model."),
                ("pr_*.png / roc_*.png", "Đường PR/ROC theo từng nhãn và macro."),
                ("validation_loss_all_models.png / validation_accuracy_all_models.png", "Learning curve trên validation."),
                ("system_architecture.png", "Sơ đồ kiến trúc toàn hệ thống: 2 nhánh (AI mDeBERTa-v3 + multi-view head + rule engine lexicon 668 term) hợp nhất qua fusion p'=p+βr (AI trọng số 1, rule hạ trọng số β) rồi ngưỡng đóng băng - dành cho mục Methodology của bài báo."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "File ở đây là **bản copy**; số liệu gốc nằm ở `outputs/evaluation/comparison/` (riêng `system_architecture.png` do script 30 sinh trực tiếp).",
            ),
            keep_policy="Sinh lại mỗi lần chạy `03_compare_models.py`; sửa tay sẽ bị ghi đè.",
        ),
        "papers/template": FolderDoc(
            title="papers/template - template LaTeX LNCS của hội nghị",
            purpose="Bộ file template LNCS (Springer) dùng để build `papers/main.tex` và bài phụ lục.",
            produced_by="Tải về từ trang hội nghị; **không** script nào ghi vào đây.",
            contents=(
                ("llncs.cls / splncs04.bst", "Class file và style bibliography của LNCS."),
                ("samplepaper.tex / llncsdoc.pdf / readme.txt / history.txt", "Bài mẫu và tài liệu của template."),
                ("fig1.eps", "Hình minh hoạ của bài mẫu."),
                ("explain.md", "File này."),
            ),
            reading_guide=("Chỉ cần khi biên dịch bài; không chứa kết quả thực nghiệm.",),
            keep_policy="**KHÔNG SỬA** - giữ template gốc để build không lệ thuộc chỉnh sửa cục bộ.",
        ),
        "papers/supplementary": FolderDoc(
            title="papers/supplementary - bài phụ lục (supplementary material)",
            purpose=(
                "Bản supplementary riêng: bảng/hình chi tiết không đưa vào bài chính (calibration, confusion counts, "
                "per-class, thứ hạng model, chồng lấn lỗi...)."
            ),
            produced_by="`main.tex` viết tay; bảng/hình do `python scripts/11_supplementary_analyses.py` và `python scripts/03_compare_models.py` sinh.",
            contents=(
                ("tables/", "Bảng LaTeX sinh tự động (calibration, confusion counts, error overlap, hyperparameters, per-class, ranking...)."),
                ("figures/", "Hình sinh tự động (reliability, error overlap, confusion grid, và bản copy theo model)."),
                ("sections/", "Các mục LaTeX của bài phụ lục."),
                ("main.tex / llncs.cls", "Khung bài phụ lục, dùng template LNCS."),
                ("explain.md", "File này."),
            ),
            reading_guide=("Số liệu chi tiết cho phản biện: xem `tables/` trước, rồi `figures/`.",),
            keep_policy="Giữ `main.tex` và `sections/`; `tables/` + `figures/` sinh lại được.",
        ),
        "papers/supplementary/figures": FolderDoc(
            title="papers/supplementary/figures - hình của bài phụ lục",
            purpose="Hình cho bài phụ lục: độ tin cậy (reliability), chồng lấn lỗi giữa các hệ thống, confusion grid, threshold sweep, và bản copy hình so sánh.",
            produced_by="`python scripts/11_supplementary_analyses.py` (+ copy từ `outputs/evaluation/comparison/`)",
            contents=(
                ("reliability.png", "Biểu đồ độ tin cậy (calibration) theo bin."),
                ("error_overlap.png", "Mức chồng lấn lỗi giữa các model."),
                ("confusion_grid.png", "Lưới confusion matrix."),
                ("threshold_sweep.png", "macro-F1 theo ngưỡng."),
                ("comparison/", "Bản copy hình so sánh giữa các model."),
                ("per_model/", "Hình riêng từng model (M1..M6)."),
                ("explain.md", "File này."),
            ),
            reading_guide=("`reliability.png` + `error_overlap.png` là phần bổ sung quan trọng khi phản biện về hiệu chuẩn và mức giống nhau giữa các model.",),
            keep_policy="Sinh lại được; không sửa tay.",
        ),
        "papers/supplementary/figures/*": FolderDoc(
            title="papers/supplementary/figures/<nhóm> - hình phụ lục theo nhóm",
            purpose="Thư mục con của hình phụ lục: `comparison/` (so giữa các model) và `per_model/` (từng model).",
            produced_by="`python scripts/03_compare_models.py` / `python scripts/11_supplementary_analyses.py`",
            contents=(
                ("*.png", "Hình cần cho bài phụ lục (macro-F1, per-class F1, PR/ROC, learning curve, confusion matrix)."),
                ("per_model/M*/", "Mỗi model một thư mục con (M1..M6)."),
                ("explain.md", "File này."),
            ),
            reading_guide=("Tên file nói rõ nội dung; số liệu gốc nằm trong `outputs/evaluation/`.",),
            keep_policy="Sinh lại được; không sửa tay.",
        ),
        "papers/supplementary/figures/per_model/*": FolderDoc(
            title="papers/supplementary/figures/per_model/MX - hình của một model",
            purpose="Hình riêng cho từng model AI (M1..M6): confusion matrix chuẩn hoá, PR/ROC và learning curve.",
            produced_by="`python scripts/11_supplementary_analyses.py`",
            contents=(
                ("confusion_matrix_normalized.png", "Confusion matrix chuẩn hoá theo hàng."),
                ("pr_curve.png / roc_curve.png", "Đường PR/ROC theo từng nhãn."),
                ("val_loss_curve.png / val_accuracy_curve.png", "Learning curve trên validation."),
                ("explain.md", "File này."),
            ),
            reading_guide=("So `val_loss_curve.png` giữa các model để thấy model nào hội tụ sớm/muộn.",),
            keep_policy="Sinh lại được; không sửa tay.",
        ),
        "papers/supplementary/sections": FolderDoc(
            title="papers/supplementary/sections - các mục LaTeX của bài phụ lục",
            purpose="Nội dung bài phụ lục tách thành file nhỏ, chủ yếu `\\input` các bảng/hình đã sinh.",
            produced_by="Viết tay.",
            contents=(
                ("comparison_figures.tex", "Mục so sánh giữa các model."),
                ("per_system_figures.tex", "Mục trình bày từng hệ thống."),
                ("explain.md", "File này."),
            ),
            reading_guide=("Sửa nội dung chữ ở đây; số liệu lấy từ `papers/supplementary/tables/`.",),
            keep_policy="**KHÔNG XÓA** - tên file phải khớp lệnh `\\input` trong `papers/supplementary/main.tex`.",
        ),
        "papers/supplementary/tables": FolderDoc(
            title="papers/supplementary/tables - bảng LaTeX sinh tự động của bài phụ lục",
            purpose="Bảng chi tiết cho phản biện: calibration, confusion counts, error overlap, hyperparameters, per-class, thứ hạng model, kiểm định từng cặp.",
            produced_by="`python scripts/11_supplementary_analyses.py` (+ `python scripts/06_analyze_significance.py`)",
            contents=(
                ("calibration.tex", "Sai số hiệu chuẩn (ECE...) theo model."),
                ("confusion_counts.tex", "Số lượng trong confusion matrix."),
                ("error_overlap.tex", "Mức chồng lấn lỗi giữa các model."),
                ("hyperparameters.tex", "Hyperparameter của các run."),
                ("per_class.tex", "P/R/F1 theo từng lớp cho mọi model."),
                ("ranking.tex", "Xếp hạng model theo macro-F1."),
                ("pairs_all.tex", "Kiểm định từng cặp model."),
                ("explain.md", "File này."),
            ),
            reading_guide=("Số cho phần 'phân tích bổ sung' của bài chính: `per_class.tex`, `pairs_all.tex`.",),
            keep_policy="Sinh lại mỗi lần chạy script; sửa tay sẽ bị ghi đè.",
        ),
        "papers": FolderDoc(
            title="paper - khung bài báo",
            purpose="Khung bài báo LaTeX. Bảng và hình được **sinh tự động** từ kết quả thực nghiệm, không gõ tay.",
            produced_by="`python scripts/03_compare_models.py` (sinh bảng + hình); `main.tex` và `references.bib` viết tay.",
            contents=(
                ("main.tex", "File chính: tiêu đề, tác giả, abstract, các mục, và lệnh `\\input` các bảng/hình đã sinh."),
                ("references.bib", "13 tài liệu tham khảo (ViHSD, mDeBERTa-v3, focal loss, curriculum, PR vs ROC, ...)."),
                ("explain.md", "File này."),
            ),
            children=(
                ("sections/", "Các mục của bài báo - hiện là placeholder, cần điền sau khi có kết quả thật."),
                ("tables/", "Bảng LaTeX **sinh tự động**. Sửa tay sẽ bị ghi đè lần `03` kế tiếp."),
                ("figures/", "Hình PNG **copy tự động** từ `evaluation/comparison/`."),
            ),
            keep_policy=(
                "**GIỮ NGUYÊN `main.tex`, `references.bib` và `sections/`.** Riêng `tables/` và `figures/` "
                "được sinh lại mỗi lần chạy `03_compare_models.py`."
            ),
        ),
        "papers/sections": FolderDoc(
            title="paper/sections - các mục bài báo",
            purpose="Mỗi mục của bài báo nằm trong một file riêng để `main.tex` gọn và dễ sửa từng phần.",
            produced_by="Chỉnh tay. Không có script nào ghi vào thư mục này.",
            contents=(
                ("abstract.tex", "Tóm tắt."),
                ("introduction.tex", "Mở đầu, khoảng trống nghiên cứu, đóng góp."),
                ("related_work.tex", "Tài liệu liên quan."),
                ("dataset.tex", "Dữ liệu ViHSD và tiền xử lý."),
                ("methodology.tex", "Phương pháp: encoder, head, loss, 4 hệ thống."),
                ("dlr_two_stage.tex", "Kỹ thuật DLR và huấn luyện 2 stage."),
                ("curriculum.tex", "Kỹ thuật curriculum concept 3 stage."),
                ("setup.tex", "Thiết lập thực nghiệm và quy chuẩn công bằng."),
                ("results.tex", "Kết quả trên test, một lần chạy duy nhất."),
                ("discussion.tex", "Thảo luận và giới hạn."),
                ("conclusion.tex", "Kết luận."),
                ("explain.md", "File này."),
            ),
            keep_policy=(
                "**KHÔNG XÓA.** Tên file phải khớp với lệnh `\\input` trong `paper/main.tex`; "
                "thêm mục mới thì thêm cả file `.tex` và lệnh `\\input` tương ứng."
            ),
        ),
        "papers/tables": FolderDoc(
            title="paper/tables - bảng LaTeX sinh tự động",
            purpose="Ba bảng cho bài báo, sinh từ kết quả thực nghiệm nên **không bao giờ có sai số nhập tay**.",
            produced_by="`python scripts/03_compare_models.py`",
            contents=(
                ("model_comparison.tex", "Bảng so sánh 4 model; giá trị tốt nhất mỗi cột được in đậm."),
                ("classification_results.tex", "Precision / Recall / F1 theo từng lớp của từng model."),
                ("ablation_results.tex", "Bảng ablation: biến thể, head, DLR, số stage, loss - trả lời đóng góp của từng kỹ thuật."),
                ("explain.md", "File này."),
            ),
            keep_policy=(
                "**NỘI DUNG ĐƯỢC GHI ĐÈ MỖI LẦN CHẠY `03`.** Muốn chỉnh bảng, hãy sửa "
                "`src/comparison/compare_models.py` chứ đừng sửa file trong này."
            ),
        ),
        # ---------------- docs ---------------- #
        "docs": FolderDoc(
            title="docs - tài liệu sinh tự động",
            purpose=(
                "Tài liệu do script sinh ra, không sửa tay. Hiện có `folder_index.md` - bảng tra cứu "
                "mọi thư mục trong dự án."
            ),
            produced_by="`python scripts/05_document_folders.py --index docs/folder_index.md`",
            contents=(
                ("folder_index.md", "Chỉ mục toàn bộ thư mục: tên, đường dẫn `explain.md`, kích thước, số file."),
                ("explain.md", "File này."),
            ),
            reading_guide=(
                "Muốn biết một thư mục dùng để làm gì: mở `folder_index.md` rồi bấm vào `explain.md` tương ứng.",
                "Muốn tài liệu đầy đủ hơn `README.md`: xem `README.md` ở thư mục gốc.",
            ),
            keep_policy="**ĐƯỢC GHI ĐÈ** mỗi lần chạy `05_document_folders.py`. Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py`.",
        ),
    }


# --------------------------------------------------------------------------- #
# Pattern matching
# --------------------------------------------------------------------------- #
def _match(rel_path: str, specs: Dict[str, FolderDoc]) -> Optional[FolderDoc]:
    if rel_path in specs:
        return specs[rel_path]
    for key, doc in specs.items():
        if "*" not in key:
            continue
        if fnmatch.fnmatchcase(rel_path, key.rstrip("/")):
            return doc
    return None


def spec_for(rel_path: str) -> Optional[FolderDoc]:
    """Find the registry entry that documents ``rel_path`` (POSIX, relative to root).

    Exact matches win, then ``*`` patterns (each ``*`` = exactly one directory
    level).  Smoke-test trees such as ``reports_smoke/`` reuse the description of
    their real counterpart (``reports/``) so the two never drift apart.
    """
    specs = _specs()
    normalised = rel_path.strip("/")
    doc = _match(normalised, specs)
    if doc is not None:
        return doc
    if normalised.startswith("outputs/"):
        stripped = normalised[len("outputs/"):].strip("/")
        doc = _match(stripped, specs)
        if doc is not None:
            return doc
        if "_smoke" in stripped:
            doc = _match(stripped.replace("_smoke", ""), specs)
            if doc is not None:
                return doc
    if normalised.startswith("outputs_smoke/"):
        stripped = normalised[len("outputs_smoke/"):].strip("/")
        doc = _match(stripped, specs)
        if doc is not None:
            return doc
        if "_smoke" in stripped:
            doc = _match(stripped.replace("_smoke", ""), specs)
            if doc is not None:
                return doc
    if "_smoke" in normalised:
        return _match(normalised.replace("_smoke", ""), specs)
    return None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt_size(size: int) -> str:
    step = 1024.0
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < step or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} GB"


def _fmt_mtime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return "-"


def describe_children(folder: Path) -> List[Tuple[str, str, str]]:
    """Return ``(name, kind, detail)`` for every direct child, folders first."""
    rows: List[Tuple[str, str, str]] = []
    try:
        entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError:
        return rows
    for entry in entries:
        if entry.is_dir():
            n_files = sum(1 for p in entry.rglob("*") if p.is_file())
            size = sum(p.stat().st_size for p in entry.rglob("*") if p.is_file())
            rows.append((entry.name, "thư mục", f"{n_files} file, {_fmt_size(size)}"))
        else:
            rows.append((entry.name, "file", f"{_fmt_size(entry.stat().st_size)} · {_fmt_mtime(entry)}"))
    return rows


def _contents_table(items: Sequence[Tuple[str, str]]) -> List[str]:
    if not items:
        return []
    lines = ["| File / thư mục | Ý nghĩa |", "|---|---|"]
    lines.extend(f"| `{name}` | {desc} |" for name, desc in items)
    return lines


def _child_table(items: Sequence[Tuple[str, str]]) -> List[str]:
    if not items:
        return []
    lines = ["| Thư mục con | Vai trò |", "|---|---|"]
    lines.extend(f"| `{name}` | {desc} |" for name, desc in items)
    return lines


def _run_index_block(folder: Path) -> List[str]:
    """Table of ``run_*`` subfolders with status - only for model-level folders."""
    runs = sorted([p for p in folder.glob("run_*") if p.is_dir()])
    if not runs:
        return []
    lines = [
        "## Danh sách các lần chạy (`run_XXX`)",
        "",
        "Cột **Trạng thái**: `đã chấm test` = có `metrics.json` · `đã train` = có `training_summary.json` · "
        "`dở dang` = thư mục đã tạo nhưng chưa xong · `chỉ dry-run` = **không phải lần chạy thật** "
        "(chỉ có `config.yaml`, không có `training_log.csv`; các thư mục này do `--dry-run` để lại trước khi "
        "dry-run được sửa để không ghi file).",
        "",
        "| Run | Sửa lần cuối | Số epoch đã log | Best val macro F1 | Trạng thái |",
        "|---|---|---:|---|---|",
    ]
    for run in runs:
        summary = run / "training_summary.json"
        csv_path = run / "training_log.csv"
        epochs = "-"
        best = "-"
        state = "chỉ dry-run" if not (summary.is_file() or csv_path.is_file() or (run / "metrics.json").is_file()) else "dở dang"
        if (run / "metrics.json").is_file():
            state = "đã chấm test"
        if summary.is_file():
            state = "đã train" if state != "đã chấm test" else state
            try:
                payload = json.loads(summary.read_text(encoding="utf-8"))
                # best_* / total_epochs nằm trong khối "training" của summary
                training = payload.get("training") or {}
                history = payload.get("history") or training.get("history") or []
                epochs = str(len(history)) if history else str(training.get("epochs_run", "-"))
                raw_best = training.get("best_value")
                if isinstance(raw_best, (int, float)):
                    best = f"{float(raw_best):.4f}"
            except (OSError, ValueError, TypeError):
                pass
        if csv_path.is_file():
            try:
                epochs = str(sum(1 for _ in csv_path.open(encoding="utf-8")) - 1)
            except OSError:
                pass
        lines.append(f"| `{run.name}` | {_fmt_mtime(run)} | {epochs} | {best} | {state} |")
    lines.append("")
    lines.append(
        "Cột *Sửa lần cuối* là thời điểm sửa file trong thư mục run - dùng để biết lần chạy nào mới nhất."
    )
    lines.append("")
    lines.append(
        "> Bảng này do `scripts/05_document_folders.py` sinh lại. Cần làm mới sau mỗi lần train: "
        "`python scripts/05_document_folders.py`"
    )
    return lines


def render_explain(folder: Path, rel_path: str) -> str:
    """Build the markdown text for ``folder``."""
    doc = spec_for(rel_path)
    title = doc.title if doc else f"`{rel_path or '.'}`"
    lines: List[str] = [GENERATED_MARKER, f"# {title}", ""]

    if doc is None:
        lines += [
            "> Folder này **không có** mô tả trong registry `src/common/folder_docs.py`.",
            "> Nhiều khả năng là thư mục sinh ra ngoài dự kiến - hãy thêm mô tả vào registry.",
            "",
        ]
    else:
        lines += [f"> **Dùng để làm gì:** {doc.purpose}", ""]
        lines += [f"> **Sinh ra bởi:** {doc.produced_by}", ""]
        lines += [f"> **Quy tắc giữ lại:** {doc.keep_policy}", ""]

    if "_smoke" in rel_path:
        lines += [
            "> ⚠️ **Đây là output của SMOKE TEST** (dữ liệu cắn nhỏ, 1 epoch, checkpoint tắt). "
            "Dùng để kiểm tra code chạy được hay không - **không phải kết quả nghiên cứu**, "
            "không được trích dẫn. Giữ lại để đối chiếu khi code thay đổi.",
            "",
        ]

    if doc and doc.contents:
        lines += ["## File cố định của folder này", ""] + _contents_table(doc.contents) + [""]

    if doc and doc.children:
        lines += ["## Thư mục con và vai trò", ""] + _child_table(doc.children) + [""]

    index = _run_index_block(folder)
    if index:
        lines += index

    rows = describe_children(folder)
    if rows:
        lines += [
            "",
            "## Nội dung hiện có trên đĩa",
            "",
            "| Tên | Loại | Kích thước · sửa lần cuối |",
            "|---|---|---|",
        ]
        lines += [f"| `{name}` | {kind} | {detail} |" for name, kind, detail in rows]
        lines += [""]

    if doc and doc.reading_guide:
        lines += ["## Cách đọc", ""]
        lines += [f"{i}. {step}" for i, step in enumerate(doc.reading_guide, start=1)]
        lines += [""]

    if doc and doc.details:
        lines += ["## Chi tiết", ""]
        for block in doc.details:
            lines.append(block)
        lines += [""]

    if doc and doc.extra_notes:
        lines += ["## Ghi chú", ""]
        lines += [f"- {note}" for note in doc.extra_notes]
        lines += [""]

    lines += [
        "---",
        "",
        f"_Sinh tự động bởi `scripts/05_document_folders.py` lúc "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
        "Muốn đổi nội dung, sửa registry trong `src/common/folder_docs.py` rồi chạy lại script._",
    ]
    return "\n".join(lines) + "\n"


def write_explain(folder: str | Path, rel_path: Optional[str] = None) -> Path:
    """Write ``explain.md`` into ``folder`` (always refreshed)."""
    path = Path(folder)
    ensure_dir(path)
    if rel_path is None:
        rel_path = relative_to_root(path)
    out = path / EXPLAIN_FILENAME
    out.write_text(render_explain(path, rel_path), encoding="utf-8")
    return out


def ensure_explain(folder: str | Path, rel_path: Optional[str] = None) -> Optional[Path]:
    """Create ``explain.md`` only when it is missing (used when a run starts)."""
    path = Path(folder)
    ensure_dir(path)
    out = path / EXPLAIN_FILENAME
    if out.is_file():
        return None
    return write_explain(path, rel_path)


def relative_to_root(path: Path) -> str:
    """POSIX path of ``path`` relative to the project root (``.`` for the root)."""
    try:
        rel = Path(path).resolve().relative_to(Path(PROJECT_ROOT).resolve())
    except ValueError:
        rel = Path(path).resolve().relative_to(Path(path).anchor)
    return "." if str(rel) in (".", "") else rel.as_posix()


def should_skip(folder: Path) -> bool:
    """Folders that must not be documented or walked into."""
    name = folder.name
    if name in {"__pycache__", ".git", ".idea", ".pytest_cache", ".mypy_cache"}:
        return True
    return name.startswith(".")


def iter_project_folders(root: Optional[Path] = None, max_depth: int = 6) -> Iterable[Path]:
    """Yield the project root and every documentable folder under it."""
    base = Path(root) if root else Path(PROJECT_ROOT)
    yield base
    stack: List[Tuple[Path, int]] = [(base, 0)]
    while stack:
        current, depth = stack.pop()
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except OSError:
            continue
        for child in children:
            if should_skip(child):
                continue
            yield child
            stack.append((child, depth + 1))


def document_tree(root: Optional[Path] = None, max_depth: int = 6) -> List[Path]:
    """Write ``explain.md`` in every folder of the tree; return the files written."""
    written: List[Path] = []
    for folder in iter_project_folders(root, max_depth=max_depth):
        written.append(write_explain(folder))
    return written


def index_markdown(root: Optional[Path] = None) -> str:
    """Compact index of the whole project, listing every documented folder."""
    base = Path(root) if root else Path(PROJECT_ROOT)
    lines = [
        "| Folder | Mục đích |",
        "|---|---|",
    ]
    for folder in iter_project_folders(base):
        rel = relative_to_root(folder)
        doc = spec_for(rel)
        purpose = " ".join((doc.purpose if doc else "(chưa có mô tả)").split())
        if len(purpose) > 140:
            purpose = purpose[:137] + "..."
        lines.append(f"| [`{rel}/`]({rel}/explain.md) | {purpose} |")
    return "\n".join(lines) + "\n"


__all__ = [
    "EXPLAIN_FILENAME",
    "GENERATED_MARKER",
    "FolderDoc",
    "document_tree",
    "ensure_explain",
    "index_markdown",
    "iter_project_folders",
    "relative_to_root",
    "render_explain",
    "spec_for",
    "write_explain",
]

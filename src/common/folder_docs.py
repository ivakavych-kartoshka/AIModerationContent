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
                ("configs/", "5 file YAML: data split + 4 cấu hình model."),
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
                ("preprocess.py", "Làm sạch + remap nhãn canonical + ghi `preprocessing_report.json`."),
                ("split.py", "Tạo split, tính SHA-256 từng file, kiểm tra toàn vẹn, ghi `split_manifest.json`."),
                ("dataset.py", "Tokenize `train/validation/test`, `TokenizedTextDataset`, `collate_fn` an toàn Windows (num_workers>0)."),
                ("prepare_data.py", "CLI cho `scripts/00_prepare_data.py`."),
                ("explain.md", "File này."),
            ),
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
                ("report.py", "`model_summary.txt` và khối thông tin môi trường để tái lập."),
                ("run_evaluation.py", "CLI đánh giá một model; **lưu trữ** kết quả cũ vào `history/` thay vì xóa."),
                ("explain.md", "File này."),
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
            children=(("vihsd/", "3 file CSV gốc của ViHSD: `train.csv`, `dev.csv`, `test.csv`."),),
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
        "data/processed": FolderDoc(
            title="data/processed - dữ liệu sau tiền xử lý",
            purpose="Kết quả làm sạch + remap nhãn canonical. Phục vụ kiểm tra, không phải nơi mô hình đọc.",
            produced_by="`python scripts/00_prepare_data.py --stage preprocess`",
            children=(("vihsd/", "CSV đã làm sạch + báo cáo tiền xử lý."),),
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
                ("comparison/", "Bảng và biểu đồ so sánh 4 model."),
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
                "cùng các biểu đồ PR/ROC/validation dùng cho bài báo."
            ),
            produced_by="`python scripts/03_compare_models.py`",
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
                "Bảng quan trọng nhất: `model_comparison.md` (macro F1, weighted F1, per-class F1, latency).",
                "Giải thích vì sao accuracy không phải số liệu chính: xem mục 2.2 của `README.md`.",
            ),
            keep_policy=KEEP_POLICY,
        ),
        # ---------------- paper ---------------- #
        "paper": FolderDoc(
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
        "paper/sections": FolderDoc(
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
        "paper/tables": FolderDoc(
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
"paper/figures": FolderDoc(
            title="paper/figures - hình cho bài báo",
            purpose="Bản sao PNG của các biểu đồ so sánh, đặt đúng tên để `main.tex` `\\includegraphics` được.",
            produced_by="`python scripts/03_compare_models.py` (copy từ `evaluation/comparison/`)",
            keep_policy="**NỘI DUNG ĐƯỢC GHI ĐÈ MỖI LẦN CHẠY `03`.** Bản gốc còn nguyên ở `evaluation/comparison/`.",
        ),
        "paper/supplementary": FolderDoc(
            title="paper/supplementary - tài liệu bổ sung",
            purpose="Chỗ để các tài liệu đi kèm bài báo: bảng chi tiết hơn, log đầy đủ, và thông tin tái lập.",
            produced_by="Chỉnh tay. Không có script nào ghi vào thư mục này.",
            keep_policy="**KHÔNG XÓA.** Trỏ tới đây từ `paper/main.tex` nếu dùng đến.",
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

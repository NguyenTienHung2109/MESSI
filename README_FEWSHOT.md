# Few-shot OOD: PACS single-source → single-target

Branch này cung cấp 7 pipeline CLIP đã tích hợp cho main table: CLIP zero-shot, Tip-Adapter-F, GDA+CLIP, LP++, CLAP, ProKeR và TIMO. Các pipeline dùng CLIP ViT-B/16 image + text, frozen encoder, và **inductive** inference. ProKeR có thêm solver RFF. LaplacianShot, TIM, α-TIM, PADDLE, EM-Dirichlet và Ours có sẵn dòng trống trong workbook để tổng hợp kết quả; branch này không cung cấp runner đã kiểm thử cho chúng.

## 1. Protocol

PACS gồm `art_painting`, `cartoon`, `photo`, `sketch`. Với mỗi K ∈ {1,2,4,8,16} và seed ∈ {42,43,44,45,46}, lấy ngẫu nhiên có tái lập **K ảnh có nhãn mỗi lớp tại một source**. Cùng source/seed/K dùng chung support cho tất cả methods và ba target còn lại. Mỗi target được chấm riêng, thành 12 hướng × 5 seed = 60 accuracy cho mỗi method và K. Các ảnh augmentation của support không tính là nhãn mới. CLIP zero-shot không dùng support, nên điểm mỗi target lặp lại trên 5 seed.

Mỗi run có immutable support manifest và checksum; target labels chỉ được đọc sau dự đoán. Điểm tổng hợp là **mean ± sample SD trên 5 seed**, với mỗi seed lấy trung bình đều 12 hướng. Đây là single-source/`total`; không lẫn với multi-source.

Các method CLIP dùng cấu hình OOD cố định, không chọn theo target labels. Những thay đổi so với validation/augmentation trong paper gốc:

| Key | Chính sách chính |
|---|---|
| `clip_zeroshot` | một prompt cố định `a photo of a {class}.` |
| `tip_adapter_f_imagenet_fixed` | ImageNet fixed α=1.17, β=1, source views |
| `gda_clip` | GDA từ source views + CLIP text logits, fixed α=1 |
| `lp_pp_frozen` | checkpoint chọn bằng source-support views; không có validation labels bổ sung |
| `clap_frozen` | CLAP hiệu chỉnh; 10 distinct views được lặp thành 20-entry training pool |
| `proker_imagenet_fixed` | exact solve; ImageNet/RN50 coefficients cố định theo K áp dụng cho ViT-B/16 |
| `proker_rff_imagenet_fixed` | xấp xỉ RFF, default 4096 components; báo riêng với exact |
| `timo_fixed` | 8 generic prompt templates, fixed fusion; không phải TIMO-S |

Không lấy các kết quả từ `legacy_*` làm CLAP/TaskRes đã sửa. Một số method khác còn được registry nhận diện, nhưng **7 pipeline trên là phạm vi được smoke-test end-to-end cho branch này**.

## 2. Chuẩn bị môi trường và dữ liệu

Python 3.10+; cài PyTorch và torchvision tương thích CUDA theo môi trường của bạn, sau đó:

```bash
python -m venv .venv
source .venv/bin/activate
# Cài torch, torchvision tương thích máy theo https://pytorch.org/get-started/locally/
pip install torch torchvision
pip install -r requirements-fewshot.txt
```

Chuẩn bị PACS theo DomainBed, khoảng 9.991 ảnh; đặt `DATA_DIR` là thư mục cha của `PACS`:

```text
$DATA_DIR/PACS/art_painting/<class>/*.jpg
$DATA_DIR/PACS/cartoon/<class>/*.jpg
$DATA_DIR/PACS/photo/<class>/*.jpg
$DATA_DIR/PACS/sketch/<class>/*.jpg
```

Bảy class và tên domain phải khớp giữa các domain. Dùng bản PACS DomainBed do nhóm cung cấp: script kiểm tra 9.991 ảnh, 4 domain, 7 lớp. Dữ liệu không được upload cùng code.

Tải checkpoint **OpenAI CLIP ViT-B/16** (335 MiB), hoặc dùng file có sẵn của nhóm:

```bash
mkdir -p checkpoints
curl -fL --retry 3 -o checkpoints/ViT-B-16.pt \
  https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
export DATA_DIR=/absolute/path/to/domainbed-data
export CLIP_CHECKPOINT="$PWD/checkpoints/ViT-B-16.pt"
export CUDA_VISIBLE_DEVICES=1  # physical GPU; cuda:0 trong process tương ứng GPU 1
export NUM_CPU_THREADS=1      # tổng các job của user không quá 8 threads
```

Không commit dataset, checkpoint hoặc cache. Kiểm tra tài nguyên và giới hạn của nhóm trước khi chạy: tối đa 2 GPU vật lý, ưu tiên GPU 1; tối đa 8 CPU threads và 64 GiB RAM tổng cộng; dung lượng backup dưới 300 GB (ưu tiên dưới 200 GB). Script prepare dùng một GPU, default batch 8; giảm `--batch-size` khi GPU đang được dùng chung.

## 3. Tạo cache và manifest

Chạy **một lần**, trước khi chạy methods:

```bash
scripts/run_pacs_fewshot.sh prepare --batch-size 8
```

Lệnh audit dataset và checkpoint checksum, tạo CLIP image features, text, 8 prompt embeddings, 10 source views và support manifests cho toàn bộ K/seed. Cache nằm ở `results/fewshot_ood/`; không cần đặt dataset trong repo. Thời gian và dung lượng phụ thuộc dataset. Không đổi ảnh/dataset, class mapping, seed hay augmentation giữa methods.

Để chuẩn bị một tập con K/seed cho kiểm thử nhanh:

```bash
scripts/run_pacs_fewshot.sh prepare --shots 1 --seeds 42 --batch-size 2
```

**Dùng worktree/cache riêng cho thử nhanh.** Lệnh prepare từ chối ghi đè cache đã tồn tại. Các tham số K/seed được khai báo khi prepare phải bao phủ các job sẽ chạy, vì view cache chỉ chứa những support IDs đã chọn. Không sửa file audit sau khi tạo candidate pools/runs: checksum được gắn vào kết quả. `--allow-nonstandard-dataset` chỉ dành cho smoke test, không dùng khi tạo kết quả paper.

## 4. Chạy benchmark

```bash
# 7 methods × 5 K × 5 seeds, tuần tự; các source trong mỗi job được xử lý riêng.
scripts/run_pacs_fewshot.sh run
```

Mặc định train các head trên CPU (`TRAINING_DEVICE=cpu`). Nếu còn GPU phù hợp, đặt `TRAINING_DEVICE=cuda:0`; `CUDA_VISIBLE_DEVICES` vẫn phải ghi rõ GPU vật lý. Với máy dùng chung, giảm batch khi extract và đảm bảo tổng tài nguyên của mọi job nằm trong giới hạn. Script `run` mặc định thực hiện full matrix, không phải một quick test.

Chạy một K/seed hoặc một method qua cùng wrapper (tham số phía sau thay thế danh sách mặc định):

```bash
scripts/run_pacs_fewshot.sh run --shots 1 --seeds 42
scripts/run_pacs_fewshot.sh run --methods gda_clip --shots 1 --seeds 42
```

Chạy ProKeR RFF tùy chọn thay cho exact (không gộp hai solver thành một kết quả):

```bash
python scripts/run_fewshot_matrix.py \
  --datasets PACS --backbones clip_vit_b16 --settings single_source --scopes total \
  --methods proker_rff_imagenet_fixed --shots 1 2 4 8 16 --seeds 42 43 44 45 46 \
  --rff-components 4096 --rff-device cpu
```

Khi dùng dataset/số lớp lớn ngoài PACS, exact ProKeR tạo ma trận kernel theo số views và có thể vượt 64 GiB; không khởi chạy nếu chưa ước tính RAM, dùng RFF khi phù hợp và ghi rõ solver. Cấu hình 512 components dùng trong smoke test chỉ kiểm tra thực thi, không thay thế 4096 trong benchmark.

Kết quả nằm ở:

```text
results/fewshot_ood/runs/PACS/clip_vit_b16/single_source/total/<K>/<seed>/<method>/<source-fold>.json
results/fewshot_ood/manifests/PACS/single_source/total/<K>/<seed>/<source-fold>.json
results/fewshot_ood/matrix_logs/
```

Trong mỗi JSON, `target_metrics` có một metric riêng cho mỗi target và `manifest_sha256` xác định support. Controller bỏ qua fold đã hoàn thành; không thay đổi tham số/feature rồi reuse output cũ. Dùng một checkout/cache kết quả mới cho cấu hình khác.

## 5. File Excel

Mở `fewshot_pacs_results.xlsx` bằng Excel/LibreOffice. Các sheet `K1`, `K2`, `K4`, `K8`, `K16` có 12 hướng cho từng method và 5 cột seed. Nhập **top-1 accuracy theo %, từ 0 đến 100**. Sheet `Summary` tự hiện mean ± SD khi đủ cả 60 ô; ô thiếu không tự tính trung bình từng phần. 14 methods bao gồm 7 pipeline được kiểm thử, ProKeR RFF, 5 baseline chưa tích hợp đầy đủ và Ours.

Có thể tạo một bản mới từ kết quả đã chạy:

```bash
python scripts/create_fewshot_workbook.py --import-runs results/fewshot_ood/runs --output results/pacs_filled.xlsx
```

Lệnh chỉ import các JSON PACS/CLIP đúng single-source, K và seed đã khai báo. Các dòng DINO/transductive/Ours vẫn nhập thủ công. Tạo workbook trống: `python scripts/create_fewshot_workbook.py --output results/pacs_blank.xlsx`. Excel sẽ tính công thức khi mở file; không coi ô trống là 0.

## 6. Kiểm thử

Branch đã được chạy smoke trên 56 ảnh PACS thật, 4 domain × 7 class × 2 ảnh, CLIP ViT-B/16, K=1, seed 42, 10 support views, đủ 4 source và 3 target/source, với cả bảy methods. Các con số smoke không phải điểm paper. Test protocol/classifier:

```bash
pip install pytest
python -m pytest -q tests/fewshot
```

Nếu controller dừng, đọc đường dẫn log được in ra; không dùng các ô thiếu như kết quả đầy đủ. Bản này không chứa ảnh, checkpoint, caches hay JSON kết quả smoke.

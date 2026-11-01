# Few-shot OOD: single-source → single-target, 5 datasets

Benchmark này hỗ trợ **PACS, VLCS, OfficeHome, TerraIncognita và DomainNet**. Bảy pipeline CLIP được tích hợp: CLIP zero-shot, Tip-Adapter-F, GDA+CLIP, LP++, CLAP, ProKeR và TIMO. Chúng dùng CLIP ViT-B/16 image + text, frozen encoder, và inductive inference. Excel còn có chỗ cho các phương pháp transductive và Ours; branch này chưa cung cấp runner đầy đủ cho các method đó.

## Protocol và số hướng

| Dataset | Folder trong `DATA_DIR` | Domain | Số source→target |
|---|---|---|---:|
| PACS | `PACS` | art_painting, cartoon, photo, sketch | 12 |
| VLCS | `VLCS` | C, L, S, V | 12 |
| OfficeHome | `office_home` | Art, Clipart, Product, Real World | 12 |
| TerraIncognita | `terra_incognita` | location_100, location_38, location_43, location_46 | 12 |
| DomainNet | `domain_net` | clipart, infograph, painting, quickdraw, real, sketch | 30 |

**DomainNet bao gồm Quickdraw cho tất cả method đã tích hợp**: 5 hướng `quickdraw → domain khác` và 5 hướng `domain khác → quickdraw`, cùng 20 hướng giữa các domain còn lại. Cùng một manifest support theo source/K/seed được dùng cho mọi method và target. Với K-shot không đủ ảnh ở bất kỳ class nào của source, cả 5 hướng từ source đó được đánh dấu `infeasible`, kể cả khi source là Quickdraw. Target luôn dùng toàn bộ ảnh hợp lệ; không loại Quickdraw khỏi test.

Audit yêu cầu đúng sáu tên domain trên. Controller kiểm tra mỗi file fold đã chạy có kết quả cho đủ năm target; file cũ thiếu Quickdraw sẽ báo lỗi thay vì được tính là hoàn tất. Nếu gặp file cũ thiếu target, chạy lại trong namespace kết quả mới; không sửa hoặc gộp thủ công kết quả cũ. Các setting này áp dụng cho `clip_zeroshot`, `tip_adapter_f_imagenet_fixed`, `gda_clip`, `lp_pp_frozen`, `clap_frozen`, `proker_imagenet_fixed` (hoặc key RFF riêng) và `timo_fixed`. Ours và patch-local vẫn chưa có runner trong nhánh này; không diễn giải workbook trống là kết quả đã chạy.

Với từng K ∈ {1,2,4,8,16}, seed ∈ {42,43,44,45,46} và source, lấy K ảnh có nhãn **mỗi lớp** trong source. Một support manifest được dùng chung cho tất cả methods và tất cả target còn lại. K ảnh được tính theo ID ảnh gốc; augmentation không thêm nhãn. Điểm chính là mean accuracy qua các hướng, rồi mean ± sample SD qua 5 seed. CLIP zero-shot không dùng support, vì vậy điểm một target lặp lại ở các seed.

**Không đủ K:** một source có lớp dưới K ảnh sau khi loại trùng được ghi là `infeasible`; không lấy ảnh từ domain khác, không chọn lớp con. Khi đó mọi hướng có source này được loại cho tất cả methods ở K tương ứng. Kết quả `† (m/n)` trong workbook là trung bình trên m hướng khả thi trong tổng n hướng. Nếu không có source khả thi, không có điểm. Không so sánh các method trên tập hướng khác nhau. Các target giữ phân bố lớp tự nhiên.

## Phương pháp và cách diễn giải

| Method key | Thiết lập OOD |
|---|---|
| `clip_zeroshot` | một prompt `a photo of a {class}.` |
| `tip_adapter_f_imagenet_fixed` | ImageNet fixed α=1.17, β=1, source views |
| `gda_clip` | **GDA visual + CLIP text logits**, fixed α=1 |
| `lp_pp_frozen` | checkpoint chọn bằng source-support views |
| `clap_frozen` | 10 distinct source views lặp thành pool 20 entries |
| `proker_imagenet_fixed` | exact solver, ImageNet/RN50 coefficients cố định theo K |
| `proker_rff_imagenet_fixed` | RFF approximation, 4096 components mặc định; báo solver riêng |
| `timo_fixed` | 8 generic prompt templates, fixed fusion; không phải TIMO-S |

Các fixed coefficients, augmentation và source-only checkpoint policy là **adaptation cho OOD protocol này**, không phải toàn bộ paper validation protocol gốc. Không dùng target labels để chọn hyperparameter. Excel có thêm LaplacianShot, TIM, α-TIM, PADDLE, EM-Dirichlet và Ours để nhập kết quả sau; chúng không thuộc 7 CLIP runner đã tích hợp ở đây.

## 1. Môi trường, dữ liệu và giới hạn

Python 3.10+. Cài PyTorch/torchvision tương thích CUDA, sau đó:

```bash
python -m venv .venv
source .venv/bin/activate
# Chọn torch/torchvision phù hợp tại https://pytorch.org/get-started/locally/
pip install torch torchvision
pip install -r requirements-fewshot.txt
```

Chuẩn bị các dataset theo cấu trúc DomainBed:

```text
$DATA_DIR/PACS/<domain>/<class>/<image>
$DATA_DIR/VLCS/<domain>/<class>/<image>
$DATA_DIR/office_home/<domain>/<class>/<image>
$DATA_DIR/terra_incognita/<domain>/<class>/<image>
$DATA_DIR/domain_net/<domain>/<class>/<image>
```

Dùng cùng bản dữ liệu cho toàn nhóm. Script audit kiểm tra số domain/lớp và số ảnh bản tham chiếu. Với OfficeHome, script hash nội dung để loại cross-domain byte duplicates. Dataset, checkpoints và caches không có trong Git.

Tải checkpoint OpenAI CLIP ViT-B/16 hoặc dùng file nhóm đã có:

```bash
mkdir -p checkpoints
curl -fL --retry 3 -o checkpoints/ViT-B-16.pt \
  https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
export DATA_DIR=/absolute/path/to/domainbed-data
export CLIP_CHECKPOINT="$PWD/checkpoints/ViT-B-16.pt"
export CUDA_VISIBLE_DEVICES=1  # ID GPU vật lý; trong process là cuda:0
export NUM_CPU_THREADS=1
```

Trước khi chạy, kiểm tra và phối hợp các job của user: tối đa 2 GPU vật lý, ưu tiên GPU 1; tổng cộng ≤8 CPU threads và ≤64 GiB RAM; dữ liệu backup <300 GB, ưu tiên <200 GB. Cache DomainNet và việc extract source views có thể mất nhiều thời gian/dung lượng. Script extract dùng một GPU, mặc định batch 8; dùng `--batch-size 1` hoặc `2` khi GPU đang bận. Khi resource không đủ, dừng và chạy sau.

## 2. Tạo cache và support manifests

Cho cả 5 dataset, **tuần tự**:

```bash
scripts/run_fewshot_ood.sh prepare all --batch-size 8
```

Hoặc một dataset:

```bash
scripts/run_fewshot_ood.sh prepare PACS --batch-size 8
scripts/run_fewshot_ood.sh prepare VLCS --batch-size 8
scripts/run_fewshot_ood.sh prepare OfficeHome --batch-size 8
scripts/run_fewshot_ood.sh prepare TerraIncognita --batch-size 8
scripts/run_fewshot_ood.sh prepare DomainNet --batch-size 8
```

Lệnh kiểm tra checksum checkpoint, audit dữ liệu, trích xuất image features và text embeddings, tạo 10 support views, và lưu manifest của tất cả seed/K **khả thi**. Lệnh từ chối ghi đè cache đã tồn tại. Nếu muốn kiểm thử một K/seed, chuẩn bị trong checkout/cache riêng:

```bash
scripts/run_fewshot_ood.sh prepare PACS --shots 1 --seeds 42 --batch-size 2
```

K/seed khi prepare phải bao phủ các run sẽ thực hiện. `--allow-nonstandard-dataset` chỉ dành cho smoke test; không dùng cho kết quả paper. Sau khi bắt đầu benchmark, không thay dataset, ảnh, audit hoặc tham số trong cùng namespace kết quả. Audit được hash riêng theo dataset, vì vậy chuẩn bị dataset khác không làm thay đổi provenance của dataset đã chạy.

## 3. Chạy các method

Mặc định chạy 5 dataset, 7 methods, 5 K và 5 seeds **tuần tự**:

```bash
scripts/run_fewshot_ood.sh run
```

Mặc định train heads trên CPU (`TRAINING_DEVICE=cpu`). Để chạy heads trên GPU đã chọn: `export TRAINING_DEVICE=cuda:0`. Nếu chỉ chạy một dataset hoặc method:

```bash
scripts/run_fewshot_ood.sh run --datasets OfficeHome --shots 1 --seeds 42
scripts/run_fewshot_ood.sh run --datasets PACS VLCS --methods gda_clip --shots 1 2
scripts/run_fewshot_ood.sh run --datasets DomainNet --methods gda_clip --shots 1 2 4 8 16 --seeds 42 43 44 45 46
```

ProKeR exact tạo ma trận kernel theo số support views. Wrapper ước lượng ba ma trận FP64: nếu vượt **48 GiB**, nó chuyển run đó sang key `proker_rff_imagenet_fixed` để giữ khoảng trống trong giới hạn 64 GiB; các key và kết quả được lưu riêng, không gộp vô điều kiện. RFF mặc định 4096 components và chạy CPU; có thể đặt `RFF_DEVICE=cuda:0` nếu đủ GPU. Các job lớn vẫn cần theo dõi tổng RAM/GPU. Để cố định solver riêng:

```bash
scripts/run_fewshot_ood.sh run --datasets DomainNet --methods proker_rff_imagenet_fixed --rff-components 4096
scripts/run_fewshot_ood.sh run --datasets PACS --methods proker_imagenet_fixed
```

Kết quả:

```text
results/fewshot_ood/runs/<dataset>/clip_vit_b16/single_source/total/<K>/<seed>/<method>/<source-fold>.json
results/fewshot_ood/manifests/<dataset>/single_source/total/<K>/<seed>/<source-fold>.json
results/fewshot_ood/infeasible/<dataset>/single_source/total/<K>/<seed>/<source-fold>.json
results/fewshot_ood/matrix_logs/
```

`target_metrics` trong một JSON chứa kết quả từng target; `manifest_sha256` xác định support. Controller resume fold đã có. Các source infeasible có file giải thích riêng và tính vào coverage; không giả định là 0% hoặc bỏ sót. Nếu run thất bại đọc log trước khi chạy lại. Không dùng namespace cũ với cấu hình/feature đã đổi. Bản sửa này hash audit theo từng dataset, vì vậy candidate pools tạo bằng version trước với hash toàn bộ file audit không tương thích; dùng namespace kết quả mới thay vì trộn.

## 4. Excel

Mở `fewshot_ood_results.xlsx`: sheet `Summary` có từng dataset × method và cột 1/2/4/8/16-shot; **25 sheet** `<dataset>_K<shot>` ghi chi tiết từng hướng và từng seed. Nhập top-1 accuracy theo %, ví dụ `83.25`. Công thức chỉ hiện kết quả khi mỗi hướng đã có đủ 5 seed, hoặc cả 5 ô ghi `N/A` cho source không khả thi. `† (m/n)` báo chỉ trung bình trên m hướng khả thi trong n hướng; nếu toàn bộ source không khả thi, summary để trống. Các ô chưa chạy để trống, không phải 0.

Import kết quả được chuẩn bị bởi runner:

```bash
python scripts/create_fewshot_workbook.py --import-runs results/fewshot_ood/runs --output results/fewshot_filled.xlsx
```

Lệnh import JSON CLIP của **cả 5 dataset** và tự đọc `results/fewshot_ood/infeasible` để đánh dấu `N/A` cho mọi method ở cùng hướng/seed. Các method DINO/transductive/Ours có thể điền thủ công. Excel/LibreOffice tính công thức khi mở. Tạo workbook trống bằng `python scripts/create_fewshot_workbook.py --output results/fewshot_blank.xlsx`.

## 5. Kiểm thử

```bash
pip install pytest
python -m pytest -q tests/fewshot
```

Các smoke test dùng số ảnh/domain thu nhỏ chỉ xác minh pipeline; không đưa độ chính xác smoke vào paper. Kết quả chạy đầy đủ cần kiểm tra đủ method, hướng khả thi, 5 seed và đúng manifest trước khi tổng hợp.

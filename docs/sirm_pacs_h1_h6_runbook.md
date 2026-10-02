# SIRM trên PACS: setup và kế hoạch H1–H6

Ngày: 2026-10-02. Branch: `SIRM`. Code nền được kiểm tra: `e3a422e`.

## 1. Phạm vi và trạng thái

Đây là runbook bàn giao để triển khai/chạy trên máy khác, không phải báo cáo kết quả.
Branch đã có pipeline train SIRM, config PACS và tests. **Chưa có bộ runner hoàn chỉnh H1–H6**, split 60/20/20, frozen-bank router fitting hoặc Q-Regret theo thiết kế dưới đây. Các tên file trong mục TODO là đề xuất, không phải CLI đã tồn tại.

Tài liệu này không đồng bộ các chỉnh sửa local chưa commit, checkpoint, dataset hoặc script chẩn đoán chưa được track. Các biến thể HardQ mới trong workspace máy cũ không thuộc snapshot nền này. Không dùng tên run HardQ nếu chưa tự bổ sung chúng.

PACS không có nhãn cơ chế thật: expert suitability, biến đổi ảnh và invariance gap chỉ là proxy. Không gọi subset cố định hoặc expert-loss oracle là ground-truth mechanism.

## 2. Setup máy mới

```bash
git clone --branch SIRM --single-branch https://github.com/NguyenTienHung2109/messi.git
cd messi
git rev-parse HEAD
conda create -n sirm-pacs python=3.9 -y
conda activate sirm-pacs
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install 'git+https://github.com/microsoft/tutel@69b0540970dd989c6f0b46f4c93754338182802c'
python -m pip check
nvidia-smi
```

Đây là đường setup theo dependency snapshot của repo, chưa được cài thử trong môi trường sạch cho tài liệu này. Cần Linux/NVIDIA, driver tương thích CUDA 12.8; build Tutel có thể cần compiler/CUDA toolkit. Không dùng hướng dẫn CUDA 11.6 cũ trong README cùng snapshot PyTorch này. Nếu GPU khác yêu cầu build khác, ghi rõ thay đổi và khóa environment trước khi so sánh.

```bash
python - <<'PY'
import torch
from domainbed import algorithms, datasets
print(torch.__version__, torch.version.cuda)
assert torch.cuda.is_available(), 'Thí nghiệm này yêu cầu CUDA'
print(torch.cuda.get_device_name(0))
PY
python -m domainbed.scripts.train_subset_irm_pacs --help
python -m unittest -q domainbed.test.test_subset_irm
```

### Dữ liệu và pretrained weights

Chuyển/tải PACS từ nguồn chính thức: https://domaingeneralization.github.io/ . Runner nhận thư mục CHA của `PACS`, ví dụ:

```text
domainbed/data/PACS/
  art_painting/<class>/*
  cartoon/<class>/*
  photo/<class>/*
  sketch/<class>/*
```

Kiểm tra bốn domain có cùng 7 tên class; lưu file inventory và hashes/split IDs. Domain IDs: 0 art_painting, 1 cartoon, 2 photo, 3 sketch. Không đặt thêm tầng `PACS_Original` bên trong mà không điều chỉnh đường dẫn.

Backbone DeiT-Small có thể tải weights qua mạng. Máy offline: chuyển file `deit_small_distilled_patch16_224-649709d9.pth` vào `domainbed/pretrained/`; xem README tại đó. Không chuyển checkpoint huấn luyện vào Git.

### Smoke thực sự chạy được trên code nền

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
python -m domainbed.scripts.train_subset_irm_pacs \
  --config configs/subset_irm_pacs.json --data-dir ./domainbed/data \
  --run Smoke5 --target-env 1 --steps 30 --checkpoint-freq 10 \
  --skip-target-eval --output-dir subset_irm_outputs/hypotheses/setup_smoke
```

Dùng output-dir mới cho mỗi lần chạy. `--skip-target-eval` cũng bỏ đánh giá target cuối run: phải thêm evaluator riêng sau khi khóa checkpoint. Không suy luận accuracy từ smoke 30 bước.

Ví dụ tạo bank pilot cũ (80/20, CHƯA phải protocol H chính thức):

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
python -m domainbed.scripts.train_subset_irm_pacs \
  --config configs/subset_irm_pacs.json --data-dir ./domainbed/data \
  --run SIRMCurrentNoExpertQCapOff --target-env 1 \
  --steps 5001 --checkpoint-freq 500 --skip-target-eval \
  --output-dir subset_irm_outputs/hypotheses/pilot_env1_seed0
```

Code nền đọc seed từ JSON, không có `--seed`: tạo bản sao config với trường `seed` tương ứng trước khi chạy seed khác. Lưu ý run pilot trên có lịch regularizer riêng; không dùng nó làm bằng chứng so sánh với method có lịch khác. Checkpoint: `checkpoints/best.pkl`, `checkpoints/last.pkl`.

## 3. Protocol chính cần triển khai trước full runs

- Bốn lượt leave-one-domain-out; target hoàn toàn không tham gia chọn model/hyperparameter.
- Pilot seeds 0,1,2; xác nhận seeds 0–4. Cùng split/initialization cho so sánh theo cặp.
- Trong từng source/class: 60% fit, 20% probe, 20% validation, không giao nhau. Mọi biến thể của một ảnh nằm cùng split.
- Fit: encoder/experts/head. Probe: teacher, statistics, auxiliary router fitting. Validation: chọn checkpoint/hyperparameter. Probe là training data.
- Cùng DeiT-Small pretrained, 6 experts, classifier riêng, Top-2, 5001 steps, dense warmup 500; batch 32/source = 96 tổng. LR khởi điểm 3e-5, weight decay 1e-6. Khóa config trước target.
- Chọn checkpoint bằng accuracy trung bình đều qua ba source-validation domains, không theo target. Quy tắc tie: checkpoint sớm hơn.
- Target chính: toàn bộ ảnh domain held-out, chỉ sau khi khóa model. Đánh dấu rõ nếu đối chiếu kết quả cũ chỉ dùng target-in.
- Frozen-bank: encoder/experts/heads ở eval, requires_grad=False; chỉ fit router trên source-probe.
- Joint-training: CE và SIRM cập nhật như thiết kế; loss phụ router dùng `router(z_moe.detach())` và teacher/statistics detach. Không detach CE chính ngoài ý muốn.
- SIRM vẫn cập nhật encoder/projection/experts/classifier nếu không freeze; Q và routing responsibilities detach trong penalty.
- Không dùng test-label oracle để fit, chọn checkpoint, chọn temperature hay chọn assignment.

Lưu effective config, git SHA, split manifest, package versions, pretrained hash, GPU, seed, thời gian, VRAM. Dùng `pip freeze > environment.freeze.txt` trong thư mục output.

## 4. Cache và metrics dùng chung

Cache bằng deterministic eval transform, theo image ID:
`domain, split, class, expert_logits[M,C], router_dense[M], router_topk[M], projected_feature[D]`.
Thêm transform ID/parent image ID cho paired interventions. Source và target cache tách riêng. Không dùng target cache để fit bất cứ thành phần nào.

Đặt `ell_im = CE(expert_logits_im, y_i)`:
- Accuracy/NLL của mixture logits.
- Selection loss = mean_i sum_m pi_im ell_im.
- Routing regret = mean_i(sum_m pi_im ell_im - min_m ell_im).
- Top-1 expert hit-rate, oracle single-expert accuracy, expert loads, entropy, ESS.

Selection loss khác CE của logits trộn. Oracle tối ưu CE khác oracle tối ưu correctness; đặt tên riêng. Không coi single-expert oracle là upper bound của mọi mixture.

## 5. H1: Input dự đoán được expert suitability

1. Đóng băng bank, tạo teacher `softmax(-ell/T)` từ source-probe.
2. Fit router với KL teacher-to-router; temperature và early stopping chọn từ source-validation.
3. So sánh uniform, global learned weights, native router, teacher-trained input router, label oracle (chẩn đoán).
4. Standardize feature bằng thống kê source-probe; train router riêng với Gaussian noise sigma = 0, 0.25, 0.5, 1. Negative control: shuffle feature tại eval.
5. Đo accuracy/NLL/regret trên từng target; báo cáo expert-loss margin để xử lý ties.

Ủng hộ: input router hơn global weights trên target, lợi ích mất khi shuffle. Oracle tốt nhưng router kém => chưa học được tín hiệu chuyển miền. Oracle không hơn global => bank chưa tạo headroom. Đây là suitability, không phải nhận diện causal mechanism.

## 6. H2: Bền với thay đổi tỷ lệ dạng input

1. Tạo cùng ảnh dưới ba dạng: gốc, grayscale, blur nhẹ; chọn blur trên source, lưu kernel/sigma và thư viện.
2. Eval mọi dạng một lần; weighted metrics với rho = (1/3,1/3,1/3), (0.8,0.1,0.1), (0.1,0.8,0.1), (0.1,0.1,0.8). Giữ class proportions.
3. H2a: bank train augmentation chuẩn. H2b: train lại mọi đối chứng với ba dạng xuất hiện đều trên source; chỉ đổi tỷ lệ khi test.
4. So input router với global weights/uniform/oracle trên cùng bank.
5. Báo cáo worst-mixture accuracy, NLL, regret từng dạng và chênh lệch với global weights.

Ủng hộ: lợi ích routing duy trì khi rho đổi. Đây là controlled observation-mixture shift, không phải chứng minh latent mechanism shift. Không mặc định grayscale/blur giữ nguyên cơ chế; kiểm tra khả năng nhận diện nhãn trên source.

## 7. H3: Router không chỉ dựa vào style shortcut

A. Paired interventions với brightness/contrast/saturation nhẹ, mức cố định từ source:
- Gốc: sum_m pi_m(x) f_m(x).
- Router-only: sum_m pi_m(Tx) f_m(x), giữ expert logits gốc.
- Full: sum_m pi_m(Tx) f_m(Tx).

Đo JS divergence routing, Top-2 support change, regret trên logits gốc và accuracy drop. Routing đổi không tự nó là lỗi; competence có thể đổi trong nhánh full.

B. Train stress-test riêng với viền màu nhỏ ngoài nội dung: source domain có màu cố định, màu không mã hóa class. So với training không viền. Validation: đúng mapping/tráo/bỏ viền. Target: màu độc lập class, không có khái niệm màu target đúng. Báo cáo riêng PACS modified.

Ủng hộ: router-only can thiệp nuisance không làm mất mạnh chất lượng. Domain-probe accuracy cao riêng lẻ không chứng minh shortcut. Không có paired ground-truth mechanism intervention trong PACS.

## 8. H4: Routing theo sample cần thiết

1. Trên cùng frozen bank, fit/eval: uniform-6, global learned weights, fixed pair với weights học, input Top-2, input dense.
2. Chọn cặp trong 15 cặp và fit weights chỉ từ source-probe; hyperparameter từ validation.
3. Shuffle routing toàn bộ, và shuffle trong cùng (domain,class), giữ nguyên expert outputs. Nhãn chỉ dùng cho chẩn đoán shuffle.
4. Đo paired accuracy/NLL/regret delta trên từng target/class.

Ủng hộ mạnh: input router hơn global/fixed pair, và shuffle trong class/domain vẫn gây hại. Nếu không, chưa cần sample-level routing. Chạy H4 đầu tiên vì rẻ và trực tiếp.

## 9. H5: Experts bổ sung nhau

1. Frozen-bank: Top-1, Top-2, dense, uniform, best global expert, label-oracle single expert.
2. Diagnostic pair oracle: 15 cặp, weights grid 0,0.1,...,1, tối thiểu CE từng sample; không dùng train/tuning. Đây chỉ là grid oracle.
3. Bỏ từng expert rồi renormalize: đo accuracy drop ngay lập tức; fit lại router trên source-probe để đo khả năng thay thế.
4. Joint confirmation: train Top-1 và Top-2 từ cùng initialization/split/schedule. Bank train Top-2 có thể ưu ái Top-2 trong frozen evaluation.
5. Nếu thử feature-concatenation head, thêm head-capacity control; đó là kiến trúc khác.

Đo NLL/accuracy, error disagreement, ablation drops, params/FLOPs/VRAM/time. Code hiện tại compute-dense dù gradient-routing sparse; không gán savings FLOPs theo Top-k lý thuyết.

Ủng hộ: combination hơn selection, nhiều expert có đóng góp khó thay thế. Ensemble gain chưa chứng minh đã tách đúng Venn mechanisms.

## 10. H6: Tiêu chí assignment phù hợp invariance

Đối chứng: Q-Random cố định, Q-Risk, Q-Grad, Q-Regret, Q-FixedPairs, Q-Global.

### Kiểm soát cấu trúc

Nhánh so tiêu chí: 2 assignments/domain; mọi expert hoạt động phải có >=2 domain. Tổng 6 edges nghĩa là tối đa 3 experts hoạt động, KHÔNG thể 6 experts đều có >=2 domain. Dùng cùng feasible support set cho Random/Risk/Grad/Regret; cùng quy tắc weights, update interval và schedules. Q-Global nằm ở nhánh cấu trúc khác, không gán mọi chênh lệch cho assignment criterion.

FixedPairs dùng 3 expert slots cho {D1,D2}, {D1,D3}, {D2,D3}; giữ tổng kiến trúc 6 experts và báo cáo active capacity. Đây không phải ground truth.

### Statistics

- Risk: CE trên source-probe.
- Grad: gradient theo classifier parameters của cùng expert, tính theo (domain,class); trung bình `relu(-cos)` qua classes đủ mẫu. Bỏ gradient norm quá nhỏ; lưu thresholds và support. Full-expert gradient là sensitivity analysis riêng.
- Regret: representation frozen; fit linear probes riêng từng domain và probe chung cho từng subset trên source-fit; đánh giá trên source-probe. Gap = weighted mean CE chung - weighted mean CE riêng. Cùng regularization, capacity, convergence criterion. Gap hữu hạn có thể âm; lưu raw và uncertainty.
- Risk + compatibility/gap là objective chọn Q. Chốt trọng số bằng source-validation, cùng tuning budget. Statistics và Q detach.
- Cập nhật mỗi 500 bước cho mọi tiêu chí. Probe fitting không được cập nhật backbone/expert hoặc dùng target. Cached features phải được refresh khi representation thay đổi.

### Train và đánh giá

Frozen-bank trước, joint-training sau. Cùng KL teacher loss với router input detach, cùng CE/SIRM và warmup. Validation invariance metrics phải tính trên source-validation độc lập với probe dùng chọn Q.

Đo OOD accuracy, held-out shared-classifier gap, gradient compatibility, assignment churn, Q supports, loads, ESS, SIRM active rate, compute. Không dùng score trên chính probe tạo Q làm bằng chứng duy nhất.

Ủng hộ: tăng OOD đồng thời cải thiện invariance metric độc lập. PACS không cho phép tuyên bố recovery đúng subset cơ chế thật.

## 11. TODO triển khai trên máy mới (chưa có CLI)

Tên đề xuất:
1. `configs/sirm_hypotheses_pacs.json`: protocol/sweeps/thresholds, không override âm thầm config cũ.
2. `domainbed/scripts/train_sirm_hypotheses.py`: split manifest 60/20/20, seeds 0–4, source-only selection, target không eval trong train.
3. `scripts/cache_sirm_hypotheses.py`: load selected checkpoint, export deterministic source/target cache riêng.
4. `scripts/fit_sirm_router_hypotheses.py`: frozen-bank global/input teachers và H1/H4 controls.
5. `scripts/eval_sirm_hypotheses.py`: H1–H5 metrics/interventions/oracles, selected checkpoint only.
6. `domainbed/sirm_assignment_criteria.py`: Q-Risk/Grad/Regret cùng feasible supports, detached teacher.
7. `scripts/report_sirm_hypotheses.py`: aggregate paired runs, confidence intervals và tables.

Trước full runs phải kiểm tra: splits không overlap; target không được đọc khi fit/select; cached vs direct logits khớp; auxiliary loss chỉ có router gradient; SIRM vẫn có encoder/expert gradient; shuffle giữ đúng expert logits; metrics khớp tính tay trên toy tensors; resume không đổi split/config. Test numerical/gradient behavior, không chỉ mirror implementation.

## 12. Thứ tự chạy và artifacts

1. Setup + Smoke5, khóa environment.
2. Triển khai splits/cache/frozen evaluator; H4 -> H1 -> H5 frozen.
3. H2/H3 trên bank đã khóa; retrain augmentation/shortcut cohorts riêng.
4. Triển khai H6 rồi joint H6/H5.
5. Pilot 4 targets x 3 seeds mỗi full-training variant; confirm variants đã chốt với 5 seeds. Có 6 H6 variants => 72 pilot runs nếu chạy toàn ma trận; không nhân số frozen controls thành full runs.

Output đề xuất (cấu trúc mới cần triển khai):
```text
outputs/sirm_hypotheses/<protocol>/<variant>/env<id>/seed<n>/
  manifest.json
  effective_config.json
  split_manifest.json
  environment.freeze.txt
  checkpoints/best.pkl
  source_metrics.json
  target_metrics.json
  routing_metrics.csv
  interventions.csv
```

Report per-target và mean 4 targets, mean±SD qua seeds, paired deltas cùng split/seed; bootstrap ảnh theo parent ID để các transform không giả làm samples độc lập. Độ bất định training lấy qua seeds; không coi hàng nghìn ảnh là hàng nghìn training runs. Không chọn cấu hình từ kết quả target pilot rồi gọi confirm trên cùng target là hoàn toàn untouched.

## 13. Tài liệu tham khảo

- PACS: https://domaingeneralization.github.io/
- DomainBed/model selection: https://github.com/facebookresearch/DomainBed
- Partial invariance (prior work trực tiếp): https://arxiv.org/abs/2301.12067

Tiêu chí thành công: bank có headroom -> router suy ra suitability từ input -> lợi ích chuyển target -> chịu được nuisance/mixture shifts -> assignment cải thiện invariance độc lập lẫn OOD. Không cần mọi H đều đúng; kết quả âm giúp quyết định sửa bank, router hay hypothesis.

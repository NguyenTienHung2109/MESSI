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

## 14. Follow-up Q / IRMv1 sau diagnosis H2 (2026-10-05)

Workspace hiện có runner riêng tại `experiments/sirm_q_followup/`; đây là phần bổ sung local cho protocol 80/20 đã chạy, không thay trạng thái snapshot nền hoặc protocol 60/20/20 đề xuất ở trên.

Hai phép đo không retrain dùng 48 checkpoint A/B/C/D đã chọn từ source, với cache gốc deterministic. Báo cáo tại `results/sirm_q_followup/diagnosis/summary.md`, số liệu cell/rank/ESS tại CSV cùng thư mục. H2/OFF: G không trọng số Q/non-Q = 0.06885/0.24080; G có trọng số routing và đủ ESS = 0.06899/0.06433; Spearman source-out CE ↔ target CE trung bình 0.69524; target loss gap non-selected minus selected = +0.16651. Risk-EMA ranking thực sự tạo Q được xuất riêng trong `rho_ema`, không đồng nhất với source-out competence.

Kết quả chưa ủng hộ khẳng định H2 tự thỏa IRMv1 gần bằng 0, và chưa ủng hộ khẳng định Q chỉ chọn source competence mà không có transfer signal. Chưa đo early pre-activation; domain-mean derivative có thể triệt tiêu và không bằng mean squared minibatch penalty. Không coi scalar IRMv1 stationarity là bằng chứng causal invariance.

Pilot đã khóa: H2 trên Cartoon/Sketch × 3 seeds, No-IRM/Learned-Q/Random-Q/Global-Q, 5001 steps. Tái sử dụng B/D sau audit, chỉ train mới 12 run. Random-Q là đồ thị ba cặp cố định trên ba expert slots ngẫu nhiên, độc lập risk; Global-Q dense normalized. Chỉ đổi Q cho penalty, teacher router vẫn learned-Q; giữ ESS và điều kiện >=2 source domains. Actual activation/support được log, không mặc định Global-Q hoặc Random-Q luôn có penalty active. Learned/random có thể khác active capacity và weights, nên không diễn giải mọi delta là thuần assignment criterion.

CLI và giới hạn diễn giải: `experiments/sirm_q_followup/README.md`. Báo cáo pilot tự cập nhật tại `results/sirm_q_followup/pilot/summary.md`; chỉ đọc là hoàn tất khi `progress.json` có `stage=completed` và đủ 24 cells được evaluated. Target chỉ được eval sau source-lock và paired-control audit. Soft-Q, ramp bổ sung và residual architecture chưa thuộc pilot này.

Bổ sung theo target cho H2/OFF: Spearman Art/Cartoon/Photo/Sketch = 0.867/0.714/0.981/0.219; target gap dương ở 11/12 run. Spearman theo saved risk-EMA tạo Q = 0.529 trung bình. Sketch vẫn hỗ trợ nghi ngờ source ranking chuyển target yếu; không suy diễn từ trung bình chung rằng Q transfer tốt trên mọi target.

## 15. Global expert quality / interaction / coverage (2026-10-05)

Đã chạy đủ bốn post-hoc analyses mới trên 48 checkpoint bằng CPU, không retrain hoặc chỉnh loss. Runner: `experiments/sirm_h2_interaction/run.py`; mô tả và CLI tại `experiments/sirm_h2_interaction/README.md`; báo cáo tại `results/sirm_q_followup/interaction/summary.md`.

- Double-centered source-out competence: B raw Q advantage 0.24327 = expert-effect 0.23502 + interaction 0.00825; D raw advantage 0.24793 = expert-effect 0.24389 + interaction 0.00404. Interaction đóng góp 3.39%/1.63% của aggregate gap B/D. Expert-effect chiếm 89.59%/89.99% sum-of-squares; interaction 3.00%/2.72%. Đây là decomposition mô tả, không phải causal ANOVA.
- Jaccard Q/current global Top-2 theo source-out CE B/D = 0.55556/0.43519. Không nói Q gần như trùng global Top-2. Theo saved risk EMA: interaction gap B/D = 0.02255/0.02077, Jaccard = 0.64815/0.64815; EMA đã tạo Q nên không dùng nó làm validation độc lập.
- Absolute target CE với mỗi model dùng own-Q: B selected/non-selected = 0.49110/0.65761; D = 0.52832/0.70656. Giữ mask B cố định: selected tăng 0.04067, non-selected tăng 0.04722. Cả hai nhóm đều xấu đi, không ra pattern Q giữ nguyên và chỉ non-Q bị hy sinh.
- Phân nhóm theo s_B: target CE delta low-frequency (0/1) = +0.04838, high-frequency (2/3) = +0.05089; native target mass low group B/D = 20.44%/22.50%. Dùng s_D cũng cho low/high tăng gần nhau. Sketch riêng có low-frequency delta +0.08819 so với high +0.03317, nhưng SD giữa ba seeds lớn; không suy diễn pattern này thành cơ chế chung.
- Cached probability/logit swaps: B logits/B probabilities = 88.927%; D/D = 88.084%; B logits/D probabilities = 88.958%; D logits/B probabilities = 88.169%. Đổi probabilities không phục hồi bank D về mức B. Đây là output-composition diagnostic, không phải router chạy trên representation khác hoặc causal mediation experiment.

Diễn giải: Q advantage chủ yếu là globally strong experts được hỗ trợ; selective low-Q-frequency sacrifice không được hỗ trợ trên toàn bộ cohort. Hiện dữ liệu nghiêng về suy giảm rộng ở expert-logit bank sau SIRM. Chưa tách được backbone/expert/head/calibration; chưa xác nhận một fix interaction-aware soft-Q sẽ có ích OOD. Không đổi loss/assignment từ target diagnostics. Pilot Random/Global-Q vẫn tiếp tục với code/config đã khóa.

Các checks đã pass: reconstruction và additive-shift invariance của residual, global-only negative control, fixed-mask composition; checkpoint/cache provenance, source-domain ordering, paired initialization/splits/schedules/target image IDs/labels, cached native accuracy khớp kết quả gốc. CSV có masks B/D cố định, transitions, per-target means/SD và paired uncertainty theo seeds. Chỉ có 3 seeds và source-out đã dùng chọn checkpoint; kết quả không phải untouched source holdout.

## 16. SIRM-Discover v2: triển khai + tests + ON/OFF smoke (2026-10-05)

Theo phạm vi người dùng đã chọn, đã triển khai prototype riêng tại `experiments/sirm_discover/`, chạy tests và smoke ON/OFF; không chạy full pilot. Pilot Q cũ vẫn `stage=stopped`.

Bốn thay đổi: global predictor + class-centered residual experts; bỏ standalone expert CE và dùng multiple-choice soft-min; Q từ leave-one-out marginal utility không renormalize routing; soft/NULL Q với delayed/ramped component IRMv1. Router có NULL slot và teacher từ single-correction utility; Q dùng marginal utility của composition, là hai statistics khác nhau. Q/teacher/context/routing được detach trong penalty. Cặp source dùng q_i,m*q_j,m; divisor cố định giữ nguyên attenuation của confidence, không chia cho tổng Q.

Prototype chọn freeze backbone/global head sau stage 1. Candidate budget 1000 global + 1000 discovery + 3001 invariance = 5001 steps, ramp 500, vẫn protocol source 80/20 và H2. Config có utility/interaction effect floors và continuous evidence gates, confidence bounded; đây là heuristics được ghi công thức rõ trong README, không phải calibrated evidence về latent mechanisms. Các budget/weights là defaults chưa được validate bằng OOD.

12 tests numerical/gradient/stage/state pass; paired smoke 20 steps/arm hoàn tất tại `results/sirm_discover/smoke_verified/summary.md`. Initialization/split/minibatch schedule/capacity và pre-penalty trajectory khớp chính xác; OFF penalty = 0, ON có active pair steps sau ramp. Smoke Q gần như NULL; source scores tie và selected checkpoints ở step 12 (ramp bắt đầu bằng 0). Không dùng accuracy smoke làm evidence ON > OFF. Synthetic tests kiểm tra active penalty không tầm thường, NULL/no-interaction, gradient isolation và confidence scaling.

Thiết kế đang kiểm tra discovery objective, không khẳng định factor recovery. Global branch stage 1 train CE, nên zero residual chỉ tương đương global predictor về kiến trúc, chưa phải global IRM về objective. Scalar component IRMv1 chưa chứng minh shared optimal classifier vectors. Tiêu chuẩn research ON > OFF, held-out interaction/marginal-utility advantage và ground-truth factor recovery chưa được đánh giá.

CLI, mathematical choices, resume/provenance và giới hạn diễn giải: `experiments/sirm_discover/README.md`. Legacy factorial evaluator/cache không áp dụng trực tiếp vì prediction đã có global + residual composition. Không restart pilot cũ hoặc thêm loss/criterion ngoài nhánh prototype này.

## 17. SIRM-v2 gate: Sketch / L38 × R0/R1/R2 (2026-10-05)

Theo yêu cầu mới, runner `experiments/sirm_discover_gate/` triển khai gate 6 run, seed 0: PACS Sketch và TerraIncognita location_38 × residual discovery ON với SIRM OFF / learned-Q ON / random-Q ON. Global là ERM + H2, freeze sau stage 1; IRM chỉ áp residual contribution. Budget 1000/1000/3001, ramp 500, 5001 steps/run. Pilot cũ vẫn dừng.

Gate khóa learned marginal-utility Q tại đầu stage 3 khi ba trajectory còn giống nhau. R2 dùng expert-column derangement theo seed cố định của cùng ma trận, giữ row weights/sparsity/NULL mass và tổng pair strength. Đây là fixed-Q control, khác prototype refresh Q liên tục; EMA vẫn đo drift. Random control kiểm tra alignment giữa graph và experts, không randomize graph overlap. Audits kiểm tra initialization/split/schedule/capacity, trajectory trước penalty, Q lock/permutation và OFF penalty bằng 0.

Xuất U/double-centered U, interaction advantage, Jaccard global Top-K, residual norms/CE gain, correction/global ratio, exact component G selected/non-selected; checkpoints before/activation/middle/final; target leave-one-expert-out accuracy và Spearman source utility với target importance. Target chỉ đọc sau khóa checkpoint source-selected của cả ba arm và audit. R0/R1 dùng cùng mask khi so G; có snapshots cùng step để tránh đồng nhất selected checkpoints khác thời điểm.

CLI và chi tiết: `experiments/sirm_discover_gate/README.md`. Outputs full tại `results/sirm_discover_gate/phase1/`, smoke pipeline tại `results/sirm_discover_gate/smoke/`. Full chỉ hoàn tất khi progress=completed và 6/6 cells evaluated. OfficeHome/DomainNet chưa có data local; chưa chạy R3 hoặc tự động scale 16 run. Old PACS chỉ reuse khi metadata/split/full target count khớp; old Terra H2 chưa verified nên N/A. Một seed chỉ là exploratory signal.

## 18. Yêu cầu cho các lần train mới: protocol và logging DomainBed (2026-10-06)

Theo yêu cầu người dùng, từ lần chạy mới tiếp theo dùng training protocol/runner của DomainBed (`domainbed/scripts/train.py`), tích hợp algorithm SIRM-v2 vào luồng đó thay vì tiếp tục dùng runner riêng chỉ in source mean. Giữ các lựa chọn architecture/loss/augmentation đã được chỉ định trong protocol của từng experiment; không tự chuyển H2 thành augmentation mặc định chỉ vì đổi runner.

Log bảng theo step trong terminal và lưu `results.jsonl`: step/epoch, losses và accuracy từng environment với các cột `env{i}_in_acc`, `env{i}_out_acc` theo DomainBed. Ghi mapping env index → domain name rõ trong manifest/report. Các hàng score phải được cập nhật đều theo `checkpoint_freq` đã ghi trong config, bao gồm step đầu/cuối; không chỉ tổng hợp accuracy cuối run. Accuracy được tính tại các mốc evaluation, không phải full-dataset evaluation ở mọi optimizer update. Log loss/update diagnostics giữa các mốc khi cần.

Ghi rõ rule chọn checkpoint bằng source validation; target scores dùng để theo dõi/post-hoc theo protocol đã công bố, không dùng để chọn checkpoint/hyperparameter. Nếu một experiment yêu cầu target chỉ đọc sau source-lock, giữ ngoại lệ đó và ghi rõ các target columns chưa được evaluate trong training. Đổi sang protocol mới cần khóa manifest/splits/schedule/config và audit lại các paired controls trước chạy.

Yêu cầu này áp dụng các run mới. Cặp dynamic-Q Terra L38 seed 0 đang chạy vẫn dùng code/config đã khóa; không đổi runner hoặc restart giữa chừng.

## 19. Terra L38: raw gradient và binary SIRM gate (2026-10-06)

Dynamic-Q pair hoàn tất: shared/eligible support và nonzero penalty bắt đầu step 2150; active 1050/3001 steps, nhưng max weighted gradient ~1.53e-11 và model/logits ON/OFF khớp tuyệt đối. Đây là numerical attenuation, chưa có treatment đủ mạnh để đánh giá lợi ích IRM.

Theo yêu cầu mới, no-retrain diagnosis tại `experiments/sirm_discover_binary/inspect_current.py` xuất raw pair loss/gradient và Q-weighted gradient cho 23 eligible pairs. Dùng weights của checkpoint cuối step 5000 với Q/minibatches source theo các refresh cũ; không có checkpoint weights tại từng refresh nên không gọi đây là historical step gradients. Raw grad norm 4.51e-4–0.211, Q weights 3.31e-13–1.91e-8; báo cáo tại `results/sirm_discover_binary/raw_diagnosis/summary.md`.

Test mới chỉ Terra L38 seed 0 × R0 OFF / R1 binary ON, 5001 steps/arm. Binary chỉ đổi weighting/divisor: weight=1 cho term pass nguyên eligibility hiện tại, mean theo số active terms; không đổi Q/threshold/ESS/architecture/discovery/lambda/ramp. Confidence mode cũ vẫn được giữ để tái hiện historical results.

Runner mới là DomainBed `domainbed/scripts/train.py` với algorithm `SIRMDiscoverV2`; có env*_in_acc/env*_out_acc và results.jsonl. Scheduled-loader adapter giữ nguyên source minibatch và augmentation H2 cũ; source splits và initialization được audit. Source-out evaluation tại Q refresh không tạo checkpoint selection mới. Target columns chỉ monitoring, checkpoint chọn bằng source-out và full-target evaluation sau khóa. Pipeline 1-step đã kiểm tra đủ 8 acc columns và init/step-0 loss khớp chính xác old Terra. 21 tests binary/gradient/dynamic/stage/control pass.

Full pair outputs tại `results/sirm_discover_binary/terra_L38_seed0/`. Không tự thay lambda/ramp kể cả gradient ratio >1; không mở rộng datasets/seeds. Ratio gradient đo trên residual parameters giống diagnosis trước. Xem README trong experiment để biết provenance, quy tắc selection và giới hạn resume DomainBed.

## 20. Binary pair: báo cáo 5 nhóm và kiểm tra effective strength

Theo yêu cầu bổ sung, cặp binary hiện tại có watcher độc lập tại `experiments/sirm_discover_binary_analysis/`; không sửa code/config training đã khóa. Khi cả R0/R1 hoàn tất, watcher xuất `results/sirm_discover_binary/terra_L38_seed0/analysis/summary.md`: accuracy và paired delta; median/mean/p90 cùng lịch sử gradient ratio; active fraction trên mọi Stage-3 step; Q evolution/feedback giữa hai arm; residual cosine, correction/global RMS và router NULL trên source-out/target. G được so trên cùng mask R1-selected và union các cells thực sự nhận IRM.

Ratio dùng weighted gradient đã bao gồm lambda/ramp, chia cho gradient của base objective trên cùng residual parameters. Norm chỉ được đo tại Q refresh có penalty active, không phải mọi optimizer step. Báo hai distribution riêng: active refreshes và toàn bộ refreshes với inactive=0; missing active measurement không bị thay bằng 0. Router NULL và Q NULL là hai quantities khác nhau. Cosine là similarity của residual outputs, không chứng minh causal factors hoặc collapse.

Nếu R1 giảm accuracy trong khi ratio lớn (ví dụ 5/10/20), báo khả năng effective regularization quá mạnh sau đổi normalization; không dùng kết quả đó một mình để phủ định binary gate hoặc IRMv1. Giữ nguyên lambda/ramp/thresholds trong test hiện tại. Đây vẫn chỉ là Terra L38 seed 0; không mở rộng hoặc tự tune từ target score.

Bổ sung theo yêu cầu người dùng: dừng bản DomainBed R0 trùng tại step 3750 sau khi toàn bộ 3751 step loss/EMA/Q/router NULL/residual RMS khớp tuyệt đối R0 cũ; initialization/splits/schedules/source validation cũng khớp. Tái sử dụng R0 cũ hoàn chỉnh và source-selected checkpoint step 5000, chỉ train mới R1. Bản R0 bị dừng được giữ tại `banks/R0_domainbed_partial/`. `experiments/sirm_discover_binary_analysis/reuse_r0.py` điều phối R1 với command/config đã khóa, không sửa các training source files. Chuyển checkpoint R0 sang wrapper format chỉ thêm prefix `model.` và metadata; đã kiểm tra mọi tensor giữ nguyên. Provenance: `R0_reuse_plan.json`; original plan và inputs không bị đổi. R0 lịch sử có source-score logs; chỉ partial R0 và R1 mới có đầy đủ DomainBed per-domain in/out logs theo cadence đã chạy.

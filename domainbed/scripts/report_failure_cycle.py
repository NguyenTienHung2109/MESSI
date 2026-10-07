"""Rebuild the seed-0 failure investigation tables and scientific figure."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def table(df):
    rows=[list(df.columns),['---']*len(df.columns)]
    rows += [[f'{v:.4f}' if isinstance(v,float) else str(v) for v in row] for row in df.itertuples(index=False,name=None)]
    return '\n'.join('| '+' | '.join(row)+' |' for row in rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='predictive_support_outputs/pacs_failure_cycle2_seed0');args=p.parse_args();root=Path(args.output)
    scale=pd.read_csv(root/'scale_and_bank.csv');tokens=pd.read_csv(root/'token_ablation/metrics.csv');banks=pd.read_csv(root/'token_banks/metrics.csv');control=pd.read_csv(root/'global_bank_control/metrics.csv');variance=pd.read_csv(root/'variance_decomposition.csv')
    for df in [tokens,banks]:df['evaluation']=df.fold.map(lambda f:'PACS target' if f=='main' else 'Source-exclusion')
    token_summary=tokens[tokens.stage=='held'].groupby(['mode','evaluation'],as_index=False)[['accuracy','ce']].mean()
    bank_summary=banks[banks.stage=='held'].groupby(['variant','evaluation','control'],as_index=False)[['accuracy','ce']].mean()
    control_summary=control.groupby(['variant','stage','bank','scaling'],as_index=False)[['accuracy','ce']].mean()
    scale_summary=scale[(scale.fold=='main') & scale.control.isin(['single_global','all'])].groupby(['variant','stage','control','scaling'],as_index=False)[['accuracy','ce']].mean()
    variance_summary=variance[variance.fold=='main'].groupby(['variant','stage'],as_index=False)[['between_class_fraction','domain_given_class_fraction','within_class_domain_residual']].mean()
    for name,df in [('token_summary',token_summary),('bank_summary',bank_summary),('global_control_summary',control_summary),('scale_summary',scale_summary),('variance_summary',variance_summary)]:df.to_csv(root/f'{name}.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    t=token_summary.pivot(index='mode',columns='evaluation',values='accuracy')*100;t.plot.bar(ax=axes[0,0]);axes[0,0].set(title='Fixed-budget token ablation',ylabel='Accuracy (%)',ylim=(55,76),xlabel='')
    s=scale_summary[(scale_summary.variant=='erm')&(scale_summary.stage=='source_selection')].pivot(index='control',columns='scaling',values='ce');s.plot.bar(ax=axes[0,1]);axes[0,1].set(title='Source-selection CE: scale changes ordering',ylabel='CE',xlabel='')
    c=control_summary[(control_summary.stage=='held')&(control_summary.scaling=='raw')].pivot(index='variant',columns='bank',values='accuracy')*100;c.plot.bar(ax=axes[1,0]);axes[1,0].set(title='Five experts, matched sample budgets',ylabel='Target accuracy (%)',ylim=(65,75),xlabel='')
    v=variance_summary[variance_summary.stage=='source_selection'].set_index('variant');v[['between_class_fraction','domain_given_class_fraction']].plot.bar(ax=axes[1,1]);axes[1,1].set(title='Source feature variance (equal-domain weighting)',ylabel='Fraction of total variance',xlabel='')
    for ax in axes.flat:ax.tick_params(axis='x',rotation=0);ax.grid(axis='y',alpha=.2)
    fig.savefig(root/'failure_diagnostics.png',dpi=180);fig.savefig(root/'failure_diagnostics.pdf');plt.close(fig)
    cert=pd.read_csv(root/'token_banks/certificates.csv')
    selections=list((root/'token_banks').glob('*/env*/selection.json'));chosen=[json.loads(x.read_text())['selected'] for x in selections]
    text='''# PACS: vòng kiểm tra failure tiếp theo — seed 0

Đã hoàn tất: 48 fold phân tích cơ chế, 80 fold token ablation, 48 fold full expert bank và 12 đối chứng bank có cùng số expert. Các fold có thể dùng lại fit khi tập adaptation trùng nhau; đây không phải 188 lần huấn luyện độc lập. Các domain PACS đã được quan sát trong các vòng trước: kết quả là exploratory, không phải validation mới hay bảo đảm tổng quát hóa.

## Kết luận thực nghiệm

1. **Failure ở cách đánh giá contribution:** với ERM, ensemble tăng source-selection accuracy 92.17% → 92.97% nhưng CE thô xấu hơn 1.1471 → 1.1876. Sau fit một hệ số dương trên source calibration, CE đảo thứ tự: global 0.3076, ensemble 0.2475. Accuracy không đổi. Vì vậy không thể diễn giải CE thô âm là không có diversity hữu ích. Confidence bound vẫn là rào cản riêng; scaling không tự sửa certificate.
2. **Feature extractor bỏ qua thông tin có ích:** frozen DIST token tăng global PACS mean 68.67% → 71.36%, source-exclusion 64.39% → 66.16%. Mean(CLS,DIST) cũng tăng source-exclusion lên 66.26%. Lặp CLS để tăng input width không giúp (62.34%). Cùng MLP, budget và seed; chưa chọn DIST thay mean dựa trên target.
3. **Rank tăng không đồng nghĩa feature tốt hơn:** tỷ lệ variance giữa class trên source giảm ERM 0.8423 → within 0.8195 → cross 0.8090; tỷ lệ domain-given-class tăng 0.0068 → 0.0132/0.0122. Đây là dấu hiệu thêm biến thiên nội lớp, không phải chứng minh quan hệ nhân quả hay lý do tự động thêm alignment.
4. **Subset có tín hiệu vượt ensemble size:** cùng năm experts và budget train/calibration mỗi expert, pair-bank hơn all-global-bank 0.91 điểm % (CLS), 0.61 (DIST), 0.90 (mean). Seed đơn và chưa có đối chứng này trên một tập source-exclusion mới với ít nhất ba adaptation domains, nên chưa đủ để khẳng định cơ chế subset tổng quát.
5. **Selection vẫn chưa khai thác lợi ích:** DIST uniform bank đạt 72.41%, nhưng selected global chỉ 71.36%; mean uniform đạt 72.18%, selected 70.42%. Không đổi selection bằng cách nhìn target. Không triển khai temperature như predictor đã được chứng nhận.

## Tính toàn vẹn và kiểm soát

Weights khớp checkpoint pretrained; token custom khớp timm reference trên batch kiểm tra mỗi domain với sai số tối đa < 1.7e-5, CLS cache khớp tuyệt đối. Loại trừ giả thuyết nạp sai pretrained trong kiểm tra này. CLS ablation tái lập đúng map baseline. Train/calibration/audit/selection tách biệt; held domain không tham gia adaptation. Temperature chỉ fit trên source calibration. Bank đối chứng thay pair maps bằng global maps cùng seed, kiến trúc, 500 bước, 128 mẫu mỗi bước; calibration tổng số mẫu giữ nguyên. Domain coverage khác nhau là yếu tố đang kiểm tra.

## Cơ sở toán học và phạm vi diễn giải

Với logits l và beta > 0, argmax(beta*l)=argmax(l). R(beta)=E[log sum exp(beta*l_c)-beta*l_y] có R''(beta)=E[Var_softmax(beta*l)(l_c)] >= 0. Vì vậy scalar calibration là bài toán lồi một chiều và tách được confidence khỏi quyết định phân lớp. Jensen chỉ cho CE(mean logits) <= mean CE(logits), không cho CE(ensemble) <= CE(global). Temperature fit trên calibration làm predictor thay đổi, nên phải xây lại chứng nhận với range tương ứng và dữ liệu audit độc lập nếu muốn sử dụng làm method.

Với phân phối trọng số đều theo domain, luật phân rã variance cho T = Var(E[Z|Y]) + E Var(E[Z|Y,D]|Y) + E Var(Z|Y,D). Script kiểm tra đúng đẳng thức; rank hay tổng variance không đo riêng class signal. Concat(CLS,DIST) chứa lớp hàm CLS-only vì lớp tuyến tính đầu có thể đặt trọng số DIST bằng 0; điều này không bảo đảm tối ưu hay generalization tốt hơn. DIST/mean có cùng chiều đầu vào với CLS; repeat-CLS là đối chứng tăng số tham số.

Reference-hull certificate chỉ nói về hull của các reference heads đã định nghĩa, không phải toàn bộ row-ball. Alpha của một family không bao phủ mọi vòng nghiên cứu thích nghi. Không diễn giải các kết quả đã xem target như một bảo đảm 95% mới.

## Hướng sửa được bằng chứng ủng hộ

Ưu tiên bảo toàn thông tin DIST/mean trong feature pipeline thử nghiệm và tách calibration của predictor khỏi phép kiểm tra contribution. Chưa có cơ sở để thêm cross-domain ridge objective vào default; chưa có cơ sở bỏ confidence bound để chọn ensemble. Bước xác nhận tiếp theo cần cố định lựa chọn bằng source-only protocol và đánh giá trên dữ liệu chưa dùng để thiết kế. Các default production chưa bị thay đổi bởi ablation này.

## Kết quả chi tiết

Accuracy bên dưới tính theo [0,1]; lấy trung bình đều theo fold/domain, không gộp số ảnh. Các source-exclusion fold chồng lặp, không được coi là mẫu thống kê độc lập.
'''
    text+=f'\nSelection chọn global {chosen.count("global")}/{len(chosen)} fold. Full-class audit accepted {int(cert.full_class_accepted.sum())}/{len(cert)} domain checks.\n'
    for title,df in [('Token ablation',token_summary),('Full expert banks',bank_summary),('Đối chứng năm global experts',control_summary),('CE trước/sau calibration',scale_summary),('Phân rã feature variance',variance_summary)]:text+='\n### '+title+'\n\n'+table(df)+'\n'
    text+='\n![Diagnostics](failure_diagnostics.png)\n\nReproduce summary: `python -m domainbed.scripts.report_failure_cycle`. Protocols, source snapshots, per-domain metrics, checkpoints và indices nằm trong các thư mục token_ablation, token_banks và global_bank_control.\n'
    (root/'report.md').write_text(text)

if __name__=='__main__':main()

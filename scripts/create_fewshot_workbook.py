"""Create the five-dataset single-source workbook, optionally importing results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    ('CLIP zero-shot / ICML 2021', 'clip_vit_b16', 'Inductive', 'clip_zeroshot'),
    ('Tip-Adapter-F / ECCV 2022', 'clip_vit_b16', 'Inductive', 'tip_adapter_f_imagenet_fixed'),
    ('GDA+CLIP / ICLR 2024', 'clip_vit_b16', 'Inductive', 'gda_clip'),
    ('LP++ / CVPR 2024', 'clip_vit_b16', 'Inductive', 'lp_pp_frozen'),
    ('CLAP / CVPR 2024', 'clip_vit_b16', 'Inductive', 'clap_frozen'),
    ('ProKeR exact / CVPR 2025', 'clip_vit_b16', 'Inductive', 'proker_imagenet_fixed'),
    ('ProKeR RFF / CVPR 2025', 'clip_vit_b16', 'Inductive', 'proker_rff_imagenet_fixed'),
    ('TIMO / AAAI 2025', 'clip_vit_b16', 'Inductive', 'timo_fixed'),
    ('LaplacianShot / ICML 2020', 'DINOv3-B/16', 'Transductive', 'laplacianshot'),
    ('TIM / NeurIPS 2020', 'DINOv3-B/16', 'Transductive', 'tim'),
    ('alpha-TIM / NeurIPS 2021', 'DINOv3-B/16', 'Transductive', 'alpha_tim'),
    ('PADDLE / NeurIPS 2022', 'DINOv3-B/16', 'Transductive', 'paddle'),
    ('EM-Dirichlet / CVPR 2024', 'clip_vit_b16', 'Transductive', 'em_dirichlet'),
    ('Ours', 'DINOv3-B/16', 'Transductive', 'ours'),
]
DATASETS = {
    'PACS': ('art_painting', 'cartoon', 'photo', 'sketch'),
    'VLCS': ('C', 'L', 'S', 'V'),
    'OfficeHome': ('Art', 'Clipart', 'Product', 'Real World'),
    'TerraIncognita': ('location_100', 'location_38', 'location_43', 'location_46'),
    'DomainNet': ('clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'),
}
SHOTS = (1, 2, 4, 8, 16)
SEEDS = (42, 43, 44, 45, 46)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'fewshot_ood_results.xlsx')
    p.add_argument('--import-runs', type=Path, help='results/fewshot_ood/runs directory')
    p.add_argument('--import-infeasible', type=Path, help='infeasible directory; defaults to sibling of runs')
    a = p.parse_args()
    wb = Workbook(); summary = wb.active; summary.title = 'Summary'
    summary.append(['Dataset', 'Method / venue', 'Backbone', 'Protocol', *[f'{k}-shot' for k in SHOTS]])
    row_map = {}
    for dataset, domains in DATASETS.items():
        n_pairs = len(domains) * (len(domains) - 1)
        for k in SHOTS:
            ws = wb.create_sheet(f'{dataset}_K{k}')
            ws.append(['Method / venue', 'Backbone', 'Source', 'Target', *[f'Seed {s} (%)' for s in SEEDS], 'Mean 5 seeds (%)'])
            for name, backbone, _, key in METHODS:
                for source in domains:
                    for target in domains:
                        if source == target: continue
                        row = ws.max_row + 1
                        ws.append([name, backbone, source, target, None, None, None, None, None,
                                   f'=IF(COUNTIF(E{row}:I{row},"N/A")=5,"Infeasible",IF(COUNT(E{row}:I{row})=5,AVERAGE(E{row}:I{row}),""))'])
                        row_map[(dataset, k, key, source, target)] = row
            validation = DataValidation(type='custom', formula1='OR(E2="",E2="N/A",AND(ISNUMBER(E2),E2>=0,E2<=100))')
            validation.allow_blank = True; validation.showErrorMessage = True
            validation.errorTitle = 'Accuracy phải trong khoảng 0–100'
            validation.error = 'Nhập % accuracy hoặc N/A cho một source không khả thi.'
            ws.add_data_validation(validation); validation.add(f'E2:I{ws.max_row}')
            ws.freeze_panes = 'E2'; ws.auto_filter.ref = f'A1:J{ws.max_row}'
            for c, width in {'A':33,'B':17,'C':19,'D':19,'E':16,'F':16,'G':16,'H':16,'I':16,'J':20}.items(): ws.column_dimensions[c].width = width
            for row in ws.iter_rows(min_row=2, min_col=5, max_col=10):
                for cell in row: cell.number_format = '0.00'
        for idx, (name, backbone, protocol, key) in enumerate(METHODS):
            row = summary.max_row + 1
            for col, value in enumerate((dataset, name, backbone, protocol), 1): summary.cell(row, col, value)
            first, last = 2 + idx * n_pairs, 1 + (idx + 1) * n_pairs
            for col, k in enumerate(SHOTS, 5):
                sheet = f"'{dataset}_K{k}'"
                data = f'{sheet}!E{first}:I{last}'
                statuses = f'{sheet}!J{first}:J{last}'
                seed_means = ','.join(f'AVERAGE({sheet}!{s}{first}:{s}{last})' for s in 'EFGHI')
                summary.cell(row, col, f'=IF(AND(COUNT({statuses})+COUNTIF({statuses},"Infeasible")={n_pairs},COUNT({statuses})>0),TEXT(AVERAGE({data}),"0.00")&" ± "&TEXT(STDEV({seed_means}),"0.00")&IF(COUNTIF({statuses},"Infeasible")>0," † ("&COUNT({statuses})&"/{n_pairs})",""),"")')
    summary.freeze_panes = 'E2'; summary.auto_filter.ref = f'A1:I{summary.max_row}'
    for col, width in {'A':19,'B':33,'C':18,'D':18,'E':18,'F':18,'G':18,'H':18,'I':18}.items(): summary.column_dimensions[col].width = width
    note = summary.max_row + 3
    notes = [
        'Nhập accuracy 0–100 vào các sheet <dataset>_K<shot>.',
        'Summary chỉ hiện khi mỗi hướng có đủ 5 seed hoặc đủ 5 ô N/A; SD tính trên 5 mean theo seed.',
        'N/A: source không đủ K ảnh ở ít nhất một lớp. †: trung bình chỉ trên các hướng khả thi.',
        'Các method phải dùng chung tập hướng khả thi. CLIP zero-shot có cùng kết quả ở các seed.',
        'ProKeR exact và RFF là hai solver; ghi rõ solver khi báo cáo. Ô trống là chưa có kết quả.',
        'Kết quả pilot/smoke không phải kết quả paper.',
    ]
    for i, value in enumerate(notes): summary.cell(note+i, 1, value)
    for ws in wb.worksheets:
        for cell in ws[1]:
            cell.fill = PatternFill('solid', fgColor='17365D'); cell.font = Font(bold=True, color='FFFFFF')
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        ws.row_dimensions[1].height = 32
    if a.import_runs:
        used = 0
        for path in a.import_runs.rglob('*.json'):
            record = json.loads(path.read_text())
            dataset = record.get('dataset')
            if (dataset not in DATASETS or record.get('setting') != 'single_source'
                or record.get('shot_scope') != 'total' or record.get('seed') not in SEEDS
                or record.get('shots') not in SHOTS or record.get('backbone') != 'clip_vit_b16'):
                continue
            sources = record.get('source_domains', [])
            if len(sources) != 1: raise ValueError(f'Invalid source in {path}')
            for target, metric in record['target_metrics'].items():
                key = (dataset, record['shots'], record['method'], sources[0], target)
                if key not in row_map: continue
                value = float(metric['accuracy'])*100
                if not 0 <= value <= 100: raise ValueError(f'Invalid accuracy: {path}')
                cell = wb[f'{dataset}_K{record["shots"]}'].cell(row_map[key], 5+SEEDS.index(record['seed']))
                if cell.value is not None: raise ValueError(f'Duplicate result for {key}, seed {record["seed"]}')
                cell.value = value; used += 1
        print(f'Imported {used} individual source-target-seed accuracies')
    infeasible_root = a.import_infeasible or (a.import_runs.parent / 'infeasible' if a.import_runs else None)
    if infeasible_root and infeasible_root.exists():
        marked = 0
        for path in infeasible_root.rglob('*.json'):
            record = json.loads(path.read_text()); dataset = record.get('dataset')
            if (dataset not in DATASETS or record.get('setting') != 'single_source'
                or record.get('shot_scope') != 'total' or record.get('seed') not in SEEDS
                or record.get('shots') not in SHOTS): continue
            sources = record.get('source_domains', [])
            if len(sources) != 1: raise ValueError(f'Invalid infeasible source in {path}')
            for target in record['target_domains']:
                for _, _, _, method in METHODS:
                    key = (dataset, record['shots'], method, sources[0], target)
                    if key not in row_map: raise ValueError(f'Unknown infeasible pair {key}')
                    cell = wb[f'{dataset}_K{record["shots"]}'].cell(row_map[key], 5+SEEDS.index(record['seed']))
                    if cell.value is not None: raise ValueError(f'Result conflicts with infeasible record: {path}, {key}')
                    cell.value = 'N/A'; marked += 1
        print(f'Marked {marked} method/source/target/seed cells infeasible')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(a.output)
    print(f'Workbook written: {a.output}')

if __name__ == '__main__': main()

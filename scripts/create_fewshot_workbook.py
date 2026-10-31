"""Create a PACS single-source workbook; optionally load benchmark JSON results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

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
DOMAINS = ('art_painting', 'cartoon', 'photo', 'sketch')
SHOTS = (1, 2, 4, 8, 16)
SEEDS = (42, 43, 44, 45, 46)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'fewshot_pacs_results.xlsx')
    p.add_argument('--import-runs', type=Path, help='Optional results/fewshot_ood/runs directory')
    a = p.parse_args()
    wb = Workbook(); summary = wb.active; summary.title = 'Summary'
    summary.append(['Method / venue', 'Backbone', 'Protocol', *[f'{k}-shot' for k in SHOTS]])
    row_map = {}
    for k in SHOTS:
        ws = wb.create_sheet(f'K{k}')
        ws.append(['Method / venue', 'Backbone', 'Source', 'Target', *[f'Seed {s} (%)' for s in SEEDS], 'Mean 5 seeds (%)'])
        for name, backbone, _, key in METHODS:
            for source in DOMAINS:
                for target in DOMAINS:
                    if source == target: continue
                    row = ws.max_row + 1
                    ws.append([name, backbone, source, target, None, None, None, None, None,
                               f'=IF(COUNT(E{row}:I{row})=5,AVERAGE(E{row}:I{row}),"")'])
                    row_map[(k, key, source, target)] = row
        validation = DataValidation(type='decimal', operator='between', formula1=0, formula2=100)
        validation.allow_blank = True; validation.showErrorMessage = True
        validation.errorTitle = 'Accuracy phải trong khoảng 0–100'
        validation.error = 'Nhập top-1 accuracy theo phần trăm, ví dụ 83.25.'
        ws.add_data_validation(validation); validation.add(f'E2:I{ws.max_row}')
        ws.freeze_panes = 'E2'; ws.auto_filter.ref = f'A1:J{ws.max_row}'
        for c, width in {'A':31,'B':17,'C':17,'D':17,'E':16,'F':16,'G':16,'H':16,'I':16,'J':20}.items(): ws.column_dimensions[c].width = width
        for row in ws.iter_rows(min_row=2, min_col=5, max_col=10):
            for cell in row: cell.number_format = '0.00'
    for idx, (name, backbone, protocol, key) in enumerate(METHODS, 2):
        summary.cell(idx, 1, name); summary.cell(idx, 2, backbone); summary.cell(idx, 3, protocol)
        first, last = 2 + (idx-2)*12, 13 + (idx-2)*12
        for col, k in enumerate(SHOTS, 4):
            data = f"'K{k}'!E{first}:I{last}"
            seed_means = ','.join(f"AVERAGE('K{k}'!{s}{first}:{s}{last})" for s in 'EFGHI')
            summary.cell(idx, col, f'=IF(COUNT({data})=60,TEXT(AVERAGE({data}),"0.00")&" ± "&TEXT(STDEV({seed_means}),"0.00"),"")')
    summary.freeze_panes = 'D2'; summary.auto_filter.ref = f'A1:H{summary.max_row}'
    for col, width in {'A':33,'B':18,'C':18,'D':18,'E':18,'F':18,'G':18,'H':18}.items(): summary.column_dimensions[col].width = width
    note = summary.max_row + 3
    summary[f'A{note}'] = 'Nhập accuracy 0–100 vào các ô seed ở sheet K1, K2, K4, K8, K16.'
    summary[f'A{note+1}'] = 'Summary chỉ hiện khi có đủ 12 hướng × 5 seed; SD tính trên 5 mean theo seed.'
    summary[f'A{note+2}'] = 'Mỗi cell là mean ± SD (điểm phần trăm). CLIP zero-shot có cùng kết quả ở các seed.'
    summary[f'A{note+3}'] = 'ProKeR exact và RFF là hai solver; ghi rõ nếu chỉ báo cáo một solver.'
    summary[f'A{note+4}'] = 'Kết quả pilot/smoke không phải kết quả paper. Các ô trống là chưa có dữ liệu.'
    for ws in wb.worksheets:
        for cell in ws[1]:
            cell.fill = PatternFill('solid', fgColor='17365D'); cell.font = Font(bold=True, color='FFFFFF')
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        ws.row_dimensions[1].height = 32
    if a.import_runs:
        used = 0
        for path in a.import_runs.rglob('*.json'):
            record = json.loads(path.read_text())
            if (record.get('dataset') != 'PACS' or record.get('setting') != 'single_source'
                or record.get('shot_scope') != 'total' or record.get('seed') not in SEEDS
                or record.get('shots') not in SHOTS or record.get('backbone') != 'clip_vit_b16'):
                continue
            sources = record.get('source_domains', [])
            if len(sources) != 1: raise ValueError(f'Invalid source in {path}')
            for target, metric in record['target_metrics'].items():
                key = (record['shots'], record['method'], sources[0], target)
                if key not in row_map: continue
                value = float(metric['accuracy'])*100
                if not 0 <= value <= 100: raise ValueError(f'Invalid accuracy: {path}')
                cell = wb[f'K{record["shots"]}'].cell(row_map[key], 5+SEEDS.index(record['seed']))
                if cell.value is not None: raise ValueError(f'Duplicate result for {key}, seed {record["seed"]}')
                cell.value = value; used += 1
        print(f'Imported {used} individual source-target-seed values')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(a.output)
    print(f'Workbook written: {a.output}')

if __name__ == '__main__': main()

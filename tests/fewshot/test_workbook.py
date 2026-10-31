"""Workbook layout, incomplete-result guard and real-result import."""
import json
import subprocess
import sys
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]


def test_workbook_import_and_incomplete_guard(tmp_path):
    runs = tmp_path / 'runs'
    runs.mkdir()
    (runs / 'result.json').write_text(json.dumps({
        'dataset': 'PACS', 'setting': 'single_source', 'shot_scope': 'total',
        'seed': 42, 'shots': 1, 'backbone': 'clip_vit_b16',
        'source_domains': ['photo'], 'method': 'gda_clip',
        'target_metrics': {'sketch': {'accuracy': .875}},
    }))
    output = tmp_path / 'workbook.xlsx'
    subprocess.run([sys.executable, str(ROOT / 'scripts/create_fewshot_workbook.py'),
                    '--output', str(output), '--import-runs', str(runs)], check=True)
    workbook = load_workbook(output, data_only=False)
    assert workbook.sheetnames == ['Summary', 'K1', 'K2', 'K4', 'K8', 'K16']
    rows = [row for row in workbook['K1'].iter_rows(min_row=2, values_only=True)
            if row[0] == 'GDA+CLIP / ICLR 2024' and row[2:4] == ('photo', 'sketch')]
    assert len(rows) == 1 and rows[0][4] == 87.5 and rows[0][5] is None
    assert 'COUNT(' in workbook['Summary']['D4'].value
    assert '=60' in workbook['Summary']['D4'].value
    assert workbook['K1'].max_row == 1 + 14 * 12

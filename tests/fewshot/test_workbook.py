"""Workbook layout, incomplete-result guard, import, and infeasible source."""
import json
import subprocess
import sys
from pathlib import Path
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]


def test_five_dataset_workbook_and_import(tmp_path):
    runs = tmp_path / 'runs'; runs.mkdir()
    infeasible = tmp_path / 'infeasible'; infeasible.mkdir()
    def result(dataset, source, target):
        return {'dataset': dataset, 'setting': 'single_source', 'shot_scope': 'total',
                'seed': 42, 'shots': 1, 'backbone': 'clip_vit_b16',
                'source_domains': [source], 'method': 'gda_clip',
                'target_metrics': {target: {'accuracy': .875}}}
    (runs / 'pacs.json').write_text(json.dumps(result('PACS', 'photo', 'sketch')))
    (runs / 'office.json').write_text(json.dumps(result('OfficeHome', 'Art', 'Real World')))
    (runs / 'domainnet.json').write_text(json.dumps(result('DomainNet', 'quickdraw', 'sketch')))
    (infeasible / 'terra.json').write_text(json.dumps({
        'dataset': 'TerraIncognita', 'setting': 'single_source', 'shot_scope': 'total',
        'seed': 42, 'shots': 1, 'source_domains': ['location_38'],
        'target_domains': ['location_100', 'location_43', 'location_46'],
    }))
    output = tmp_path / 'workbook.xlsx'
    subprocess.run([sys.executable, str(ROOT / 'scripts/create_fewshot_workbook.py'),
                    '--output', str(output), '--import-runs', str(runs)], check=True)
    workbook = load_workbook(output, data_only=False)
    assert len(workbook.sheetnames) == 26
    assert workbook['PACS_K1'].max_row == 1 + 14 * 12
    assert workbook['DomainNet_K16'].max_row == 1 + 14 * 30
    for sheet, source, target in [('PACS_K1', 'photo', 'sketch'),
                                  ('OfficeHome_K1', 'Art', 'Real World'),
                                  ('DomainNet_K1', 'quickdraw', 'sketch')]:
        rows = [row for row in workbook[sheet].iter_rows(min_row=2, values_only=True)
                if row[0] == 'GDA+CLIP / ICLR 2024' and row[2:4] == (source, target)]
        assert len(rows) == 1 and rows[0][4] == 87.5 and rows[0][5] is None
    rows = [row for row in workbook['TerraIncognita_K1'].iter_rows(min_row=2, values_only=True)
            if row[0] == 'GDA+CLIP / ICLR 2024' and row[2:4] == ('location_38', 'location_43')]
    assert rows[0][4] == 'N/A' and rows[0][5] is None
    assert 'COUNTIF(' in workbook['Summary']['E4'].value
    assert '=12' in workbook['Summary']['E4'].value
    assert '=30' in workbook['Summary']['E60'].value

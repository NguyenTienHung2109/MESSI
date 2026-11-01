"""Large exact ProKeR jobs select the separately recorded RFF method."""
import json
import sys
from types import SimpleNamespace
import pytest
from scripts import run_fewshot_matrix as matrix
from domainbed.fewshot.protocols import folds


def test_large_domainnet_uses_rff_key(tmp_path, monkeypatch):
    dataset = 'DomainNet'; backbone = 'clip_vit_b16'
    names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    audit_path = tmp_path / 'results/fewshot_ood/data_audit.json'
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(json.dumps({dataset: {'domains': {n: {} for n in names},
                                                'canonical_classes': list(range(345))}}))
    view_path = tmp_path / 'results/fewshot_ood/support_views' / dataset / backbone / 'complete.json'
    view_path.parent.mkdir(parents=True); view_path.write_text(json.dumps({'n_views': 10}))
    seen = []
    def fake_run(command, cwd, stdout, stderr):
        seen.append(command)
        result_dir = (tmp_path / 'results/fewshot_ood/runs' / dataset / backbone /
                      'single_source/total/16/42/proker_rff_imagenet_fixed')
        result_dir.mkdir(parents=True)
        for fold in folds(names, 'single_source'):
            (result_dir / (fold.identifier + '.json')).write_text(json.dumps({
                'source_domains': fold.source_domains, 'target_domains': fold.target_domains,
                'target_metrics': {target: {} for target in fold.target_domains}}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(matrix, 'ROOT', tmp_path)
    monkeypatch.setattr(matrix.subprocess, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['run_fewshot_matrix', '--datasets', dataset,
        '--backbones', backbone, '--methods', 'proker_imagenet_fixed', '--settings', 'single_source',
        '--scopes', 'total', '--shots', '16', '--seeds', '42', '--proker-rff-on-large'])
    matrix.main()
    assert len(seen) == 1
    assert seen[0][seen[0].index('--method')+1] == 'proker_rff_imagenet_fixed'
    assert seen[0][seen[0].index('--rff-components')+1] == '4096'


def test_matrix_rejects_completed_fold_missing_quickdraw(tmp_path, monkeypatch):
    names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    audit = tmp_path / 'results/fewshot_ood/data_audit.json'
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({'DomainNet': {'domains': {name: {} for name in names}}}))
    fold = folds(names, 'single_source')[0]
    result_dir = tmp_path / 'results/fewshot_ood/runs/DomainNet/clip_vit_b16/single_source/total/1/42/gda_clip'
    result_dir.mkdir(parents=True)
    result = {'source_domains': fold.source_domains, 'target_domains': fold.target_domains,
              'target_metrics': {target: {} for target in fold.target_domains if target != 'quickdraw'}}
    (result_dir / (fold.identifier + '.json')).write_text(json.dumps(result))
    monkeypatch.setattr(matrix, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['run_fewshot_matrix', '--datasets', 'DomainNet',
        '--backbones', 'clip_vit_b16', '--methods', 'gda_clip', '--settings', 'single_source',
        '--scopes', 'total', '--shots', '1', '--seeds', '42'])
    with pytest.raises(RuntimeError, match='Incomplete result'):
        matrix.main()

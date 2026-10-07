"""Reuse the audited historical OFF control and run only the locked binary R1."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch

from experiments.sirm_discover_binary import run as binary
from experiments.sirm_h1_h6 import run as base
from domainbed.scripts.train_subset_irm_pacs import make_schedule
from .run import rows


COMPARE_KEYS = [
    'loss', 'loss_cls', 'loss_discover', 'loss_router', 'loss_balance',
    'loss_sirm', 'effective_lambda_sirm', 'sirm_active', 'sirm_pair_count',
    'residual_rms', 'router_null_mass', 'utility_ema', 'utility_interaction',
    'q', 'q_null', 'q_confidence', 'utility_updates', 'G', 'q_learned',
]


def read(path):
    return json.loads(path.read_text())


def convert_checkpoint(source, destination, config, selected_step, source_score):
    """Only add the DomainBed wrapper prefix; preserve every tensor and config."""
    sha = base.digest(source)
    native = base.load(source)
    state = native['model_dict']
    payload = dict(
        model_dict={'model.' + key: value for key, value in state.items()},
        config=config,
        selected_step=selected_step,
        step=native['step'],
        source_val_mean=source_score,
        reused_from=str(source),
        original_checkpoint_sha256=sha,
    )
    torch.save(payload, destination)
    recovered = base.load(destination)
    assert set(recovered['model_dict']) == {'model.' + key for key in state}
    assert all(torch.equal(value, recovered['model_dict']['model.' + key])
               for key, value in state.items())
    assert recovered['config'] == config
    assert base.digest(source) == sha
    return dict(
        source=str(source), original_sha256=sha,
        destination=str(destination), destination_sha256=base.digest(destination),
        tensor_count=len(state), tensor_elements=sum(v.numel() for v in state.values()),
        all_tensor_contents_exact=True, change='model. key prefix and provenance metadata only',
    )


def prepare(root, historical):
    original_plan = read(root / 'plan.json')
    assert binary.code_hashes() == original_plan['hashes']
    assert read(root / 'progress.json')['stage'] == 'switching_to_reused_R0'
    request = read(root / 'R0_reuse_request.json')
    manifest = read(historical / 'manifest.json')
    summary = read(historical / 'summary.json')
    split = read(root / 'source_split_manifest.json')
    partial = root / 'banks/R0'
    archive = root / 'banks/R0_domainbed_partial'
    stopped = rows(partial / 'diagnostics.jsonl')
    complete = rows(historical / 'diagnostics.jsonl')
    assert len(complete) == 5001 and summary['steps'] == 5001
    assert [r['step'] for r in complete] == list(range(5001))
    assert all(r['loss_sirm'] == 0 and r['effective_lambda_sirm'] == 0
               and not r['sirm_active'] for r in complete)
    assert stopped[-1]['step'] == request['stopped_R0_last_step']
    assert all(r[key] == complete[r['step']][key]
               for r in stopped for key in COMPARE_KEYS)
    assert split == read(historical / 'split_manifest.json')
    assert base.canonical_hash(split) == original_plan['split_hash'] == manifest['split_hash']
    initialization = read(partial / 'initialization.json')
    assert initialization['weights_sha256'] == manifest['initialization_sha256']
    assert initialization['parameters'] == manifest['parameters_total']
    wanted = original_plan['config']['R0']['sirm_discover_config']
    assert {k: v for k, v in wanted.items() if k != 'sirm_weighting'} == manifest['config']
    schedules = read(partial / 'source_schedules.json')
    chunks = []
    for env, entry in sorted(split['sources'].items(), key=lambda item: int(item[0])):
        schedule = make_schedule(list(range(len(entry['in']))), 5001,
                                 wanted['batch_size'], 0, int(env))
        chunk = schedule.tobytes()
        assert schedules[env]['split_keys'] == entry['in']
        assert schedules[env]['schedule_sha256'] == hashlib.sha256(chunk).hexdigest()
        chunks.append(chunk)
    assert hashlib.sha256(b''.join(chunks)).hexdigest() == manifest['schedule_sha256']
    previous_refresh = {r['step']: r for r in rows(historical / 'q_refresh.jsonl')}
    for r in rows(partial / 'q_refresh.jsonl'):
        for key in ['source_validation_accuracy', 'source_validation_per_domain']:
            assert r[key] == previous_refresh[r['step']][key]
    assert not archive.exists() and not (root / 'banks/R1').exists()
    partial.rename(archive)
    log = root / 'R0.log'
    if log.exists():
        log.rename(root / 'R0_domainbed_partial.log')
    control = root / 'banks/R0'
    control.mkdir()
    copied = {}
    for name in ['diagnostics.jsonl', 'q_refresh.jsonl', 'train_log.jsonl',
                 'source_diagnostics.json', 'manifest.json', 'effective_config.json',
                 'split_manifest.json', 'summary.json', 'environment.freeze.txt']:
        source = historical / name
        if source.exists():
            sha = base.digest(source)
            shutil.copy2(source, control / name)
            assert base.digest(control / name) == sha == base.digest(source)
            copied[name] = dict(source=str(source), sha256=sha)
    for name in ['initialization.json', 'source_schedules.json']:
        shutil.copy2(archive / name, control / name)
    converted = {}
    for name, native in [('source_best', 'best'), ('last', 'last')]:
        converted[name] = convert_checkpoint(
            historical / 'checkpoints' / (native + '.pkl'),
            control / (name + '.pkl'), manifest['config'],
            summary['selected_step'], summary['source_mean'],
        )
    selection = dict(selected_step=summary['selected_step'],
                     source_val_mean=summary['source_mean'], target_used=False,
                     reused=True, historical_bank=str(historical))
    base.write(control / 'selection.json', selection)
    provenance = dict(
        user_requested=True, R0_reused=True, R0_retrained=False,
        stopped_duplicate_last_step=stopped[-1]['step'],
        exact_matching_steps=len(stopped), compared_keys=COMPARE_KEYS,
        historical_bank=str(historical), archived_partial_bank=str(archive),
        copied_files=copied, converted_checkpoints=converted,
        original_plan_sha256=base.digest(root / 'plan.json'),
        unchanged_training_hashes=original_plan['hashes'],
        controller_sha256=base.digest(Path(__file__)),
        initialization_split_schedule_validation_matched=True,
        target_used_to_choose_reuse=False,
        note='Reuse was requested after exact OFF trajectory/control audit. '
             'Historical source-selected checkpoints are unchanged. R0 has '
             'historical source-score logs; only the archived partial R0 and '
             'fresh R1 have DomainBed per-domain in/out logs.',
    )
    base.write(control / 'reuse_manifest.json', provenance)
    base.write(root / 'R0_reuse_plan.json', provenance)
    base.write(root / 'progress.json', dict(stage='R0_reuse_prepared', R0_reused=True))
    print(json.dumps(dict(R0_reused=True, stopped_step=stopped[-1]['step'],
                          matching_steps=len(stopped), historical_selected_step=summary['selected_step'])),
          flush=True)


def run(root, historical, data_dir, prepare_only=False):
    torch.set_num_threads(4)
    if not (root / 'R0_reuse_plan.json').exists():
        prepare(root, historical)
    if prepare_only:
        return
    plan = read(root / 'plan.json')
    reuse = read(root / 'R0_reuse_plan.json')
    assert base.digest(root / 'plan.json') == reuse['original_plan_sha256']
    assert binary.code_hashes() == plan['hashes']
    output = root / 'banks/R1'
    assert binary.params('R1', output) == plan['config']['R1']
    output.mkdir()  # fresh R1 only; no weight-only resume
    start = time.time()
    (root / 'summary.md').write_text(
        '# Terra L38 binary-SIRM: historical R0 / fresh R1\n\n'
        f'R0 duplicate stopped at step {reuse["stopped_duplicate_last_step"]} '
        'after exact control/trajectory audit, at user request. Completed '
        'historical R0 is reused; fresh R1 uses the unchanged locked DomainBed '
        'binary configuration. Final paired results pending.\n\n'
        'Provenance: R0_reuse_plan.json; partial audit: '
        'banks/R0_domainbed_partial/.\n')
    base.write(root / 'progress.json', dict(stage='training', arm='R1',
               pid=os.getpid(), R0_reused=True, elapsed=0.))
    with (root / 'R1.log').open('w') as stream:
        result = subprocess.run(binary.command('R1', output, data_dir, 5001),
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        base.write(root / 'progress.json', dict(stage='failed', arm='R1',
                   returncode=result.returncode, R0_reused=True))
        raise RuntimeError((root / 'R1.log').read_text()[-4000:])
    assert binary.code_hashes() == plan['hashes']
    base.write(root / 'progress.json', dict(stage='evaluation', pid=os.getpid(), R0_reused=True))
    binary.finish(root, data_dir)
    behavior = read(root / 'behavior.json')
    behavior.update(R0_reused=True,
                    R0_duplicate_stopped_step=reuse['stopped_duplicate_last_step'],
                    R0_reuse_exact_matching_steps=reuse['exact_matching_steps'])
    base.write(root / 'behavior.json', behavior)
    report = root / 'summary.md'
    text = report.read_text().replace(
        'Both arms freshly train; old R0 is not reused.',
        'Completed historical R0 is reused at user request after exact '
        'initialization/split/schedule/trajectory/validation audit through '
        f'step {reuse["stopped_duplicate_last_step"]}. Only R1 trains fresh.')
    text = text.replace(
        'Per-domain scores: banks/R0/results.jsonl and banks/R1/results.jsonl; console tables in out.txt.',
        'R1 DomainBed per-domain scores: banks/R1/results.jsonl and out.txt. '
        'R0 historical source scores: banks/R0/train_log.jsonl and q_refresh.jsonl. '
        'The stopped duplicate DomainBed R0 is retained at banks/R0_domainbed_partial/.')
    report.write_text(text + '\nR0 reuse and checkpoint tensor-preserving '
                      'format conversion: R0_reuse_plan.json.\n')
    base.write(root / 'progress.json', dict(stage='completed', elapsed=time.time()-start,
               fresh_runs=1, reused_runs=1, R0_reused=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/sirm_discover_binary/terra_L38_seed0')
    parser.add_argument('--historical', default='results/sirm_discover_dynamic/terra_L38_seed0/TerraIncognita/banks/R0')
    parser.add_argument('--data-dir', default='domainbed/data')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(Path(args.root).resolve(), Path(args.historical).resolve(), args.data_dir, args.prepare_only)


if __name__ == '__main__':
    main()

"""Fresh DomainBed binary R1 with an explicitly requested lambda; reuse audited R0."""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch
from experiments.sirm_discover_binary import run as binary
from experiments.sirm_h1_h6 import run as base


def requested_params(arm, output, strength):
    params = binary.params(arm, output)
    if arm == 'R1':
        params['sirm_discover_config']['lambda_sirm'] = strength
    return params


def requested_command(output, data_dir, strength):
    command = binary.command('R1', output, data_dir, 5001)
    index = command.index('--hparams') + 1
    command[index] = json.dumps(requested_params('R1', output, strength))
    return command


def run(root, previous, data_dir, strength):
    assert strength > 0
    assert json.loads((previous/'progress.json').read_text())['stage'] == 'completed'
    old_plan = json.loads((previous/'plan.json').read_text())
    assert binary.code_hashes() == old_plan['hashes'], 'training sources changed'
    root.mkdir(parents=True, exist_ok=False)
    (root/'banks').mkdir()
    shutil.copytree(previous/'banks/R0', root/'banks/R0')
    shutil.copy2(previous/'source_split_manifest.json', root/'source_split_manifest.json')
    output = root/'banks/R1'
    configs = {arm: requested_params(arm, root/'banks'/arm, strength) for arm in ['R0', 'R1']}
    original = old_plan['config']['R1']['sirm_discover_config']
    new = configs['R1']['sirm_discover_config']
    assert {k:v for k,v in new.items() if k != 'lambda_sirm'} == {k:v for k,v in original.items() if k != 'lambda_sirm'}
    plan = dict(seed=0, target=1, dataset='TerraIncognita', arms=['R0','R1'],
                hashes=binary.code_hashes(), config=configs, split_hash=old_plan['split_hash'],
                previous_run=str(previous), previous_lambda=original['lambda_sirm'],
                requested_lambda=strength, changed='lambda_sirm only; fresh R1, audited historical R0 reused',
                controller_sha256=base.digest(Path(__file__)), target_used_for_selection=False)
    base.write(root/'plan.json', plan)
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for name in plan['hashes']:
        (snapshot/name.replace('/','__')).write_bytes(Path(name).read_bytes())
    (snapshot/'lambda_run.py').write_bytes(Path(__file__).read_bytes())
    shutil.copy2(previous/'environment.freeze.txt', root/'environment.freeze.txt')
    provenance=json.loads((previous/'R0_reuse_plan.json').read_text())
    provenance.update(reused_from_completed_run=str(previous), no_duplicate_R0_started=True,
                      original_plan_sha256=base.digest(root/'plan.json'))
    for name in ['source_best.pkl','last.pkl','diagnostics.jsonl','q_refresh.jsonl','source_schedules.json']:
        assert base.digest(root/'banks/R0'/name) == base.digest(previous/'banks/R0'/name)
    base.write(root/'R0_reuse_plan.json', provenance)
    output.mkdir()
    (root/'summary.md').write_text(f'# Terra L38 seed 0: binary SIRM lambda={strength}\n\nFresh R1 running with DomainBed, 5001 steps; audited R0 reused. Only lambda changes from {original["lambda_sirm"]} to {strength}; architecture, Q thresholds/temperature, ESS, 50-step refresh, 500-step ramp, seed/splits/schedule and source selection unchanged.\n')
    start=time.time()
    base.write(root/'progress.json', dict(stage='training',arm='R1',pid=os.getpid(),R0_reused=True,lambda_sirm=strength))
    print(json.dumps(dict(started='R1', lambda_sirm=strength, root=str(root), R0_reused=True)),flush=True)
    with (root/'R1.log').open('w') as stream:
        result=subprocess.run(requested_command(output,data_dir,strength),stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        base.write(root/'progress.json',dict(stage='failed',arm='R1',returncode=result.returncode))
        raise RuntimeError((root/'R1.log').read_text()[-4000:])
    assert binary.code_hashes()==plan['hashes']
    base.write(root/'progress.json',dict(stage='evaluation',pid=os.getpid(),R0_reused=True,lambda_sirm=strength))
    binary.finish(root,data_dir)
    behavior=json.loads((root/'behavior.json').read_text())
    behavior.update(R0_reused=True,lambda_sirm=strength,previous_lambda=original['lambda_sirm'],user_requested_lambda_change=True)
    base.write(root/'behavior.json',behavior)
    report=root/'summary.md';text=report.read_text()
    text=text.replace('lambda=0.1,',f'lambda={strength},').replace('Both arms freshly train; old R0 is not reused.','Only R1 trains fresh; audited historical R0 is reused.')
    text=text.replace('Per-domain scores: banks/R0/results.jsonl and banks/R1/results.jsonl; console tables in out.txt.','R1 per-domain scores: banks/R1/results.jsonl and out.txt. R0 historical source scores: banks/R0/train_log.jsonl and q_refresh.jsonl.')
    report.write_text(text+f'\nUser-requested lambda-only follow-up to {previous}. No other settings changed.\n')
    base.write(root/'progress.json',dict(stage='completed',elapsed=time.time()-start,fresh_runs=1,reused_runs=1,R0_reused=True,lambda_sirm=strength))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--previous',default='results/sirm_discover_binary/terra_L38_seed0')
    parser.add_argument('--data-dir',default='domainbed/data')
    parser.add_argument('--lambda-sirm',type=float,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    run(Path(args.output).resolve(),Path(args.previous).resolve(),args.data_dir,args.lambda_sirm)


if __name__=='__main__':
    main()

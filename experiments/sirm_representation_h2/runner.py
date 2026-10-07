"""CLI orchestrating existing train code and one shared factorial evaluator."""
import argparse,json,os,shlex,subprocess,sys,time,traceback
from pathlib import Path
import torch
from experiments.sirm_h1_h6 import run as base
from .protocol import CONDITIONS,config_for,train,validate_reuse,assert_only_sirm_changed
from .diagnostics import evaluate_run,json_write
from .summarize import summarize

DEFAULT_REUSE='experiments/sirm_h1_h6/artifacts/full_pilot_20261002_085859'

def job(args):
    torch.set_num_threads(4)
    aug,enabled=CONDITIONS[args.condition];root=Path(args.output).resolve()
    dest=root/'runs'/args.condition/f'env{args.target_env}'/f'seed{args.seed}'
    dest.mkdir(parents=True,exist_ok=True)
    reuse=enabled and not args.smoke
    if reuse:
        bank=Path(args.reuse_root).resolve()/('native' if args.condition=='C' else 'balanced_views')/f'env{args.target_env}'/f'seed{args.seed}'
        evidence=validate_reuse(bank,args.condition,args.target_env,args.seed);json_write(dest/'reuse_audit.json',evidence)
    else:
        bank=root/'banks'/args.condition/f'env{args.target_env}'/f'seed{args.seed}'
    if (dest/'metrics.json').exists():
        old=json.loads((dest/'metrics.json').read_text());p=old['provenance']
        if (p['condition'],p['target_env'],p['seed'],p['smoke'])!=(args.condition,args.target_env,args.seed,args.smoke):raise ValueError('completed job identity mismatch')
        if p['checkpoint_sha256']!=base.digest(bank/'checkpoints/best.pkl'):raise ValueError('completed checkpoint changed')
        print('already complete: '+str(dest),flush=True);return
    if not reuse and not (bank/'summary.json').exists():
        resume=(bank/'resume.pt').exists()
        train(args.condition,bank,args.target_env,args.seed,args.smoke,resume=resume)
    if not reuse:
        manifest=json.loads((bank/'manifest.json').read_text());eff=json.loads((bank/'effective_config.json').read_text())
        assert eff['hparams']['subset_irm_lambda_sirm']==(.1 if enabled else 0.)
        if not manifest['target_read_during_training'] is False:raise ValueError('target used during training')
        rows=[json.loads(x) for x in (bank/'train_log.jsonl').read_text().splitlines()]
        best=max(rows,key=lambda r:r['source_mean']);selected=json.loads((bank/'summary.json').read_text())
        if (selected['selected_step'],selected['source_mean'])!=(best['step'],best['source_mean']):raise ValueError('checkpoint selection differs from source mean earliest tie')
        if not enabled and any(r['weighted_loss_sirm']!=0 or r['effective_lambda_sirm']!=0 for r in rows):raise ValueError('OFF SIRM term not zero')
    evaluate_run(bank,dest,args.condition,args.target_env,args.seed,reuse,args.smoke)
    for source in Path(__file__).parent.glob('*.py'):
        snapshot=dest/'source_snapshot';snapshot.mkdir(exist_ok=True);(snapshot/source.name).write_bytes(source.read_bytes())

def suite(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);targets=[0] if args.smoke else list(range(4));seeds=[0] if args.smoke else [0,1,2]
    jobs=[dict(condition=c,target=t,seed=s,reused=c in ['C','D'] and not args.smoke) for t in targets for s in seeds for c in CONDITIONS]
    assert_only_sirm_changed(config_for(True),config_for(False))
    reuse_evidence=[]
    if not args.smoke:
        for j in jobs:
            if j['reused']:
                bank=Path(args.reuse_root)/('native' if j['condition']=='C' else 'balanced_views')/f'env{j["target"]}'/f'seed{j["seed"]}'
                reuse_evidence.append(dict(**j,**validate_reuse(bank,j['condition'],j['target'],j['seed'])))
    plan=dict(dataset='PACS',targets=targets,seeds=seeds,conditions={c:dict(augmentation=a,sirm_enabled=s) for c,(a,s) in CONDITIONS.items()},
        jobs=jobs,smoke=args.smoke,total_runs=len(jobs),new_training_runs=sum(not j['reused'] for j in jobs),steps=6 if args.smoke else 5001,
        architecture='existing HypothesesSIRM / MESSISubsetIRM, unchanged',off_ablation='only subset_irm_lambda_sirm = 0',
        primary_metrics=['acc_original_native','worst_view_native','worst_view_global','acc_gray_blur_native','acc_gray_blur_global'],
        checkpoint_selection='mean source-out accuracy; earliest tie',global_weights='existing source-in mixture CE fit, source-out NLL selection',
        probe='CLS frozen; source-in LBFGS linear classifier; L2 grid [.001,.01,.1] selected by equal-domain source-out NLL',
        practical_interaction_margin=.005,source_only_selection=True,reuse_evidence=reuse_evidence,
        source_code_hashes={str(p):base.digest(p) for p in Path(__file__).parent.glob('*.py')},
        baseline_config=config_for(True),disabled_config=config_for(False))
    if (root/'plan.json').exists():
        old=json.loads((root/'plan.json').read_text())
        if old!=plan:raise ValueError('refuse to change a locked experiment plan')
    else:json_write(root/'plan.json',plan)
    # Fail before spending training compute if any proposed reuse is not truly equivalent.
    commands=[]
    for j in jobs:
        command=[sys.executable,'-u','-m','experiments.sirm_representation_h2.runner','job','--output',str(root),'--reuse-root',str(Path(args.reuse_root).resolve()),
            '--condition',j['condition'],'--target-env',str(j['target']),'--seed',str(j['seed'])]
        if args.smoke:command.append('--smoke')
        commands.append((j,command))
    (root/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(Path.cwd()))+'\n'+'\n'.join(shlex.join(c) for _,c in commands)+'\n')
    summarize(root);start=time.time();failures=[];completed=0
    # Source/reuse diagnostics are evaluated under the same locked plan; no target-driven adaptation.
    for j,command in commands:
        dest=root/'runs'/j['condition']/f'env{j["target"]}'/f'seed{j["seed"]}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'metrics.json').exists():completed+=1;continue
        json_write(root/'progress.json',dict(status='running',completed=completed,total=len(jobs),current=j,pid=os.getpid(),elapsed_seconds=time.time()-start,failed=len(failures)))
        print(f'start {completed+len(failures)+1}/{len(jobs)} {j}',flush=True)
        with (dest/'job.log').open('a') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            tail='\n'.join((dest/'job.log').read_text().splitlines()[-30:]);failure=dict(**j,error=tail,returncode=result.returncode)
            failures.append(failure);json_write(dest/'failure.json',failure);json_write(root/'failures.json',failures)
            print(f'FAILED {j}: {tail}',flush=True)
            summarize(root)
            if args.smoke:raise RuntimeError('smoke failed; full suite is not authorized to launch until fixed')
        else:
            completed+=1;print(f'completed {completed}/{len(jobs)} {j}',flush=True);summarize(root)
    status=summarize(root)
    if not failures:
        # Architecture and sampling controls cover all four factorial arms within each split/seed.
        audit=[]
        for target in targets:
            for seed in seeds:
                manifests=[]
                for c in CONDITIONS:
                    bank=Path(json.loads((root/'runs'/c/f'env{target}'/f'seed{seed}'/'config.json').read_text())['bank'])
                    manifests.append(json.loads((bank/'manifest.json').read_text()))
                for key in ['initialization_sha256','schedule_sha256','split_hash','parameters_total','parameters_trainable','pretrained_sha256']:
                    if len({m[key] for m in manifests})!=1:raise ValueError(f'unmatched control {key} env{target} seed{seed}')
                audit.append(dict(target=target,seed=seed,all_four_initialization_schedule_split_capacity_matched=True))
        json_write(root/'paired_controls_audit.json',audit)
    json_write(root/'progress.json',dict(status='completed' if not failures else 'completed_with_failures',**status,total=len(jobs),elapsed_seconds=time.time()-start))
    print(json.dumps(status),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ['suite','job']:
        q=sub.add_parser(name);q.add_argument('--output',required=True);q.add_argument('--reuse-root',default=DEFAULT_REUSE);q.add_argument('--smoke',action='store_true')
        if name=='job':
            q.add_argument('--condition',choices=list(CONDITIONS),required=True);q.add_argument('--target-env',type=int,choices=range(4),required=True);q.add_argument('--seed',type=int,choices=range(3),required=True)
    q=sub.add_parser('train');q.add_argument('--output',required=True);q.add_argument('--aug_mode',choices=['baseline','h2_balanced'],required=True)
    q.add_argument('--sirm',choices=['on','off'],required=True);q.add_argument('--target-env',type=int,choices=range(4),required=True);q.add_argument('--seed',type=int,choices=range(3),required=True);q.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.command=='train':
        c=next(c for c,(aug,on) in CONDITIONS.items() if aug==args.aug_mode and on==(args.sirm=='on'));torch.set_num_threads(4);train(c,args.output,args.target_env,args.seed,args.smoke)
    elif args.command=='job':job(args)
    else:suite(args)

if __name__=='__main__':main()

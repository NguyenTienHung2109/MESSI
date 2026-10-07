"""Source-selected lambda/activation pilot; legacy training code is unchanged."""
import argparse,json,os,subprocess,sys,time,statistics,shlex
from pathlib import Path
import torch
from experiments.sirm_h1_h6 import run as base
from experiments.sirm_representation_h2.protocol import config_for,ViewTransform
from experiments.sirm_representation_h2.diagnostics import json_write,evaluate_run
from experiments.sirm_representation_h2.summarize import write_csv,describe,paired_bootstrap
from .model import DiagnosticSIRM,ReplayComplete

ARMS={'off':(0.,0),'l001':(.01,0),'l0025':(.025,0),'l005':(.05,0),'l01':(.1,0),'ramp01':(.1,500)}
OLD=Path('results/sirm_representation_h2/full').resolve()

def audit_replay(root,job):
    a=root/'replays'/job['arm']/f'env{job["target"]}'/f'seed{job["seed"]}'
    c='B' if job['arm']=='off' else 'D';b=Path(json.loads((OLD/'runs'/c/f'env{job["target"]}'/f'seed{job["seed"]}'/'config.json').read_text())['bank'])
    previous={r['step']:r for r in map(json.loads,(b/'train_log.jsonl').read_text().splitlines())}
    for line in (a/'train_log.jsonl').read_text().splitlines():
        row=json.loads(line);old=previous[row['step']]
        for metric in ['loss_cls','loss_route','loss_sirm','source_mean']:
            assert abs(row[metric]-old[metric])<1e-4,(job,row['step'],metric,row[metric],old[metric])
    return dict(**job,replay_matches_logged_reference=True)

def train(args):
    torch.set_num_threads(4);root=Path(args.output).resolve();dest=root/('replays' if args.replay else 'banks')/args.arm/f'env{args.target}'/f'seed{args.seed}'
    if (dest/('replay_complete.json' if args.replay else 'summary.json')).exists():return
    config=config_for(True);strength,ramp=ARMS[args.arm]
    config['hparams']['subset_irm_lambda_sirm']=strength
    if ramp:config['hparams']['subset_irm_sirm_ramp_steps']=2 if args.smoke else ramp
    DiagnosticSIRM.output=dest;DiagnosticSIRM.replay=args.replay;DiagnosticSIRM.smoke=args.smoke
    base.HypothesesSIRM=DiagnosticSIRM;base.ViewTransform=ViewTransform
    resume=(dest/'resume.pt').exists()
    if resume and (dest/'dense_diagnostics.jsonl').exists():
        next_step=base.load(dest/'resume.pt')['next_step']
        rows=[line for line in (dest/'dense_diagnostics.jsonl').read_text().splitlines() if json.loads(line)['step']<next_step]
        (dest/'dense_diagnostics.jsonl').write_text(''.join(line+'\n' for line in rows))
    namespace=argparse.Namespace(output=str(dest),data_dir='domainbed/data',seed=args.seed,target_env=args.target,
        criterion=None,assignment_weight=1.,variant='h2_activation_'+args.arm,cohort='balanced_views',topk=None,
        steps=6 if args.smoke else None,smoke=args.smoke,limit=8 if args.smoke else None,resume=resume,stop_after=None)
    try:base.train(namespace,config)
    except ReplayComplete:
        if not args.replay:raise
    for source in Path(__file__).parent.glob('*.py'):(dest/'source_snapshot'/('activation_'+source.name)).write_bytes(source.read_bytes())
    if args.replay:json_write(dest/'replay_complete.json',dict(steps=6 if args.smoke else 1051,source_only=True,accuracy_comparison=False))

def report(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());rows=[];trajectory=[]
    for job in plan['training_jobs']:
        bank=root/'banks'/job['arm']/f'env{job["target"]}'/f'seed{job["seed"]}'
        source='new';key=dict(arm=job['arm'],target=job['target'],seed=job['seed'],lambda_sirm=ARMS[job['arm']][0],ramp_steps=ARMS[job['arm']][1])
        if job['reuse']:
            c='B' if job['arm']=='off' else 'D';cfg=json.loads((OLD/'runs'/c/f'env{job["target"]}'/f'seed{job["seed"]}'/'config.json').read_text());bank=Path(cfg['bank']);source='reused'
        row=dict(**key,status='pending',bank=str(bank),source=source)
        if (bank/'summary.json').exists():
            selected=json.loads((bank/'summary.json').read_text());row.update(status='trained',source_val=selected['source_mean'],selected_step=selected['selected_step'])
        metrics=root/'evaluation'/job['arm']/f'env{job["target"]}'/f'seed{job["seed"]}'/'metrics.json'
        if metrics.exists():row.update(json.loads(metrics.read_text())['metrics']);row['status']='evaluated'
        rows.append(row)
    for job in plan['replay_jobs']:
        bank=root/'replays'/job['arm']/f'env{job["target"]}'/f'seed{job["seed"]}'
        if (bank/'dense_diagnostics.jsonl').exists():
            for line in (bank/'dense_diagnostics.jsonl').read_text().splitlines():
                r=json.loads(line);r.update(arm=job['arm'],target=job['target'],seed=job['seed'])
                for key,value in list(r.items()):
                    if isinstance(value,(list,dict)):r[key]=json.dumps(value)
                trajectory.append(r)
    write_csv(root/'runs.csv',rows);write_csv(root/'onset_trajectories.csv',trajectory)
    onset=[]
    for job in plan['replay_jobs']:
        part=[r for r in trajectory if r['arm']==job['arm'] and r['target']==job['target'] and r['seed']==job['seed']]
        for event in [500,1000]:
            window=[r for r in part if event-10<=r['step']<=event+10 and r.get('diagnostic_cosine_backbone') is not None]
            if window:
                onset.append(dict(arm=job['arm'],target=job['target'],seed=job['seed'],event=event,
                    backbone_cosine_mean=statistics.mean(r['diagnostic_cosine_backbone'] for r in window),
                    backbone_negative_count=sum(r['diagnostic_cosine_backbone']<0 for r in window),samples=len(window),
                    max_weighted_ratio=max(r['diagnostic_weighted_ratio_backbone'] for r in window),
                    max_raw_ratio=max(r['diagnostic_raw_ratio_backbone'] for r in window)))
    write_csv(root/'onset_summary.csv',onset)
    aggregate=[];effects=[]
    important=['source_val','acc_original_native','worst_view_native','worst_view_global','acc_gray_blur_native','acc_gray_blur_global']
    lookup={(r['arm'],r['target'],r['seed']):r for r in rows}
    for arm in ARMS:
        for metric in important:
            part=[r for r in rows if r['arm']==arm and metric in r]
            for target in plan['targets']:
                values=[r[metric] for r in part if r['target']==target]
                if values:aggregate.append(dict(arm=arm,metric=metric,target=target,**describe(values)))
            means=[statistics.mean(r[metric] for r in part if r['seed']==seed) for seed in plan['seeds'] if sum(r['seed']==seed for r in part)==len(plan['targets'])]
            if means:aggregate.append(dict(arm=arm,metric=metric,target='all',unit='equal target mean then seed SD',**describe(means)))
            for reference in ['off','l01']:
                units=[];values=[]
                for target in plan['targets']:
                    for seed in plan['seeds']:
                        a=lookup[arm,target,seed];b=lookup[reference,target,seed]
                        if metric in a and metric in b:
                            delta=a[metric]-b[metric];values.append(delta);units.append((target,seed))
                            effects.append(dict(arm=arm,reference=reference,metric=metric,target=target,seed=seed,delta=delta,scope='paired_cell'))
                if values:effects.append(dict(arm=arm,reference=reference,metric=metric,target='all',scope='paired_overall',**describe(values),**paired_bootstrap(values,units)))
    write_csv(root/'aggregate.csv',aggregate);write_csv(root/'paired_effects.csv',effects)
    text=['# H2 SIRM activation pilot','',f'Trained {sum(r["status"]!="pending" for r in rows)}/{len(rows)} cells; evaluated {sum(r["status"]=="evaluated" for r in rows)}.',
        '', 'Target Art/Sketch × seeds0/1/2. Same5001-step budget, architecture, H2, splits, router/expert losses. Lambda and SIRM ramp are the only objective interventions. Setting choice uses equal target/seed mean source-out selected-checkpoint accuracy; ties prefer less penalty, then earlier ramp. Target scores are evaluated only after locking this choice.',
        '', 'Onset replay is a logging diagnostic, not a shortened accuracy experiment. Q updates still occur every500 steps; logging does not increase assignment-update frequency. Historical Q snapshots and raw/counterfactual gradients are recorded per actual minibatch around activation.',
        '', 'This is exploratory after earlier PACS target inspection. Source-val hyperparameter/model selection can overfit validation; this pilot does not establish generalization to other datasets.',
        '', '| Arm | Lambda | Ramp | Source val mean | Target original mean |','|---|---:|---:|---:|---:|']
    for arm,(strength,ramp) in ARMS.items():
        part=[r for r in rows if r['arm']==arm];sv=[r['source_val'] for r in part if 'source_val' in r];target=[r['acc_original_native'] for r in part if 'acc_original_native' in r]
        text.append(f'| {arm} | {strength} | {ramp} | '+(f'{100*statistics.mean(sv):.3f}' if len(sv)==len(part) else 'pending')+' | '+(f'{100*statistics.mean(target):.3f}' if len(target)==len(part) else 'pending')+' |')
    (root/'summary.md').write_text('\n'.join(text)+'\n');return rows

def suite(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True);targets=[0] if args.smoke else [0,3];seeds=[0] if args.smoke else [0,1,2]
    replay=[dict(arm=a,target=t,seed=s,replay=True,reuse=False) for t in targets for s in seeds for a in ['off','l01']]
    jobs=[dict(arm=a,target=t,seed=s,replay=False,reuse=(a in ['off','l01'] and not args.smoke)) for t in targets for s in seeds for a in ARMS]
    plan=dict(smoke=args.smoke,targets=targets,seeds=seeds,arms=ARMS,replay_jobs=replay,training_jobs=jobs,
        steps=6 if args.smoke else 5001,replay_end=1050,target_selection=False,
        source_selection='mean source-out accuracy across six cells; ties lower lambda then ramp; locked before new target evaluation',
        hashes={p.name:base.digest(p) for p in Path(__file__).parent.glob('*.py')})
    if (root/'plan.json').exists():
        assert json.loads((root/'plan.json').read_text())==json.loads(json.dumps(plan)),'locked plan changed'
    else:json_write(root/'plan.json',plan)
    commands=[];failures=[];start=time.time();audits=[]
    for job in replay+jobs:
        if job['reuse']:continue
        command=[sys.executable,'-u','-m','experiments.sirm_h2_activation.run','train','--output',str(root),'--arm',job['arm'],'--target',str(job['target']),'--seed',str(job['seed'])]
        if job['replay']:command.append('--replay')
        if args.smoke:command.append('--smoke')
        commands.append(shlex.join(command))
    (root/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(Path.cwd()))+'\n'+'\n'.join(commands)+'\n')
    for job in replay+jobs:
        if job['reuse']:continue
        command=[sys.executable,'-u','-m','experiments.sirm_h2_activation.run','train','--output',str(root),'--arm',job['arm'],'--target',str(job['target']),'--seed',str(job['seed'])]
        if job['replay']:command.append('--replay')
        if args.smoke:command.append('--smoke')
        log=root/'logs'/f'{"replay" if job["replay"] else "train"}_{job["arm"]}_env{job["target"]}_seed{job["seed"]}.log';log.parent.mkdir(exist_ok=True)
        json_write(root/'progress.json',dict(stage='replay' if job['replay'] else 'training',current=job,pid=os.getpid(),elapsed=time.time()-start))
        with log.open('a') as stream:code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT).returncode
        if code:
            failure=dict(job=job,returncode=code,log=str(log),error='\n'.join(log.read_text().splitlines()[-25:]));failures.append(failure);json_write(root/'failures.json',failures);raise RuntimeError(failure)
        if job['replay'] and not args.smoke:
            audits.append(audit_replay(root,job));json_write(root/'replay_audit.json',audits)
        report(root);print('finished '+json.dumps(job),flush=True)
    rows=report(root);assert all(r['status']!='pending' for r in rows)
    controls=[]
    for target in targets:
        for seed in seeds:
            manifests=[json.loads((Path(r['bank'])/'manifest.json').read_text()) for r in rows if r['target']==target and r['seed']==seed]
            for key in ['initialization_sha256','split_hash','schedule_sha256','parameters_total','parameters_trainable','pretrained_sha256']:
                assert len({m[key] for m in manifests})==1,(target,seed,key)
            controls.append(dict(target=target,seed=seed,all_six_initialization_split_schedule_capacity_matched=True))
    json_write(root/'paired_controls_audit.json',controls)
    rankings=[]
    for arm in ARMS:
        part=[r for r in rows if r['arm']==arm];rankings.append(dict(arm=arm,mean_source_val=statistics.mean(r['source_val'] for r in part),lambda_sirm=ARMS[arm][0],ramp_steps=ARMS[arm][1]))
    winner=min(rankings,key=lambda r:(-r['mean_source_val'],r['lambda_sirm'],r['ramp_steps']))
    json_write(root/'source_selection_lock.json',dict(winner=winner,rankings=rankings,target_used=False))
    for r in rows:
        dest=root/'evaluation'/r['arm']/f'env{r["target"]}'/f'seed{r["seed"]}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'metrics.json').exists():continue
        json_write(root/'progress.json',dict(stage='evaluation',current=r,pid=os.getpid(),elapsed=time.time()-start))
        if r['source']=='reused':
            c='B' if r['arm']=='off' else 'D';result=json.loads((OLD/'runs'/c/f'env{r["target"]}'/f'seed{r["seed"]}'/'metrics.json').read_text())
            result['activation_arm']=r['arm'];result['reused']=True;json_write(dest/'metrics.json',result)
        else:
            result=evaluate_run(r['bank'],dest,'B' if r['arm']=='off' else 'D',r['target'],r['seed'],False,args.smoke)
            result['activation_arm']=r['arm'];json_write(dest/'metrics.json',result)
        report(root)
    # Logging must preserve the original trajectory on the replayed reference settings.
    if not args.smoke:
        audits=[audit_replay(root,job) for job in replay]
        json_write(root/'replay_audit.json',audits)
    json_write(root/'progress.json',dict(stage='completed',failures=0,total_cells=len(rows),elapsed=time.time()-start))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['suite','train']);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--arm',choices=list(ARMS));p.add_argument('--target',type=int);p.add_argument('--seed',type=int);p.add_argument('--replay',action='store_true');args=p.parse_args()
    train(args) if args.command=='train' else suite(args)

if __name__=='__main__':main()

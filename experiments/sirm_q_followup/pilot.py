"""H2 penalty-assignment control: preserve learned-Q router supervision."""
import argparse, json, os, shlex, subprocess, sys, time
from pathlib import Path
import torch
from domainbed import subset_irm as core
from experiments.sirm_h1_h6 import run as base
from experiments.sirm_representation_h2.protocol import config_for, ViewTransform, validate_reuse
from experiments.sirm_representation_h2.diagnostics import evaluate_run, json_write
from experiments.sirm_representation_h2.summarize import write_csv, describe, paired_bootstrap

OLD=Path('results/sirm_representation_h2/full').resolve()
ARMS=['off','learned','random','global']
ORIGINAL=core.subset_irm_penalty


def penalty_q(q,arm,permutation):
    if arm=='random':
        # Fixed random feasible pair graph, independent of competence/Q.
        result=torch.zeros_like(q)
        for d,pair in enumerate([(0,1),(1,2),(0,2)]):
            result[d,permutation[list(pair)]]=0.5
        return result.detach()
    if arm=='global': return torch.full_like(q,1/q.shape[1]).detach()
    return q.detach()


def install(arm,seed,target,dest):
    # Local generator: drawing the fixed random expert relabeling cannot disturb
    # model initialization, augmentation or training minibatch RNG streams.
    gen=torch.Generator().manual_seed(918273+seed*101+target*1009)
    perm=torch.randperm(6,generator=gen)
    def wrapper(logits,labels,domains,gamma,q,min_effective_samples=2.,detach_responsibility=True,eps=core.EPS):
        actual=penalty_q(q,arm,perm.to(q.device))
        loss,info=ORIGINAL(logits,labels,domains,gamma,actual,min_effective_samples,detach_responsibility,eps)
        if torch.is_grad_enabled():
            # Includes diagnostic-gradient calls; record explicitly, not as steps.
            with (dest/'penalty_calls.jsonl').open('a') as f:
                f.write(json.dumps(dict(arm=arm,q_learned=q.tolist(),q_penalty=actual.tolist(),active=info['active'],active_cells=info['active_cells'].tolist(),ess=info['ess'].tolist(),loss=float(loss.detach())))+'\n')
        return loss,info
    core.subset_irm_penalty=wrapper
    return perm.tolist()


def train(args):
    torch.set_num_threads(4);dest=Path(args.output).resolve()/'banks'/args.arm/f'env{args.target}'/f'seed{args.seed}'
    if (dest/'summary.json').exists():return
    config=config_for(args.arm!='off');base.ViewTransform=ViewTransform
    perm=install(args.arm,args.seed,args.target,dest)
    ns=argparse.Namespace(output=str(dest),data_dir='domainbed/data',seed=args.seed,target_env=args.target,criterion=None,assignment_weight=1.,variant='q_followup_'+args.arm,cohort='balanced_views',topk=None,steps=6 if args.smoke else None,smoke=args.smoke,limit=8 if args.smoke else None,resume=(dest/'resume.pt').exists(),stop_after=None)
    base.train(ns,config)
    json_write(dest/'penalty_control.json',dict(arm=args.arm,permutation=perm,router_teacher='unchanged learned-Q',penalty_only=True))
    for p in Path(__file__).parent.glob('*.py'):(dest/'source_snapshot'/('q_followup_'+p.name)).write_bytes(p.read_bytes())


def report(root,plan):
    rows=[]
    for j in plan['jobs']:
        bank=Path(j['bank']);r=dict(**j,status='pending')
        if (bank/'summary.json').exists():
            s=json.loads((bank/'summary.json').read_text());r.update(status='trained',source_mean=s['source_mean'],selected_step=s['selected_step'])
        p=root/'evaluation'/j['arm']/f'env{j["target"]}'/f'seed{j["seed"]}'/'metrics.json'
        if p.exists():r.update(status='evaluated',**json.loads(p.read_text())['metrics'])
        rows.append(r)
    write_csv(root/'runs.csv',rows)
    lines=['# H2 Q controls','',f'Trained {sum(r["status"]!="pending" for r in rows)}/{len(rows)}; evaluated {sum(r["status"]=="evaluated" for r in rows)}/{len(rows)}.', '', 'Cartoon/Sketch × seeds 0/1/2; 5001 steps. OFF and learned reuse verified identical factorial banks. Random selects three expert slots with a local seeded permutation, then assigns the three source pairs to those slots: exactly two experts per domain, each selected expert shared by two domains, fixed throughout training, weights 0.5. This is a feasible random pair graph independent of source competence. Learned-Q incidence and soft weights can differ, so differences are not pure assignment-criterion effects. Global uses normalized dense Q. Only penalty Q changes; router distillation retains learned Q in every arm. Penalty normalization, ESS and two-domain eligibility remain unchanged. Global and random may still have inactive routed cells; no routing support is forced. Actual Q and ESS are logged for every penalty call.', '', 'Checkpoints use source-out mean accuracy, earliest tie. Targets are evaluated after all four arms finish and matched controls pass. No target-driven model/hyperparameter choice.', '', '| Arm | Source mean | Target original |','|---|---:|---:|']
    import statistics
    for arm in ARMS:
        rs=[r for r in rows if r['arm']==arm]
        vals=[]
        for metric in ['source_mean','acc_original_native']:
            v=[r[metric] for r in rs if metric in r];vals.append(f'{100*statistics.mean(v):.3f}' if len(v)==len(rs) else 'pending')
        lines.append('| '+arm+' | '+' | '.join(vals)+' |')
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    effects=[]
    for r in rows:
        ref=next(x for x in rows if x['arm']=='off' and x['target']==r['target'] and x['seed']==r['seed'])
        for metric in ['source_mean','acc_original_native','worst_view_native']:
            if metric in r and metric in ref:effects.append(dict(arm=r['arm'],target=r['target'],seed=r['seed'],metric=metric,delta_vs_off=r[metric]-ref[metric]))
    if effects:write_csv(root/'paired_effects.csv',effects)
    aggregate=[]
    for arm in ARMS:
        for metric in ['source_mean','acc_original_native','worst_view_native','worst_view_global']:
            for target in sorted({r['target'] for r in rows}):
                vals=[r[metric] for r in rows if r['arm']==arm and r['target']==target and metric in r]
                if vals:aggregate.append(dict(arm=arm,target=target,metric=metric,**describe(vals)))
    if aggregate:write_csv(root/'aggregate.csv',aggregate)
    overall=[]
    for arm in ARMS:
        for metric in ['source_mean','acc_original_native','worst_view_native']:
            part=[r for r in effects if r['arm']==arm and r['metric']==metric]
            if part:overall.append(dict(arm=arm,metric=metric,**describe([r['delta_vs_off'] for r in part]),**paired_bootstrap([r['delta_vs_off'] for r in part],[(r['target'],r['seed']) for r in part])))
    if overall:write_csv(root/'paired_overall.csv',overall)
    return rows


def suite(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    targets=[1] if args.smoke else [1,3];seeds=[0] if args.smoke else [0,1,2];jobs=[]
    for t in targets:
        for s in seeds:
            for arm in ARMS:
                reuse=not args.smoke and arm in ['off','learned']
                bank=root/'banks'/arm/f'env{t}'/f'seed{s}'
                if reuse:
                    cfg=json.loads((OLD/'runs'/('B' if arm=='off' else 'D')/f'env{t}'/f'seed{s}'/'config.json').read_text());bank=Path(cfg['bank'])
                    eff=json.loads((bank/'effective_config.json').read_text())
                    assert eff['config']==config_for(arm!='off') and eff['steps']==5001 and eff['cohort']=='balanced_views'
                    if arm=='learned':validate_reuse(bank,'D',t,s)
                    assert base.digest(bank/'checkpoints/best.pkl')==cfg['checkpoint_sha256']
                    assert (eff['target_env'],eff['seed'])==(t,s)
                jobs.append(dict(arm=arm,target=t,seed=s,reuse=reuse,bank=str(bank)))
    plan=dict(jobs=jobs,smoke=args.smoke,steps=6 if args.smoke else 5001,target_selection=False,random_rule='fixed random feasible three-pair graph; three randomly selected expert slots; two per domain; weight 0.5',router_teacher='learned Q unchanged',hashes={p.name:base.digest(p) for p in Path(__file__).parent.glob('*.py')})
    if (root/'plan.json').exists():assert json.loads((root/'plan.json').read_text())==plan,'locked plan changed'
    else:json_write(root/'plan.json',plan)
    commands=[]
    for j in jobs:
        if j['reuse']:continue
        cmd=[sys.executable,'-u','-m','experiments.sirm_q_followup.pilot','train','--output',str(root),'--arm',j['arm'],'--target',str(j['target']),'--seed',str(j['seed'])]
        if args.smoke:cmd.append('--smoke')
        commands.append((j,cmd))
    (root/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+'\n'.join(shlex.join(c) for j,c in commands)+'\n')
    report(root,plan);start=time.time()
    for j,cmd in commands:
        log=root/'logs'/f'{j["arm"]}_env{j["target"]}_seed{j["seed"]}.log';log.parent.mkdir(exist_ok=True)
        json_write(root/'progress.json',dict(stage='training',current=j,pid=os.getpid(),elapsed=time.time()-start))
        with log.open('a') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
        if code:
            json_write(root/'progress.json',dict(stage='failed',current=j,log=str(log),returncode=code));raise RuntimeError(log.read_text()[-4000:])
        report(root,plan);print('trained '+json.dumps(j),flush=True)
    rows=report(root,plan);audits=[]
    for t in targets:
        for s in seeds:
            ms=[json.loads((Path(r['bank'])/'manifest.json').read_text()) for r in rows if r['target']==t and r['seed']==s]
            for k in ['initialization_sha256','split_hash','schedule_sha256','parameters_total','parameters_trainable','pretrained_sha256']:assert len({m[k] for m in ms})==1,(t,s,k)
            audits.append(dict(target=t,seed=s,initialization_split_schedule_capacity_matched=True))
    json_write(root/'paired_controls_audit.json',audits)
    json_write(root/'source_lock.json',dict(checkpoints=[dict(arm=r['arm'],target=r['target'],seed=r['seed'],checkpoint_sha256=base.digest(Path(r['bank'])/'checkpoints/best.pkl'),selected_step=r['selected_step']) for r in rows],target_used=False))
    for r in rows:
        dest=root/'evaluation'/r['arm']/f'env{r["target"]}'/f'seed{r["seed"]}';dest.mkdir(parents=True,exist_ok=True)
        json_write(root/'progress.json',dict(stage='evaluation',current=r,pid=os.getpid(),elapsed=time.time()-start))
        if r['reuse']:
            src=OLD/'runs'/('B' if r['arm']=='off' else 'D')/f'env{r["target"]}'/f'seed{r["seed"]}'/'metrics.json'
            json_write(dest/'metrics.json',json.loads(src.read_text()))
        elif not (dest/'metrics.json').exists():evaluate_run(r['bank'],dest,'B' if r['arm']=='off' else 'D',r['target'],r['seed'],False,args.smoke)
        report(root,plan)
    json_write(root/'progress.json',dict(stage='completed',cells=len(rows),elapsed=time.time()-start))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['suite','train']);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--arm',choices=ARMS);p.add_argument('--target',type=int);p.add_argument('--seed',type=int);args=p.parse_args()
    try:train(args) if args.command=='train' else suite(args)
    except Exception as e:
        root=Path(args.output);root.mkdir(parents=True,exist_ok=True);json_write(root/'failure.json',dict(error=repr(e)));raise

if __name__=='__main__':main()

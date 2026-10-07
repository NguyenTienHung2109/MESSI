"""Six-run PACS Sketch / Terra L38 gate, seed 0. No automatic scale-up."""
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from scipy.stats import spearmanr
from experiments.sirm_h1_h6 import run as base
from experiments.sirm_discover import run as v2
from experiments.sirm_representation_h2.summarize import write_csv
from .data import Adapter,SPECS
from .model import GateSIRM
from .metrics import evaluate

ARMS={'R0':('off','learned'),'R1':('on','learned'),'R2':('random','random')}
CONFIG_FOR=v2.config_for


def hashes():
    files=list(Path(__file__).parent.glob('*.py'))+list(Path('experiments/sirm_discover').glob('*.py'))+[Path('experiments/sirm_discover/config.json'),Path('experiments/sirm_h1_h6/run.py')]
    return {str(p):base.digest(p) for p in files}


def install(args,adapter):
    base.inventory=adapter.inventory;base.datasets_for=adapter.datasets_for;base.source_data=adapter.source_data
    policy=ARMS[args.arm][1]
    def build(config):
        from domainbed.deit_transformer import DeiTFeaturizer
        backbone=DeiTFeaturizer(pretrained=config['pretrained'],model_name=config['backbone'])
        return GateSIRM(backbone,backbone.n_outputs,len(adapter.classes),3,config,policy=policy,assignment_seed=args.seed).cuda()
    v2.build=build
    def config(smoke,arm):
        result=CONFIG_FOR(smoke,'off' if args.arm=='R0' else 'on');result['protocol']='sirm_discover_gate_'+args.dataset
        result['q_freeze_stage3']=True # historical fixed-Q protocol only
        start=result['global_steps']+result['discover_steps'];last=start+result['invariance_steps']-1
        result['diagnostic_steps']=[start-1,start,(start+last)//2,last]
        if result['discover_steps']%result['utility_interval']:raise ValueError('stage3 must start at a utility update')
        return result
    v2.config_for=config
    def measured(model,data,config,diagnostics=False):
        step=int(model.step)-1;start=config['global_steps']+config['discover_steps'];last=start+config['invariance_steps']-1
        events={start-1:'before',start:'activation',(start+last)//2:'middle',last:'final'}
        result=evaluate(model,data,config,diagnostics or step in events)
        if step in events:
            folder=Path(args.output)/'milestones';folder.mkdir(exist_ok=True)
            lam=config['lambda_sirm']*min(max((step-start)/config['sirm_ramp_steps'],0.),1.)
            base.write(folder/(events[step]+'.json'),dict(step=step,event=events[step],effective_lambda=lam,measurement='post-update deterministic source-out',**result))
        return result
    v2.evaluate=measured
    return build


def train(args):
    if args.plan:
        plan=json.loads(Path(args.plan).read_text());assert plan['hashes']==hashes(),'executed code differs from locked gate'
    adapter=Adapter(args.dataset);install(args,adapter)
    ns=argparse.Namespace(output=args.output,smoke=args.smoke,arm=ARMS[args.arm][0],config=None,resume=args.resume,target=SPECS[args.dataset]['target'],seed=args.seed,data_dir=args.data_dir)
    v2.train(ns)
    effective=json.loads((Path(args.output)/'effective_config.json').read_text())
    base.write(Path(args.output)/'gate_control.json',dict(dataset=args.dataset,arm=args.arm,policy=ARMS[args.arm][1],q_locked_at_stage3=True,hashes=hashes(),effective=effective))


def restore(bank,dataset,arm,seed,data_dir):
    adapter=Adapter(dataset);effective=json.loads((bank/'effective_config.json').read_text());split=json.loads((bank/'split_manifest.json').read_text());adapter.classes=split['class_names']
    ns=argparse.Namespace(output=str(bank),dataset=dataset,arm=arm,seed=seed)
    builder=install(ns,adapter);model=builder(effective['config']);payload=base.load(bank/'checkpoints/best.pkl');model.load_state_dict(payload['model_dict']);model.configure_stage();model.eval()
    return model,adapter,effective,split


def target_eval(root,dataset,jobs,smoke,data_dir):
    locks=[]
    for job in jobs:
        bank=Path(job['bank']);locks.append(dict(arm=job['arm'],sha256=base.digest(bank/'checkpoints/best.pkl'),selected_step=json.loads((bank/'summary.json').read_text())['selected_step']))
    lock=root/dataset/'target_lock.json';base.write(lock,dict(checkpoints=locks,target_used_for_selection=False));lockhash=base.digest(lock)
    sources={}
    for job in jobs:
        dest=root/dataset/'evaluation'/job['arm'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'metrics.json').exists():continue
        bank=Path(job['bank']);model,adapter,effective,split=restore(bank,dataset,job['arm'],0,data_dir);config=effective['config']
        assert base.digest(bank/'checkpoints/best.pkl')==next(x['sha256'] for x in locks if x['arm']==job['arm'])
        target=SPECS[dataset]['target'];data=ImageFolder(adapter.folder(data_dir,target),base.ViewTransform(config,'original',False,target,'balanced_views'))
        assert data.classes==split['class_names'];ids=list(range(len(data)))
        if smoke:ids=ids[:14]
        measured=evaluate(model,{target:Subset(data,ids)},config,diagnostics=True)
        source_diag=json.loads((bank/'source_diagnostics.json').read_text());src_u=np.array(source_diag['marginal_utility']).mean(0);target_u=np.array(measured['marginal_utility'])[0];importance=np.array(measured['leave_one_out_accuracy_drop'])
        def rho(a,b):return float(spearmanr(a,b).statistic) if np.ptp(a)>1e-12 and np.ptp(b)>1e-12 else None
        result=dict(dataset=dataset,target=SPECS[dataset]['domains'][target],arm=job['arm'],accuracy=measured['source_mean'],global_accuracy=measured['global_mean'],source_utility_target_utility_spearman=rho(src_u,target_u),source_utility_target_ablation_spearman=rho(src_u,importance),target_samples=len(ids),checkpoint_sha256=base.digest(bank/'checkpoints/best.pkl'),source_selected=True,target_used_for_selection=False,diagnostics=measured)
        base.write(dest/'metrics.json',result);assert base.digest(lock)==lockhash
        del model;torch.cuda.empty_cache()


def audit(jobs,config):
    ms=[json.loads((Path(j['bank'])/'manifest.json').read_text()) for j in jobs]
    for key in ['initialization_sha256','split_hash','schedule_sha256','parameters_total','pretrained_sha256']:assert len({m[key] for m in ms})==1,key
    logs={j['arm']:[json.loads(x) for x in (Path(j['bank'])/'diagnostics.jsonl').read_text().splitlines()] for j in jobs}
    start=config['global_steps']+config['discover_steps']
    for step in range(start+1):
        for arm in ['R1','R2']:
            for key in ['loss_cls','loss_discover','loss_router','loss_balance','utility_ema']:
                assert logs['R0'][step][key]==logs[arm][step][key],(arm,step,key)
    for arm in ['R0','R1','R2']:
        model=base.load(Path(next(j['bank'] for j in jobs if j['arm']==arm))/'checkpoints/last.pkl')['model_dict']
        if arm=='R0':template=model['q_locked'];null=model['null_locked']
        else:torch.testing.assert_close(template,model['q_locked'],rtol=0,atol=0);torch.testing.assert_close(null,model['null_locked'],rtol=0,atol=0)
        expected=template[:,model['assignment_permutation']] if arm=='R2' else template
        torch.testing.assert_close(expected,model['q'],rtol=0,atol=0)
        torch.testing.assert_close(template.sort(1).values,model['q'].sort(1).values,rtol=0,atol=0)
        for row in logs[arm][start:]:
            assert row['q']==expected.tolist() and row['q_learned']==template.tolist(),(arm,row['step'],'Q drift')
            assert row['q_null']==null.tolist(),(arm,row['step'],'NULL drift')
    assert all(r['loss_sirm']==0 and r['effective_lambda_sirm']==0 for r in logs['R0'])
    return dict(initialization_split_schedule_capacity_matched=True,pre_activation_trajectory_identical=True,Q_reference_and_NULL_identical=True,R2_permutation_fixed=True,OFF_penalty_zero=True,active_steps={arm:sum(r['sirm_active'] for r in log) for arm,log in logs.items()})


def mechanism_report(root,dataset,jobs):
    rows=[];experts=[];diags={}
    for job in jobs:
        bank=Path(job['bank']);arm=job['arm'];d=json.loads((bank/'source_diagnostics.json').read_text());diags[arm]=d
        t=json.loads((root/dataset/'evaluation'/arm/'metrics.json').read_text())
        rows.append(dict(arm=arm,selected_step=json.loads((bank/'summary.json').read_text())['selected_step'],source_accuracy=d['source_mean'],target_accuracy=t['accuracy'],G_selected=d['G_selected'],G_nonselected=d['G_nonselected'],interaction_advantage=d['interaction_advantage'],jaccard_global_topk=d['q_global_topk_jaccard'],correction_global_ratio=d['correction_global_norm_ratio_rms'],q_mass_mean=float(np.array(d['q']).sum(1).mean()),null_rows=d['null_rows'],source_target_importance_spearman=t['source_utility_target_ablation_spearman']))
        for m in range(len(d['residual_norm_per_expert'])):
            experts.append(dict(arm=arm,expert=m,source_utility=float(np.array(d['marginal_utility'])[:,m].mean()),target_utility=t['diagnostics']['marginal_utility'][0][m],target_accuracy_drop=t['diagnostics']['leave_one_out_accuracy_drop'][m],source_residual_norm=d['residual_norm_per_expert'][m],source_single_correction_gain=d['single_logit_improvement'][m],target_residual_norm=t['diagnostics']['residual_norm_per_expert'][m]))
    write_csv(root/dataset/'mechanisms.csv',rows);write_csv(root/dataset/'expert_importance.csv',experts)
    mask=np.array(diags['R1']['q'])>0
    gs={a:float(np.array(d['G_matrix'])[mask].mean()) if mask.any() else None for a,d in diags.items()}
    base.write(root/dataset/'fixed_selected_G.json',dict(mask_source='R1 source-in learned Q; identical to R0 Q',G_under_same_mask=gs,R1_minus_R0=gs['R1']-gs['R0'] if mask.any() else None,no_selected_cells=not bool(mask.any()),checkpoints='source-selected independently; milestone JSONs provide equal-step comparisons'))


def report(root,plan):
    rows=[];datasets=['PACS','TerraIncognita']
    for j in plan['jobs']:
        r=dict(**j,status='pending');bank=Path(j['bank'])
        if (bank/'summary.json').exists():r.update(status='trained',**json.loads((bank/'summary.json').read_text()))
        p=root/j['dataset']/'evaluation'/j['arm']/'metrics.json'
        if p.exists():r.update(status='evaluated',target_accuracy=json.loads(p.read_text())['accuracy'])
        rows.append(r)
    write_csv(root/'runs.csv',rows)
    text=['# SIRM-v2 six-run gate','',f'Trained {sum(r["status"]!="pending" for r in rows)}/6, evaluated {sum(r["status"]=="evaluated" for r in rows)}/6. '+('Pipeline smoke only.' if plan['smoke'] else 'Exploratory hardest-target pilot, seed 0.'), '', 'Global ERM/H2 anchor; freeze after 1000 global steps, discover for 1000, then 3001 invariance steps with 500-step ramp. Stage-3 Q is locked at activation in all arms; R2 applies one fixed seeded expert permutation of the same Q. This preserves every row weight/sparsity/NULL mass and cross-domain incidence. This is a fixed-Q gate of the redesign, not the continually refreshed-Q protocol. Targets are read only after source-selected checkpoints and matched-control audits.', '', '| Dataset / target | Old OFF | Old ON | R0 OFF | R1 learned Q | R2 random Q | R3 global Q |','|---|---:|---:|---:|---:|---:|---:|']
    deltas=[];decisions=[]
    for dataset in datasets:
        part={r['arm']:r for r in rows if r['dataset']==dataset};oldoff=oldon=None
        if dataset=='PACS' and not plan['smoke']:
            for arm,c in [('off','B'),('on','D')]:
                old=Path('results/sirm_representation_h2/full/runs')/c/'env3/seed0';oldcfg=json.loads((old/'config.json').read_text())
                bank=Path(part['R0']['bank'])
                metrics=json.loads((old/'metrics.json').read_text())
                old_ok=(oldcfg['dataset']=='PACS' and oldcfg['target_env']==3 and oldcfg['seed']==0 and oldcfg['augmentation']=='h2_balanced' and oldcfg['total_steps']==5001 and not oldcfg['smoke'] and not oldcfg['target_used_for_model_or_hyperparameter_selection'] and metrics['views']['original']['native']['samples']==3929)
                if old_ok and (bank/'manifest.json').exists() and json.loads((bank/'manifest.json').read_text())['split_hash']==oldcfg['split_hash']:
                    acc=metrics['metrics']['acc_original_native']
                    if arm=='off':oldoff=acc
                    else:oldon=acc
        value=lambda x:f'{100*x:.3f}' if x is not None else 'N/A'
        text.append('| '+dataset+' / '+SPECS[dataset]['domains'][SPECS[dataset]['target']]+' | '+' | '.join([value(oldoff),value(oldon)]+[value(part[a].get('target_accuracy')) for a in ['R0','R1','R2']]+['not run'])+' |')
        if all('target_accuracy' in r for r in part.values()):
            a={arm:r['target_accuracy'] for arm,r in part.items()};delta=dict(dataset=dataset,delta_architecture=a['R0']-oldoff if oldoff is not None else None,delta_SIRM=a['R1']-a['R0'],delta_Q=a['R1']-a['R2'],delta_subset=None);deltas.append(delta)
            decisions.append(dict(dataset=dataset,positive_SIRM=delta['delta_SIRM']>0,learned_beats_random=delta['delta_Q']>0))
    if deltas:write_csv(root/'deltas.csv',deltas)
    text.extend(['', 'Accuracies are percent; deltas.csv uses fractions. Terra old matching H2 OFF/ON is unverified and omitted. Old PACS rows require identical split hashes, target/seed, H2 augmentation and full target evaluation; old/new training schedules and objectives differ, so architecture delta compares whole methods. R3, OfficeHome and DomainNet are deferred. Six runs use one seed: this is a scale-up signal, not confirmatory evidence.', '', 'Per-arm source diagnostics and before/activation/middle/final snapshots include U, double-centered utility, Q/NULL, exact routed-component G, residual norms, correction/global logit ratio and source validation. Target diagnostics include per-expert CE utility, single-correction utility and leave-one-expert-out accuracy drops. U is utility, so interaction advantage is selected minus non-selected, unlike old risk gaps. The two diagnostic quantities are not identical.', '', 'All-NULL/no shared Q implies an inactive invariance test and is reported as such; it is not converted to forced assignment or tuned using target labels. No automatic 16-run scale-up is launched.'])
    (root/'summary.md').write_text('\n'.join(text)+'\n')
    if len(decisions)==2:
        promising=any(x['positive_SIRM'] for x in decisions) and all(x['learned_beats_random'] for x in decisions)
        stop=all(not x['positive_SIRM'] for x in decisions)
        base.write(root/'gate_decision.json',dict(per_dataset=decisions,promising=promising,stop_and_inspect=stop,full16_launched=False,reason='OfficeHome/DomainNet data unavailable; scale-up requires a concrete prepared protocol'))
    return rows


def suite(args):
    torch.set_num_threads(4);root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True);jobs=[]
    for dataset in SPECS:
        assert (Path(args.data_dir)/SPECS[dataset]['directory']).is_dir()
        for arm in ARMS:jobs.append(dict(dataset=dataset,arm=arm,seed=0,target=SPECS[dataset]['target'],bank=str(root/dataset/'banks'/arm)))
    plan=dict(jobs=jobs,smoke=args.smoke,seed=0,data_dir=args.data_dir,hashes=hashes(),config=CONFIG_FOR(args.smoke,'on'),Q_policy='fixed learned Q at start of stage 3; fixed seeded expert permutation for R2',target_checkpoint_selection=False)
    if (root/'plan.json').exists():assert json.loads((root/'plan.json').read_text())==plan,'locked plan changed'
    else:
        base.write(root/'plan.json',plan);snapshot=root/'source_snapshot';snapshot.mkdir()
        for p in plan['hashes']:(snapshot/p.replace('/','__')).write_bytes(Path(p).read_bytes())
    begin=time.time();report(root,plan)
    for dataset in SPECS:
        part=[j for j in jobs if j['dataset']==dataset]
        for j in part:
            if (Path(j['bank'])/'summary.json').exists():continue
            cmd=[sys.executable,'-u','-m','experiments.sirm_discover_gate.run','train','--dataset',dataset,'--arm',j['arm'],'--output',j['bank'],'--data-dir',args.data_dir,'--plan',str(root/'plan.json')]
            if args.smoke:cmd.append('--smoke')
            if (Path(j['bank'])/'resume.pt').exists():cmd.append('--resume')
            log=root/'logs'/f'{dataset}_{j["arm"]}.log';log.parent.mkdir(exist_ok=True)
            base.write(root/'progress.json',dict(stage='training',current=j,pid=os.getpid(),elapsed=time.time()-begin))
            with log.open('a') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
            if code:
                base.write(root/'progress.json',dict(stage='failed',current=j,returncode=code,log=str(log)));raise RuntimeError(log.read_text()[-4000:])
            report(root,plan);print('trained '+str(j),flush=True)
        control=audit(part,plan['config']);base.write(root/dataset/'paired_audit.json',control)
        base.write(root/'progress.json',dict(stage='evaluation',dataset=dataset,pid=os.getpid(),elapsed=time.time()-begin))
        target_eval(root,dataset,part,args.smoke,args.data_dir);mechanism_report(root,dataset,part);report(root,plan)
    base.write(root/'progress.json',dict(stage='completed',runs=6,elapsed=time.time()-begin))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['suite','train']);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--data-dir',default='domainbed/data');p.add_argument('--dataset',choices=list(SPECS));p.add_argument('--arm',choices=list(ARMS));p.add_argument('--seed',type=int,default=0);p.add_argument('--resume',action='store_true');p.add_argument('--plan');a=p.parse_args()
    train(a) if a.command=='train' else suite(a)

if __name__=='__main__':main()

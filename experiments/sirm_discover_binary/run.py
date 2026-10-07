"""DomainBed Terra L38 seed-0 OFF / binary-SIRM matched behavior test."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from experiments.sirm_h1_h6 import run as base
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover_dynamic.run import config_for
from experiments.sirm_discover_gate.model import GateSIRM
from experiments.sirm_discover_gate.metrics import evaluate
from experiments.sirm_discover_gate.data import Adapter
from experiments.sirm_discover_analysis.run import tensors_compare
from experiments.sirm_representation_h2.summarize import write_csv

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def code_hashes():
    files=list(Path(__file__).parent.glob('*.py'))+list(Path('experiments/sirm_discover').glob('*.py'))+[Path('experiments/sirm_discover/config.json'),Path('experiments/sirm_discover_gate/model.py'),Path('experiments/sirm_h1_h6/run.py'),Path('domainbed/sirm_discover.py'),Path('domainbed/scripts/train.py'),Path('domainbed/algorithms.py'),Path('domainbed/datasets.py')]
    return {str(p):base.digest(p) for p in files}

def params(arm,output):
    c=config_for(False,arm);c['sirm_weighting']='binary'
    return dict(sirm_discover_config=c,sirm_output_dir=str(output),sirm_source_envs=[0,2,3],class_balanced=False,data_augmentation=True,lr=c['lr'],weight_decay=c['weight_decay'],batch_size=c['batch_size'],pretrained=c['pretrained'],model=c['backbone'])

def command(arm,output,data_dir,steps):
    return [sys.executable,'-u','-m','domainbed.scripts.train','--dataset','TerraIncognita','--algorithm','SIRMDiscoverV2','--test_envs','1','--source_envs','0','2','3','--seed','0','--trial_seed','0','--hparams_seed','0','--steps',str(steps),'--checkpoint_freq','250','--num_workers','2','--eval_num_workers','2','--eval_batch_size','64','--data_dir',data_dir,'--output_dir',str(output),'--hparams',json.dumps(params(arm,output))]

def restore(bank):
    p=base.load(bank/'source_best.pkl');c=p['config'];backbone=DeiTFeaturizer(pretrained=c['pretrained'],model_name=c['backbone'])
    m=GateSIRM(backbone,backbone.n_outputs,10,3,c).cuda();m.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()});m.configure_stage();m.eval();return m,c,p

def finish(root,data_dir):
    banks={arm:root/'banks'/arm for arm in ['R0','R1']};logs={a:rows(p/'diagnostics.jsonl') for a,p in banks.items()}
    for k in ['weights_sha256','parameters']:assert read(banks['R0']/'initialization.json')[k]==read(banks['R1']/'initialization.json')[k]
    assert read(banks['R0']/'source_schedules.json')==read(banks['R1']/'source_schedules.json')
    for step in range(2001):
        for k in ['loss_cls','loss_discover','loss_router','loss_balance','utility_ema','q']:assert logs['R0'][step][k]==logs['R1'][step][k],(step,k)
    on=[r for r in logs['R1'] if r['stage']=='invariance'];active=[r for r in on if r['sirm_active']];refresh=[r for r in rows(banks['R1']/'q_refresh.jsonl') if r['stage']=='invariance'];measurements=[r for r in refresh if r['weighted_sirm_grad_norm'] is not None]
    for r in measurements:r['gradient_ratio']=r['weighted_sirm_grad_norm']/r['base_residual_grad_norm'] if r['base_residual_grad_norm'] else None
    write_csv(root/'active_refresh_gradients.csv',measurements)
    changes=[]
    for name in ['source_best','last']:
        a=base.load(banks['R0']/(name+'.pkl'))['model_dict'];b=base.load(banks['R1']/(name+'.pkl'))['model_dict']
        modules={m:tensors_compare({k:v for k,v in a.items() if k.startswith('model.'+m+'.')},{k:v for k,v in b.items() if k.startswith('model.'+m+'.')}) for m in ['featurizer','global_head','residuals','router']};changes.append(dict(checkpoint=name,modules=modules));del a,b
    base.write(root/'parameter_comparison.json',changes)
    locks={a:dict(sha256=base.digest(p/'source_best.pkl'),selection=read(p/'selection.json')) for a,p in banks.items()};base.write(root/'target_lock.json',dict(checkpoints=locks,target_used_for_selection=False));target={};logits={}
    for a,p in banks.items():
        m,c,payload=restore(p);adapter=Adapter('TerraIncognita');data=ImageFolder(adapter.folder(data_dir,1),base.ViewTransform(c,'original',False,1,'balanced_views'))
        target[a]=evaluate(m,{1:data},c,True);parts=[]
        with torch.no_grad():
            for x,y in DataLoader(data,batch_size=64,num_workers=2):parts.append(m(x.cuda())['logits'].cpu().numpy())
        logits[a]=np.concatenate(parts);np.save(root/f'target_logits_{a}.npy',logits[a]);assert base.digest(p/'source_best.pkl')==locks[a]['sha256'];del m;torch.cuda.empty_cache()
    base.write(root/'target_metrics.json',target)
    delta=logits['R1'].astype(np.float64)-logits['R0'].astype(np.float64)
    first=lambda rs,k:next((r['step'] for r in rs if r[k]>0),None)
    ratios=[r['gradient_ratio'] for r in measurements if r['gradient_ratio'] is not None]
    result=dict(pre_activation_identical=True,first_shared_support_step=first(refresh,'shared_experts'),first_nonzero_SIRM_step=first(on,'loss_sirm'),active_steps=len(active),stage3_steps=len(on),active_fraction=len(active)/len(on),max_SIRM_loss=max(r['loss_sirm'] for r in on),weighted_gradient_norm_max=max((r['weighted_sirm_grad_norm'] for r in measurements),default=None),gradient_ratio_min=min(ratios,default=None),gradient_ratio_max=max(ratios,default=None),gradient_ratio_median=float(np.median(ratios)) if ratios else None,target_accuracy={a:d['source_mean'] for a,d in target.items()},different_target_predictions=int((logits['R0'].argmax(1)!=logits['R1'].argmax(1)).sum()),target_max_abs_logit_difference=float(np.abs(delta).max()),parameter_divergence=any(m['different_elements']>0 for p in changes for m in p['modules'].values()),first_baseline_trajectory_difference=next((i for i in range(2001,5001) if any(logs['R0'][i][k]!=logs['R1'][i][k] for k in ['loss_cls','loss_discover','loss_router','loss_balance'])),None),weighting='binary eligible-term mean',hyperparameters_tuned=False)
    base.write(root/'behavior.json',result)
    text=['# DomainBed Terra L38 binary-SIRM behavior','', 'Seed 0, R0 OFF / R1 binary ON. Q formula, thresholds, dynamic refresh interval 50, ESS, residual architecture, discovery, lambda=0.1, 500-step ramp and 1000/1000/3001 budgets unchanged. Only active-term weighting/normalization changed: each eligible pair weight=1 and loss is its mean. Standard DomainBed train loop/splits/per-domain score columns; scheduled-source-loader adapter preserves the previous exact batch and H2 augmentation schedule. Both arms freshly train; old R0 is not reused.','', 'Source-only selection at the original 250-step cadence with stage-end/final checkpoints. Extra 50-step Q refresh evaluations do not select checkpoints. Target columns printed during DomainBed evaluation are monitoring only; final full-target scores use locked source-selected checkpoints.','', '| Behavior | Value |','|---|---:|']
    for k,v in result.items():text.append(f'| {k} | {v} |')
    text.extend(['','Per-domain scores: banks/R0/results.jsonl and banks/R1/results.jsonl; console tables in out.txt. Full exact refresh diagnostics: q_refresh.jsonl. [Raw-gradient diagnosis](../raw_diagnosis/summary.md) uses historical Q/batches at the old final checkpoint, not historical parameter gradients. Reported gradient ratios use residual parameters, matching the previous dynamic-Q diagnostic.', '', 'No automatic scaling or lambda/ramp change is applied even if a gradient ratio exceeds one. Report any dominance to the user before a separate treatment change. Accuracy is exploratory; this test establishes whether binary selection produces a measurable penalty/update.'])
    (root/'summary.md').write_text('\n'.join(text)+'\n');print(json.dumps(result),flush=True)

def run(a):
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    if a.command=='pipeline':
        out=root/'pipeline';out.mkdir(exist_ok=False);returncode=subprocess.run(command('R0',out,a.data_dir,1)).returncode
        if returncode:raise RuntimeError('DomainBed pipeline failed')
        r=rows(out/'results.jsonl')[-1];assert all(f'env{i}_{part}_acc' in r for i in range(4) for part in ['in','out']);base.write(root/'pipeline_check.json',dict(per_domain_acc_columns=True,steps=1,pipeline_only=True));return
    diagnosis=read(Path(a.diagnosis)/'diagnosis.json');assert diagnosis['raw_gradient_larger_than_1e_minus10'],'raw gradient is not healthy; inspect before training'
    if (root/'plan.json').exists():raise FileExistsError('fresh output required; no weight-only resume for paired test')
    adapter=Adapter('TerraIncognita');split=adapter.inventory(a.data_dir,[0,2,3],0,params('R0',root/'banks/R0')['sirm_discover_config']);base.write(root/'source_split_manifest.json',split)
    old=read(Path('results/sirm_discover_dynamic/terra_L38_seed0/TerraIncognita/banks/R0/manifest.json'));assert base.canonical_hash(split)==old['split_hash']
    plan=dict(seed=0,target=1,dataset='TerraIncognita',arms=['R0','R1'],hashes=code_hashes(),config={arm:params(arm,root/'banks'/arm) for arm in ['R0','R1']},raw_diagnosis=diagnosis,split_hash=old['split_hash'],changed='binary active-term weighting only; DomainBed runner integration preserves schedule',target_used_for_selection=False)
    base.write(root/'plan.json',plan);snapshot=root/'source_snapshot';snapshot.mkdir()
    (root/'environment.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    for p in plan['hashes']:(snapshot/p.replace('/','__')).write_bytes(Path(p).read_bytes())
    (root/'summary.md').write_text('# DomainBed Terra L38 binary-SIRM test\n\nIn progress: fresh R0 OFF and R1 binary ON; full original numerical settings. Results pending.\n')
    begin=time.time()
    for arm in ['R0','R1']:
        assert code_hashes()==plan['hashes'],'source changed during test'
        out=root/'banks'/arm;out.mkdir(parents=True)
        base.write(root/'progress.json',dict(stage='training',arm=arm,pid=os.getpid(),elapsed=time.time()-begin))
        with (root/(arm+'.log')).open('w') as f:code=subprocess.run(command(arm,out,a.data_dir,5001),stdout=f,stderr=subprocess.STDOUT).returncode
        if code:
            base.write(root/'progress.json',dict(stage='failed',arm=arm,returncode=code));raise RuntimeError((root/(arm+'.log')).read_text()[-4000:])
        print('completed '+arm,flush=True)
    base.write(root/'progress.json',dict(stage='evaluation',pid=os.getpid()));finish(root,a.data_dir);base.write(root/'progress.json',dict(stage='completed',elapsed=time.time()-begin,runs=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['pipeline','suite']);p.add_argument('--output',required=True);p.add_argument('--data-dir',default='domainbed/data');p.add_argument('--diagnosis',default='results/sirm_discover_binary/raw_diagnosis');a=p.parse_args();torch.set_num_threads(4);run(a)

if __name__=='__main__':main()

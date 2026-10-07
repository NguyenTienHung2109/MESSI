"""Terra L38 seed-0 dynamic-Q behavioral pair; original numerical settings."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from experiments.sirm_h1_h6 import run as base
from experiments.sirm_discover import run as v2
from experiments.sirm_discover_gate import run as gate
from experiments.sirm_discover_gate.data import Adapter,SPECS
from experiments.sirm_discover_analysis.run import tensors_compare
from experiments.sirm_representation_h2.summarize import write_csv

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def hashes():return dict(gate.hashes(),**{str(p):base.digest(p) for p in Path(__file__).parent.glob('*.py')})

def config_for(smoke,arm):
    c=gate.CONFIG_FOR(False,'off' if arm=='R0' else 'on')
    c.update(protocol='sirm_discover_dynamic_terra_L38_seed0',q_freeze_stage3=False,validate_q_refresh=True,log_sirm_gradient_norm=True,diagnostic_steps=[1999,2000,3500,5000])
    return c

def train(a):
    plan=read(Path(a.plan));assert plan['hashes']==hashes(),'code differs from locked plan'
    a.dataset='TerraIncognita';a.seed=0
    adapter=Adapter(a.dataset);gate.install(a,adapter)
    v2.config_for=lambda smoke,arm:config_for(False,a.arm)
    v2.train(argparse.Namespace(output=a.output,smoke=False,arm='off' if a.arm=='R0' else 'on',config=None,resume=a.resume,target=1,seed=0,data_dir=a.data_dir))

def compare_targets(root,jobs,data_dir):
    outputs={}
    for j in jobs:
        model,adapter,effective,split=gate.restore(Path(j['bank']),'TerraIncognita',j['arm'],0,data_dir);c=effective['config']
        data=ImageFolder(adapter.folder(data_dir,1),base.ViewTransform(c,'original',False,1,'balanced_views'));parts=[]
        with torch.no_grad():
            for x,y in DataLoader(data,batch_size=c['eval_batch_size'],num_workers=2):parts.append(model(x.cuda())['logits'].cpu().numpy())
        outputs[j['arm']]=np.concatenate(parts);np.save(root/f'target_logits_{j["arm"]}.npy',outputs[j['arm']]);del model;torch.cuda.empty_cache()
    delta=outputs['R1'].astype(np.float64)-outputs['R0'].astype(np.float64)
    return dict(max_abs_logit_difference=float(np.abs(delta).max()),RMS_logit_difference=float(np.sqrt(np.mean(delta*delta))),different_logit_elements=int(np.count_nonzero(delta)),different_predictions=int(np.count_nonzero(outputs['R0'].argmax(1)!=outputs['R1'].argmax(1))))

def finish(root,jobs,data_dir):
    banks={j['arm']:Path(j['bank']) for j in jobs};logs={a:rows(p/'diagnostics.jsonl') for a,p in banks.items()}
    manifests={a:read(p/'manifest.json') for a,p in banks.items()}
    for key in ['initialization_sha256','split_hash','schedule_sha256','parameters_total','pretrained_sha256']:assert manifests['R0'][key]==manifests['R1'][key],key
    keys=['loss','loss_cls','loss_discover','loss_router','loss_balance','utility_ema','q']
    for step in range(2001):
        for k in keys:assert logs['R0'][step][k]==logs['R1'][step][k],(step,k)
    assert all(r['loss_sirm']==0 for r in logs['R0'])
    on=[r for r in logs['R1'] if r['stage']=='invariance'];active=[r for r in on if r['sirm_active']];nonzero=[r for r in on if r['loss_sirm']>0]
    refresh={a:rows(p/'q_refresh.jsonl') for a,p in banks.items()};stage_refresh=[r for r in refresh['R1'] if r['stage']=='invariance']
    assert len(stage_refresh)==61 and len(on)==3001
    evolution=[]
    for a,rs in refresh.items():
        for r in rs:
            evolution.append(dict(arm=a,step=r['step'],stage=r['stage'],selected_cells=r['selected_cells'],shared_experts=r['shared_experts'],structural_pair_expert_count=r['structural_pair_expert_count'],eligible_pair_expert_count=r['eligible_pair_expert_count'],executed_pair_expert_count=r['sirm_pair_count'],loss_sirm=r['loss_sirm'],lambda_sirm=r['effective_lambda_sirm'],weighted_sirm_grad_norm=r['weighted_sirm_grad_norm'],base_residual_grad_norm=r['base_residual_grad_norm'],sirm_changes_gradient_elements=r['sirm_changes_gradient_elements'],source_accuracy=r['source_validation_accuracy'],q=r['q'],utility_ema=r['utility_ema']))
    write_csv(root/'Q_evolution.csv',evolution)
    states=[]
    for checkpoint in ['best','last']:
        a=base.load(banks['R0']/'checkpoints'/f'{checkpoint}.pkl')['model_dict'];b=base.load(banks['R1']/'checkpoints'/f'{checkpoint}.pkl')['model_dict']
        modules={m:tensors_compare({k:v for k,v in a.items() if k.startswith(m+'.')},{k:v for k,v in b.items() if k.startswith(m+'.')}) for m in ['featurizer','global_head','residuals','router']}
        states.append(dict(checkpoint=checkpoint,modules=modules));del a,b
    base.write(root/'parameter_comparison.json',states)
    audit=dict(initialization_split_schedule_capacity_matched=True,pre_activation_trajectory_exact=True,OFF_penalty_zero=True,refresh_cadence_stage3=50,stage3_refreshes=61,checkpoint_selection_cadence_unchanged=True)
    base.write(root/'paired_audit.json',audit)
    gate.target_eval(root,'TerraIncognita',jobs,False,data_dir)
    target=compare_targets(root,jobs,data_dir)
    accuracy={a:read(root/'TerraIncognita/evaluation'/a/'metrics.json')['accuracy'] for a in banks}
    first=lambda rs,key:next((r['step'] for r in rs if r[key]>0),None)
    changed=any(stage_refresh[i]['q']!=stage_refresh[i-1]['q'] for i in range(1,len(stage_refresh)))
    gradient=[r for r in stage_refresh if r['weighted_sirm_grad_norm'] is not None]
    first_trajectory=next((i for i in range(2001,5001) if any(logs['R0'][i][k]!=logs['R1'][i][k] for k in ['loss_cls','loss_discover','loss_router','loss_balance','residual_rms','router_null_mass'])),None)
    block=dict(no_selected_cells_refreshes=sum(r['selected_cells']==0 for r in stage_refresh),selected_but_no_shared_refreshes=sum(r['selected_cells']>0 and r['structural_pair_expert_count']==0 for r in stage_refresh),shared_but_no_eligible_refreshes=sum(r['structural_pair_expert_count']>0 and r['eligible_pair_expert_count']==0 for r in stage_refresh))
    result=dict(first_shared_support_step=first(stage_refresh,'shared_experts'),first_eligible_refresh_step=first(stage_refresh,'eligible_pair_expert_count'),first_nonzero_SIRM_step=first(on,'loss_sirm'),stage3_active_steps=len(active),stage3_steps=len(on),active_fraction=len(active)/len(on),nonzero_loss_steps=len(nonzero),Q_changes_in_stage3=changed,distinct_Q_matrices=len({json.dumps(r['q']) for r in stage_refresh}),target_accuracy=accuracy,target_comparison=target,first_baseline_trajectory_difference_step=first_trajectory,parameter_divergence=any(m['different_elements']>0 for s in states for m in s['modules'].values()),max_SIRM_loss=max(r['loss_sirm'] for r in on),max_weighted_SIRM_grad_norm=max((r['weighted_sirm_grad_norm'] for r in gradient),default=None),max_base_residual_grad_norm=max((r['base_residual_grad_norm'] for r in gradient),default=None),blockers=block,hyperparameters_tuned=False,target_used_for_selection=False)
    base.write(root/'behavior.json',result)
    text=['# Terra L38 dynamic-Q behavioral test','', 'Seed 0, R0 OFF / R1 ON, 5001 steps each. Architecture, discovery objective, all thresholds/temperatures, ESS, utility definition and SIRM formula unchanged. Q refreshes every 50 steps during discovery and Stage 3 under no_grad. Global remains ERM/H2 and frozen after Stage 1. No hyperparameter tuning.','', '## Exact code changes','', '- GateSIRM: Stage-3 freezing is an explicit historical option; default dynamic updates no longer restore q_locked. q_learned follows the current detached Q.','- component_eligibility: extracted the original count/routing-mass/ESS checks without numerical changes, so OFF can report eligible support too.','- SIRMDiscover.update: refresh/support/ESS logging; optional weighted penalty and baseline residual gradient norms via autograd.grad, without assigning .grad or changing the objective.','- Training runner: deterministic source validation at refreshes, q_refresh.jsonl; these extra evaluations do not create selection checkpoints. Existing 250-step selection cadence remains.','- New two-arm Terra runner locks source code/config, audits controls, evaluates source-selected targets and reports parameter/logit differences.','', '## Behavior','', '| Measurement | Value |','|---|---:|']
    for k in ['first_shared_support_step','first_eligible_refresh_step','first_nonzero_SIRM_step','stage3_active_steps','stage3_steps','active_fraction','nonzero_loss_steps','Q_changes_in_stage3','distinct_Q_matrices','parameter_divergence','first_baseline_trajectory_difference_step','max_SIRM_loss','max_weighted_SIRM_grad_norm']:text.append(f'| {k} | {result[k]} |')
    text.extend(['',f'R0/R1 target accuracy: **{100*accuracy["R0"]:.6f}% / {100*accuracy["R1"]:.6f}%**. Exploratory behavior check only, not evidence of an OOD improvement.', '',f'Target logits: {json.dumps(target)}','',f'Refresh blockers: {json.dumps(block)}','', '## Q evolution','', '| Step | Selected cells | Shared experts | Eligible pairs | Executed pairs | SIRM loss | Source acc |','|---|---:|---:|---:|---:|---:|---:|'])
    for r in stage_refresh:text.append(f'| {r["step"]} | {r["selected_cells"]} | {r["shared_experts"]} | {r["eligible_pair_expert_count"]} | {r["sirm_pair_count"]} | {r["loss_sirm"]:.8g} | {100*r["source_validation_accuracy"]:.4f}% |')
    text.extend(['','Full matrices, EMA, ESS and refresh gradient measurements: banks/R0 and R1/q_refresh.jsonl and Q_evolution.csv. Parameter content comparison is in parameter_comparison.json; metadata/buffer differences alone do not count as learned parameter divergence. Stage-3 step counts include the zero-ramp activation step.', '', 'Very small pair weights may produce genuine nonzero penalty gradients but little or no float32 update difference. If weights/logits remain identical, report that numerical attenuation explicitly rather than asserting divergence from a nonzero loss. Eligible-but-OFF terms are diagnostic counts only; R0 never executes the penalty.'])
    (root/'summary.md').write_text('\n'.join(text)+'\n');print(json.dumps(result),flush=True)

def suite(a):
    torch.set_num_threads(4);root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    jobs=[dict(dataset='TerraIncognita',arm=arm,seed=0,target=1,bank=str(root/'TerraIncognita/banks'/arm)) for arm in ['R0','R1']]
    plan=dict(jobs=jobs,data_dir=a.data_dir,configs={arm:config_for(False,arm) for arm in ['R0','R1']},hashes=hashes(),target_used_for_selection=False)
    if (root/'plan.json').exists():assert read(root/'plan.json')==plan,'locked plan changed'
    else:
        base.write(root/'plan.json',plan);snap=root/'source_snapshot';snap.mkdir()
        for p in plan['hashes']:(snap/p.replace('/','__')).write_bytes(Path(p).read_bytes())
    (root/'summary.md').write_text('# Terra L38 dynamic-Q behavioral test\n\nIn progress: two 5001-step runs, R0 OFF and R1 ON. Q refresh interval 50; original numerical settings unchanged. Results are pending.\n')
    start=time.time()
    for j in jobs:
        bank=Path(j['bank'])
        if (bank/'summary.json').exists():continue
        base.write(root/'progress.json',dict(stage='training',current=j,pid=os.getpid(),elapsed=time.time()-start))
        cmd=[sys.executable,'-u','-m','experiments.sirm_discover_dynamic.run','train','--arm',j['arm'],'--output',str(bank),'--plan',str(root/'plan.json'),'--data-dir',a.data_dir]
        if (bank/'resume.pt').exists():cmd.append('--resume')
        with (root/(j['arm']+'.log')).open('a') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
        if code:
            base.write(root/'progress.json',dict(stage='failed',current=j,returncode=code));raise RuntimeError((root/(j['arm']+'.log')).read_text()[-4000:])
        print('completed '+j['arm'],flush=True)
    base.write(root/'progress.json',dict(stage='auditing_and_evaluating',pid=os.getpid()))
    finish(root,jobs,a.data_dir);base.write(root/'progress.json',dict(stage='completed',elapsed=time.time()-start,runs=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['suite','train']);p.add_argument('--output',required=True);p.add_argument('--arm',choices=['R0','R1']);p.add_argument('--data-dir',default='domainbed/data');p.add_argument('--plan');p.add_argument('--resume',action='store_true');a=p.parse_args();train(a) if a.command=='train' else suite(a)

if __name__=='__main__':main()

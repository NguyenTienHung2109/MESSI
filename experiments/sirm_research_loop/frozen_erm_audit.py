"""B0 endpoint replay and controls for frozen-ERM400 experiments."""
import shutil,json
from pathlib import Path
import torch
from experiments.terra_10pct_factor_diagnosis.run import write,digest,ROOT
from experiments.terra_10pct_factor_diagnosis.report import data_for,plain_scores,rows

def evaluate_B0(experiment,anchor,manifest):
    bank=experiment/'runs/L38/ERM_H2';dest=bank/'posthoc';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        assert json.loads((dest/'complete.json').read_text())['checkpoint_sha256']==digest(anchor);return
    sha=digest(anchor);write(dest/'selection_lock.json',dict(checkpoint_sha256=sha,target_used_for_selection=False,rule='predeclared completed full-budget endpoint; no early stopping or target ranking'))
    payload=torch.load(anchor,map_location='cpu',weights_only=False);c=payload['config'];step=payload['step'];assert step==500
    from experiments.terra_10pct_factor_diagnosis.erm import ERMH2Diagnostic
    model=ERMH2Diagnostic((3,224,224),10,3,dict(sirm_discover_config=c,sirm_output_dir=str(bank),sirm_source_envs=[0,2,3]))
    model.load_state_dict(payload['model_dict']);model.requires_grad_(False);model=model.cuda().eval()
    record=dict(target='L38',target_env=1,method='ERM_H2',selected_step=500,checkpoint_sha256=sha,selection='fixed full-budget endpoint',splits={})
    for split,envs,key in [('source_out',[0,2,3],'out'),('target_in',[1],'in'),('target_out',[1],'out')]:record['splits'][split]=plain_scores(model,data_for(manifest,envs,c,key),c)
    assert sha==digest(anchor)
    shutil.copyfile(anchor,bank/'source_best.pkl')
    write(dest/'complete.json',record);write(bank/'anchor_provenance.json',dict(original_checkpoint=str(anchor),sha256=sha,original_logged_step=500,original_update_count=501,Stage2_updates=0,all_parameters_frozen=True))
    del model;torch.cuda.empty_cache()

def audit_anchor(bank,anchor):
    a=torch.load(anchor,map_location='cpu',weights_only=False)['model_dict'];result={}
    for filename in ['source_best.pkl','last.pkl']:
        b=torch.load(bank/filename,map_location='cpu',weights_only=False)['model_dict'];modules={}
        for name in ['featurizer','global_head']:
            source={k:v for k,v in a.items() if k.startswith(name+'.')}
            modules[name]=all(torch.equal(v,b['model.'+k]) for k,v in source.items());assert modules[name]
        result[filename]=modules
    checks=rows(bank/'frozen_anchor_checks.jsonl');assert [x['step'] for x in checks]==list(range(400))
    assert all(x['requires_grad_false'] and x['gradient_absent'] and x['parameter_versions_unchanged'] for x in checks)
    write(bank/'strong_anchor_audit.json',dict(passed=True,anchor_sha256=digest(anchor),checkpoint_tensor_equality=result,all_400_updates_frozen=True))

def audit_pair(trio):
    d={m:rows(trio/m/'diagnostics.jsonl') for m in ['OFF','ON']}
    assert json.loads((trio/'OFF/initialization.json').read_text())==json.loads((trio/'ON/initialization.json').read_text())
    assert json.loads((trio/'OFF/source_schedules.json').read_text())==json.loads((trio/'ON/source_schedules.json').read_text())
    for m,logs in d.items():
        assert [x['step'] for x in logs]==list(range(400))
        for x in logs:
            assert x['stage']==('discover' if x['step']<200 else 'invariance')
            assert x['q_refreshed']==(x['step']%5==0)
            lam=0. if m=='OFF' or x['step']<200 else min(1.,(x['step']-200)/50)
            assert abs(x['effective_lambda_sirm']-lam)<1e-12
            if m=='OFF' or x['step']<200:assert not x['sirm_active'] and x['loss_sirm']==0
        for i,x in enumerate(logs):
            if i and not x['q_refreshed']:assert x['q']==logs[i-1]['q']
        vals=rows(trio/m/'results.jsonl');assert all(not any(k.startswith('env1_') for k in x) for x in vals)
        best=max((x for x in vals if (x['step']%25==0 or x['step']==399) and x['step']>=200 and 'source_val_mean' in x),key=lambda x:(x['source_val_mean'],-x['step']))
        s=json.loads((trio/m/'selection.json').read_text());assert not s['target_used'] and s['selected_step']==best['step'] and s['source_val_mean']==best['source_val_mean']
    for a,b in zip(d['OFF'][:200],d['ON'][:200]):
        assert all(a[k]==b[k] for k in ['loss_cls','loss_discover','loss_router','loss_balance','utility_ema','q'])
    write(trio/'control_audit.json',dict(passed=True,identical_strong_initialization=True,identical_source_schedules=True,updates400=True,discovery200_invariance200=True,preactivation_exact=True,source_only_earliest_best=True))

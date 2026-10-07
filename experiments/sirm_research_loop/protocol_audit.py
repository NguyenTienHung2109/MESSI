"""Read-only full-trajectory checks, independent of training's audit flags."""
import argparse,json,math
from pathlib import Path
from .shrinkage import digest,write

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def finite(value):
    if isinstance(value,(int,float)):return math.isfinite(value)
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,list):return all(finite(v) for v in value)
    return True

def audit(bank,target=1,complete=True,schedule='short501'):
    d=rows(bank/'diagnostics.jsonl');v=rows(bank/'results.jsonl');q=rows(bank/'q_refresh.jsonl')
    assert d and v
    hp=v[0]['hparams'];c=hp['sirm_discover_config'];lam=c['lambda_sirm']
    expected={'short501':(100,100,301),'frozen400':(0,200,200)}[schedule]
    global_steps,discover_steps,invariance_steps=expected
    activation=global_steps+discover_steps
    updates=sum(expected)
    checks={
        'contiguous_steps':[x['step'] for x in d]==list(range(len(d))),
        'finite_loss_and_mechanism_arrays':all(finite(x) for x in d+q),
        'fixed_requested_schedule':(c['global_steps'],c['discover_steps'],c['invariance_steps'])==expected,
        'dynamic_Q':c['q_freeze_stage3'] is False,
        'binary_SIRM':c['sirm_weighting']=='binary',
        'target_not_in_source_training':hp['sirm_source_envs']==[e for e in range(4) if e!=target],
        'target_scores_absent_from_training':all(not any(k.startswith(f'env{target}_') for k in x) for x in v),
        'stages_correct':all(x['stage']==('global' if x['step']<global_steps else 'discover' if x['step']<activation else 'invariance') for x in d),
        'ramp_correct':c['sirm_ramp_steps']==50 and all(abs(x['effective_lambda_sirm']-(0. if x['step']<activation else lam*min(1.,(x['step']-activation)/50)))<1e-12 for x in d),
        'refresh_cadence':c['utility_interval']==5 and all(bool(x['q_refreshed'])==(x['step']>=global_steps and (x['step']-global_steps)%5==0) for x in d),
        'Q_held_between_refreshes':all(x['q']==d[i-1]['q'] for i,x in enumerate(d) if i and not x['q_refreshed']),
        'no_frozen_assignment':all(not x.get('assignment_locked',False) for x in d),
        'zero_penalty_before_activation':all(not x['sirm_active'] and x['loss_sirm']==0 for x in d if x['step']<activation),
        'OFF_penalty_zero':lam!=0 or all(not x['sirm_active'] and x['loss_sirm']==0 for x in d),
        'active_penalty_positive':all(x['sirm_pair_count']>0 and x['loss_sirm']>0 for x in d if x['sirm_active']),
    }
    selection=None
    if complete:
        checks['all_requested_updates']=len(d)==updates and (bank/'done').exists()
        s=json.loads((bank/'selection.json').read_text())
        candidates=[x for x in v if activation<=x['step']<updates and (x['step']%25==0 or x['step']==updates-1) and 'source_val_mean' in x]
        # Duplicate refresh rows cannot add a selection opportunity.
        best=max(candidates,key=lambda x:(x['source_val_mean'],-x['step']))
        checks['selection_source_only']=s['target_used'] is False
        checks['selected_earliest_source_best']=s['selected_step']==best['step'] and abs(s['source_val_mean']-best['source_val_mean'])<1e-12
        report=json.loads((bank/'posthoc/complete.json').read_text())
        lock=json.loads((bank/'posthoc/selection_lock.json').read_text())
        checks['checkpoint_lock_matches']=report['checkpoint_sha256']==lock['checkpoint_sha256']==digest(bank/'source_best.pkl')
        selection=dict(step=s['selected_step'],active_steps_before_selected=sum(x['sirm_active'] for x in d if x['step']<=s['selected_step']),first_active=next((x['step'] for x in d if x['sirm_active']),None))
    return dict(passed=all(checks.values()),requested_schedule=schedule,checks=checks,failed_checks=[k for k,v in checks.items() if not v],through_step=d[-1]['step'],selection=selection,scope='full saved trajectory/config/source-selection rows; no training, inference, fitting or selection changes')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);p.add_argument('--target',type=int,default=1);p.add_argument('--live',action='store_true');p.add_argument('--schedule',choices=['short501','frozen400'],default='short501');a=p.parse_args()
    result=audit(a.bank,a.target,not a.live,a.schedule);a.dest.parent.mkdir(parents=True,exist_ok=True);write(a.dest,result)
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)

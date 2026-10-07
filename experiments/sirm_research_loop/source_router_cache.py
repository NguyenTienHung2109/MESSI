"""Source-only label-teacher and matched-head routing diagnostics on caches."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import residual_utilities,router_teacher

def audit(dest,baseline=None):
    a=np.load(dest/'eda/source_out_replay.npz');g=torch.tensor(a['OFF_g']);delta=torch.tensor(a['OFF_delta']);pi=torch.tensor(a['OFF_pi']);y=torch.tensor(a['y']);env=a['domains']
    full,_,marginal,single=residual_utilities(g,delta,pi,y);teacher=router_teacher(single,.1,.002);oracle=g+(teacher[:,:6,None]*delta).sum(1);per_domain={}
    for d in sorted(set(env.tolist())):
        mask=torch.tensor(env==d);row={}
        for name,logits in [('global',g),('actual',full),('label_teacher_diagnostic',oracle)]:row[name]=dict(accuracy=float(logits[mask].argmax(1).eq(y[mask]).double().mean()),CE=float(F.cross_entropy(logits[mask],y[mask])))
        row.update(router_NULL=float(pi[mask,-1].mean()),teacher_NULL=float(teacher[mask,-1].mean()),router_negative_utility_mass=float((pi[mask,:6]*(marginal[mask]<0)).sum(1).mean()),teacher_negative_utility_mass=float((teacher[mask,:6]*(marginal[mask]<0)).sum(1).mean()),teacher_KL=float(F.kl_div(pi[mask].log(),teacher[mask],reduction='batchmean')));per_domain[str(d)]=row
    result=dict(scope='source-out only; label teacher is diagnostic, not deployable or target oracle; source-selected checkpoints; no fitting/updates',per_domain=per_domain,equal_domain_mean={name:{k:float(np.mean([row[name][k] for row in per_domain.values()])) for k in ['accuracy','CE']} for name in ['global','actual','label_teacher_diagnostic']})
    if baseline is not None:
        old=np.load(baseline/'eda/source_out_replay.npz');assert np.array_equal(a['ids'],old['ids']) and np.array_equal(a['y'],old['y']) and np.array_equal(a['OFF_g'],old['OFF_g'])
        olddelta=torch.tensor(old['OFF_delta']);oldpi=torch.tensor(old['OFF_pi']);_,_,oldU,oldsingle=residual_utilities(g,olddelta,oldpi,y);fixed=router_teacher(oldsingle,.1,.002);new_router_old_heads=g+(pi[:,:6,None]*olddelta).sum(1);old_router_new_heads=g+(oldpi[:,:6,None]*delta).sum(1)
        matched={}
        for d in sorted(set(env.tolist())):
            mask=torch.tensor(env==d);row={}
            for name,logits in [('new_router_old_heads',new_router_old_heads),('old_router_new_heads',old_router_new_heads)]:row[name]=dict(accuracy=float(logits[mask].argmax(1).eq(y[mask]).double().mean()),CE=float(F.cross_entropy(logits[mask],y[mask])))
            row.update(baseline_router_fixed_teacher_KL=float(F.kl_div(oldpi[mask].log(),fixed[mask],reduction='batchmean')),new_router_fixed_teacher_KL=float(F.kl_div(pi[mask].log(),fixed[mask],reduction='batchmean')),baseline_negative_utility_mass=float((oldpi[mask,:6]*(oldU[mask]<0)).sum(1).mean()),new_router_baseline_negative_utility_mass=float((pi[mask,:6]*(oldU[mask]<0)).sum(1).mean()));matched[str(d)]=row
        result['fixed_baseline_teacher_and_head_controls']=matched
    (dest/'source_router_alignment.json').write_text(json.dumps(result,indent=2)+'\n');return result

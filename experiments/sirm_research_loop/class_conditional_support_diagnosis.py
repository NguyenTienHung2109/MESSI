"""Counterfactual per-label source support/reliability; no actual Q or model update."""
import json,importlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .shrinkage import ROOT,digest,write
qfun=importlib.import_module('experiments.sirm_discover.math').soft_null_q


def groups(a,name,subset):
    y=torch.tensor(a['y']);d=torch.tensor(a['domains']);f=torch.tensor(a[name+'_full']);delta=torch.tensor(a[name+'_delta']);pi=torch.tensor(a[name+'_pi']);ce=F.cross_entropy(f,y,reduction='none')
    u=torch.stack([F.cross_entropy(f-pi[:,j,None]*delta[:,j],y,reduction='none')-ce for j in range(6)],1)
    result={}
    for c in range(10):
        counts=[];means=[]
        for domain in [0,2,3]:
            mask=y.eq(c)&d.eq(domain)&torch.tensor(subset);counts.append(int(mask.sum()));means.append(u[mask].mean(0) if mask.any() else torch.zeros(6))
        v=torch.stack(means);valid=torch.tensor([[n>=2]*6 for n in counts]);q,null,conf,interaction=qfun(v,valid,temperature=.05,utility_floor=.002,interaction_floor=.002)
        result[str(c)]=dict(counts=counts,complete_minimum_two=bool(valid.all()),U=v.tolist(),V=interaction.tolist(),Q=q.tolist(),selected_cells=int((q>0).sum()),shared_experts=(q.gt(0).sum(0)>=2).nonzero().flatten().tolist())
    return result


def main():
    torch.set_num_threads(4);root=ROOT/'results/sirm_research_loop/iteration_12';out=root/'class_conditional_support_diagnosis';out.mkdir(exist_ok=True)
    plan=json.loads((root/'plan.json').read_text());c=plan['hparams']['OFF']['sirm_discover_config'];assert c['utility_floor']==.002 and c['interaction_floor']==.002 and c['q_temperature']==.05
    caches={s:dict(np.load(root/'source_comparison'/f'{s}_replay.npz')) for s in ['source_in','source_out']};results={}
    for name in ['11_selected','12_selected','12_last']:
        results[name]={}
        for split,a in caches.items():
            mask=np.ones(len(a['y']),dtype=bool);results[name][split]=groups(a,name,mask)
        a=caches['source_out'];halves=[np.zeros(len(a['y']),dtype=bool) for _ in range(2)]
        for domain in [0,2,3]:
            for label in range(10):
                ix=np.flatnonzero((a['y']==label)&(a['domains']==domain));ix=sorted(ix,key=lambda i:str(a['ids'][i]));
                for j,i in enumerate(ix):halves[j%2][i]=True
        assert np.all(halves[0]^halves[1]);results[name]['source_out_halves']=[groups(a,name,h) for h in halves]
        reliability=[]
        for label in range(10):
            k=str(label);tr=results[name]['source_in'][k];val=results[name]['source_out'][k];hs=[v[k] for v in results[name]['source_out_halves']]
            sharedsets=[set(v['shared_experts']) for v in [tr,val]+hs];stable=sorted(set.intersection(*sharedsets))
            reliability.append(dict(label=label,source_in_shared=tr['shared_experts'],source_out_shared=val['shared_experts'],half_shared=[v['shared_experts'] for v in hs],
                half_complete_minimum_two=[v['complete_minimum_two'] for v in hs],stable_across_train_out_and_both_halves=stable))
        results[name]['reliability']=reliability
    result=dict(training=False,fitting=False,target_read=False,actual_Q_changed=False,selection_changes=False,relabeling=False,
        thresholds={'U':.002,'V':.002,'temperature':.05,'per_domain_class_minimum':2},source_replay_hashes={s:digest(root/'source_comparison'/f'{s}_replay.npz') for s in caches},code_sha256=digest(Path(__file__)),results=results,
        scope='Class-conditioned full-matrix Q is a saved-state source-only diagnostic, not current EMA Q or a trained policy. Classes with any source group below n2 are invalidated as a complete matrix. Stratified alternating-ID source-out halves are reliability proxies, not independent fitting/selection data. No missing class imputation or threshold relaxation. A one-class risk changes IRMv1 semantics and would need a separately justified objective before training.')
    write(out/'results.json',result)
    print(json.dumps({k:v['reliability'] for k,v in results.items()},indent=2))

if __name__=='__main__':main()

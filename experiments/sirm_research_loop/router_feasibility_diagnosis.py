"""Source-only convex-routing capacity at a fixed saved checkpoint; label oracle only."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog
from domainbed.deit_transformer import DeiTFeaturizer
from .centered_model import CenteredGateSIRM
from .hard_router_diagnosis import outputs
from .shrinkage import ROOT, digest, write


def maximum_margin(g, delta, y):
    rivals=np.array([i for i in range(len(g)) if i!=y])
    slopes=delta[:,y,None]-delta[:,rivals]
    gaps=g[y]-g[rivals]
    # maximize t subject to every true-class margin >= t; NULL uses remaining mass.
    a=np.column_stack([-slopes.T,np.ones(len(rivals))])
    a=np.concatenate([a,np.array([[1]*len(delta)+[0]])])
    b=np.r_[gaps,1.]
    r=linprog(np.r_[np.zeros(len(delta)),-1.],A_ub=a,b_ub=b,
              bounds=[(0,None)]*len(delta)+[(None,None)],method='highs')
    assert r.success,r.message
    w=r.x[:-1];logit=g+w@delta
    margin=float(np.min(logit[y]-logit[rivals]))
    assert abs(margin-r.x[-1])<1e-6
    assert w.min()>=-1e-7 and w.sum()<=1+1e-7
    return margin,w


def main():
    torch.set_num_threads(4)
    dest=ROOT/'results/sirm_research_loop/iteration_11'
    out=dest/'router_feasibility_diagnosis';out.mkdir(exist_ok=True)
    path=dest/'experiment/runs/L38/OFF/source_best.pkl'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    assert payload['selected_step']==399
    c=payload['config'];model=CenteredGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()})
    model.configure_stage();model.requires_grad_(False);model.eval()
    cache=ROOT/'results/sirm_research_loop/iteration_09/source_router_diagnosis'
    plan=dict(code_sha256=digest(Path(__file__)),checkpoint_sha256=digest(path),target_read=False,
              fitting=False,training=False,selection=False,margin_tolerance=1e-5,
              scope='Exact convex label-oracle routing feasibility, simplex mass <=1 including NULL. '
                    'Not a label-free router or evidence of learnability; fixed saved source-only bank.')
    records=[]; summaries={}
    identities={}
    for split in ['source_in','source_out']:
        p=cache/f'{split}_original.npz'
        with np.load(p) as z:
            features,y,domains=[torch.from_numpy(z[k]) for k in ['z','y','domains']];ids=z['ids'].tolist()
        assert set(domains.tolist())=={0,2,3}
        identities[split]=set(ids)
        if split=='source_in':
            frozen=torch.load(dest/'source_feature_cache.pt',map_location='cpu',weights_only=False)
            assert frozen['ids']==ids and torch.equal(features,frozen['features'])
        with torch.no_grad():f=outputs(model,features,y)
        g,delta,pi,full=[f[k].numpy().astype('float64') for k in ['g','delta','pi','full']]
        rows=[]
        for i in range(len(y)):
            label=int(y[i]); global_correct=g[i].argmax()==label
            single_correct=(g[i]+delta[i]).argmax(1)==label
            routed_correct=(g[i]+pi[i,:6,None]*delta[i]).argmax(1)==label
            margin,w=maximum_margin(g[i],delta[i],label)
            feasible=margin>plan['margin_tolerance']
            thresholds=[]
            if not global_correct:
                rivals=np.array([a for a in range(10) if a!=label])
                gaps=g[i,label]-g[i,rivals]
                for expert in np.where(single_correct)[0]:
                    slopes=delta[i,expert,label]-delta[i,expert,rivals]
                    lower=max(0.,float(np.max(-gaps[slopes>0]/slopes[slopes>0])))
                    assert lower<=1+1e-6
                    thresholds.append(dict(expert=int(expert),minimum_single_scale=lower,
                        actual_scale=float(pi[i,expert]),
                        actual_over_minimum=float(pi[i,expert]/lower) if lower>0 else None))
            row=dict(split=split,id=ids[i],domain=int(domains[i]),label=label,
                     global_correct=bool(global_correct),full_correct=bool(full[i].argmax()==label),
                     single_repair_available=bool(single_correct.any()),
                     routed_single_repair_available=bool(routed_correct.any()),
                     maximum_convex_margin=margin,convex_repair_available=bool(feasible),
                     repairing_expert_scale_thresholds=thresholds,
                     oracle_weights=w.tolist(),actual_weights=pi[i].tolist())
            rows.append(row);records.append(row)
        def counts(rows):
            wrong=[r for r in rows if not r['global_correct']]
            return dict(n=len(rows),global_wrong=len(wrong),
                unscaled_single_capacity=sum(r['single_repair_available'] for r in wrong),
                routed_single_capacity=sum(r['routed_single_repair_available'] for r in wrong),
                convex_capacity=sum(r['convex_repair_available'] for r in wrong),
                mixture_only_capacity=sum(r['convex_repair_available'] and not r['single_repair_available'] for r in wrong),
                unrepairable_in_simplex=sum(not r['convex_repair_available'] for r in wrong),
                actual_repairs=sum(r['full_correct'] for r in wrong),
                actual_harms=sum(r['global_correct'] and not r['full_correct'] for r in rows))
        summaries[split]={'all':counts(rows),**{str(d):counts([r for r in rows if r['domain']==d]) for d in [0,2,3]}}
        plan.setdefault('cache_sha256',{})[split]=digest(p)
    assert not identities['source_in'] & identities['source_out']
    assert digest(path)==plan['checkpoint_sha256']
    write(out/'results.json',dict(plan=plan,summaries=summaries,records=records))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()

"""Case comparison of fitted input-only router and label-aware ceiling; source only."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.sirm_discover.math import router_teacher
from .source_router_teacher_fit import OUT, DEST, ROOT
from .source_specialist_routing_audit import replay
from .routed_conditional_utility import load
from .source_probe_penalty_direction import outputs
from .shrinkage import digest, write

def run():
    torch.set_num_threads(4);assert not (OUT/'case_analysis.json').exists()
    audit=json.loads((DEST/'final_analysis_audit.json').read_text())
    for rel,h in audit['sha256'].items():assert digest(DEST/rel)==h,rel
    data={};cache=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis'
    for split in ['source_in','source_out']:
        a=dict(np.load(cache/f'{split}_tokens.npz'));data[split]=(torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains']),a['ids'])
    manifest=json.loads((DEST/'experiment/subset_manifest.json').read_text());names=manifest['class_names'];states={}
    for state in ['selected','last']:
        m,p,path=load(DEST,state);before={k:v.clone() for k,v in m.state_dict().items()};m.calibrate_features(data['source_in'][0],data['source_in'][2]);splits={}
        for split,(x,y,dom,ids) in data.items():
            fitted=np.load(OUT/f'{state}_{split}_replay.npz');assert np.array_equal(fitted['ids'],ids)
            with torch.no_grad():
                r=replay(m,x,y);teacher=router_teacher(r['raw_gain'],m.config['router_temperature'],m.config['utility_floor']);g,delta,oracle=outputs(m,x,teacher)
                fit=torch.tensor(fitted['fitted_logits']);pi=torch.tensor(fitted['fitted_router']);assert torch.equal(g,torch.tensor(fitted['global_logits']))
                gc=g.argmax(1).eq(y);oc=oracle.argmax(1).eq(y);fc=fit.argmax(1).eq(y);l1=(pi-teacher).abs().sum(1)
                cohorts={}
                for name,mask in [('global_correct',gc),('global_wrong',~gc),('oracle_repairs',~gc&oc)]:
                    cohorts[name]=dict(n=int(mask.sum()),teacher_fit_L1_mean=float(l1[mask].mean()),
                        fitted_correct=int(fc[mask].sum()),teacher_NULL_mean=float(teacher[mask,-1].mean()),fitted_NULL_mean=float(pi[mask,-1].mean()))
                cases=[]
                mask=(~gc&oc)|(gc&~fc)
                for i in torch.nonzero(mask).flatten().tolist():
                    cases.append(dict(image_id=str(ids[i]),label=names[int(y[i])],global_prediction=names[int(g[i].argmax())],
                        native_prediction=names[int(r['full'][i].argmax())],fitted_prediction=names[int(fit[i].argmax())],oracle_prediction=names[int(oracle[i].argmax())],
                        oracle_repairs=bool(~gc[i]&oc[i]),fitted_harms=bool(gc[i]&~fc[i]),
                        teacher=teacher[i].tolist(),fitted_router=pi[i].tolist(),single_expert_CE_gain=r['raw_gain'][i].tolist(),
                        teacher_fit_L1=float(l1[i])))
            splits[split]=dict(cohorts=cohorts,cases=cases)
        m.load_state_dict(before);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());assert all(p.grad is None for p in m.parameters());states[state]=splits
    write(OUT/'case_analysis.json',dict(states=states,source_only=True,target_read=False,fitting=False,
        code_sha256=digest(Path(__file__)),limits='Oracle uses labels, fitted inference uses features only. Cohorts defined by observed source outcomes; exploratory descriptive, not new fit/selection data.'))
    (OUT/'case_analysis_code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for path in OUT.iterdir():audit['sha256'][str(path.relative_to(DEST))]=digest(path)
    write(DEST/'final_analysis_audit.json',audit)
    print(json.dumps({state:{split:r['cohorts'] for split,r in v.items()} for state,v in states.items()},indent=2))

if __name__=='__main__':run()

"""Source-only label-aware teacher ceiling, explicitly not an inference method."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.sirm_discover.math import router_teacher
from .source_specialist_routing_audit import DEST, OUT, ROOT, replay
from .routed_conditional_utility import load
from .source_probe_penalty_direction import outputs
from .shrinkage import digest, write

def score(g, f, y, domains):
    gc, fc = g.argmax(1).eq(y), f.argmax(1).eq(y)
    return dict(accuracy_equal_domain=float(torch.stack([fc[domains.eq(d)].double().mean() for d in domains.unique()]).mean()),
        repairs=int((~gc & fc).sum()), harms=int((gc & ~fc).sum()),
        correct_images=int(fc.sum()), total_images=len(y),
        CE_equal_domain=float(torch.stack([torch.nn.functional.cross_entropy(f[domains.eq(d)],y[domains.eq(d)]) for d in domains.unique()]).mean()))

def run():
    torch.set_num_threads(4)
    assert not (OUT/'teacher_ceiling.json').exists()
    audit=json.loads((DEST/'final_analysis_audit.json').read_text())
    for rel,h in audit['sha256'].items():assert digest(DEST/rel)==h,rel
    data={};cache=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis'
    for split in ['source_in','source_out']:
        a=dict(np.load(cache/f'{split}_tokens.npz'))
        data[split]=(torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains']))
    states={};locks={}
    for state in ['selected','last']:
        m,p,path=load(DEST,state);before={k:v.clone() for k,v in m.state_dict().items()};locks[str(path)]=digest(path)
        m.calibrate_features(data['source_in'][0],data['source_in'][2]);result={}
        with torch.no_grad():
            for split,(x,y,dom) in data.items():
                r=replay(m,x,y);pi=r['pi'];teacher=router_teacher(r['raw_gain'],m.config['router_temperature'],m.config['utility_floor'])
                g,d,f=outputs(m,x,pi);_,_,tf=outputs(m,x,teacher)
                policies={'global':g,'native':f,'label_aware_teacher':tf}
                for e in range(6): policies[f'remove_E{e}']=f-pi[:,e,None]*d[:,e]
                result[split]={key:score(g,value,y,dom) for key,value in policies.items()}
        m.load_state_dict(before);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());assert all(q.grad is None for q in m.parameters())
        states[state]=result
    for path,h in locks.items():assert digest(Path(path))==h
    write(OUT/'teacher_ceiling.json',dict(states=states,source_only=True,target_read=False,training=False,
        input_sha256=locks,code_sha256=digest(Path(__file__)),
        limitations='Teacher sees true source labels and is not deployable. Its risk-based oracle ceiling is not independent validation or proof of a learnable inference router. All six removals reported without selecting a production policy. Original eval centering used throughout.'))
    (OUT/'teacher_ceiling_code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for path in OUT.iterdir():audit['sha256'][str(path.relative_to(DEST))]=digest(path)
    write(DEST/'final_analysis_audit.json',audit)
    print(json.dumps({state:vals['source_out'] for state,vals in states.items()},indent=2))

if __name__=='__main__':run()

"""Compare native label-conditioned teacher with deployed router on source only."""
import json
from pathlib import Path
import torch
from experiments.sirm_discover.math import router_teacher
from .source_specialist_routing_audit import DEST, OUT, ROOT, replay
from .routed_conditional_utility import load
from .shrinkage import digest, write

def summarize(r, y, names, temperature, floor):
    teacher = router_teacher(r['raw_gain'], temperature, floor)
    result = {}
    for c, name in enumerate(names):
        mask = y.eq(c)
        if not mask.any(): continue
        zero = teacher[mask, 3].eq(0)
        result[name] = dict(n=int(mask.sum()), teacher_mean=teacher[mask].mean(0).tolist(),
            router_mean=r['pi'][mask].mean(0).tolist(),
            E3_teacher_zero_fraction=float(zero.float().mean()),
            E3_router_mass_when_teacher_zero=float(r['pi'][mask, 3][zero].mean()) if zero.any() else None,
            teacher_router_L1=float((teacher[mask]-r['pi'][mask]).abs().sum(1).mean()),
            E3_positive_single_gain_fraction=float(r['raw_gain'][mask, 3].gt(floor).float().mean()))
    return result

def run():
    torch.set_num_threads(4)
    assert not (OUT/'teacher_comparison.json').exists()
    audit = json.loads((DEST/'final_analysis_audit.json').read_text())
    for rel, expected in audit['sha256'].items(): assert digest(DEST/rel)==expected, rel
    original = json.loads((OUT/'results.json').read_text())
    import numpy as np
    cache = ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis'
    data={}
    for split in ['source_in','source_out']:
        a=dict(np.load(cache/f'{split}_tokens.npz'))
        data[split]=(torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains']))
    names=original['class_names'];states={};locks={}
    h2dir=ROOT/'results/sirm_research_loop/iteration_15/source_conditional_H2'
    h2audit=json.loads((h2dir/'extraction_audit.json').read_text())
    for state in ['selected','last']:
        m,p,path=load(DEST,state);saved={k:v.clone() for k,v in m.state_dict().items()};locks[str(path)]=digest(path)
        m.calibrate_features(data['source_in'][0],data['source_in'][2])
        temperature=m.config['router_temperature'];floor=m.config['utility_floor'];result={}
        with torch.no_grad():
            for split,(x,y,d) in data.items():result[split]=summarize(replay(m,x,y),y,names,temperature,floor)
            parts=[];ys=[]
            for row in h2audit['rows']:
                f=h2dir/f"H2_step{row['step']}.pt";assert digest(f)==row['sha256'];locks[str(f)]=digest(f)
                b=torch.load(f,map_location='cpu',weights_only=False)
                parts.append(replay(m,b['raw'],b['y'],training=True));ys.append(b['y'])
            r={k:torch.cat([v[k] for v in parts]) for k in parts[0]}
            result['scheduled_H2_batch_centered']=summarize(r,torch.cat(ys),names,temperature,floor)
        m.load_state_dict(saved);assert all(torch.equal(v,m.state_dict()[k]) for k,v in saved.items());assert all(p.grad is None for p in m.parameters())
        states[state]=dict(teacher_temperature=temperature,utility_floor=floor,groups=result)
    for path,h in locks.items():assert digest(Path(path))==h
    write(OUT/'teacher_comparison.json',dict(states=states,source_only=True,target_read=False,training=False,
        input_sha256=locks,code_sha256=digest(Path(__file__)),
        interpretation='Teacher is label-conditioned oracle training target, not an inference policy. H2 uses actual batch centering at saved states, not historical teacher/router. Original source views use eval centering. KL mismatch does not prove a trainable router can realize teacher without source overfit.'))
    (OUT/'teacher_code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for path in OUT.iterdir():audit['sha256'][str(path.relative_to(DEST))]=digest(path)
    write(DEST/'final_analysis_audit.json',audit)
    print('Teacher audit complete; source only.')

if __name__=='__main__':run()

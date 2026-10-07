"""Error-conditioned source-only teacher capacity; no target reads or fitting."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import residual_utilities,router_teacher

def run(dest):
    torch.set_num_threads(4)
    config=json.loads((dest/'plan.json').read_text())['hparams']['OFF']['sirm_discover_config']
    with np.load(dest/'eda/source_out_replay.npz') as a:
        g=torch.tensor(a['OFF_g']);delta=torch.tensor(a['OFF_delta']);pi=torch.tensor(a['OFF_pi']);y=torch.tensor(a['y']);domains=a['domains'];ids=a['ids'].copy()
    full,_,utility,single=residual_utilities(g,delta,pi,y)
    teacher=router_teacher(single,config['router_temperature'],config['utility_floor'])
    oracle=g+(teacher[:,:delta.shape[1],None]*delta).sum(1)
    outputs={'global':g,'actual':full,'label_teacher':oracle}
    correct={name:logits.argmax(1).eq(y) for name,logits in outputs.items()}
    ce={name:F.cross_entropy(logits,y,reduction='none') for name,logits in outputs.items()}
    rows=[]
    for i in range(len(y)):
        row=dict(image_id=str(ids[i]),environment=int(domains[i]),true_class=int(y[i]),actual_NULL=float(pi[i,-1]),teacher_NULL=float(teacher[i,-1]),negative_utility_mass=float((pi[i,:6]*(utility[i]<0)).sum()),max_single_utility=float(single[i].max()))
        for name,logits in outputs.items():row.update({name+'_prediction':int(logits[i].argmax()),name+'_correct':bool(correct[name][i]),name+'_CE':float(ce[name][i])})
        rows.append(row)
    groups={}
    for domain in sorted(set(domains.tolist())):
        for cls in sorted(set(y.tolist())):
            for state in ['all','global_correct','global_wrong']:
                mask=torch.tensor(domains==domain)&y.eq(cls)
                if state!='all':mask &= correct['global'] if state=='global_correct' else ~correct['global']
                if not mask.any():continue
                summary=dict(n=int(mask.sum()),actual_NULL=float(pi[mask,-1].mean()),teacher_NULL=float(teacher[mask,-1].mean()),negative_utility_mass=float((pi[mask,:6]*(utility[mask]<0)).sum(1).mean()))
                for name in outputs:summary[name]=dict(accuracy=float(correct[name][mask].double().mean()),CE=float(ce[name][mask].mean()),repairs_global=int((~correct['global']&correct[name]&mask).sum()),harms_global=int((correct['global']&~correct[name]&mask).sum()))
                groups[f'{domain}/{cls}/{state}']=summary
    result=dict(scope='Source-out original views only; error/class groups are descriptive at source-selected checkpoints, not independent validation. Teacher uses labels and is not deployable. No target cache is read.',images=len(rows),teacher_temperature=config['router_temperature'],utility_floor=config['utility_floor'],groups=groups,
      pooled_counts={name:dict(correct=int(correct[name].sum()),repairs_global=int((~correct['global']&correct[name]).sum()),harms_global=int((correct['global']&~correct[name]).sum())) for name in outputs})
    out=dest/'source_router_case_review';out.mkdir(exist_ok=True)
    with (out/'cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('destination',type=Path);args=parser.parse_args();print(json.dumps(run(args.destination)['pooled_counts']))

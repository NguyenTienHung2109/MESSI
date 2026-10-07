"""Fixed source-only class-composition/utility generalization comparison."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.sirm_discover.math import residual_utilities
from .class_utility_diagnosis import decomposition
from .shrinkage import ROOT, digest, write


def main():
    torch.set_num_threads(4);dest=ROOT/'results/sirm_research_loop/iteration_12/source_comparison'
    proof=json.loads((dest/'results.json').read_text());assert proof['target_read'] is False
    assert all(digest(Path(v['path']))==v['checkpoint_sha256'] for v in proof['locks'].values())
    caches={}
    for split in ['source_in','source_out']:
        with np.load(dest/(split+'_replay.npz')) as a:caches[split]={k:torch.from_numpy(a[k]) for k in a.files if k!='ids'}
    result=dict(code_sha256=digest(Path(__file__)),comparison_sha256=digest(dest/'results.json'),
                target_read=False,training=False,fitting=False,selection_changes=False,
                scope='Source original caches only; class/domain means at fixed selected or399 states. '
                      'Two-per-cell is a descriptive inclusion rule, not a changed training/Q/ESS condition. '
                      'Fresh-Q averaging controls are posthoc matrices, not trained trajectories or threshold tuning.',states={})
    for state in ['selected','last']:
        result['states'][state]={}
        for iteration in ['11','12']:
            key=iteration+'_'+state;values={};utility={}
            for split,a in caches.items():
                g,delta,pi=[a[key+'_'+k] for k in ['g','delta','pi']]
                values[split]=decomposition(g,delta,pi,a['y'],a['domains'])
                _,_,utility[split],_=residual_utilities(g,delta,pi,a['y'])
            pairs=[]
            for d in [0,2,3]:
                for label in range(10):
                    masks={s:(a['domains']==d)&(a['y']==label) for s,a in caches.items()}
                    if any(int(mask.sum())<2 for mask in masks.values()):continue
                    means={s:utility[s][mask].mean(0).numpy() for s,mask in masks.items()}
                    pairs.append(dict(domain=d,label=label,counts={s:int(mask.sum()) for s,mask in masks.items()},
                                      source_in_U=means['source_in'].tolist(),source_out_U=means['source_out'].tolist()))
            train=np.array([p['source_in_U'] for p in pairs]);val=np.array([p['source_out_U'] for p in pairs])
            stats=dict(class_domain_groups=len(pairs),expert_cells=train.size,
                       train_positive=int((train>0).sum()),positive_both=int(((train>0)&(val>0)).sum()),
                       train_positive_validation_negative=int(((train>0)&(val<0)).sum()),
                       train_above_original_floor=int((train>.002).sum()),above_floor_both=int(((train>.002)&(val>.002)).sum()),
                       flattened_pearson=float(np.corrcoef(train.flatten(),val.flatten())[0,1]))
            result['states'][state][iteration]=dict(step=proof['locks'][key]['step'],decomposition=values,
                conditional_utility_generalization=stats,matched_class_domain_pairs=pairs)
    write(dest/'conditional_utility.json',result)
    print(json.dumps({s:{k:v['conditional_utility_generalization'] for k,v in r.items()} for s,r in result['states'].items()},indent=2))

if __name__=='__main__':main()

"""Fixed source-in norm matching; source replay only, no fitted/tuned scale."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .shrinkage import ROOT, digest, write


def assess(g,delta,pi,y,domains):
    full=g+(pi[:,:6,None]*delta).sum(1)
    wrong=g.argmax(1)!=y;correct=full.argmax(1)==y
    capacity=((g[:,None]+delta).argmax(2)==y[:,None]).any(1)&wrong
    masks=[domains==d for d in domains.unique()]
    return dict(equal_domain_accuracy=float(torch.stack([correct[m].double().mean() for m in masks]).mean()),
                equal_domain_CE=float(torch.stack([F.cross_entropy(full[m],y[m]) for m in masks]).mean()),
                repairs=int((wrong&correct).sum()),harms=int((~wrong&~correct).sum()),
                unscaled_repair_capacity=int(capacity.sum()),
                expert_RMS=float(delta.double().square().mean().sqrt()),
                correction_RMS=float((full-g).double().square().mean().sqrt()))


def main():
    torch.set_num_threads(4);dest=ROOT/'results/sirm_research_loop/iteration_12/source_comparison'
    path=dest/'results.json';r=json.loads(path.read_text())
    assert r['target_read'] is False
    assert all(digest(Path(v['path']))==v['checkpoint_sha256'] for v in r['locks'].values())
    caches={}
    for split in ['source_in','source_out']:
        with np.load(dest/(split+'_replay.npz')) as a:caches[split]={k:torch.from_numpy(a[k]) for k in a.files if k!='ids'}
    results={}
    for state in ['selected','last']:
        old='11_'+state;new='12_'+state;a=caches['source_in']
        expert_alpha=float(a[new+'_delta'].double().square().mean().sqrt()/a[old+'_delta'].double().square().mean().sqrt())
        def composed(key):return (a[key+'_pi'][:,:6,None]*a[key+'_delta']).sum(1).double().square().mean().sqrt()
        correction_alpha=float(composed(new)/composed(old))
        rows={}
        for split,cache in caches.items():
            rows[split]={}
            for label,key,alpha in [('old',old,1.),('new',new,1.),('old_expert_norm_matched',old,expert_alpha),('old_correction_norm_matched',old,correction_alpha)]:
                rows[split][label]=assess(cache[key+'_g'],alpha*cache[key+'_delta'],cache[key+'_pi'],cache['y'],cache['domains'])
        results[state]=dict(expert_alpha_source_in=expert_alpha,correction_alpha_source_in=correction_alpha,splits=rows)
    write(dest/'scale_control.json',dict(code_sha256=digest(Path(__file__)),comparison_sha256=digest(path),
        target_read=False,training=False,fitting=False,selection_changes=False,
        scope='Separate fixed source-in pooled expert-RMS and composed-correction-RMS matches. No grid or target ranking. '
              'Old routing stays fixed; selected times may differ, while last states both step399. These controls do not prove semantic factor learning.',results=results))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()

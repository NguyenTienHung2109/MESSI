"""Source-only case capacity in matched spatial and repeated-CLS banks."""
import json,csv
from pathlib import Path
import numpy as np
from .spatial_bank_run import DEST
from .shrinkage import write,digest

def run(dest=DEST):
    sources={'spatial':dest/'eda/source_out_replay.npz','CLS_control':dest/'matched_CLS_control/eda/source_out_replay.npz'}
    rows=[];summary={};identities=None
    for name,path in sources.items():
        with np.load(path) as a:
            ids=a['ids'].copy();y=a['y'];domain=a['domains'];g=a['OFF_g'];delta=a['OFF_delta'];pi=a['OFF_pi'];logits=a['B1'];glob=g.argmax(1);pred=logits.argmax(1)
            key=(ids.tolist(),y.tolist(),domain.tolist())
            if identities is None:identities=key
            else:assert key==identities
            single=g[:,None,:]+delta;capacity=(single.argmax(-1)==y[:,None]).any(1)
            globalgap=g[np.arange(len(y)),y]-g[np.arange(len(y)),glob]
            corr=logits-g;added=corr[np.arange(len(y)),y]-corr[np.arange(len(y)),glob]
            for i,image in enumerate(ids):
                rows.append(dict(bank=name,image_id=str(image),environment=int(domain[i]),true_class=int(y[i]),global_prediction=int(glob[i]),bank_prediction=int(pred[i]),global_correct=bool(glob[i]==y[i]),bank_correct=bool(pred[i]==y[i]),unscaled_single_expert_can_repair=bool(capacity[i]),global_true_vs_pred_gap=float(globalgap[i]),routed_true_vs_global_pred_correction=float(added[i]),router_NULL=float(pi[i,-1])))
            groups={}
            for d in sorted(set(domain.tolist())):
                for c in sorted(set(y.tolist())):
                    mask=(domain==d)&(y==c);n=int(mask.sum())
                    if n<2:continue
                    wrong=mask&(glob!=y);right=mask&(glob==y)
                    groups[f'{d}/{c}']=dict(n=n,global_wrong=int(wrong.sum()),single_expert_capacity=int((wrong&capacity).sum()),actual_repairs=int((wrong&(pred==y)).sum()),actual_harms=int((right&(pred!=y)).sum()),wrong_gap_median=float(np.median(globalgap[wrong])) if wrong.any() else None,wrong_correction_median=float(np.median(added[wrong])) if wrong.any() else None)
            wrong=glob!=y
            summary[name]=dict(cache_sha256=digest(path),images=len(y),global_correct=int((glob==y).sum()),bank_correct=int((pred==y).sum()),global_wrong=int(wrong.sum()),single_expert_capacity=int((wrong&capacity).sum()),repairs=int((wrong&(pred==y)).sum()),harms=int(((~wrong)&(pred!=y)).sum()),wrong_gap_median=float(np.median(globalgap[wrong])),wrong_correction_median=float(np.median(added[wrong])),groups=groups)
    out=dest/'source_spatial_capacity';out.mkdir(exist_ok=True)
    write(out/'results.json',dict(source_only=True,target_read=False,minimum_group_samples=2,scope='Locked selected checkpoints; oracle capacity uses unscaled individual residuals and true labels, is not deployable or proof of useful factors. Compare actual repairs/harms and class/domain groups.',banks=summary))
    with (out/'cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return summary

if __name__=='__main__':print(json.dumps(run()))

"""Common readout regularization selected inside source-in, never source-out."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from .frozen_token_readout_diagnosis import MODES,prepare,fit,score
from .shrinkage import ROOT,digest,write


def main():
    torch.set_num_threads(4);out=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';audit=json.loads((out/'extraction_audit.json').read_text());arrays={s:dict(np.load(out/(s+'_tokens.npz'))) for s in ['source_in','source_out']};tr=arrays['source_in'];meta=ROOT/'results/sirm_research_loop/iteration_08/source_annotation_audit/selected_source_results.json';lookup={r['local_id']:r['official_sequence'] for r in json.loads(meta.read_text())['records']};sequences=[lookup[str(i)] for i in tr['ids']];assert None not in sequences
    fold=np.array([int(hashlib.sha256(s.encode()).hexdigest(),16)%3 for s in sequences]);assert set(fold)=={0,1,2};grid=[.001,.01,.1,1.];plan=dict(code_sha256=digest(Path(__file__)),readout_code_sha256=digest(ROOT/'experiments/sirm_research_loop/frozen_token_readout_diagnosis.py'),feature_cache_hashes={r['split']:r['cache_sha256'] for r in audit['rows']},metadata_sha256=digest(meta),fold_rule='sha256(official source-in event ID) modulo3; source-out not used to define folds',fold_sizes={str(i):int((fold==i).sum()) for i in range(3)},L2_grid=grid,selection='One common L2 minimizing mean of three inner-fold equal-domain CE across all five predeclared modes; ties choose earliest grid entry. No mode selection. Normalize on inner fitting subset only.',target_read=False,source_out_used_for_selection=False,scope='Inner readout labels/events held out only. Frozen anchor previously trained on full source-in; not independent representation validation. All modes/endpoint fits remain exploratory; no primary SIRM settings changed.')
    assert all(digest(out/(s+'_tokens.npz'))==plan['feature_cache_hashes'][s] for s in arrays);write(out/'inner_validation_plan.json',plan);records=[]
    for index in range(3):
        mask=fold!=index;held=~mask;assert not set(np.array(sequences)[mask])&set(np.array(sequences)[held]);assert set(tr['domains'][mask])=={0,2,3}
        # Source-out feature values are deliberately absent from all CV fit inputs.
        innerarrays={'source_in':tr}
        for mode in MODES:
            z,_=prepare(tr,innerarrays,mask,mode)
            for l2 in grid:
                w,b,info=fit(z['source_in'],tr,mask,l2);logits=torch.tensor(tr['global_logits'],dtype=torch.double)+z['source_in']@w.T+b
                metric=score(logits,tr,held);records.append(dict(fold=index,mode=mode,l2=l2,fit=info,held=metric));assert info['converged_gradient']
        write(out/'inner_validation_progress.json',dict(status='cross_validating',completed_folds=list(range(index+1))))
    means={str(l2):float(np.mean([r['held']['equal_domain_CE'] for r in records if r['l2']==l2])) for l2 in grid};chosen=min(grid,key=lambda l2:(means[str(l2)],grid.index(l2)));write(out/'inner_selection_lock.json',dict(plan=plan,common_l2=chosen,inner_CV_mean_CE=means,records=records,source_out_read_by_selection=False))
    results={};locks={};mask=np.ones(len(tr['y']),bool)
    for mode in MODES:
        z,norm=prepare(tr,arrays,mask,mode);w,b,info=fit(z['source_in'],tr,mask,chosen);assert info['converged_gradient'];pred={s:torch.tensor(a['global_logits'],dtype=torch.double)+z[s]@w.T+b for s,a in arrays.items()};metrics={s:score(pred[s],a,np.ones(len(a['y']),bool)) for s,a in arrays.items()};results[mode]=dict(fit=info,splits=metrics)
        model=out/('inner_selected_'+mode+'.pt');torch.save(dict(weight=w,bias=b,normalization=norm,common_l2=chosen,selection_lock_sha256=digest(out/'inner_selection_lock.json')),model);locks[mode]=digest(model);np.savez_compressed(out/('inner_selected_'+mode+'_replay.npz'),**{s+'_logits':v.numpy() for s,v in pred.items()});print(mode,'l2',chosen,'out',metrics['source_out'],flush=True)
    assert digest(meta)==plan['metadata_sha256'];write(out/'inner_validated_readout_results.json',dict(plan=plan,common_l2=chosen,inner_CV_mean_CE=means,selection_lock_sha256=digest(out/'inner_selection_lock.json'),results=results,model_hashes=locks,target_read=False,primary_method_modified=False));write(out/'inner_validation_progress.json',dict(status='complete',completed_folds=[0,1,2],final_modes=list(results)))

if __name__=='__main__':main()

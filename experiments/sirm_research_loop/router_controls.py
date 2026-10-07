"""Two source-calibrated magnitude controls for the router intervention.

Old OFF -> new OFF isolates composition/discovery gain; new OFF -> new ON
isolates SIRM gain. Calibration is locked before target access in each case.
"""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover_gate.model import GateSIRM
from experiments.terra_10pct_factor_diagnosis.report import data_for
from .shrinkage import ROOT,head,quantities,read,digest,write

DEST=ROOT/'results/sirm_research_loop/iteration_02'

@torch.no_grad()
def run():
    torch.set_num_threads(4)
    paths={'baseline_OFF':ROOT/'results/terra_10pct_factor_diagnosis/runs/L38/OFF','candidate_OFF':DEST/'experiment/runs/L38/OFF','candidate_ON':DEST/'experiment/runs/L38/ON'}
    saved={k:read(p/'posthoc/complete.json') for k,p in paths.items()}
    locks={k:digest(p/'source_best.pkl') for k,p in paths.items()}
    models={}
    for k,p in paths.items():
        assert locks[k]==saved[k]['checkpoint_sha256']
        payload=torch.load(p/'source_best.pkl',map_location='cpu',weights_only=False);c=payload['config']
        backbone=DeiTFeaturizer(pretrained=False,model_name=c['backbone']);model=GateSIRM(backbone,backbone.n_outputs,10,3,c)
        model.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()});model.configure_stage();models[k]=model.eval()
    for name in ['featurizer','global_head']:
        anchor=getattr(models['baseline_OFF'],name).state_dict()
        for k,m in models.items():
            other=getattr(m,name).state_dict()
            assert anchor.keys()==other.keys() and all(torch.equal(anchor[p],other[p]) for p in anchor),f'{k}: global anchor changed'
    manifest=read(ROOT/'subsets/terra_10pct_seed0/manifest.json')
    write(DEST/'control_checkpoint_locks.json',dict(checkpoints=locks,manifest_sha256=digest(ROOT/'subsets/terra_10pct_seed0/manifest.json'),target_selected=False))
    mask=np.array(saved['candidate_ON']['mechanism']['union_mask'],bool);splits={};alpha={}
    for split,envs,key in [('source_out',[0,2,3],'out'),('target_in',[1],'in'),('target_out',[1],'out')]:
        columns={k:{v:[] for v in ['g','delta','pi']} for k in models};ys=[];domains=[]
        for d,(env,ds) in enumerate(sorted(data_for(manifest,envs,c,key).items())):
            for x,y in DataLoader(ds,batch_size=64,num_workers=2):
                z=models['baseline_OFF'].featurizer(x)
                for k,m in models.items():
                    for v,t in zip(['g','delta','pi'],head(m,z)):columns[k][v].append(t)
                ys.append(y);domains.append(torch.full_like(y,d))
        arrays={k:{v:torch.cat(t) for v,t in a.items()} for k,a in columns.items()};y=torch.cat(ys);dd=torch.cat(domains)
        measured={k:quantities(**a,y=y,domains=dd,alpha=1.) for k,a in arrays.items()}
        for k in models:assert abs(measured[k]['accuracy']-saved[k]['splits'][split]['source_mean'])<1e-12
        if split=='source_out':
            alpha={'baseline_OFF_matched_candidate_OFF':measured['candidate_OFF']['correction_RMS']/measured['baseline_OFF']['correction_RMS'],'candidate_OFF_matched_candidate_ON':measured['candidate_ON']['correction_RMS']/measured['candidate_OFF']['correction_RMS']}
            write(DEST/'router_scale_lock.json',dict(alpha=alpha,calibration='source-out correction RMS, never target; alpha>1 is amplification, not shrinkage',target_used=False))
        for name,value in alpha.items():
            owner='baseline_OFF' if name.startswith('baseline') else 'candidate_OFF'
            measured[name]=quantities(**arrays[owner],y=y,domains=dd,alpha=value)
        measured['global_only']=quantities(**arrays['baseline_OFF'],y=y,domains=dd,alpha=0.)
        comparisons={}
        for a,b,scaled in [('baseline_OFF','candidate_OFF','baseline_OFF_matched_candidate_OFF'),('candidate_OFF','candidate_ON','candidate_OFF_matched_candidate_ON')]:
            p=measured[a]['predictions'];q=measured[b]['predictions'];r=measured[scaled]['predictions']
            comparisons[a+' -> '+b]=dict(accuracy_gain=measured[b]['accuracy']-measured[a]['accuracy'],scaled_gain=measured[scaled]['accuracy']-measured[a]['accuracy'],candidate_changed=int((p!=q).sum()),scaled_changed=int((p!=r).sum()),candidate_scaled_disagreement=int((q!=r).sum()),candidate_changes_reproduced=int(((p!=q)&(q==r)).sum()))
        for k,v in measured.items():
            v.pop('predictions')
            if split=='source_out':v['G_common_executed']=float(np.array(v['G_matrix'])[mask].mean()) if mask.any() else None
            else:
                su=np.array(splits['source_out']['methods'][k]['utility_matrix']).mean(0);tu=np.array(v['utility_matrix'])[0]
                for label,values in [('source_target_CE_spearman',tu),('source_target_accuracy_spearman',v['expert_accuracy_drop'])]:
                    r=float(spearmanr(su,values).statistic);v[label]=r if np.isfinite(r) else None
        splits[split]=dict(samples=len(y),methods=measured,comparisons=comparisons)
    for k,p in paths.items():assert digest(p/'source_best.pkl')==locks[k]
    result=dict(checkpoint_hashes=locks,alpha=alpha,common_mask=mask.tolist(),splits=splits,target_selected=False,training=False,scope='uniform source-RMS controls; does not rule out nonuniform suppression or router-specific global fallback')
    write(DEST/'router_control_results.json',result)
    return result

if __name__=='__main__':run()

"""Source-calibrated inference controls and descriptive utility stability."""
import hashlib,json
import numpy as np
import torch
from torch.utils.data import DataLoader
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover_gate.model import GateSIRM
from experiments.terra_10pct_factor_diagnosis.report import data_for
from .shrinkage import head,read,digest,write
from .anchor_run import DEST,EXPERIMENT

def scores(logits,y,domains):
    x=torch.as_tensor(logits).double();y=torch.as_tensor(y)
    pred=x.argmax(1).numpy();ce=torch.nn.functional.cross_entropy(x,y,reduction='none').numpy()
    ds=sorted(set(domains.tolist()));acc={str(d):float((pred[domains==d]==y.numpy()[domains==d]).mean()) for d in ds}
    loss={str(d):float(ce[domains==d].mean()) for d in ds}
    return dict(accuracy=float(np.mean(list(acc.values()))),CE=float(np.mean(list(loss.values()))),per_domain_accuracy=acc,per_domain_CE=loss),pred

def stability(a):
    y=a['y'];domains=a['domains'];ids=a['ids'];u=a['B1_utility'];out={};fold=np.zeros(len(y),int)
    for d in sorted(set(domains.tolist())):
        for c in sorted(set(y[domains==d].tolist())):
            ii=np.where((domains==d)&(y==c))[0]
            ii=sorted(ii,key=lambda i:hashlib.sha256(str(ids[i]).encode()).hexdigest())
            fold[ii]=np.arange(len(ii))%2
    matrices=[]
    for f in [0,1]:
        ds=sorted(set(domains.tolist()));U=np.array([u[(domains==d)&(fold==f)].mean(0) for d in ds])
        V=U-U.mean(1,keepdims=True)-U.mean(0,keepdims=True)+U.mean();mask=(U>.002)&(V>.002)
        out[str(f)]=dict(U=U.tolist(),V=V.tolist(),selected=mask.tolist(),shared_experts=int((mask.sum(0)>=2).sum()),domain_counts={str(d):int(((domains==d)&(fold==f)).sum()) for d in ds});matrices.append(U)
    out['positive_in_both_halves']=((matrices[0]>0)&(matrices[1]>0)).tolist()
    out['U_floor_in_both_halves']=((matrices[0]>.002)&(matrices[1]>.002)).tolist()
    out['scope']='Source-selected bank; deterministic class/domain halves are descriptive and not independent selection or event-level replicates.'
    return out

@torch.no_grad()
def run():
    torch.set_num_threads(4);dest=DEST/'bias_control';dest.mkdir(exist_ok=True)
    write(dest/'progress.json',dict(status='source_in_calibration'))
    bank=EXPERIMENT/'runs/L38/OFF';sha=digest(bank/'source_best.pkl');manifest=read(EXPERIMENT/'subset_manifest.json')
    assert sha==read(bank/'posthoc/complete.json')['checkpoint_sha256']
    payload=torch.load(bank/'source_best.pkl',map_location='cpu',weights_only=False);c=payload['config'];backbone=DeiTFeaturizer(pretrained=False,model_name=c['backbone'])
    model=GateSIRM(backbone,backbone.n_outputs,10,3,c);model.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()});model.requires_grad_(False);model.eval()
    means={};ids=[]
    for d,ds in sorted(data_for(manifest,[0,2,3],c,'in').items()):
        corrections=[]
        for x,y in DataLoader(ds,batch_size=64,num_workers=2):
            g,delta,pi=head(model,model.featurizer(x));corrections.append((delta*pi[:,:6,None]).sum(1).double())
        means[d]=torch.cat(corrections).mean(0).numpy()
        ids.extend(manifest['sources'][str(d)]['images'][i]['id'] for i in ds.indices)
    b=np.array(list(means.values())).mean(0)
    lock=dict(vector=b.tolist(),domain_means={str(k):v.tolist() for k,v in means.items()},source_in_ids=ids,checkpoint_sha256=sha,manifest_sha256=digest(EXPERIMENT/'subset_manifest.json'),target_used_to_calibrate=False,optimization=False,rule='equal-domain source-in composed correction mean')
    write(dest/'source_bias_lock.json',lock);write(dest/'progress.json',dict(status='locked_bias_replay'))
    results={};a_source=None
    for split in ['source_out','target_in','target_out']:
        a=np.load(DEST/'eda'/f'{split}_replay.npz');g=a['B0'].astype(float);full=a['B1'].astype(float);y=a['y'];domains=a['domains'];variants={'global':g,'residual':full,'constant_bias':g+b,'centered_instance':full-b};stats={};pred={}
        for name,logits in variants.items():stats[name],pred[name]=scores(logits,y,domains)
        repairs=(pred['global']!=y)&(pred['residual']==y);harms=(pred['global']==y)&(pred['residual']!=y)
        transitions={name:dict(repairs_global=int(((pred['global']!=y)&(pp==y)).sum()),harms_global=int(((pred['global']==y)&(pp!=y)).sum()),residual_repairs_retained=int((repairs&(pp==y)).sum()),residual_harms_retained=int((harms&(pp!=y)).sum()),disagreement_with_residual=int((pp!=pred['residual']).sum())) for name,pp in pred.items()}
        results[split]=dict(n=len(y),scores=stats,transitions=transitions)
        np.savez_compressed(dest/f'{split}_controls.npz',ids=a['ids'],y=y,domains=domains,**pred)
        if split=='source_out':a_source={k:a[k] for k in a.files};write(dest/'source_utility_stability.json',stability(a_source))
    held=[]
    for d in sorted(means):
        mask=a_source['domains']==d;bb=np.mean([v for k,v in means.items() if k!=d],0)
        sc={}
        for name,logits in {'global':a_source['B0'][mask],'residual':a_source['B1'][mask],'other_domain_bias':a_source['B0'][mask]+bb}.items():sc[name],_=scores(logits,a_source['y'][mask],a_source['domains'][mask])
        held.append(dict(held_calibration_domain=d,calibration_domains=[k for k in means if k!=d],scores=sc))
    assert digest(bank/'source_best.pkl')==sha
    result=dict(splits=results,source_held_calibration=held,source_bias_lock=lock,training=False,checkpoint_selection_changed=False,target_tuning=False,exploratory_after_target_EDA=True)
    write(dest/'results.json',result);write(dest/'progress.json',dict(status='completed',remaining='interpretation and next preregistered experiment'))
    return result

if __name__=='__main__':run()

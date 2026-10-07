"""Source-only E2 mean/instance and composition controls; no deployable oracle."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .class_balanced_run import DEST
from .shrinkage import digest,write

def score(logits,y,dom,g):
    ce=F.cross_entropy(logits,y,reduction='none');correct=logits.argmax(1).eq(y);gc=g.argmax(1).eq(y)
    return {'CE_equal_domain':float(torch.stack([ce[dom==d].mean() for d in [0,2,3]]).mean()),'accuracy_equal_domain':float(torch.stack([correct[dom==d].double().mean() for d in [0,2,3]]).mean()),'repairs':int((~gc&correct).sum()),'harms':int((gc&~correct).sum()),'bird_CE':float(ce[y==0].mean()),'bird_correct':int((correct&(y==0)).sum())}

def run():
    torch.set_num_threads(4);out=DEST/'source_component_diagnosis';out.mkdir(exist_ok=True)
    plan={'source_only':True,'target_read':False,'training':False,'optimization':False,'method_change':False,'expert':2,'calibration':'Equal-domain source-in means, global and within true class; no source-out mean fitting. Class-conditioned means are label-informed diagnostic controls, not deployable inference. Other-domain calibration holdouts remove calibration exposure only, not bank training exposure.','states':['original_OFF_selected','original_OFF_last','equal_OFF_selected','equal_OFF_last'],'scope':'Test whether E2 source bird CE utility survives source-in constant-component substitutions, and whether other components cancel its fixed-competitor margin. No threshold or Q changes. No target-based variant selection.'}
    write(out/'plan.json',plan);(out/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());results={};locks={}
    for name in plan['states']:
        path=DEST/'source_spatial_reliability'/f'{name}_source_replay.npz';locks[name]=digest(path)
        a=np.load(path);data={}
        for split in ['source_in','source_out']:
            y=torch.tensor(a[split+'_y']);dom=torch.tensor(a[split+'_domains']);delta=torch.tensor(a[split+'_bank_delta']).double();pi=torch.tensor(a[split+'_bank_pi']).double();term=delta*pi[:,:6,None];full=torch.tensor(a[split+'_bank_full']).double();g=full-term.sum(1)
            data[split]=(y,dom,term,g)
        assert not set(a['source_in_ids'])&set(a['source_out_ids'])
        yi,di,ti,gi=data['source_in'];y,dom,t,g=data['source_out'];mu=torch.stack([ti[di==d,2].mean(0) for d in [0,2,3]]).mean(0)
        cmu=torch.stack([torch.stack([ti[(di==d)&(yi==c),2].mean(0) for d in [0,2,3]]).mean(0) for c in range(10)])
        assert torch.isfinite(cmu).all();others=g+t[:,[0,1,3,4,5]].sum(1);actual=t[:,2];full=others+actual
        components={'actual':actual,'constant_global':mu[None].expand_as(actual),'centered_global':actual-mu,'constant_true_class':cmu[y],'centered_true_class':actual-cmu[y]}
        variants={}
        for key,component in components.items():
            logits=others+component;u=F.cross_entropy(others,y,reduction='none')-F.cross_entropy(logits,y,reduction='none')
            variants[key]={'composed':score(logits,y,dom,g),'E2_alone':score(g+component,y,dom,g),'U_per_domain':torch.stack([u[dom==d].mean() for d in [0,2,3]]).tolist(),'bird_U_per_domain':torch.stack([u[(dom==d)&(y==0)].mean() for d in [0,2,3]]).tolist()}
        ix=torch.arange(len(y));comp=g.clone();comp[ix,y]=-torch.inf;comp=comp.argmax(1);margin=lambda v:v[ix,y]-v[ix,comp]
        bird=[]
        for i in torch.nonzero(y==0).flatten().tolist():bird.append({'id':str(a['source_out_ids'][i]),'domain':int(dom[i]),'global_margin':float(margin(g)[i]),'E2_margin':float(margin(actual)[i]),'other_margin':float(margin(t[:,[0,1,3,4,5]].sum(1))[i]),'full_margin':float(margin(full)[i]),'global_constant_E2_margin':float(margin(components['constant_global'])[i]),'class_constant_E2_margin':float(margin(components['constant_true_class'])[i])})
        holdouts={}
        for d in [0,2,3]:
            keep=dom==d;train_domains=[x for x in [0,2,3] if x!=d];mean=torch.stack([ti[di==x,2].mean(0) for x in train_domains]).mean(0);classmean=torch.stack([torch.stack([ti[(di==x)&(yi==c),2].mean(0) for x in train_domains]).mean(0) for c in range(10)])
            vals={}
            for label,v in [('actual',actual),('constant_global',mean.expand_as(actual)),('constant_true_class',classmean[y]),('centered_true_class',actual-classmean[y])]:
                ce=F.cross_entropy(others+v,y,reduction='none');base=F.cross_entropy(others,y,reduction='none');mask=keep&(y==0);vals[label]={'bird_U':float((base-ce)[mask].mean()),'bird_CE':float(ce[mask].mean()),'n':int(mask.sum())}
            holdouts[str(d)]=vals
        results[name]={'variants':variants,'bird_cases':bird,'other_domain_calibration':holdouts,'source_in_global_mean_E2':mu.tolist(),'source_in_class_means_E2':cmu.tolist()}
        assert digest(path)==locks[name]
    write(out/'results.json',{'plan':plan,'source_cache_locks':locks,'code_sha256':digest(Path(__file__)),'results':results});print(json.dumps({k:{v:x['bird_U_per_domain'] for v,x in r['variants'].items()} for k,r in results.items()},indent=2))
if __name__=='__main__':run()

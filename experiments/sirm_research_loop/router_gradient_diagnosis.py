"""Saved-state source objective alignment; no optimizer updates/target reads."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover_gate.model import GateSIRM
from experiments.sirm_discover.math import multiple_choice_loss,residual_utilities,router_teacher
from experiments.terra_10pct_factor_diagnosis.report import data_for
import argparse
DEST=None
EXPERIMENT=None
from .shrinkage import digest,read,write

def norm(g):return sum(float(x.detach().double().square().sum()) for x in g if x is not None)**.5
def cosine(a,b):
    n=norm(a)*norm(b)
    return float(sum((x.detach().double()*y.detach().double()).sum() for x,y in zip(a,b) if x is not None and y is not None)/n) if n else None

def main():
    torch.set_num_threads(4);out=DEST/'router_gradient_diagnosis';out.mkdir(exist_ok=True)
    bank=EXPERIMENT/'runs/L38/OFF';path=bank/'source_best.pkl';sha=digest(path)
    assert sha==read(bank/'posthoc/complete.json')['checkpoint_sha256']
    payload=torch.load(path,map_location='cpu',weights_only=False);c=payload['config'];backbone=DeiTFeaturizer(pretrained=False,model_name=c['backbone'])
    model=GateSIRM(backbone,backbone.n_outputs,10,3,c);model.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()});model.configure_stage();model.eval()
    assert all(not p.requires_grad for name in ['featurizer','global_head'] for p in getattr(model,name).parameters())
    residual=list(model.residuals.parameters());router=list(model.router.parameters());parameters=residual+router
    results=[];manifest=read(EXPERIMENT/'subset_manifest.json')
    for split in ['in','out']:
        datasets=data_for(manifest,[0,2,3],c,split)
        for sample in range(3):
            xs=[];ys=[];ids=[]
            for domain,ds in sorted(datasets.items()):
                rng=np.random.default_rng(61703+sample*17+domain);idx=rng.choice(len(ds),size=min(32,len(ds)),replace=False).tolist()
                x,y=next(iter(DataLoader(Subset(ds,idx),batch_size=len(idx),num_workers=0)));xs.append(x);ys.append(y)
                ids.extend(manifest['sources'][str(domain)]['images'][ds.indices[i]]['id'] for i in idx)
            with torch.no_grad():z=model.featurizer(torch.cat(xs));g=model.global_head(z)
            y=torch.cat(ys);delta=torch.stack([expert(z) for expert in model.residuals],1);delta=delta-delta.mean(-1,keepdim=True);pi=model.router(z).softmax(1)
            full,_,_,single=residual_utilities(g,delta,pi,y)
            cls=F.cross_entropy(full,y);discover,choice=multiple_choice_loss(g,delta,y,c['discovery_temperature'])
            teacher=router_teacher(single,c['router_temperature'],c['utility_floor']);route=F.kl_div(F.log_softmax(model.router(z.detach()),1),teacher,reduction='batchmean')
            balance=(6*choice.mean(0).square().sum()-1)*(1-teacher[:,-1]).mean().detach()
            base=cls+c['lambda_discover']*discover+c['lambda_router']*route+c['lambda_balance']*balance
            grads={name:torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True) for name,loss in [('cls',cls),('weighted_discover',c['lambda_discover']*discover),('weighted_router',c['lambda_router']*route),('base',base)]}
            results.append(dict(split='source_'+split,sample=sample,n=len(y),image_ids=ids,loss_cls=float(cls.detach()),loss_discover=float(discover.detach()),norms={k:norm(v) for k,v in grads.items()},cosine_cls_discover=cosine(grads['cls'],grads['weighted_discover']),cosine_cls_router=cosine(grads['cls'],grads['weighted_router']),cosine_cls_base=cosine(grads['cls'],grads['base']),residual_only_discover_to_cls_ratio=norm(grads['weighted_discover'][:len(residual)])/norm(grads['cls'][:len(residual)]),router_only_norms={k:norm(v[len(residual):]) for k,v in grads.items()},router_only_distill_to_cls_ratio=norm(grads['weighted_router'][len(residual):])/norm(grads['cls'][len(residual):]) if norm(grads['cls'][len(residual):]) else None,router_cls_distill_cosine=cosine(grads['cls'][len(residual):],grads['weighted_router'][len(residual):]),router_distill_base_cosine=cosine(grads['weighted_router'][len(residual):],grads['base'][len(residual):])))
            assert all(p.grad is None for p in model.parameters())
    assert digest(path)==sha
    write(out/'results.json',dict(checkpoint_sha256=sha,checkpoint_step=payload['selected_step'],method='saved selected OFF state, three deterministic 32-per-domain original-view source batches per split',historical_training_gradients=False,training=False,target_read=False,parameter_updates=False,results=results))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('destination',type=__import__('pathlib').Path);args=parser.parse_args();DEST=args.destination.resolve();EXPERIMENT=DEST/'experiment';main()

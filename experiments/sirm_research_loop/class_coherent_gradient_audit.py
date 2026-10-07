"""Prelaunch saved-state source-only comparison of discovery gradients."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover.math import center_logits, expert_ce
from .centered_model import CenteredGateSIRM
from .class_coherent_discovery import class_coherent_cost
from .discovery_error_gradient_diagnosis import vector
from .shrinkage import ROOT, digest, write


def gradients(model, z, y, mode):
    with torch.no_grad():
        g = model.global_head(z); pi = model.router(z).softmax(1)
    raw = center_logits(torch.stack([e(z) for e in model.residuals], 1))
    delta = raw - (raw.mean(0, keepdim=True) if mode == 'batch' else model.expert_source_mean[None])
    losses = expert_ce(g[:, None] + delta, y)
    image = losses.amin(1).mean(); grouped = class_coherent_cost(losses, y)
    full = g + (pi[:, :6, None] * delta).sum(1)
    cls = F.cross_entropy(full, y)
    params = list(model.residuals.parameters()); w = model.config['lambda_discover']
    a,b,c = [vector(loss,params) for loss in [w*image,w*grouped,cls]]
    cos = lambda x,y: float(F.cosine_similarity(x,y,dim=0)) if x.norm() and y.norm() else None
    return dict(image_loss=float(image.detach()),class_loss=float(grouped.detach()),
                image_grad_norm=float(a.norm()),class_grad_norm=float(b.norm()),
                class_to_image_norm=float(b.norm()/a.norm()) if a.norm() else None,
                class_to_classification_norm=float(b.norm()/c.norm()) if c.norm() else None,
                class_image_gradient_cosine=cos(a,b),class_classification_cosine=cos(b,c)), (a,b,c)


def main():
    torch.set_num_threads(4)
    prior=ROOT/'results/sirm_research_loop/iteration_11'
    dest=ROOT/'results/sirm_research_loop/iteration_12'; dest.mkdir(exist_ok=True)
    path=prior/'experiment/runs/L38/OFF/source_best.pkl'
    p=torch.load(path,map_location='cpu',weights_only=False); assert p['selected_step']==399
    c=p['config']; model=CenteredGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()}); model.configure_stage();model.eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    original=[]
    proof=json.loads((prior/'composed_bias_source_diagnosis/results.json').read_text())
    for split in ['source_in','source_out']:
        file=ROOT/'results/sirm_research_loop/iteration_09/source_router_diagnosis'/f'{split}_original.npz'
        assert digest(file)==proof['plan']['source_cache_hashes'][split]
        with np.load(file) as cache: z,y=[torch.from_numpy(cache[k]) for k in ['z','y']]
        metric,vs=gradients(model,z,y,'saved')
        original.append((dict(split=split,view='original',centering='saved',cache_sha256=digest(file),metrics=metric),vs))
    reference=original[1][1][2]; records=[]
    def record(row,vs):
        for name,v in zip(['image','class'],vs[:2]):
            row['metrics'][name+'_gradient_cosine_to_source_out_classification']=float(F.cosine_similarity(v,reference,dim=0))
        records.append(row)
    for row,vs in original: record(row,vs)
    for step in [200,215,250,300,350,395]:
        file=ROOT/'results/sirm_research_loop/iteration_08/pair_shared_parameter_audit'/f'H2_step{step}.pt'
        priorrow=next(r for r in proof['paired_batches'] if r['step']==step)
        assert digest(file)==priorrow['cache_sha256']
        batch=torch.load(file,map_location='cpu',weights_only=False)
        for mode in ['saved','batch']:
            metric,vs=gradients(model,batch['z'],batch['y'],mode)
            record(dict(split='source_in_scheduled',view='H2',step=step,centering=mode,cache_sha256=digest(file),metrics=metric),vs)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in model.parameters())
    assert all(not p.requires_grad for mod in [model.featurizer,model.global_head] for p in mod.parameters())
    result=dict(checkpoint_sha256=digest(path),checkpoint_step=399,code_sha256=digest(Path(__file__)),
                target_read=False,training=False,fitting=False,parameter_updates=False,
                state_tensors_unchanged=True,scope='Residual parameter gradients at one saved bank; not historical gradients or future class-coherent training evidence.',records=records)
    write(dest/'prelaunch_gradient_audit.json',result)
    print(json.dumps([dict(split=r['split'],step=r.get('step'),centering=r['centering'],**r['metrics']) for r in records],indent=2))

if __name__=='__main__': main()

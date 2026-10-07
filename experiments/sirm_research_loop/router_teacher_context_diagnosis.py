"""Fixed-bank source-only teacher utility/context and gradient alignment controls."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover.math import router_teacher,expert_ce
from .centered_model import CenteredGateSIRM
from .hard_router_diagnosis import outputs
from .bias_control import scores
from .shrinkage import ROOT,digest,write


def grad(loss,params):
    return torch.cat([g.detach().double().flatten() for g in torch.autograd.grad(loss,params,retain_graph=True)])


def main():
    torch.set_num_threads(4);root=ROOT/'results/sirm_research_loop/iteration_11';out=root/'router_teacher_context_diagnosis';out.mkdir(exist_ok=True)
    path=root/'experiment/runs/L38/OFF/source_best.pkl';p=torch.load(path,map_location='cpu',weights_only=False);assert p['selected_step']==399
    c=p['config'];model=CenteredGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()});model.configure_stage();model.eval()
    state={k:v.clone() for k,v in model.state_dict().items()}
    params=list(model.router.parameters());records=[]
    validated=__import__('json').loads((root/'composed_bias_source_diagnosis/results.json').read_text())
    plan=dict(code_sha256=digest(Path(__file__)),checkpoint_sha256=digest(path),target_read=False,fitting=False,training=False,
        scope='Fixed399 source-only teacher/read-only router gradient diagnosis. Label-dependent teachers are not deployable. '
              'Floor/temperature unchanged; hypothetical teachers do not change Q, primary trajectory or parameters.',
        teachers=['standalone','routed_single','marginal_context','local_simplex_direction'])
    for split in ['source_in','source_out']:
        cache=ROOT/'results/sirm_research_loop/iteration_09/source_router_diagnosis'/f'{split}_original.npz'
        assert digest(cache)==validated['plan']['source_cache_hashes'][split]
        with np.load(cache) as a:z,y,d=[torch.from_numpy(a[k]) for k in ['z','y','domains']]
        f=outputs(model,z,y);g=f['g'];delta=f['delta'];pi=f['pi'];full=f['full']
        ce_g=F.cross_entropy(g,y,reduction='none');contribution=pi[:,:6,None]*delta
        routed=ce_g[:,None]-expert_ce(g[:,None]+contribution,y)
        local=((F.one_hot(y,10)-full.softmax(1))[:,None]*delta).sum(2)
        utilities=dict(standalone=f['single'],routed_single=routed,marginal_context=f['marginal'],local_simplex_direction=local)
        cls=grad(F.cross_entropy(full,y),params);wrong=g.argmax(1)!=y
        kl_input=F.log_softmax(model.router(z),1)
        choices={};targets={}
        for name,u in utilities.items():
            target=router_teacher(u,c['router_temperature'],c['utility_floor']);targets[name]=target
            logits=g+(target[:,:6,None]*delta).sum(1)
            score,pred=scores(logits.detach().numpy(),y.numpy(),d.numpy())
            route=c['lambda_router']*F.kl_div(kl_input,target,reduction='batchmean');rg=grad(route,params)
            cosine=float(F.cosine_similarity(cls,rg,dim=0)) if cls.norm()>0 and rg.norm()>0 else None
            score.update(repairs=int((wrong&torch.from_numpy(pred).eq(y)).sum()),harms=int((~wrong&torch.from_numpy(pred).ne(y)).sum()),
                NULL=float(target[:,-1].mean()),wrong_NULL=float(target[wrong,-1].mean()),
                weighted_router_gradient_norm=float(rg.norm()),classification_router_gradient_norm=float(cls.norm()),
                router_gradient_cosine_to_classification=cosine,
                fraction_negative_context_mass=float((target[:,:6]*(f['marginal'].detach()<0)).sum(1).mean()))
            choices[name]=score
        comparisons={name:dict(teacher_L1_difference=float((target-targets['standalone']).abs().sum(1).mean()),
            argmax_disagreement=float((target.argmax(1)!=targets['standalone'].argmax(1)).double().mean())) for name,target in targets.items()}
        records.append(dict(split=split,n=len(y),teachers=choices,comparisons=comparisons))
    assert all(v.grad is None for v in model.parameters())
    assert all(torch.equal(state[k],v) for k,v in model.state_dict().items()) and digest(path)==plan['checkpoint_sha256']
    write(out/'results.json',dict(plan=plan,records=records,state_tensors_unchanged=True))
    print([(r['split'],r['teachers']) for r in records])


if __name__=='__main__':main()

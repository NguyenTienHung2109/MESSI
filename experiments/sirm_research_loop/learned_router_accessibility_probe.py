"""Source-only trainable nonlinear versus linear teacher accessibility, frozen bank."""
from pathlib import Path
import json,copy,hashlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from domainbed.deit_transformer import DeiTFeaturizer
from domainbed.scripts.train_subset_irm_pacs import make_schedule
from .centered_model import CenteredGateSIRM
from .hard_router_diagnosis import outputs
from .bias_control import scores
from .shrinkage import ROOT,digest,write


class AdapterRouter(nn.Module):
    def __init__(self,linear,width):
        super().__init__();self.linear=copy.deepcopy(linear)
        self.adapter=nn.Sequential(nn.Linear(384,width),nn.ReLU(),nn.Linear(width,7))
        nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
    def forward(self,x):return self.linear(x)+self.adapter(x)


def evaluate(router,a):
    with torch.no_grad():
        pi=router(a['standardized']).softmax(1);logits=a['g']+(pi[:,:6,None]*a['delta']).sum(1)
        metric,pred=scores(logits.numpy(),a['y'].numpy(),a['domains'].numpy());base=a['g'].argmax(1);pred=torch.from_numpy(pred)
        metric.update(repairs=int(((base!=a['y'])&(pred==a['y'])).sum()),harms=int(((base==a['y'])&(pred!=a['y'])).sum()),
            teacher_KL=float(F.kl_div(pi.clamp_min(1e-30).log(),a['teacher'],reduction='batchmean')),
            NULL=float(pi[:,-1].mean()))
        return metric,pred.tolist()


def main():
    torch.set_num_threads(4);torch.manual_seed(0)
    root=ROOT/'results/sirm_research_loop/iteration_11';out=root/'learned_router_accessibility_probe';out.mkdir(exist_ok=True)
    assert not (out/'results.json').exists(),'Do not silently refit a completed probe'
    path=root/'experiment/runs/L38/OFF/source_best.pkl';p=torch.load(path,map_location='cpu',weights_only=False);assert p['selected_step']==399
    c=p['config'];model=CenteredGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()});model.configure_stage();model.eval();model.requires_grad_(False)
    bank_state={k:v.clone() for k,v in model.state_dict().items()};arrays={};locks={};ids={}
    verified=json.loads((root/'composed_bias_source_diagnosis/results.json').read_text())
    for split in ['source_in','source_out']:
        cache=ROOT/'results/sirm_research_loop/iteration_09/source_router_diagnosis'/f'{split}_original.npz'
        assert digest(cache)==verified['plan']['source_cache_hashes'][split]
        with np.load(cache) as a:z,y,d=[torch.from_numpy(a[k]) for k in ['z','y','domains']];ids[split]=a['ids'].tolist()
        with torch.no_grad():f=outputs(model,z,y)
        arrays[split]={k:f[k].detach() for k in ['g','delta','teacher']};arrays[split].update(z=z,y=y,domains=d)
        locks[split]=digest(cache)
    assert not set(ids['source_in'])&set(ids['source_out'])
    mean=arrays['source_in']['z'].mean(0);std=arrays['source_in']['z'].std(0,unbiased=False).clamp_min(1e-8)
    for a in arrays.values():a['standardized']=(a['z']-mean)/std
    linear=copy.deepcopy(model.router);linear.requires_grad_(True)
    with torch.no_grad():linear.weight.mul_(std[None]);linear.bias.add_(model.router.weight@mean)
    mlp=AdapterRouter(linear,c['residual_hidden'])
    for a in arrays.values():
        with torch.no_grad():assert torch.allclose(linear(a['standardized']),model.router(a['z']),atol=3e-6,rtol=1e-5);assert torch.equal(linear(a['standardized']),mlp(a['standardized']))
    indices={d:torch.where(arrays['source_in']['domains']==d)[0].tolist() for d in [0,2,3]}
    schedules={d:make_schedule(indices[d],400,32,0,d) for d in indices}
    plan=dict(code_sha256=digest(Path(__file__)),checkpoint_sha256=digest(path),cache_sha256=locks,
        schedule_sha256={str(d):hashlib.sha256(v.tobytes()).hexdigest() for d,v in schedules.items()},
        seed=0,updates=400,lr=c['lr'],weight_decay=c['weight_decay'],lambda_router=c['lambda_router'],width=c['residual_hidden'],
        target_read=False,bank_training=False,probe_fitting=True,checkpoint_selection=False,
        hypothesis='Trainable hidden router may access corrective expert choices missed by linear or fixed-random bases.',
        scope='Two original-source-feature teacher-only probes initialized to same deployed linear function. '
              'No H2 image training, primary cap/objective reproduction, source-out fitting or early stopping. '
              'One final400 endpoint, no hyperparameter grid; exploratory accessibility, not a primary SIRM result.')
    write(out/'plan.json',plan)
    results={};curves=[];cases=[]
    for name,router in [('linear',linear),('learned_MLP',mlp)]:
        opt=torch.optim.Adam(router.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
        for step in range(400):
            batch=np.concatenate([schedules[d][step] for d in [0,2,3]])
            a=arrays['source_in'];loss=c['lambda_router']*F.kl_div(F.log_softmax(router(a['standardized'][batch]),1),a['teacher'][batch],reduction='batchmean')
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
            if step%25==0 or step==399:
                for split,a in arrays.items():metric,_=evaluate(router,a);curves.append(dict(model=name,step=step,split=split,metrics=metric))
        results[name]={}
        for split,a in arrays.items():
            metric,pred=evaluate(router,a);results[name][split]=metric
            if split=='source_out':
                for i,image in enumerate(ids[split]):cases.append(dict(model=name,id=image,domain=int(a['domains'][i]),label=int(a['y'][i]),prediction=pred[i],global_prediction=int(a['g'][i].argmax())))
        torch.save(dict(router_state=router.state_dict(),source_mean=mean,source_std=std,plan=plan),out/(name+'.pt'))
    assert all(torch.equal(bank_state[k],v) for k,v in model.state_dict().items())
    assert digest(path)==plan['checkpoint_sha256'] and all(p.grad is None for p in model.parameters())
    write(out/'results.json',dict(plan=plan,final=results,curves=curves,cases=cases,bank_state_unchanged=True))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()

"""Source-only routing attribution at matched final Stage-2 update399."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover_gate.model import GateSIRM
from experiments.terra_10pct_factor_diagnosis.report import data_for
from .shrinkage import head,read,digest,write,ROOT
from .source_router_cache import audit

@torch.no_grad()
def run():
    torch.set_num_threads(4);dest=ROOT/'results/sirm_research_loop/iteration_05';out=dest/'router_equal_step';out.mkdir(exist_ok=True)
    manifest=read(dest/'experiment/subset_manifest.json');models={};locks={}
    for label,iteration in [('baseline',4),('candidate',5)]:
        path=ROOT/f'results/sirm_research_loop/iteration_{iteration:02d}/experiment/runs/L38/OFF/last.pkl'
        payload=torch.load(path,map_location='cpu',weights_only=False);assert payload['step']==399
        c=payload['config'];backbone=DeiTFeaturizer(pretrained=False,model_name=c['backbone']);m=GateSIRM(backbone,backbone.n_outputs,10,3,c)
        m.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()});m.configure_stage();models[label]=m.eval();locks[label]=dict(path=str(path),sha256=digest(path),step=399)
    for module in ['featurizer','global_head']:
        a=getattr(models['baseline'],module).state_dict();b=getattr(models['candidate'],module).state_dict();assert a.keys()==b.keys() and all(torch.equal(v,b[k]) for k,v in a.items())
    ids=[];ys=[];envs=[];columns={label:{k:[] for k in ['g','delta','pi']} for label in models}
    for env,ds in sorted(data_for(manifest,[0,2,3],c,'out').items()):
        ids.extend(manifest['sources'][str(env)]['images'][i]['id'] for i in ds.indices)
        for x,y in DataLoader(ds,batch_size=64,num_workers=0):
            z=models['baseline'].featurizer(x);ys.append(y);envs.append(torch.full_like(y,env))
            for label,m in models.items():
                for k,v in zip(['g','delta','pi'],head(m,z)):columns[label][k].append(v)
    for label,col in columns.items():
        cache=out/label/'eda';cache.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache/'source_out_replay.npz',ids=np.array(ids),y=torch.cat(ys).numpy(),domains=torch.cat(envs).numpy(),**{'OFF_'+k:torch.cat(v).numpy() for k,v in col.items()})
    a=audit(out/'baseline');b=audit(out/'candidate',baseline=out/'baseline')
    fixed=b['fixed_baseline_teacher_and_head_controls'];mean=lambda key:float(np.mean([v[key] for v in fixed.values()]))
    results=dict(scope='Source-out at exact same Stage-2 step399, fixed same global and image IDs; no target reads or optimization.',locks=locks,baseline=a,candidate=b,summary=dict(baseline_CE=a['equal_domain_mean']['actual']['CE'],candidate_CE=b['equal_domain_mean']['actual']['CE'],old_router_fixed_teacher_KL=mean('baseline_router_fixed_teacher_KL'),new_router_fixed_teacher_KL=mean('new_router_fixed_teacher_KL'),old_negative_mass=mean('baseline_negative_utility_mass'),new_mass_on_old_negative_terms=mean('new_router_baseline_negative_utility_mass'),new_router_old_heads_CE=float(np.mean([v['new_router_old_heads']['CE'] for v in fixed.values()])),old_router_new_heads_CE=float(np.mean([v['old_router_new_heads']['CE'] for v in fixed.values()]))))
    assert all(digest(Path(v['path']))==v['sha256'] for v in locks.values());write(out/'results.json',results);print(json.dumps(results['summary']))

if __name__=='__main__':run()

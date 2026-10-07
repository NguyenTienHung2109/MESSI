"""Source-out-only router alignment diagnostic at locked selected checkpoints."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover_gate.model import GateSIRM
from experiments.sirm_discover.math import residual_utilities,router_teacher
from .shrinkage import head,digest,write,ROOT
from experiments.terra_10pct_factor_diagnosis.report import data_for

@torch.no_grad()
def audit(bank,dest):
    torch.set_num_threads(4)
    sha=digest(bank/'source_best.pkl')
    saved=json.loads((bank/'posthoc/complete.json').read_text())
    assert sha==saved['checkpoint_sha256']
    payload=torch.load(bank/'source_best.pkl',map_location='cpu',weights_only=False);c=payload['config']
    backbone=DeiTFeaturizer(pretrained=False,model_name=c['backbone'])
    model=GateSIRM(backbone,backbone.n_outputs,10,3,c)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()})
    model.configure_stage();model.eval()
    manifest=json.loads((ROOT/'subsets/terra_10pct_seed0/manifest.json').read_text())
    records=[]
    for env,ds in sorted(data_for(manifest,[0,2,3],c,'out').items()):
        vals={k:[] for k in ['teacher_KL','router_negative_utility_mass','teacher_negative_utility_mass','router_NULL','teacher_NULL']}
        for x,y in DataLoader(ds,batch_size=64,num_workers=2):
            g,delta,pi=head(model,model.featurizer(x));_,_,marginal,single=residual_utilities(g,delta,pi,y)
            teacher=router_teacher(single,c['router_temperature'],c['utility_floor'])
            bad=marginal<0
            data={'teacher_KL':F.kl_div(pi.clamp_min(1e-30).log(),teacher,reduction='none').sum(1),'router_negative_utility_mass':(pi[:,:delta.shape[1]]*bad).sum(1),'teacher_negative_utility_mass':(teacher[:,:delta.shape[1]]*bad).sum(1),'router_NULL':pi[:,-1],'teacher_NULL':teacher[:,-1]}
            for k,v in data.items():vals[k].append(v)
        records.append(dict(environment=env,samples=len(ds),**{k:float(torch.cat(v).mean()) for k,v in vals.items()}))
    assert digest(bank/'source_best.pkl')==sha
    result=dict(scope='source-out only; equal-domain means; no target reads or fitting',checkpoint_sha256=sha,selected_step=saved['selected_step'],lambda_router=c['lambda_router'],per_domain=records,equal_domain_mean={k:float(np.mean([r[k] for r in records])) for k in vals})
    dest.parent.mkdir(parents=True,exist_ok=True);write(dest,result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args();audit(a.bank,a.dest)

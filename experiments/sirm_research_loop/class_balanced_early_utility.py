"""Completed selected OFF source U/V replication while paired ON finishes."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover.math import residual_utilities
from .class_balanced_run import DEST,ROOT
from .spatial_bank_model import SpatialBankGateSIRM
from .spatial_bank_replay import spatial_head
from .spatial_source_reliability import support
from .shrinkage import digest,write
@torch.no_grad()
def main():
 torch.set_num_threads(4);bank=DEST/'experiment/runs/L38/OFF';assert (bank/'done').exists();path=bank/'source_best.pkl';p=torch.load(path,map_location='cpu',weights_only=False);c=p['config'];m=SpatialBankGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c);m.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()});m.configure_stage();m.eval();token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';audit=json.loads((token/'extraction_audit.json').read_text());sets={};locks={}
 for split in ['source_in','source_out']:
  file=token/f'{split}_tokens.npz';assert digest(file)==next(r['cache_sha256'] for r in audit['rows'] if r['split']==split);locks[split]=digest(file)
  with np.load(file) as a:
   raw=torch.tensor(np.concatenate([a['cls'],a['patch_max']],1));y=torch.tensor(a['y']);domains=a['domains'];ids=a['ids'];g,d,pi=spatial_head(m,raw);_,_,u,_=residual_utilities(g,d,pi,y);assert np.array_equal(g.argmax(1).numpy(),a['global_logits'].argmax(1));sets[split]=support(u,domains,np.ones(len(y),bool))
   if split=='source_out':
    halves=[np.zeros(len(y),bool),np.zeros(len(y),bool)]
    for domain in [0,2,3]:
     for label in range(10):
      ix=np.flatnonzero((domains==domain)&(a['y']==label));ix=sorted(ix,key=lambda i:hashlib.sha256(str(ids[i]).encode()).hexdigest())
      for j,i in enumerate(ix):halves[j%2][i]=True
    assert np.all(halves[0]^halves[1])
    for i,h in enumerate(halves):sets[f'source_out_half{i}']=support(u,domains,h)
 matrices=[np.array(s['U']) for s in sets.values()];stable=np.logical_and.reduce([v>.002 for v in matrices]);record={'source_only':True,'target_read':False,'fitting':False,'training':False,'checkpoint_sha256':digest(path),'selected_step':p['selected_step'],'cache_sha256':locks,'sets':sets,'U_floor_stable_cells':stable.tolist(),'U_floor_stable_shared_experts':np.flatnonzero(stable.sum(0)>=2).tolist(),'scope':'Saved selected375 OFF head on original source views and deterministic class/domain out halves. Not historical H2 EMA or independently held-out validation; no Q/config/selection change.'};out=DEST/'early_OFF_source_EDA';write(out/'utility_replication.json',record);print(json.dumps({k:{'shared_U_floor':s['shared_U_floor_experts'],'shared_Q':s['shared_Q_experts'],'E2_U':np.array(s['U'])[:,2].tolist(),'E2_V':np.array(s['V'])[:,2].tolist()} for k,s in sets.items()},indent=2));print('Replicated above-floor shared experts',record['U_floor_stable_shared_experts'])
if __name__=='__main__':main()

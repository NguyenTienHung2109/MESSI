"""Paired source uncertainty and full class/case accounting; no new fitting."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .direct_router_risk_magnitude_router import OUT
from .shrinkage import digest,write

def main():
 torch.set_num_threads(4);source=json.loads((OUT/'results.json').read_text());eda=json.loads((OUT/'magnitude_and_case_eda.json').read_text());locks={};records={};rng=np.random.default_rng(0)
 for state in ['selected','last']:
  for mode in ['all_source','auxiliary_domain_held']:
   domains=[]
   for held in ([None] if mode=='all_source' else [0,2,3]):
    path=OUT/f'{state}_held{held}_confidence_only.pt';p=torch.load(path,map_location='cpu',weights_only=False);locks[str(path)]=digest(path);tr=p['replays']['source_in'];v=p['replays']['source_out'];traincorr=tr['learned_logits']-tr['global_logits'];meanpi=torch.stack([tr['learned_pi'][tr['domains'].eq(d)].mean(0) for d in tr['domains'].unique()]).mean(0);cc=(meanpi[:6][None,:,None]*tr['delta']).sum(1);scale=float(traincorr.square().mean().sqrt()/cc.square().mean().sqrt());g=v['global_logits'];y=v['y'];f=v['learned_logits'];const=g+scale*(meanpi[:6][None,:,None]*v['delta']).sum(1)
    for d in v['domains'].unique():
     mask=v['domains'].eq(d);gg=g[mask];ff=f[mask];cf=const[mask];yy=y[mask];domains.append({'n':int(mask.sum()),'domain':int(d),'acc_diff_global':(ff.argmax(1).eq(yy).double()-gg.argmax(1).eq(yy).double()).numpy(),'acc_diff_constant':(ff.argmax(1).eq(yy).double()-cf.argmax(1).eq(yy).double()).numpy(),'CE_gain_global':(F.cross_entropy(gg,yy,reduction='none')-F.cross_entropy(ff,yy,reduction='none')).numpy(),'CE_gain_constant':(F.cross_entropy(cf,yy,reduction='none')-F.cross_entropy(ff,yy,reduction='none')).numpy()})
   metrics={}
   for key in ['acc_diff_global','acc_diff_constant','CE_gain_global','CE_gain_constant']:
    sims=np.zeros(10000)
    for d in domains:
     vals=d[key];ix=rng.integers(len(vals),size=(10000,len(vals)));sims+=vals[ix].mean(1)/len(domains)
    metrics[key]={'mean':float(np.mean([d[key].mean() for d in domains])),'image_bootstrap_95_interval':np.quantile(sims,[.025,.975]).tolist()}
   records[state+'/'+mode]=metrics
 write(OUT/'paired_source_evidence.json',dict(source_only=True,target_read=False,fitting=False,sampling='10000 within-domain paired image bootstrap; no camera/time clustering or multiple-inspection correction. Expert bank/backbone saw all sources; held refers only to auxiliary router fitting.',primary='selected350/all_source; auxiliary holdouts confirm direction only; last399 secondary',records=records,input_sha256=locks,code_sha256=digest(Path(__file__))))
 print(json.dumps(records,indent=2))
if __name__=='__main__':main()

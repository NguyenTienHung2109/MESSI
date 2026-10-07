"""Paired source-only uncertainty and repair/harm overlap, no fitting."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .source_direct_mixture_reliability_router import OUT
from .shrinkage import write,digest

def main():
 torch.set_num_threads(4)
 eda=json.loads((OUT/'magnitude_and_case_eda.json').read_text());res={};rng=np.random.default_rng(0)
 for state in ['selected','last']:
  p=torch.load(OUT/f'{state}_heldNone_confidence_only.pt',weights_only=False,map_location='cpu');v=p['replays']['source_out'];g=v['global_logits'];f=v['learned_logits'];y=v['y'];d=v['domains'];train=p['replays']['source_in'];meanpi=torch.tensor(eda['states'][state]['None']['confidence_only']['source_train_mean_pi']);scale=eda['states'][state]['None']['confidence_only']['constant_source_train_RMS_match_scale'];constant=g+scale*(meanpi[:6][None,:,None]*v['delta']).sum(1);comparisons={}
  for name,other in [('global',g),('RMS_matched_constant',constant)]:
   acc=(f.argmax(1).eq(y).float()-other.argmax(1).eq(y).float()).numpy();ce=(F.cross_entropy(other,y,reduction='none')-F.cross_entropy(f,y,reduction='none')).numpy();boot=[]
   for _ in range(10000):
    indices=[rng.choice(torch.where(d.eq(dom))[0].numpy(),size=int(d.eq(dom).sum()),replace=True) for dom in d.unique()];boot.append([np.mean([acc[ix].mean() for ix in indices]),np.mean([ce[ix].mean() for ix in indices])])
   boot=np.array(boot);maskg=g.argmax(1).eq(y);maskf=f.argmax(1).eq(y);masko=other.argmax(1).eq(y)
   comparisons[name]=dict(accuracy_gain_equal_domain=float(np.mean([acc[d.numpy()==int(dom)].mean() for dom in d.unique()])),CE_gain_equal_domain=float(np.mean([ce[d.numpy()==int(dom)].mean() for dom in d.unique()])),accuracy_gain_descriptive_image_bootstrap_95=np.quantile(boot[:,0],[.025,.975]).tolist(),CE_gain_descriptive_image_bootstrap_95=np.quantile(boot[:,1],[.025,.975]).tolist(),learned_repairs_absent_in_control=int((~maskg&maskf&~masko).sum()),learned_harms_control_avoids=int((maskg&~maskf&masko).sum()))
  res[state]=comparisons
 write(OUT/'paired_source_evidence.json',dict(states=res,source_only=True,target_read=False,fitting=False,bootstrap='10000 within-domain image resamples, seed0; descriptive only, camera/time clustering unadjusted; repeated inspected source validation.',code_sha256=digest(Path(__file__))))
 print(json.dumps(res,indent=2))
if __name__=='__main__':main()

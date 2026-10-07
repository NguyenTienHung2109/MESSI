"""Source-only fixed-competitor margins and utility concentration from locked replay."""
import json
from pathlib import Path
import numpy as np
from .routed_discovery_run import DEST
from .shrinkage import write,digest

def run():
 p=DEST/'eda/source_out_replay.npz';a=np.load(p);g=a['B0'];y=a['y'];ix=np.arange(len(y));c=g.copy();c[ix,y]=-np.inf;comp=c.argmax(1);gap=g[ix,y]-g[ix,comp];d=a['OFF_delta'];pi=a['OFF_pi'];term=d*pi[:,:6,None];margins=term[ix,:,y]-term[ix,:,comp];f=a['B1'];wrong=g.argmax(1)!=y;u=a['B1_utility'];rows=[]
 for i,id in enumerate(a['ids']):
  changed=f[i].argmax()!=g[i].argmax()
  if changed or y[i]==0:
   rows.append(dict(image_id=str(id),domain=int(a['domains'][i]),label=int(y[i]),global_prediction=int(g[i].argmax()),full_prediction=int(f[i].argmax()),global_gap=float(gap[i]),composed_fixed_competitor_margin=float(margins[i].sum()),expert_margin_contributions=margins[i].tolist(),utilities=u[i].tolist(),routing=pi[i].tolist(),unscaled_capacity_experts=np.flatnonzero((g[i,None]+d[i]).argmax(1)==y[i]).tolist(),routed_capacity_experts=np.flatnonzero((g[i,None]+term[i]).argmax(1)==y[i]).tolist()))
 concentration={}
 for dom in [0,2,3]:
  z=a['domains']==dom;concentration[str(dom)]={k:dict(n=int(mask.sum()),U=u[mask].mean(0).tolist()) for k,mask in [('errors',z&wrong),('correct',z&~wrong)]}
 result=dict(source_only=True,target_read=False,fitting=False,checkpoint_selection_changed=False,replay_sha256=digest(p),cases=rows,per_domain_error_correct_utilities=concentration,scope='Margins against the frozen-global highest nontrue competitor, not necessarily current full competitor. Capacity uses known labels and is not deployable.')
 write(DEST/'eda/source_case_margins.json',result);return result
if __name__=='__main__':run()

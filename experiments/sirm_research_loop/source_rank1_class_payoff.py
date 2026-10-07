"""Class-conditional source payoff of saved diagnostic readers; no fitting."""
import json
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from .shrinkage import ROOT,write,digest
from .source_low_rank_readout import CACHE
DEST=ROOT/'results/sirm_research_loop/iteration_19/source_class_direction_payoff'
def main():
 torch.set_num_threads(4);DEST.mkdir(exist_ok=True);assert not (DEST/'results.json').exists()
 data={s:dict(np.load(CACHE/f'{s}_tokens.npz')) for s in ['source_in','source_out']};raw={s:torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)) for s,a in data.items()};root=DEST.parent;locks={str(CACHE/f'{s}_tokens.npz'):digest(CACHE/f'{s}_tokens.npz') for s in data}
 families={'risk_learned':('source_rank1_risk_axis','learned_risk_axis'),'risk_fixed':('source_rank1_risk_axis','fixed_risk_axis'),'agreement_learned':('source_rank1_domain_agreement','learned_minimum_norm_convex'),'agreement_fixed':('source_rank1_domain_agreement','fixed_minimum_norm_convex'),'random0_rank1':('source_low_rank_readout','rank1'),'random0_rank9':('source_low_rank_readout','rank9')}
 plan=dict(source_only=True,target_read=False,fitting=False,hypothesis='Positive pooled reader payoff can conceal class-specific harms and source-in to source-out utility sign reversals. Identify observed common beneficial class groups before any new discovery objective.',families=families,scope='24 saved heads:6families xpooled/3auxiliaryexclusions. Original deterministic views. Source-in fitting subset; source-out evaluated domain for exclusion. No target, model selection or updates.',minimum2='All cells retain counts; shared-positive descriptive cells require at least2images per domain/class on source-out. Not statistically independent/causal/sharedfactor proof.',utility='CE(global)-CE(global+correction). No routing or individual expert LOO here. Equal-domain primary accuracy; cell/cohort utilities descriptive.',code_sha256=digest(Path(__file__)))
 write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());out={};rows=[];cases=[]
 for family,(folder,suffix) in families.items():
  out[family]={}
  for fit in ['pooled','held0','held2','held3']:
   held=None if fit=='pooled' else int(fit[-1]);p=root/folder/f'{fit}_{suffix}.pt';locks[str(p)]=digest(p);a=torch.load(p,map_location='cpu',weights_only=False);assert a['steps']==400;w=(a['b']-a['b'].mean(0,keepdim=True))@a['a'];cells={};cohorts={}
   for split,v in data.items():
    mask=np.ones(len(v['y']),bool) if (held is None) else ((v['domains']!=held) if split=='source_in' else v['domains']==held);z=(raw[split][mask]-a['mean'])/a['std']/2**.5;g=torch.tensor(v['global_logits'][mask]);y=torch.tensor(v['y'][mask]);dom=torch.tensor(v['domains'][mask]);delta=F.linear(z,w);f=g+delta;gp=g.argmax(1);fp=f.argmax(1);gc=gp.eq(y);fc=fp.eq(y);u=F.cross_entropy(g,y,reduction='none')-F.cross_entropy(f,y,reduction='none');cells[split]={}
    for d in dom.unique():
     for c in y.unique():
      q=dom.eq(d)&y.eq(c)
      if not q.any():continue
      row=dict(family=family,fit=fit,split=split,domain=int(d),label=int(c),n=int(q.sum()),U=float(u[q].mean()),global_correct=int(gc[q].sum()),full_correct=int(fc[q].sum()),repairs=int((q&~gc&fc).sum()),harms=int((q&gc&~fc).sum()),negative_utility_mass=float((-u[q]).clamp_min(0).sum()),positive_utility_mass=float(u[q].clamp_min(0).sum()))
      cells[split][f'{int(d)},{int(c)}']=row;rows.append(row)
    cohorts[split]={}
    for name,q in [('global_correct',gc),('global_wrong',~gc)]:cohorts[split][name]=dict(n=int(q.sum()),mean_U=float(u[q].mean()),negative_utility_mass=float((-u[q]).clamp_min(0).sum()),positive_utility_mass=float(u[q].clamp_min(0).sum()))
    if split=='source_out' and held is None:
     for i in range(len(y)):
      if fp[i]!=gp[i]:cases.append(dict(family=family,id=str(v['ids'][mask][i]),label=int(y[i]),global_prediction=int(gp[i]),prediction=int(fp[i]),U=float(u[i]),repair=bool(~gc[i]&fc[i]),harm=bool(gc[i]&~fc[i])))
   common=set(cells['source_in'])&set(cells['source_out']);eligible=[k for k in common if cells['source_in'][k]['n']>=2 and cells['source_out'][k]['n']>=2];reversals=[k for k in eligible if cells['source_in'][k]['U']>0 and cells['source_out'][k]['U']<0];shared={}
   for c in range(10):
    positive=[int(k.split(',')[0]) for k,v in cells['source_out'].items() if int(k.split(',')[1])==c and v['n']>=2 and v['U']>0];shared[str(c)]=positive
   out[family][fit]=dict(cells=cells,cohorts=cohorts,eligible_train_validation_cells=len(eligible),positive_train_negative_validation_cells=reversals,source_out_positive_domains_by_class=shared)
 for p,h in locks.items():assert digest(Path(p))==h
 write(DEST/'results.json',dict(source_only=True,target_read=False,fitting=False,families=out,input_sha256=locks,scientific_goal_achieved=False));write(DEST/'cell_records.json',rows);write(DEST/'changed_cases.json',cases)
 summary={k:dict(reversals=v['pooled']['positive_train_negative_validation_cells'],cohorts=v['pooled']['cohorts'],positive_domains=v['pooled']['source_out_positive_domains_by_class']) for k,v in out.items()};write(DEST/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

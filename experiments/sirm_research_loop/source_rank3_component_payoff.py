"""Saved rank3 component LOO utility and exact predictor-preserving basis controls."""
import json,itertools
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from .source_low_rank_readout import CACHE
from .source_rank3_risk_subspace import DEST
from .shrinkage import ROOT,write,digest
OUT=DEST.parent/'source_rank3_component_payoff'
def main():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();data={s:dict(np.load(CACHE/f'{s}_tokens.npz')) for s in ['source_in','source_out']};raw={s:torch.tensor(np.concatenate([x['cls'],x['patch_max']],1)) for s,x in data.items()};locks={str(CACHE/f'{s}_tokens.npz'):digest(CACHE/f'{s}_tokens.npz') for s in data};plan=dict(source_only=True,target_read=False,fitting=False,hypothesis='Useful composition does not necessarily identify useful components: saved lowrank head factorization may alter LOO utilities under predictor-preserving rotations.',heads='8saved rank3 heads, learned/fixed xpooled/held0/2/3',bases=['parameter_columns','effective_W_SVD','SVD_rotation_seed0','SVD_rotation_seed1','SVD_rotation_seed2'],rotation='Bprime=B R,Aprime=R.T A for orthogonal3x3R; exact total correction. SVDfactorization of effectiveW canonical when singularvaluesdistinct, notsemanticfactorproof.',U='CE(full minus component)-CE(full), original views, mean per domain',support='Diagnostic U>.002 AND double-centeredU>.002, minimum2domainrows; NOT trained Q/EMA, eligibility/ESS or SIRM execution. Source-out never choosesbasis/support for deployment.',removal='Full plus all7 nonempty proper component subsets, sourceonly descriptive; no bestmask selection',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());records={};cases=[];maxerr=0
 def score(g,delta,y,dom):
  f=g+delta;fc=f.argmax(1).eq(y);gc=g.argmax(1).eq(y);ce=F.cross_entropy(f,y,reduction='none');return dict(accuracy=float(torch.stack([fc[dom.eq(d)].float().mean() for d in dom.unique()]).mean()),CE=float(torch.stack([ce[dom.eq(d)].mean() for d in dom.unique()]).mean()),repairs=int((~gc&fc).sum()),harms=int((gc&~fc).sum()))
 for fit in ['pooled','held0','held2','held3']:
  held=None if fit=='pooled' else int(fit[-1]);records[fit]={}
  for policy in ['learned','fixed']:
   path=DEST/f'{fit}_{policy}_risk_axis.pt';locks[str(path)]=digest(path);a=torch.load(path,map_location='cpu',weights_only=False);bb=a['b']-a['b'].mean(0,keepdim=True);aa=a['a'];w=bb@aa;uu,sv,vh=torch.linalg.svd(w,full_matrices=False);b=uu[:,:3];coef=sv[:3,None]*vh[:3];assert sv[3:].max()<1e-6;bases={'parameter_columns':(bb,aa),'effective_W_SVD':(b,coef)}
   for seed in [0,1,2]:
    r=torch.linalg.qr(torch.randn(3,3,generator=torch.Generator().manual_seed(seed)))[0];bases[f'SVD_rotation_seed{seed}']=(b@r,r.T@coef)
   state={}
   for name,(b,c) in bases.items():
    splits={}
    for split,d in data.items():
     mask=np.ones(len(d['y']),bool) if held is None else ((d['domains']!=held) if split=='source_in' else d['domains']==held);z=(raw[split][mask]-a['mean'])/a['std']/2**.5;g=torch.tensor(d['global_logits'][mask]);y=torch.tensor(d['y'][mask]);dom=torch.tensor(d['domains'][mask]);comp=(z@c.T)[:,:,None]*b.T[None,:,:];total=comp.sum(1);reference=F.linear(z,w);err=float((total-reference).abs().max());maxerr=max(maxerr,err);assert err<2e-5;full=g+total;ce=F.cross_entropy(full,y,reduction='none');utilities=torch.stack([F.cross_entropy(full-comp[:,m],y,reduction='none')-ce for m in range(3)],1);domains=dom.unique();u=torch.stack([utilities[dom.eq(dd)].mean(0) for dd in domains]);v=u-u.mean(1,keepdim=True)-u.mean(0,keepdim=True)+u.mean();selected=(u>.002)&(v>.002);summary=dict(U=u.tolist(),interaction=v.tolist(),domains=domains.tolist(),diagnostic_selected_cells=int(selected.sum()),diagnostic_shared_components=int((selected.sum(0)>=2).sum()),positive_U_shared_components=int((u.gt(.002).sum(0)>=2).sum()),selected_mask=selected.tolist(),full=score(g,total,y,dom),global_control=score(g,torch.zeros_like(total),y,dom),subsets={})
     for count in range(1,4):
      for subset in itertools.combinations(range(3),count):summary['subsets'][','.join(map(str,subset))]=score(g,comp[:,list(subset)].sum(1),y,dom)
     splits[split]=summary
     if split=='source_out' and fit=='pooled' and policy=='learned':
      for i,id in enumerate(d['ids'][mask]):
       pred=int(full[i].argmax());gp=int(g[i].argmax());gt=int(y[i])
       if pred!=gp:cases.append(dict(basis=name,id=str(id),label=gt,global_prediction=gp,full_prediction=pred,component_U=utilities[i].tolist(),without_component_predictions=[int((full[i]-comp[i,m]).argmax()) for m in range(3)],repair=gp!=gt and pred==gt,harm=gp==gt and pred!=gt))
    state[name]=splits
   records[fit][policy]=dict(singular_values=sv.tolist(),bases=state)
 for p,h in locks.items():assert digest(Path(p))==h
 write(OUT/'results.json',dict(source_only=True,target_read=False,fitting=False,records=records,predictor_reconstruction_max_abs_error=maxerr,input_sha256=locks,scientific_goal_achieved=False));write(OUT/'changed_cases.json',cases)
 for fit,x in records.items():
  for policy,v in x.items():print(fit,policy,[(k,r['source_out']['diagnostic_shared_components'],r['source_out']['positive_U_shared_components']) for k,r in v['bases'].items()])
if __name__=='__main__':main()

"""Rank3 geometry and train-amplitude controls at saved heads; no fitting."""
import json
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from scipy.optimize import linprog
from .source_low_rank_readout import CACHE
from .source_rank3_risk_subspace import DEST
from .shrinkage import write,digest

def main():
 torch.set_num_threads(4);data={s:dict(np.load(CACHE/f'{s}_tokens.npz')) for s in ['source_in','source_out']};raw={s:torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)) for s,a in data.items()};g=torch.tensor(data['source_out']['global_logits']);y=torch.tensor(data['source_out']['y']);dom=torch.tensor(data['source_out']['domains']);gc=g.argmax(1).eq(y);records={};locks={}
 for fit in ['pooled','held0','held2','held3']:
  held=None if fit=='pooled' else int(fit[-1]);keep=np.ones(len(data['source_in']['y']),bool) if held is None else data['source_in']['domains']!=held;take=torch.ones(len(y),dtype=torch.bool) if held is None else dom.eq(held);heads={'rank3':DEST/f'{fit}_learned_risk_axis.pt','rank1':DEST.parent/'source_rank1_risk_axis'/f'{fit}_learned_risk_axis.pt','rank9':DEST.parent/'source_low_rank_readout'/f'{fit}_rank9.pt'};deltas={};norms={}
  for name,path in heads.items():
   locks[str(path)]=digest(path);a=torch.load(path,map_location='cpu',weights_only=False);b=a['b']-a['b'].mean(0,keepdim=True);w=b@a['a'];z=(raw['source_in'][keep]-a['mean'])/a['std']/2**.5;v=(raw['source_out']-a['mean'])/a['std']/2**.5;norms[name]=float(F.linear(z,w).square().mean().sqrt());deltas[name]=F.linear(v,w)
   if name=='rank3' and held is None:basis=torch.linalg.qr(b)[0].numpy().astype(float)
  variants=list(deltas.items())+[('rank3_to_rank1_trainRMS',deltas['rank3']*norms['rank1']/norms['rank3']),('rank1_to_rank3_trainRMS',deltas['rank1']*norms['rank3']/norms['rank1']),('rank3_to_rank9_trainRMS',deltas['rank3']*norms['rank9']/norms['rank3']),('rank9_to_rank3_trainRMS',deltas['rank9']*norms['rank3']/norms['rank9'])];controls={}
  for name,d in variants:
   f=g+d;fc=f.argmax(1).eq(y);ce=F.cross_entropy(f,y,reduction='none');controls[name]=dict(accuracy=float(torch.stack([fc[take&dom.eq(dd)].float().mean() for dd in dom[take].unique()]).mean()),CE=float(torch.stack([ce[take&dom.eq(dd)].mean() for dd in dom[take].unique()]).mean()),repairs=int((take&~gc&fc).sum()),harms=int((take&gc&~fc).sum()))
  records[fit]=dict(train_RMS=norms,controls=controls)
 cases=[];feasible=[];witness_max_error=0
 for i in range(len(y)):
  gt=int(y[i]);gg=g[i].numpy().astype(float);others=[j for j in range(10) if j!=gt];mat=basis[others]-basis[gt];rhs=gg[gt]-gg[others];lp=linprog(np.zeros(3),A_ub=mat,b_ub=rhs,bounds=[(None,None)]*3,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9});assert lp.status in [0,2],(i,lp.message)
  possible=lp.status==0
  if possible:
   error=float(max(0,np.max(mat@lp.x-rhs)));assert error<1e-6;witness_max_error=max(witness_max_error,error)
  feasible.append(possible);cases.append(dict(id=str(data['source_out']['ids'][i]),label=gt,global_prediction=int(gg.argmax()),reachable_in_saved_rank3_subspace=possible,witness=lp.x.tolist() if possible else None))
 f=np.array(feasible);assert f[gc.numpy()].all();summary=dict(global_errors=70,reachable_errors=int((~gc.numpy()&f).sum()),unreachable_errors=int((~gc.numpy()&~f).sum()),witness_max_constraint_error=witness_max_error,per_class={str(c):dict(global_errors=int(((y.numpy()==c)&~gc.numpy()).sum()),reachable_errors=int(((y.numpy()==c)&~gc.numpy()&f).sum())) for c in range(10)},label_aware_oracle_not_deployable=True)
 for path,h in locks.items():assert digest(Path(path))==h
 write(DEST/'capacity_and_amplitude_controls.json',dict(source_only=True,target_read=False,fitting=False,fits=records,rank3_feasibility=summary,cases=cases,input_sha256=locks,method='Unbounded label-dependent3Dcoefficient feasibility LP, tiesallowed, checked real witnesses; only bounds saved output subspace, not deployed prediction performance. Amplitudes computed on fit-source-in only.'))
 print(json.dumps({'pooled':records['pooled'],'feasibility':summary},indent=2))
if __name__=='__main__':main()

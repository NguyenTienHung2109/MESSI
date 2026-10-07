"""Train-only CLS-conditioned patch reader with raw/CLS/rotated-subtraction controls."""
import hashlib,json
from pathlib import Path
import numpy as np,torch
from torch import nn
import torch.nn.functional as F
from .source_low_rank_readout import CACHE
from .shrinkage import ROOT,write,digest
DEST=ROOT/'results/sirm_research_loop/iteration_19/source_conditional_patch_readout'
class Head(nn.Module):
 def __init__(self,b,learned):
  super().__init__();self.a=nn.Parameter(torch.zeros(3,384));self.b=nn.Parameter(b.clone(),requires_grad=learned)
 def weight(self):return (self.b-self.b.mean(0,keepdim=True))@self.a
 def forward(self,z):return F.linear(z,self.weight())
def score(f,g,y,d):
 c=f.argmax(1).eq(y);gc=g.argmax(1).eq(y);ce=F.cross_entropy(f,y,reduction='none');return dict(accuracy=float(torch.stack([c[d.eq(dd)].float().mean() for dd in d.unique()]).mean()),CE=float(torch.stack([ce[d.eq(dd)].mean() for dd in d.unique()]).mean()),repairs=int((~gc&c).sum()),harms=int((gc&~c).sum()))
def main():
 torch.set_num_threads(4);DEST.mkdir(exist_ok=True);assert not (DEST/'results.json').exists()
 data={s:dict(np.load(CACHE/f'{s}_tokens.npz')) for s in ['source_in','source_out']};g={s:torch.tensor(v['global_logits']) for s,v in data.items()};y={s:torch.tensor(v['y']) for s,v in data.items()};dom={s:torch.tensor(v['domains']) for s,v in data.items()};raw={s:{k:torch.tensor(v[k]) for k in ['cls','patch_max']} for s,v in data.items()};assert not set(data['source_in']['ids'])&set(data['source_out']['ids'])
 plan=dict(source_only=True,target_read=False,primary_SIRM_training=False,previous_goal_turn='progress: saved rank3 channel decomposition and source-in RMS controls',hypothesis='Patch residual beyond linear CLS predictability may support complementary readout with fewer CLS-like harms. Compare raw patch, CLS, conditional patch, and norm-matched random-rotation subtraction before primary integration.',families=['CLS','patch_raw','patch_conditional','patch_sham'],fits='4 families xlearned/fixed source-risk rank3 xpooled/held0/2/3 =32fits',regression='Fit-domain source-in standardized CLS C and patch P; equal-domain ridge R=solve(C.T D C+I,C.T D P), lambda1 fixed. Conditional=P-C R. Sham=P-C R Q for seeded384orthogonal Q; predicted component norm identical before final train-only standardization.',normalization='Each384channel centered/std floor1e-6 on fit-domain source-in, divided sqrt2; conditional normalization changes implicit weight regularization. No claim of removing all shared or domain information.',basis='Top3 left SVD of equal-domain source-in (onehot_y-softmax_global) outer normalized input, centeredQR/sign convention. Zero input coefficient. Learned/fixed share initialization.',training='400updates,32/domain seed0 identical schedules; CE+KL(global||full)coef1+.01 effectiveW squared, Adam.001,wd0; final399only',selection='No source-out family/basis/scale/checkpoint selection; target unread',amplitude='All4families scaled to mean source-in correction RMS per fit/policy, report all',input_sha256={str(CACHE/f'{s}_tokens.npz'):digest(CACHE/f'{s}_tokens.npz') for s in data},code_sha256=digest(Path(__file__)))
 write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());results={}
 rotation=torch.linalg.qr(torch.randn(384,384,generator=torch.Generator().manual_seed(0)))[0]
 for held in [None,0,2,3]:
  fit='pooled' if held is None else f'held{held}';keep=torch.ones(len(y['source_in']),dtype=torch.bool) if held is None else dom['source_in'].ne(held);take=torch.ones(len(y['source_out']),dtype=torch.bool) if held is None else dom['source_out'].eq(held);idx=torch.nonzero(keep).flatten();ds=dom['source_in'][keep].unique();normalized={};stats={}
  for name in ['cls','patch_max']:
   mean=raw['source_in'][name][keep].mean(0);std=raw['source_in'][name][keep].std(0,unbiased=False).clamp_min(1e-6);stats[name]=(mean,std);normalized[name]={s:(raw[s][name]-mean)/std for s in data}
  weights=torch.zeros(int(keep.sum()),dtype=torch.float64);dk=dom['source_in'][keep]
  for dd in ds:weights[dk.eq(dd)]=1/(len(ds)*int(dk.eq(dd).sum()))
  c=normalized['cls']['source_in'][keep].double();p=normalized['patch_max']['source_in'][keep].double();ridge=torch.linalg.solve(c.T@(weights[:,None]*c)+torch.eye(384,dtype=torch.float64),c.T@(weights[:,None]*p)).float();res={s:normalized['patch_max'][s]-normalized['cls'][s]@ridge for s in data};sham={s:normalized['patch_max'][s]-(normalized['cls'][s]@ridge)@rotation for s in data}
  pre={'CLS':normalized['cls'],'patch_raw':normalized['patch_max'],'patch_conditional':res,'patch_sham':sham};z={};family_stats={}
  for name,x in pre.items():
   mean=x['source_in'][keep].mean(0);std=x['source_in'][keep].std(0,unbiased=False).clamp_min(1e-6);z[name]={s:(xx-mean)/std/2**.5 for s,xx in x.items()};family_stats[name]=(mean,std)
  gen=torch.Generator().manual_seed(0);schedule=[torch.cat([ii[torch.randint(len(ii),(32,),generator=gen)] for dd in ds for ii in [idx[dom['source_in'][idx].eq(dd)]]]) for _ in range(400)];schedule_sha=hashlib.sha256(torch.stack(schedule).numpy().tobytes()).hexdigest();reference=json.loads((DEST.parent/'source_rank3_risk_subspace/results.json').read_text())['fits'][fit]['learned_risk_axis']['schedule_sha256'];assert schedule_sha==reference
  gradient_audit={};fitres={'regression':{},'families':{},'schedule_sha256':schedule_sha}
  for split,mask in [('source_in',keep),('source_out',take)]:
   pp=normalized['patch_max'][split][mask];cr=normalized['cls'][split][mask]@ridge;fitres['regression'][split]={'raw_patch_MSE':float(pp.square().mean()),'conditional_patch_MSE':float((pp-cr).square().mean()),'sham_patch_MSE':float((pp-cr@rotation).square().mean()),'subtracted_norm_rotation_max_error':float((cr.norm(dim=1)-(cr@rotation).norm(dim=1)).abs().max())}
  deltas={};norms={}
  for name in pre:
   train=z[name]['source_in'];desired=F.one_hot(y['source_in'],10).float()-F.softmax(g['source_in'],1);moment=torch.stack([desired[keep&dom['source_in'].eq(dd)].T@train[keep&dom['source_in'].eq(dd)]/int((keep&dom['source_in'].eq(dd)).sum()) for dd in ds]).mean(0);u,sv,vh=torch.linalg.svd(moment,full_matrices=False);b=u[:,:3];b=torch.linalg.qr(b-b.mean(0,keepdim=True))[0]
   for j in range(3):
    if b[b[:,j].abs().argmax(),j]<0:b[:,j]*=-1
   fitres['families'][name]={}
   for policy in ['learned','fixed']:
    h=Head(b,policy=='learned');assert h(train[:2]).count_nonzero()==0;opt=torch.optim.Adam([p for p in h.parameters() if p.requires_grad],lr=.001)
    for ix in schedule:
     f=g['source_in'][ix]+h(train[ix]);loss=F.cross_entropy(f,y['source_in'][ix])+F.kl_div(F.log_softmax(f,1),F.softmax(g['source_in'][ix],1),reduction='batchmean')+.01*h.weight().square().sum();assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
     dt=h(train[keep]);dv=h(z[name]['source_out'][take]);nr=float(dt.square().mean().sqrt());r={'source_in':score(g['source_in'][keep]+dt,g['source_in'][keep],y['source_in'][keep],dom['source_in'][keep]),'source_out':score(g['source_out'][take]+dv,g['source_out'][take],y['source_out'][take],dom['source_out'][take]),'source_in_RMS':nr,'risk_gradient_top3_energy':float(sv[:3].square().sum()/sv.square().sum())};fitres['families'][name][policy]=r;deltas[name,policy]=dv;norms[name,policy]=nr
    torch.save(dict(a=h.a.detach(),b=h.b.detach(),input_stats=stats,family_stats=family_stats[name],ridge=ridge,rotation=rotation,family=name,steps=400,held=held,policy=policy,schedule_sha256=schedule_sha),DEST/f'{fit}_{name}_{policy}.pt');np.savez_compressed(DEST/f'{fit}_{name}_{policy}_source_out.npz',ids=data['source_out']['ids'][take.numpy()],y=y['source_out'][take].numpy(),domains=dom['source_out'][take].numpy(),global_logits=g['source_out'][take].numpy(),full_logits=(g['source_out'][take]+dv).numpy());write(DEST/'progress.json',dict(status='fit_complete',fit=fit,family=name,policy=policy));print(fit,name,policy,r['source_out'],flush=True)
  for policy in ['learned','fixed']:
   common=np.mean([norms[name,policy] for name in pre])
   for name in pre:
    scale=common/max(norms[name,policy],1e-12);fitres['families'][name][policy]['common_source_in_RMS_control']={'scale':float(scale),'source_out':score(g['source_out'][take]+deltas[name,policy]*scale,g['source_out'][take],y['source_out'][take],dom['source_out'][take])}
  fitres['global']=score(g['source_out'][take],g['source_out'][take],y['source_out'][take],dom['source_out'][take]);results[fit]=fitres;write(DEST/'partial_results.json',results)
 for p,h in plan['input_sha256'].items():assert digest(Path(p))==h
 write(DEST/'results.json',dict(source_only=True,target_read=False,primary_SIRM_training=False,fits=results,input_sha256=plan['input_sha256'],scientific_goal_achieved=False));write(DEST/'progress.json',dict(status='complete'))
if __name__=='__main__':main()

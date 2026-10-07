"""Fixed-axis control for learned lowrank source-only additive readers."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
from torch import nn
import torch.nn.functional as F
from .shrinkage import ROOT,write,digest
DEST=ROOT/'results/sirm_research_loop/iteration_19/source_low_rank_readout_fixed_axis';CACHE=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis'
class Reader(nn.Module):
 def __init__(self,rank):
  super().__init__();self.a=nn.Parameter(torch.zeros(rank,768));basis=torch.zeros(10,9)
  for k in range(9):basis[:k+1,k]=1/((k+1)*(k+2))**.5;basis[k+1,k]=-(k+1)/((k+1)*(k+2))**.5
  # Fixed seeded rotation gives a non-class-selected rank1 axis, shared with rank9.
  gen=torch.Generator().manual_seed(0);rot=torch.linalg.qr(torch.randn(9,9,generator=gen))[0];self.b=nn.Parameter((basis@rot)[:,:rank].clone(),requires_grad=False)
 def weight(self):return (self.b-self.b.mean(0,keepdim=True))@self.a
 def forward(self,x):return F.linear(x,self.weight())
def main():
 torch.set_num_threads(4);DEST.mkdir(exist_ok=True);data={s:dict(np.load(CACHE/f'{s}_tokens.npz')) for s in ['source_in','source_out']};x={s:torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)) for s,a in data.items()};g={s:torch.tensor(a['global_logits']) for s,a in data.items()};y={s:torch.tensor(a['y']) for s,a in data.items()};dom={s:torch.tensor(a['domains']) for s,a in data.items()};assert not set(data['source_in']['ids'])&set(data['source_out']['ids'])
 plan=dict(hypothesis='Control whether the observed lowrank results require a learned output direction rather than the shared seeded fixed axis',source_only=True,target_read=False,training='Diagnostic readout fits only, not new bank/discovery/SIRM',fits='rank1/rank9 xpooled/3auxiliary-heldsource domains; same indices/loss/fixed400update/final399',architecture='delta=(B-centerclasses(B))Ax; no intercept; A zero, B seeded rotated Helmert columns. TrainA only; B frozen at exact shared initial axis. Global untouched. Rank9 control uses samefirstaxis plus8remaining.',normalization='Excluded-fit-source-in mean/std floor1e-6/sqrt2',loss='Equal-domain CE(global+delta)+KL(global||global+delta) coefficient1 +.01 effective-W Frobenius squared; effective-W regularization avoids factor-rescaling penalty confound',optimizer='Adam.001 weight_decay0,400updates,32 images/domain withreplacement, seed0',minimum_samples=2,rank_selected_by_validation=False,annotation_used=False,selection='Final400update only, source-out reporting after fit; no target',input_sha256={s:digest(CACHE/f'{s}_tokens.npz') for s in data},code_sha256=digest(Path(__file__)))
 assert not (DEST/'results.json').exists()
 write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());result={}
 for held in [None,0,2,3]:
  fit='pooled' if held is None else f'held{held}';keep=torch.ones(len(y['source_in']),dtype=torch.bool) if held is None else dom['source_in'].ne(held);idx=torch.nonzero(keep).flatten();domains=dom['source_in'][keep].unique();assert all(int(dom['source_in'][keep].eq(d).sum())>=2 for d in domains);mean=x['source_in'][keep].mean(0);std=x['source_in'][keep].std(0,unbiased=False).clamp_min(1e-6);z=(x['source_in']-mean)/std/2**.5;v=(x['source_out']-mean)/std/2**.5;gen=torch.Generator().manual_seed(0);schedule=[torch.cat([ii[torch.randint(len(ii),(32,),generator=gen)] for d in domains for ii in [idx[dom['source_in'][idx].eq(d)] ]]) for _ in range(400)];schedule_hash=hashlib.sha256(torch.stack(schedule).numpy().tobytes()).hexdigest();take=torch.ones(len(v),dtype=torch.bool) if held is None else dom['source_out'].eq(held);states={};trainnorm={};validationdelta={}
  for rank in [1,9]:
   head=Reader(rank);assert torch.equal(head(z[:2]),torch.zeros(2,10));opt=torch.optim.Adam(head.parameters(),lr=.001);curve=[]
   for step,ix in enumerate(schedule):
    base=g['source_in'][ix];f=base+head(z[ix]);ce=F.cross_entropy(f,y['source_in'][ix]);kl=F.kl_div(F.log_softmax(f,1),F.softmax(base,1),reduction='batchmean');reg=.01*head.weight().square().sum();loss=ce+kl+reg;assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
    if step%25==0:curve.append(dict(step=step,CE=float(ce),KL=float(kl),reg=float(reg)))
   with torch.no_grad():dt=head(z[keep]);delta=head(v);trainnorm[rank]=float(dt.square().mean().sqrt());validationdelta[rank]=delta;ft=g['source_in'][keep]+dt;sv=torch.linalg.svdvals(head.weight());training=dict(accuracy=float(ft.argmax(1).eq(y['source_in'][keep]).float().mean()),CE=float(F.cross_entropy(ft,y['source_in'][keep])),singular_values=sv.tolist())
   torch.save(dict(a=head.a.detach(),b=head.b.detach(),mean=mean,std=std,held=held,rank=rank,schedule_sha256=schedule_hash,steps=400),DEST/f'{fit}_rank{rank}.pt');states[f'rank{rank}']=dict(training=training,curve=curve)
  controls={};base=g['source_out'];yy=y['source_out'];dd=dom['source_out'];gc=base.argmax(1).eq(yy)
  for name,delta in [('global',torch.zeros_like(base)),('rank1',validationdelta[1]),('rank9',validationdelta[9]),('rank1_to_rank9_trainRMS',validationdelta[1]*trainnorm[9]/max(trainnorm[1],1e-12)),('rank9_to_rank1_trainRMS',validationdelta[9]*trainnorm[1]/max(trainnorm[9],1e-12))]:
   f=base+delta;fc=f.argmax(1).eq(yy);ce=F.cross_entropy(f,yy,reduction='none');evalds=dd[take].unique();controls[name]=dict(accuracy=float(torch.stack([fc[take&dd.eq(d)].float().mean() for d in evalds]).mean()),CE=float(torch.stack([ce[take&dd.eq(d)].mean() for d in evalds]).mean()),repairs=int((take&~gc&fc).sum()),harms=int((take&gc&~fc).sum()),correction_RMS=float(delta[take].square().mean().sqrt()));np.savez_compressed(DEST/f'{fit}_{name}_source_out.npz',ids=data['source_out']['ids'][take.numpy()],y=yy[take].numpy(),domains=dd[take].numpy(),global_logits=base[take].numpy(),full_logits=f[take].numpy(),delta=delta[take].numpy())
  result[fit]=dict(schedule_sha256=schedule_hash,train_correction_RMS=trainnorm,states=states,controls=controls);write(DEST/'progress.json',dict(status='fit_complete',fit=fit));print(fit,controls,flush=True)
 for s,sha in plan['input_sha256'].items():assert digest(CACHE/f'{s}_tokens.npz')==sha
 write(DEST/'results.json',dict(source_only=True,target_read=False,annotation_used=False,primary_SIRM_training=False,rank_selected_by_validation=False,plan_sha256=digest(DEST/'plan.json'),fits=result,scientific_goal_achieved=False));write(DEST/'progress.json',dict(status='complete'));print('Learned lowrank diagnostic complete',flush=True)
if __name__=='__main__':main()

"""Source-only top3 risk-gradient subspace test; no target or rank selection."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from .source_low_rank_readout import Reader,CACHE
from .shrinkage import ROOT,write,digest
DEST=ROOT/'results/sirm_research_loop/iteration_19/source_rank3_risk_subspace'
def main():
 torch.set_num_threads(4);DEST.mkdir(exist_ok=True);assert not (DEST/'results.json').exists()
 data={s:dict(np.load(CACHE/f'{s}_tokens.npz')) for s in ['source_in','source_out']};raw={s:torch.tensor(np.concatenate([v['cls'],v['patch_max']],1)) for s,v in data.items()};g={s:torch.tensor(v['global_logits']) for s,v in data.items()};y={s:torch.tensor(v['y']) for s,v in data.items()};dom={s:torch.tensor(v['domains']) for s,v in data.items()}
 plan=dict(source_only=True,target_read=False,annotation_used=False,primary_SIRM_training=False,hypothesis='Rank1 saved axes cannot reach38of70ERM errors while rank9 harms generalization. Test prespecified top3 source-in risk-gradient directions, learned/fixed, with held-domain and amplitude controls.',axis_seeds=[],axis_rule='Train-only feature normalization; M=mean_d mean_i((onehot_y-softmax_global) outer z). B=top3 left singular vectors, centered orthonormal QR, deterministic column signs. A remains zero. No source-out axis selection',random_controls='Existing learned/fixed risk-rank1 and random0-rank9, same schedules. Rank3 prespecified before fitting, no target rank search',previous_axis_seed=0,training_schedule_seed=0,policies=['learned','fixed'],rank=3,fits='pooled and exclusions0/2/3 xlearned/fixed,8fits,400updates',selection='Final399 for every fit; report all seeds, never select seed/rank by validation or target',loss='Same CE+KL(global||full) coef1+.01 effectiveW squared; Adam.001,wd0',inputs={s:digest(CACHE/f'{s}_tokens.npz') for s in data},code_sha256=digest(Path(__file__)))
 write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());out={}
 for held in [None,0,2,3]:
  fit='pooled' if held is None else f'held{held}';keep=torch.ones(len(y['source_in']),dtype=torch.bool) if held is None else dom['source_in'].ne(held);take=torch.ones(len(y['source_out']),dtype=torch.bool) if held is None else dom['source_out'].eq(held);idx=torch.nonzero(keep).flatten();ds=dom['source_in'][keep].unique();mean=raw['source_in'][keep].mean(0);std=raw['source_in'][keep].std(0,unbiased=False).clamp_min(1e-6);z=(raw['source_in']-mean)/std/2**.5;v=(raw['source_out']-mean)/std/2**.5;gen=torch.Generator().manual_seed(0);schedule=[torch.cat([ii[torch.randint(len(ii),(32,),generator=gen)] for d in ds for ii in [idx[dom['source_in'][idx].eq(d)]]]) for _ in range(400)];sha=hashlib.sha256(torch.stack(schedule).numpy().tobytes()).hexdigest();out[fit]={}
  with torch.no_grad():
   desired=F.one_hot(y['source_in'][keep],10).float()-F.softmax(g['source_in'][keep],1)
   zk=z[keep];dk=dom['source_in'][keep];moment=torch.stack([desired[dk.eq(d)].T@zk[dk.eq(d)]/int(dk.eq(d).sum()) for d in ds]).mean(0)
   uu,sv,vh=torch.linalg.svd(moment,full_matrices=False);b0=uu[:,:3];b0=b0-b0.mean(0,keepdim=True);b0=torch.linalg.qr(b0)[0]
   for column in range(3):
    if b0[b0[:,column].abs().argmax(),column]<0:b0[:,column]*=-1
   assert torch.allclose(b0.T@b0,torch.eye(3),atol=1e-6)
   initial_axis_audit=dict(singular_values=sv.tolist(),captured_squared_gradient_fraction=float(sv[:3].square().sum()/sv.square().sum()),class_basis=b0.tolist(),gradient_matrix=moment.tolist())
  for seed in [0]:
   for policy in ['learned','fixed']:
    name=f'{policy}_risk_axis';head=Reader(3)
    with torch.no_grad():head.b.copy_(b0)
    head.b.requires_grad_(policy=='learned');assert head(z[:2]).count_nonzero()==0;opt=torch.optim.Adam([p for p in head.parameters() if p.requires_grad],lr=.001)
    for ix in schedule:
     f=g['source_in'][ix]+head(z[ix]);loss=F.cross_entropy(f,y['source_in'][ix])+F.kl_div(F.log_softmax(f,1),F.softmax(g['source_in'][ix],1),reduction='batchmean')+.01*head.weight().square().sum();assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():delta=head(v);f=g['source_out']+delta;fc=f.argmax(1).eq(y['source_out']);gc=g['source_out'].argmax(1).eq(y['source_out']);ce=F.cross_entropy(f,y['source_out'],reduction='none');evds=dom['source_out'][take].unique();bb=head.b-head.b.mean(0,keepdim=True);qq=torch.linalg.qr(bb)[0];alignment=torch.linalg.svdvals(qq.T@b0);r=dict(accuracy=float(torch.stack([fc[take&dom['source_out'].eq(d)].float().mean() for d in evds]).mean()),CE=float(torch.stack([ce[take&dom['source_out'].eq(d)].mean() for d in evds]).mean()),repairs=int((take&~gc&fc).sum()),harms=int((take&gc&~fc).sum()),train_RMS=float(head(z[keep]).square().mean().sqrt()),mean_subspace_cosine_to_initialization=float(alignment.mean()),class_basis=qq.tolist(),effective_weight_singular_values=torch.linalg.svdvals(head.weight()).tolist(),schedule_sha256=sha)
    r['initial_axis_audit']=initial_axis_audit;out[fit][name]=r;torch.save(dict(a=head.a.detach(),b=head.b.detach(),mean=mean,std=std,steps=400,axis_rule='source_in_CE_gradient',axis_audit=initial_axis_audit,policy=policy,schedule_sha256=sha),DEST/f'{fit}_{name}.pt');np.savez_compressed(DEST/f'{fit}_{name}_source_out.npz',ids=data['source_out']['ids'][take.numpy()],y=y['source_out'][take].numpy(),domains=dom['source_out'][take].numpy(),global_logits=g['source_out'][take].numpy(),full_logits=f[take].numpy());write(DEST/'progress.json',dict(status='fit_complete',fit=fit,policy=name));print(fit,name,{k:v for k,v in r.items() if k not in ['initial_axis_audit','class_basis','effective_weight_singular_values']},flush=True)
 for s,sha in plan['inputs'].items():assert digest(CACHE/f'{s}_tokens.npz')==sha
 write(DEST/'results.json',dict(source_only=True,target_read=False,primary_SIRM_training=False,seed_selection=False,fits=out,scientific_goal_achieved=False));write(DEST/'progress.json',dict(status='complete'))
if __name__=='__main__':main()

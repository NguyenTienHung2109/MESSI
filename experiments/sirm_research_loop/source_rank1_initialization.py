"""Prespecified source-only rank1 initialization sensitivity; no seed selection."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from .source_low_rank_readout import Reader,CACHE
from .shrinkage import ROOT,write,digest
DEST=ROOT/'results/sirm_research_loop/iteration_19/source_rank1_initialization'
def main():
 torch.set_num_threads(4);DEST.mkdir(exist_ok=True);assert not (DEST/'results.json').exists()
 data={s:dict(np.load(CACHE/f'{s}_tokens.npz')) for s in ['source_in','source_out']};raw={s:torch.tensor(np.concatenate([v['cls'],v['patch_max']],1)) for s,v in data.items()};g={s:torch.tensor(v['global_logits']) for s,v in data.items()};y={s:torch.tensor(v['y']) for s,v in data.items()};dom={s:torch.tensor(v['domains']) for s,v in data.items()}
 plan=dict(source_only=True,target_read=False,annotation_used=False,primary_SIRM_training=False,hypothesis='The small rank1 source gain may depend on its arbitrary initialized class contrast. Test sensitivity before claiming discovered shared factors.',axis_seeds=[1,2],previous_axis_seed=0,training_schedule_seed=0,policies=['learned','fixed'],rank=1,fits='pooled and exclusions0/2/3,16fits,400updates',selection='Final399 for every fit; report all seeds, never select seed/rank by validation or target',loss='Same CE+KL(global||full) coef1+.01 effectiveW squared; Adam.001,wd0',inputs={s:digest(CACHE/f'{s}_tokens.npz') for s in data},code_sha256=digest(Path(__file__)))
 write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());out={}
 for held in [None,0,2,3]:
  fit='pooled' if held is None else f'held{held}';keep=torch.ones(len(y['source_in']),dtype=torch.bool) if held is None else dom['source_in'].ne(held);take=torch.ones(len(y['source_out']),dtype=torch.bool) if held is None else dom['source_out'].eq(held);idx=torch.nonzero(keep).flatten();ds=dom['source_in'][keep].unique();mean=raw['source_in'][keep].mean(0);std=raw['source_in'][keep].std(0,unbiased=False).clamp_min(1e-6);z=(raw['source_in']-mean)/std/2**.5;v=(raw['source_out']-mean)/std/2**.5;gen=torch.Generator().manual_seed(0);schedule=[torch.cat([ii[torch.randint(len(ii),(32,),generator=gen)] for d in ds for ii in [idx[dom['source_in'][idx].eq(d)]]]) for _ in range(400)];sha=hashlib.sha256(torch.stack(schedule).numpy().tobytes()).hexdigest();out[fit]={}
  for seed in [1,2]:
   basis=torch.zeros(10,9)
   for k in range(9):basis[:k+1,k]=1/((k+1)*(k+2))**.5;basis[k+1,k]=-(k+1)/((k+1)*(k+2))**.5
   rot=torch.linalg.qr(torch.randn(9,9,generator=torch.Generator().manual_seed(seed)))[0];b0=(basis@rot)[:,:1]
   for policy in ['learned','fixed']:
    name=f'{policy}_seed{seed}';head=Reader(1)
    with torch.no_grad():head.b.copy_(b0)
    head.b.requires_grad_(policy=='learned');assert head(z[:2]).count_nonzero()==0;opt=torch.optim.Adam([p for p in head.parameters() if p.requires_grad],lr=.001)
    for ix in schedule:
     f=g['source_in'][ix]+head(z[ix]);loss=F.cross_entropy(f,y['source_in'][ix])+F.kl_div(F.log_softmax(f,1),F.softmax(g['source_in'][ix],1),reduction='batchmean')+.01*head.weight().square().sum();assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():delta=head(v);f=g['source_out']+delta;fc=f.argmax(1).eq(y['source_out']);gc=g['source_out'].argmax(1).eq(y['source_out']);ce=F.cross_entropy(f,y['source_out'],reduction='none');evds=dom['source_out'][take].unique();b=head.b[:,0]-head.b[:,0].mean();bn=b/b.norm();init=b0[:,0]/b0[:,0].norm();r=dict(accuracy=float(torch.stack([fc[take&dom['source_out'].eq(d)].float().mean() for d in evds]).mean()),CE=float(torch.stack([ce[take&dom['source_out'].eq(d)].mean() for d in evds]).mean()),repairs=int((take&~gc&fc).sum()),harms=int((take&gc&~fc).sum()),train_RMS=float(head(z[keep]).square().mean().sqrt()),axis_cosine_to_initialization=float(bn@init),class_axis=bn.tolist(),schedule_sha256=sha)
    out[fit][name]=r;torch.save(dict(a=head.a.detach(),b=head.b.detach(),mean=mean,std=std,steps=400,axis_seed=seed,policy=policy,schedule_sha256=sha),DEST/f'{fit}_{name}.pt');np.savez_compressed(DEST/f'{fit}_{name}_source_out.npz',ids=data['source_out']['ids'][take.numpy()],y=y['source_out'][take].numpy(),domains=dom['source_out'][take].numpy(),global_logits=g['source_out'][take].numpy(),full_logits=f[take].numpy());write(DEST/'progress.json',dict(status='fit_complete',fit=fit,policy=name));print(fit,name,r,flush=True)
 for s,sha in plan['inputs'].items():assert digest(CACHE/f'{s}_tokens.npz')==sha
 write(DEST/'results.json',dict(source_only=True,target_read=False,primary_SIRM_training=False,seed_selection=False,fits=out,scientific_goal_achieved=False));write(DEST/'progress.json',dict(status='complete'))
if __name__=='__main__':main()

"""Source-only 2x2 optimization diagnostic on immutable iteration18 banks."""
import json,math,hashlib,csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .shrinkage import ROOT,digest,write
from .source_direct_mixture_reliability_router import descriptor,FEATURES
BANK=ROOT/'results/sirm_research_loop/iteration_18'
OUT=BANK/'source_router_conditioning'

def metrics(g,delta,pi,y,dom):
 corr=(delta*pi[:,:6,None]).sum(1);f=g+corr;ok=f.argmax(1).eq(y);base=g.argmax(1).eq(y);ce=F.cross_entropy(f,y,reduction='none');relative=pi[:,:6]/pi[:,:6].sum(1,keepdim=True).clamp_min(1e-12)
 return dict(accuracy_equal_domain=float(torch.stack([ok[dom==d].double().mean() for d in dom.unique()]).mean()),CE_equal_domain=float(torch.stack([ce[dom==d].mean() for d in dom.unique()]).mean()),repairs=int((ok&~base).sum()),harms=int((~ok&base).sum()),correct=int(ok.sum()),n=len(y),router_NULL=float(pi[:,-1].mean()),correction_RMS=float(corr.square().mean().sqrt()),relative_expert_range=float((relative.max(1).values-relative.min(1).values).mean()))

def run():
 torch.set_num_threads(4);torch.manual_seed(0);OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists()
 audit=json.loads((BANK/'final_analysis_audit.json').read_text());assert audit['analysis_complete']
 for p,h in audit['sha256'].items():assert digest(BANK/p)==h,p
 paths=[BANK/'source_analysis'/f'OFF_{state}_{split}_replay.npz' for state in ['selected','last'] for split in ['source_in','source_out']];locks={str(p):digest(p) for p in paths}
 write(OUT/'plan.json',dict(source_only=True,target_read=False,hypothesis='Near-uniform18 router may reflect optimizer step scale or descriptor conditioning, not absence of accessible correction reliability.',primary_bank='selected200',secondary_bank='last399, not selection',factorial=dict(lr=[.00003,.01],normalization=['fixed_anchor_units','train_standardized']),initialization='All12 shared11->1 weights/bias zero;NULL score0. Identical within2x2.',training='400 appended fixed-bank diagnostic updates; actual mixture CE +0.01 weight L2, Adam, original deterministic cached features. Source domain16 correct+16 error per update,minimum2. No early stopping.',holdouts='All-source and3 auxiliary router-fit domain exclusions; backbone/expert bank already saw all3sources.',normalization='Fixed: entropies/log10, margin/delta features/sourcein frozen global class-centeredRMS. Standardized: train-only pooled image/expert descriptor mean/std, recomputed excluding held domain. No source-out fitting.',controls='Native, uniform and learned; native/uniform scaled to learned source-in correctionRMS; learned scaled to native sourceinRMS. Aggregate matching not per-image.',limits='Changes LR/conditioning in diagnostic only. Additional budget, original-only views, L2 and balanced error sampling differ from primary bank. Not active SIRM, factor discovery or OOD confirmation. No target reading or diagnostic-weight reuse.',input_sha256=locks,code_sha256=digest(Path(__file__))))
 (OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());results={};curves=[]
 for state in ['selected','last']:
  data={}
  for split in ['source_in','source_out']:
   a=np.load(BANK/'source_analysis'/f'OFF_{state}_{split}_replay.npz');g=torch.from_numpy(a['global_logits']).double();d=torch.from_numpy(a['residual_logits']).double();data[split]=dict(g=g,d=d,pi=torch.from_numpy(a['router']).double(),y=torch.from_numpy(a['y']).long(),dom=torch.from_numpy(a['domains']).long(),ids=a['ids'],x=descriptor(g,d,d)[:,:,:11])
  unit=(data['source_in']['g']-data['source_in']['g'].mean(1,keepdim=True)).square().mean().sqrt();fits={}
  for held in [None,0,2,3]:
   train=data['source_in'];keep=torch.ones(len(train['y']),dtype=torch.bool) if held is None else train['dom'].ne(held);g=train['g'][keep];d=train['d'][keep];y=train['y'][keep];dom=train['dom'][keep];correct=g.argmax(1).eq(y);gen=torch.Generator().manual_seed(0);cohorts=[]
   for v in dom.unique():
    for flag in [False,True]:
     ix=torch.nonzero((dom==v)&(correct==flag)).flatten();assert len(ix)>=2;cohorts.append(ix)
   schedule=[torch.cat([ix[torch.randint(len(ix),(16,),generator=gen)] for ix in cohorts]) for _ in range(400)];fit={}
   for norm in ['fixed_anchor_units','train_standardized']:
    raw=train['x'][keep];mean=raw.mean((0,1));std=raw.std((0,1),unbiased=False).clamp_min(1e-8)
    def transform(z):
     if norm=='train_standardized':return (z-mean)/std
     z=z.clone();z[:,:,[1,4]]/=math.log(10);z[:,:,[2,5,7,9,10]]/=unit;return z
    x=transform(raw)
    for lr in [.00003,.01]:
     head=torch.nn.Linear(11,1).double();head.weight.data.zero_();head.bias.data.zero_();opt=torch.optim.Adam(head.parameters(),lr=lr);key=f'{norm}_lr{lr:g}'
     def policy(xx):
      s=head(xx).squeeze(-1);return torch.cat([s,s.new_zeros(len(s),1)],1).softmax(1)
     for step,ix in enumerate(schedule):
      pi=policy(x[ix]);loss=F.cross_entropy(g[ix]+(d[ix]*pi[:,:6,None]).sum(1),y[ix])+.01*head.weight.square().sum();opt.zero_grad();loss.backward();grad=float(torch.cat([p.grad.flatten() for p in head.parameters()]).norm());before=torch.cat([p.detach().flatten().clone() for p in head.parameters()]);opt.step()
      if step%50==0 or step==399:
       with torch.no_grad():
        s=head(x).squeeze(-1);m=metrics(g,d,policy(x),y,dom);curves.append(dict(state=state,held=str(held),normalization=norm,lr=lr,step=step,minibatch_loss=float(loss),gradient_norm=grad,update_norm=float((torch.cat([p.detach().flatten() for p in head.parameters()])-before).norm()),expert_contrast_RMS=float((s-s.mean(1,keepdim=True)).square().mean().sqrt()),common_score_mean=float(s.mean()),**m))
     with torch.no_grad():
      trainpi=policy(x);fitrms=((d*trainpi[:,:6,None]).sum(1)).square().mean().sqrt();nativepi=train['pi'][keep];nativer=((d*nativepi[:,:6,None]).sum(1)).square().mean().sqrt();uniform=torch.full_like(nativepi,1/7);unifr=((d*uniform[:,:6,None]).sum(1)).square().mean().sqrt();scores={};replays={}
      for split,a in data.items():
       take=keep if split=='source_in' else (torch.ones(len(a['y']),dtype=torch.bool) if held is None else a['dom'].eq(held));gg=a['g'][take];dd=a['d'][take];yy=a['y'][take];doms=a['dom'][take];pp=policy(transform(a['x'][take]));np0=a['pi'][take];up=torch.full_like(np0,1/7)
       policies={'learned':pp,'native':np0,'uniform':up,'native_same_RMS':np0.clone(),'uniform_same_RMS':up.clone(),'learned_native_RMS':pp.clone()};policies['native_same_RMS'][:,:6]*=fitrms/nativer.clamp_min(1e-12);policies['uniform_same_RMS'][:,:6]*=fitrms/unifr.clamp_min(1e-12);policies['learned_native_RMS'][:,:6]*=nativer/fitrms.clamp_min(1e-12)
       scores[split]={k:metrics(gg,dd,v,yy,doms) for k,v in policies.items()};replays[split]=dict(ids=a['ids'][take.numpy()],y=yy,domains=doms,global_logits=gg,delta=dd,policies={k:gg+(dd*v[:,:6,None]).sum(1) for k,v in policies.items()},learned_pi=pp)
      file=OUT/f'{state}_held{held}_{key}.pt';torch.save(dict(weight=head.weight.detach(),bias=head.bias.detach(),normalization=norm,mean=mean,std=std,unit=unit,replays=replays),file);fit[key]=dict(scores=scores,model_sha256=digest(file));print(state,held,key,scores['source_out']['learned'],flush=True)
   fits[str(held)]=dict(fits=fit,schedule_sha256=hashlib.sha256(torch.stack(schedule).numpy().tobytes()).hexdigest())
  results[state]=fits
 for p,h in locks.items():assert digest(Path(p))==h
 write(OUT/'results.json',dict(states=results,source_only=True,target_read=False,diagnostic_fitting=True,primary_training=False,input_sha256=locks))
 with (OUT/'training_curves.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
 write(OUT/'progress.json',dict(status='complete',fits=32,target_read=False,remaining='paired source error/RMS/holdout analysis before primary changes'))
 print('32 fixed-bank fits complete',flush=True)
if __name__=='__main__':run()

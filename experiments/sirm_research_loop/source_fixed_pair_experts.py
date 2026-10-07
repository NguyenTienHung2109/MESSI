"""Fixed independent class-pair expert identity versus same-span and random controls."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
from torch import nn
import torch.nn.functional as F
from .source_low_rank_readout import CACHE
from .shrinkage import ROOT,write,digest
DEST=ROOT/'results/sirm_research_loop/iteration_19/source_fixed_pair_experts'
class Bank(nn.Module):
 def __init__(self,axes):
  super().__init__();self.register_buffer('axes',axes);self.register_buffer('mean',torch.zeros(6));self.experts=nn.ModuleList([nn.Sequential(nn.Linear(384,64),nn.GELU(),nn.Linear(64,1,bias=False)) for _ in range(6)])
  for expert in self.experts:nn.init.zeros_(expert[-1].weight)
 def scalar(self,z):return torch.cat([e(z) for e in self.experts],1)
 def components(self,z):return (self.scalar(z)-self.mean)[:,:,None]*self.axes[None,:,:]/6**.5
 def forward(self,z):return self.components(z).sum(1)
 @torch.no_grad()
 def calibrate(self,z,domains):self.mean.copy_(torch.stack([self.scalar(z[domains.eq(dd)]).mean(0) for dd in domains.unique()]).mean(0))
def axes_from_errors(g,y,dom):
 pred=g.argmax(1);counts={}
 for a,b in zip(y,pred):
  if a!=b:
   pair=tuple(sorted([int(a),int(b)]));counts[pair]=counts.get(pair,0)+1
 pairs=[];axes=[]
 for pair in sorted(counts,key=lambda p:(-counts[p],p)):
  axis=torch.zeros(10);axis[pair[0]]=1/2**.5;axis[pair[1]]=-1/2**.5;candidate=torch.stack(axes+[axis])
  if int(torch.linalg.matrix_rank(candidate))==len(axes)+1:axes.append(axis);pairs.append(pair)
  if len(axes)==6:break
 assert len(axes)==6;axes=torch.stack(axes);assert int(torch.linalg.matrix_rank(axes))==6
 return axes,[dict(classes=list(p),count=counts[p],domain_counts={str(int(dd)):int((dom.eq(dd)&y.ne(pred)&((y.eq(p[0])&pred.eq(p[1]))|(y.eq(p[1])&pred.eq(p[0])))).sum()) for dd in dom.unique()}) for p in pairs]
def score(f,g,y,d):
 ce=F.cross_entropy(f,y,reduction='none');correct=f.argmax(1).eq(y);gc=g.argmax(1).eq(y);return dict(accuracy=float(torch.stack([correct[d.eq(dd)].float().mean() for dd in d.unique()]).mean()),CE=float(torch.stack([ce[d.eq(dd)].mean() for dd in d.unique()]).mean()),repairs=int((~gc&correct).sum()),harms=int((gc&~correct).sum()))
def main():
 torch.set_num_threads(4);DEST.mkdir(exist_ok=True);assert not (DEST/'results.json').exists();data={s:dict(np.load(CACHE/f'{s}_tokens.npz')) for s in ['source_in','source_out']};locks={str(CACHE/f'{s}_tokens.npz'):digest(CACHE/f'{s}_tokens.npz') for s in data};g={s:torch.tensor(v['global_logits']) for s,v in data.items()};y={s:torch.tensor(v['y']) for s,v in data.items()};d={s:torch.tensor(v['domains']) for s,v in data.items()};raw={s:{k:torch.tensor(v[k]) for k in ['cls','patch_max']} for s,v in data.items()}
 plan=dict(source_only=True,target_read=False,primary_SIRM_training=False,previous_goal_turn='progress:16domain-crossfit patch fits, source EDA and rejection of uniform-transfer remedy',hypothesis='Independent fixed label-pair output axes give unique scalar expert identities, unlike arbitrary rank3 factors. Determine useful class/domain utility and shared support before primary dynamic-Q integration.',novel_vs_prior='Prior24LBFGSreadouts used aggregateSVDspan of top6possibly-dependentpairs. This uses6independent fixed pair axes and6separate nonlinear scalarfunctions, same-spanSVD andrandom6axes controls,400updates. No semantic visual-factor claim from labelaxes.',pair_rule='Greedy independent unit class-pair axes ordered by fit-domain source-in global-error occurrence count, lexicographic ties; recompute for each auxiliary exclusion; minimum2source-in examples per present domain/class.',families=['fixed_pairs','same_span_SVD','random_span'],features='Existing source-in-only CLS-conditioned patch transform; reuse normalization/ridge only, no previous classifier weights',architecture='6independent384->64GELU->1 scalarMLPs; zero finalweights,no outputbias. Unit fixed centeredclass axes, coefficients/ sqrt6, uniform additive weights,no router. Source-in equal-domain scalar means refresh5detached/final.',loss='CE(global+sum components)+KL(global||full)coef1+.01mean(composed correction squared); Adam.001,wd1e-6,400updates,same32/domainseed0schedules,final399',controls='Identical scalar hidden/output initialization acrossfamilies; output-axis orientation changes gradient parameterization. SVD spanssame6classcontrastdimensions; random class-centeredorthonormal6directions seed0.',diagnostic_support='U=CE(full-component)-CE(full); selected U>.002 anddouble-centeredU>.002,shared>=2domainrows. Not learned Q/EMA/ESS/SIRM execution. Source-out never selects a mask.',selection='All12fits reported, no source-out family/basis/checkpoint/scale selection; target unread',input_sha256=locks,code_sha256=digest(Path(__file__)))
 write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());out={}
 for held in [None,0,2,3]:
  fit='pooled' if held is None else f'held{held}';keep=torch.ones(len(y['source_in']),dtype=torch.bool) if held is None else d['source_in'].ne(held);take=torch.ones(len(y['source_out']),dtype=torch.bool) if held is None else d['source_out'].eq(held);domains=d['source_in'][keep].unique();assert all(int((keep&d['source_in'].eq(dd)&y['source_in'].eq(cc)).sum())>=2 for dd in domains for cc in y['source_in'][keep&d['source_in'].eq(dd)].unique());transform=DEST.parent/'source_conditional_patch_readout'/f'{fit}_patch_conditional_learned.pt';locks[str(transform)]=digest(transform);t=torch.load(transform,map_location='cpu',weights_only=False);z={}
  for split in data:
   c,p=[(raw[split][k]-t['input_stats'][k][0])/t['input_stats'][k][1] for k in ['cls','patch_max']];res=p-c@t['ridge'];z[split]=(res-t['family_stats'][0])/t['family_stats'][1]/2**.5
  axes,pairs=axes_from_errors(g['source_in'][keep],y['source_in'][keep],d['source_in'][keep]);u,sv,vh=torch.linalg.svd(axes,full_matrices=False);same=vh[:6];rng=torch.Generator().manual_seed(0);rr=torch.randn(10,6,generator=rng);rr-=rr.mean(0);random=torch.linalg.qr(rr)[0].T;families={'fixed_pairs':axes,'same_span_SVD':same,'random_span':random};assert torch.allclose(axes.T@torch.linalg.pinv(axes.T),same.T@same,atol=1e-5);idx=torch.nonzero(keep).flatten();gen=torch.Generator().manual_seed(0);schedule=[torch.cat([ii[torch.randint(len(ii),(32,),generator=gen)] for dd in domains for ii in [idx[d['source_in'][idx].eq(dd)]]]) for _ in range(400)];schedule_sha=hashlib.sha256(torch.stack(schedule).numpy().tobytes()).hexdigest();prior=json.loads((DEST.parent/'source_conditional_patch_readout/results.json').read_text())['fits'][fit];assert schedule_sha==prior['schedule_sha256'];out[fit]={'pairs':pairs,'axis_singular_values':sv.tolist(),'families':{},'schedule_sha256':schedule_sha}
  for family,aa in families.items():
   torch.manual_seed(0);h=Bank(aa);assert h(z['source_in'][:2]).count_nonzero()==0;opt=torch.optim.Adam(h.parameters(),lr=.001,weight_decay=1e-6)
   for step,ix in enumerate(schedule):
    if step%5==0:h.calibrate(z['source_in'][keep],d['source_in'][keep])
    delta=h(z['source_in'][ix]);f=g['source_in'][ix]+delta;loss=F.cross_entropy(f,y['source_in'][ix])+F.kl_div(F.log_softmax(f,1),F.softmax(g['source_in'][ix],1),reduction='batchmean')+.01*delta.square().mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
   h.calibrate(z['source_in'][keep],d['source_in'][keep]);results={}
   with torch.no_grad():
    for split,mask in [('source_in',keep),('source_out',take)]:
     comps=h.components(z[split][mask]);gg=g[split][mask];yy=y[split][mask];dd=d[split][mask];f=gg+comps.sum(1);ce=F.cross_entropy(f,yy,reduction='none');utility=torch.stack([F.cross_entropy(f-comps[:,m],yy,reduction='none')-ce for m in range(6)],1);u=torch.stack([utility[dd.eq(domain)].mean(0) for domain in dd.unique()]);v=u-u.mean(0,keepdim=True)-u.mean(1,keepdim=True)+u.mean();selected=u.gt(.002)&v.gt(.002);results[split]=dict(full=score(f,gg,yy,dd),global_control=score(gg,gg,yy,dd),U=u.tolist(),interaction=v.tolist(),domains=dd.unique().tolist(),diagnostic_selected_mask=selected.tolist(),diagnostic_shared_experts=int((selected.sum(0)>=2).sum()),positive_U_shared_experts=int((u.gt(.002).sum(0)>=2).sum()),correction_RMS=float(comps.sum(1).square().mean().sqrt()))
     if split=='source_out':np.savez_compressed(DEST/f'{fit}_{family}_source_out.npz',ids=data[split]['ids'][mask.numpy()],y=yy.numpy(),domains=dd.numpy(),global_logits=gg.numpy(),full_logits=f.numpy(),components=comps.numpy())
   torch.save(dict(state_dict=h.state_dict(),transform_path=str(transform),held=held,steps=400,schedule_sha256=schedule_sha,family=family),DEST/f'{fit}_{family}.pt');out[fit]['families'][family]=results;write(DEST/'partial_results.json',out);write(DEST/'progress.json',dict(status='fit_complete',fit=fit,family=family));print(fit,family,results['source_out']['full'],'train/valshared',results['source_in']['diagnostic_shared_experts'],results['source_out']['diagnostic_shared_experts'],flush=True)
 for path,sha in locks.items():assert digest(Path(path))==sha
 write(DEST/'results.json',dict(source_only=True,target_read=False,primary_SIRM_training=False,fits=out,input_sha256=locks,scientific_goal_achieved=False));write(DEST/'progress.json',dict(status='complete'))
if __name__=='__main__':main()

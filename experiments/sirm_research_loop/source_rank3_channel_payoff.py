from pathlib import Path
import json, hashlib
import numpy as np, torch
import torch.nn.functional as F
root=Path('/home/ntduc/Desktop/Dat/MESSI');base=root/'results/sirm_research_loop/iteration_19';out=base/'source_rank3_channel_payoff';out.mkdir(exist_ok=True)
assert not (out/'results.json').exists()
torch.set_num_threads(4)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
cache=root/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis'
plan={'source_only':True,'target_read':False,'fitting':False,'hypothesis':'After arbitrary rank3 basis failed component identifiability, test fixed input-channel contributions: CLS versus coordinatewise patch-max. Spatial evidence usefulness requires held-domain repair without offsetting harms.','scope':'8 saved heads x source-in/out; no new fit, subset/amplitude/checkpoint selection','intervention':'Separate normalized input halves of effective W; full correction equals CLS contribution plus patch-max contribution. Zero normalized half removes its correction, not image content.','limits':'Patch-max is not object localization; input-channel ablation is not a causal pixel intervention. Standalone channel and full-model LOO utilities differ.'}
save(out/'plan.json',plan);locks={};records={};cases=[]
def score(logits,g,y,dom):
 p=logits.argmax(1);gp=g.argmax(1);c=p.eq(y);gc=gp.eq(y);ce=F.cross_entropy(logits,y,reduction='none');return {'accuracy':float(torch.stack([c[dom==d].float().mean() for d in dom.unique()]).mean()),'CE':float(torch.stack([ce[dom==d].mean() for d in dom.unique()]).mean()),'repairs':int((~gc&c).sum()),'harms':int((gc&~c).sum())}
for fit in ['pooled','held0','held2','held3']:
 records[fit]={};held=None if fit=='pooled' else int(fit[-1])
 for policy in ['learned','fixed']:
  path=base/'source_rank3_risk_subspace'/f'{fit}_{policy}_risk_axis.pt';locks[str(path)]=sha(path);a=torch.load(path,map_location='cpu',weights_only=False);w=(a['b']-a['b'].mean(0,keepdim=True))@a['a'];splits={}
  for split in ['source_in','source_out']:
   p=cache/f'{split}_tokens.npz';locks[str(p)]=sha(p);d=dict(np.load(p));mask=np.ones(len(d['y']),bool) if held is None else (d['domains']!=held if split=='source_in' else d['domains']==held);raw=torch.tensor(np.concatenate([d['cls'],d['patch_max']],1)[mask]);z=(raw-a['mean'])/a['std']/2**.5;g=torch.tensor(d['global_logits'][mask]);y=torch.tensor(d['y'][mask]);dom=torch.tensor(d['domains'][mask]);parts={'CLS':F.linear(z[:,:384],w[:,:384]),'patch_max':F.linear(z[:,384:],w[:,384:])};full=g+parts['CLS']+parts['patch_max'];reference=g+F.linear(z,w);assert (full-reference).abs().max()<2e-5;assert (full.argmax(1)!=reference.argmax(1)).sum()==0
   value={'global':score(g,g,y,dom),'full':score(full,g,y,dom),'channels':{}};ce=F.cross_entropy(full,y,reduction='none')
   for name,corr in parts.items():
    u=F.cross_entropy(full-corr,y,reduction='none')-ce;value['channels'][name]={'standalone':score(g+corr,g,y,dom),'LOO_U_by_domain':{str(int(dd)):float(u[dom==dd].mean()) for dd in dom.unique()},'correction_RMS':float(corr.square().mean().sqrt()),'LOO_U_by_class':{str(int(cc)):float(u[y==cc].mean()) for cc in y.unique()}}
   splits[split]=value
   if fit=='pooled' and policy=='learned' and split=='source_out':
    gp=g.argmax(1);fp=full.argmax(1)
    for i,id in enumerate(d['ids'][mask]):
     if gp[i]!=fp[i]:cases.append({'id':str(id),'label':int(y[i]),'global':int(gp[i]),'full':int(fp[i]),'CLS_only':int((g+parts['CLS'])[i].argmax()),'patch_max_only':int((g+parts['patch_max'])[i].argmax())})
  records[fit][policy]=splits
for p,h in locks.items():assert sha(Path(p))==h
save(out/'results.json',{'source_only':True,'target_read':False,'records':records,'input_sha256':locks});save(out/'cases.json',cases)
for fit,v in records.items():
 for policy,r in v.items():
  x=r['source_out'];print(fit,policy,'global/full',x['global'],x['full'],'channels',x['channels'])

"""Hold each source domain out of both local-head and fusion fitting."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .terra_attention_eda import ROOT,OUT,load_models
from .terra_local_fusion_control import SOURCE
from .terra_local_class_hierarchical_uniform import joint_logprob
from .terra_source_object_probe import CACHE
from .terra_object_pool_readout import score
from experiments.sirm_h1_h6.run import ViewTransform
from .shrinkage import digest,write
DEST=OUT/'source_object_probe/local_fusion_held_domain'

def run():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;DEST.mkdir(exist_ok=True);assert not (DEST/'plan.json').exists();locks={str(SOURCE/f'held_domain{d}_probe.pt'):digest(SOURCE/f'held_domain{d}_probe.pt') for d in [0,2,3]};write(DEST/'plan.json',dict(hypothesis='A small pooled source gain may reflect global-logit calibration rather than transferable local evidence. Hold a source domain out of both auxiliary local and fusion fitting and compare matched controls.',scope='Auxiliary components only are source-held-out; frozen Stage1 ERM backbone/global saw all3 sources. Not whole-pipeline unseen-domain testing.',extra_annotations=True,source_only=True,target_read=False,training='Reuse each locked400-update held-domain local probe; fit each fusion400 updates, same32/train-domain minibatches, zero linear20->10, Adam.001 L2.01, train-only normalization and correction centering, final fixed update',controls=['global_local','global_repeat','global_permuted_local'],source_selection=False,input_sha256=locks,code_sha256=digest(Path(__file__))));(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());current,historical,fw,mlocks=load_models();del historical;version=[p._version for p in current.parameters()];transform=ViewTransform(current.config,'original',False);data={}
 with torch.no_grad():
  for split in ['source_in','source_out']:
   orig=dict(np.load(CACHE/f'{split}_tokens.npz'));partial=torch.load(SOURCE/f'{split}_cache.pt',map_location='cpu',weights_only=False);lookup={id:i for i,id in enumerate(partial['ids'])};groups=[]
   for start in range(0,len(orig['ids']),32):
    ids=[str(id) for id in orig['ids'][start:start+32]];missing=[id for id in ids if id not in lookup];found={};captured=[]
    if missing:
     images=torch.stack([transform(Image.open(ROOT/'domainbed/data/terra_incognita'/id).convert('RGB')) for id in missing]).cuda();hook=current.featurizer.vit.norm.register_forward_hook(lambda m,args,out:captured.append(out))
     try:current.featurizer(images)
     finally:hook.remove()
     found={id:x[2:].cpu() for id,x in zip(missing,captured[0])}
    patches=torch.stack([partial['features'][lookup[id]] if id in lookup else found[id] for id in ids]);assert float((patches.amax(1)-torch.tensor(orig['patch_max'][start:start+len(ids)])).abs().max())<1e-4;groups.append(patches)
   data[split]=dict(features=torch.cat(groups),ids=orig['ids'].tolist(),labels=torch.tensor(orig['y']),domains=torch.tensor(orig['domains']),global_logits=torch.tensor(orig['global_logits']));torch.save(data[split],DEST/f'{split}_patch_cache.pt');print('cache',split,len(orig['ids']),flush=True)
 assert version==[p._version for p in current.parameters()] and all(p.grad is None and not p.requires_grad for p in current.parameters())
 states={};combined={name:[] for name in ['global_local','global_repeat','global_permuted_local']}
 for held in [0,2,3]:
  probe=torch.load(SOURCE/f'held_domain{held}_probe.pt',map_location='cpu',weights_only=False);raw={};perms={}
  with torch.no_grad():
   for split,a in data.items():
    qs=[]
    for start in range(0,len(a['features']),128):
     x=a['features'][start:start+128].cuda();prob=joint_logprob(((x-probe['mean'].cuda())/probe['std'].cuda())@probe['weight'].cuda().T+probe['bias'].cuda()).exp();ix=(1-prob[:,:,5]).topk(8,dim=1).indices;qs.append(prob.gather(1,ix[:,:,None].expand(-1,-1,10)).mean(1).clamp_min(1e-8).log().cpu())
    q=torch.cat(qs);g=a['global_logits'];perm=torch.arange(len(g));d=a['domains']
    for dom in sorted(d.unique().tolist()):
     ix=torch.nonzero(d.eq(dom)).flatten().tolist();ordered=sorted(ix,key=lambda i:hashlib.sha256(a['ids'][i].encode()).hexdigest());shifted=ordered[1:]+ordered[:1]
     for i,j in zip(ordered,shifted):perm[i]=j
    assert not perm.eq(torch.arange(len(perm))).any();perms[split]=perm.tolist();raw[split]={'global_local':torch.cat([g,q],1),'global_repeat':torch.cat([g,g],1),'global_permuted_local':torch.cat([g,q[perm]],1)}
  train=data['source_in'];take=train['domains'].ne(held);y=train['labels'][take];d=train['domains'][take];g=train['global_logits'][take];doms=sorted(d.unique().tolist());cohorts={dom:torch.nonzero(d.eq(dom)).flatten() for dom in doms};gen=torch.Generator().manual_seed(0);schedule=[torch.cat([cohorts[dom][torch.randint(len(cohorts[dom]),(32,),generator=gen)] for dom in doms]) for _ in range(400)];sh=hashlib.sha256(torch.stack(schedule).numpy().tobytes()).hexdigest();models={}
  for name in raw['source_in']:
   r=raw['source_in'][name][take];mean=r.mean(0);std=r.std(0,unbiased=False).clamp_min(1e-6);x=(r-mean)/std/np.sqrt(2);head=torch.nn.Linear(20,10);head.weight.data.zero_();head.bias.data.zero_();opt=torch.optim.Adam(head.parameters(),lr=.001)
   for ix in schedule:
    loss=F.cross_entropy(g[ix]+head(x[ix]),y[ix])+.01*head.weight.square().sum();opt.zero_grad();loss.backward();opt.step()
   with torch.no_grad():
    z=head(x);center=torch.stack([z[d.eq(dom)].mean(0) for dom in doms]).mean(0);splits={};replays={}
    for split,a in data.items():
     mask=a['domains'].ne(held) if split=='source_in' else a['domains'].eq(held);f=a['global_logits'][mask]+head((raw[split][name][mask]-mean)/std/np.sqrt(2))-center;yy=a['labels'][mask];dd=a['domains'][mask];gg=a['global_logits'][mask];splits[split]=score(gg,f,yy,dd);replays[split]=dict(ids=[id for id,m in zip(a['ids'],mask.tolist()) if m],labels=yy,domains=dd,global_logits=gg,logits=f)
    path=DEST/f'held{held}_{name}.pt';torch.save(dict(weight=head.weight,bias=head.bias,mean=mean,std=std,center=center,replays=replays),path);models[name]=dict(splits=splits,model_sha256=digest(path));combined[name].append(replays['source_out']);print(held,name,{s:{k:v[k] for k in ['accuracy_equal_domain','CE_equal_domain','correct','repairs','harms']} for s,v in splits.items()},flush=True)
  states[str(held)]=dict(models=models,schedule_sha256=sh,permutations=perms,local_probe_sha256=digest(SOURCE/f'held_domain{held}_probe.pt'))
 aggregate={}
 for name,rows in combined.items():
  pooled={k:sum([r[k] for r in rows],[]) if k=='ids' else torch.cat([r[k] for r in rows]) for k in rows[0]};torch.save(pooled,DEST/f'{name}_crossfit_replay.pt');aggregate[name]=score(pooled['global_logits'],pooled['logits'],pooled['labels'],pooled['domains'])
 for path,h in {**locks,**mlocks}.items():assert digest(Path(path))==h
 write(DEST/'results.json',dict(states=states,aggregate=aggregate,source_only=True,target_read=False,not_SIRM=True,auxiliary_held_domain_only=True,parameters_unchanged=True,input_sha256={**locks,**mlocks}));print('aggregate',{name:{k:v[k] for k in ['accuracy_equal_domain','CE_equal_domain','correct','repairs','harms']} for name,v in aggregate.items()},flush=True)
if __name__=='__main__':run()

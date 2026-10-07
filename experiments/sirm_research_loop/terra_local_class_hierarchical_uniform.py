"""Matched hierarchical presence/species parameterization with hard-negative empty loss."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .terra_attention_eda import ROOT,OUT,CURRENT,load_models
from .terra_source_object_probe import DEST,CACHE
from .terra_object_pool_readout import score
from experiments.sirm_h1_h6.run import ViewTransform
from .shrinkage import digest,write

OUTPUT=DEST/'local_class_hierarchical_uniform'

def joint_logprob(raw):
 # Output0 models nonempty presence; outputs1:10 model species conditional on presence.
 presence_logit=raw[...,:1]+float(np.log(9))
 presence=F.logsigmoid(presence_logit);absence=F.logsigmoid(-presence_logit);species=raw[...,1:].log_softmax(-1)+presence
 return torch.cat([species[...,:5],absence,species[...,5:]],dim=-1)

def run():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;OUTPUT.mkdir(exist_ok=True);assert not (OUTPUT/'plan.json').exists();names=json.loads((CURRENT/'experiment/subset_manifest.json').read_text())['class_names'];empty=names.index('empty');locks={};data={}
 for split in ['source_in','source_out']:
  orig_path=CACHE/f'{split}_tokens.npz';patch_path=DEST/f'{split}_patches.pt';locks[str(orig_path)]=digest(orig_path);locks[str(patch_path)]=digest(patch_path);orig=dict(np.load(orig_path));partial=torch.load(patch_path,map_location='cpu',weights_only=False);byid={id:i for i,id in enumerate(orig['ids'].tolist())};pids=[byid[id] for id in partial['ids']];eids=np.flatnonzero(orig['y']==empty).tolist();data[split]=dict(original=orig,partial=partial,positive_indices=pids,empty_indices=eids)
 counts={s:dict(boxed_positive=len(a['positive_indices']),empty_negative=len(a['empty_indices'])) for s,a in data.items()}
 write(OUTPUT/'plan.json',dict(hypothesis='A flat softmax couples empty rejection to species competition. Factorized presence times conditional9-species probabilities may preserve species discrimination under matched hard-negative empty supervision.',training='400 fixed updates per probe; train-only feature standardization; zero linear384->(1 presence +9 species), same3850 parameters as flat head; fixed log9 presence offset matches initial joint1/10 probabilities of flat softmax; Adam .01, L2 .01; no validation/hparam selection',supervision='Only class-matched box-covered source-in patches are weighted animal positives. Outside partial boxes ignored. Source-in images labeled empty provide negative/background patch labels on the worst8 patches by empty CE. Animal positive patch weighting remains identical to the mean-negative control.',weighting='Equal label classes; cycle eligible domains per class;8 images per class/update; each class/domain needs >=2 distinct source-in images.',controls='Pooled3-source and three hold-one-source-domain fits. Same fixed budget/settings. Source-out reporting only; target excluded.',inference='No boxes/labels required: select top8 patches by1-p(empty); average their joint presence-times-species probabilities. Empty probability=1-presence; conditional species competition receives no gradient from empty patches. Primary standalone prediction; fixed additive log-prob correction to frozen global with source-in equal-domain class-mean centering. Oracle ROI classification reported separately.',scope='New annotation-assisted diagnostic fitting, not SIRM bank or an accuracy-tuned inference policy.',minimum_samples=2,K=8,source_counts=counts,target_read=False,global_frozen=True,source_only=True,code_sha256=digest(Path(__file__)),input_sha256=locks));(OUTPUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
 current,_,_,model_locks=load_models();before=[p._version for p in current.parameters()];transform=ViewTransform(current.config,'original',False)
 with torch.no_grad():
  for split,a in data.items():
   orig=a['original'];partial=a['partial'];features=[partial['features']];y=[torch.tensor(orig['y'][a['positive_indices']])];d=[torch.tensor(orig['domains'][a['positive_indices']])];weight=[partial['occupancy']/partial['occupancy'].sum(1,keepdim=True)];indices=list(a['positive_indices']);empty_features=[]
   for start in range(0,len(a['empty_indices']),32):
    ix=a['empty_indices'][start:start+32];images=torch.stack([transform(Image.open(ROOT/'domainbed/data/terra_incognita'/str(orig['ids'][i])).convert('RGB')) for i in ix]).cuda();captured=[];hook=current.featurizer.vit.norm.register_forward_hook(lambda m,args,z:captured.append(z))
    try:current.featurizer(images)
    finally:hook.remove()
    patches=captured[0][:,2:].cpu();assert float((patches.amax(1)-torch.tensor(orig['patch_max'][ix])).abs().max())<1e-4;empty_features.append(patches)
   features.append(torch.cat(empty_features));y.append(torch.full((len(a['empty_indices']),),empty,dtype=torch.long));d.append(torch.tensor(orig['domains'][a['empty_indices']]));weight.append(torch.full((len(a['empty_indices']),196),1/196));indices+=a['empty_indices'];a.update(features=torch.cat(features),labels=torch.cat(y),domains=torch.cat(d),weights=torch.cat(weight),ids=[str(orig['ids'][i]) for i in indices],global_logits=torch.tensor(orig['global_logits'][indices]),is_boxed=torch.tensor([True]*len(a['positive_indices'])+[False]*len(a['empty_indices'])))
   torch.save({k:a[k] for k in ['features','labels','domains','weights','ids','global_logits','is_boxed']},OUTPUT/f'{split}_cache.pt');print('cache',split,counts[split],flush=True)
 assert before==[p._version for p in current.parameters()] and all(p.grad is None and not p.requires_grad for p in current.parameters())
 results={}
 for held in [None,0,2,3]:
  name='pooled' if held is None else f'held_domain{held}';train=data['source_in'];keep=torch.ones(len(train['labels']),dtype=torch.bool) if held is None else train['domains'].ne(held);x=train['features'][keep].cuda();y=train['labels'][keep].cuda();d=train['domains'][keep];w=train['weights'][keep].cuda();mean=x.mean((0,1));std=x.std((0,1),unbiased=False).clamp_min(1e-6);x=(x-mean)/std;cohorts={};skipped=[]
  for cls in range(10):
   cohorts[cls]={}
   for dom in sorted(d.unique().tolist()):
    ix=torch.nonzero(d.eq(dom)&y.cpu().eq(cls)).flatten()
    if len(ix)>=2:cohorts[cls][dom]=ix
    elif len(ix):skipped.append(dict(label=cls,domain=dom,count=len(ix)))
   assert cohorts[cls],(name,cls,'insufficient minimum2 support')
  torch.manual_seed(0);gen=torch.Generator().manual_seed(0);head=torch.nn.Linear(384,10).cuda();torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias);opt=torch.optim.Adam(head.parameters(),lr=.01);history=[]
  for step in range(400):
   indices=[]
   for cls in range(10):
    doms=sorted(cohorts[cls]);chosen=cohorts[cls][doms[step%len(doms)]];indices.append(chosen[torch.randint(len(chosen),(8,),generator=gen)])
   ix=torch.cat(indices).cuda();logp=joint_logprob(head(x[ix]));per_patch=-logp.gather(2,y[ix,None,None].expand(-1,196,1)).squeeze(-1);per_image=(per_patch*w[ix]).sum(1);negative=y[ix].eq(empty);per_image=torch.where(negative,per_patch.topk(8,dim=1).values.mean(1),per_image);loss=per_image.mean()+.01*head.weight.square().sum();opt.zero_grad();loss.backward();opt.step();history.append(float(loss))
  # Center the deployed log-prob correction using only the actual training domains.
  def image_logprob(tokens):
   p=joint_logprob(head((tokens.cuda()-mean)/std)).exp();rank=(1-p[:,:,empty]).topk(8,dim=1).indices;chosen=p.gather(1,rank[:,:,None].expand(-1,-1,10));return chosen.mean(1).clamp_min(1e-8).log()
  with torch.no_grad():
   center_logits=image_logprob(train['features'][keep]);center=torch.stack([center_logits[d.eq(dom).cuda()].mean(0) for dom in sorted(d.unique().tolist())]).mean(0);splits={};replays={}
   for split,a in data.items():
    take=torch.ones(len(a['labels']),dtype=torch.bool) if held is None else a['domains'].eq(held);ix=torch.nonzero(take).flatten();tokens=a['features'][take];label=a['labels'][take];domain=a['domains'][take];weights=a['weights'][take];g=a['global_logits'][take];deployed=image_logprob(tokens).cpu();full=g+deployed-center.cpu();oracle=(joint_logprob(head((tokens.cuda()-mean)/std)).exp().cpu()*weights[:,:,None]).sum(1);oc=oracle.argmax(1).eq(label);mask=a['is_boxed'][take];classes={}
    for cls in range(10):
     m=label.eq(cls);cells=[]
     for dom in sorted(domain.unique().tolist()):
      md=m&domain.eq(dom);n=int(md.sum());cells.append(dict(domain=dom,n=n,valid=n>=2,oracle_correct=int(oc[md].sum()),standalone_correct=int(deployed.argmax(1)[md].eq(cls).sum())))
     classes[names[cls]]=dict(n=int(m.sum()),oracle_correct=int(oc[m].sum()),standalone_correct=int(deployed.argmax(1)[m].eq(cls).sum()),cells=cells)
    splits[split]=dict(n=len(label),boxed=int(mask.sum()),empty=int((~mask).sum()),oracle_ROI_correct=int(oc[mask].sum()),oracle_ROI_n=int(mask.sum()),standalone=score(g,deployed,label,domain),fixed_additive=score(g,full,label,domain),global_control=score(g,g,label,domain),classes=classes)
    replays[split]=dict(ids=[a['ids'][i] for i in ix.tolist()],labels=label,domains=domain,global_logits=g,standalone_logits=deployed,additive_logits=full,oracle_probabilities=oracle,is_boxed=mask)
  path=OUTPUT/f'{name}_probe.pt';torch.save(dict(weight=head.weight.detach().cpu(),bias=head.bias.detach().cpu(),mean=mean.cpu(),std=std.cpu(),center=center.cpu(),replays=replays),path);results[name]=dict(loss_initial=history[0],loss_final=history[-1],skipped_minimum_sample_cohorts=skipped,splits=splits,probe_sha256=digest(path));print(name,{s:dict(n=v['n'],ROI=f"{v['oracle_ROI_correct']}/{v['oracle_ROI_n']}",standalone=v['standalone']['accuracy_equal_domain'],additive=v['fixed_additive']['accuracy_equal_domain'],global_acc=v['global_control']['accuracy_equal_domain'],repair=v['fixed_additive']['repairs'],harm=v['fixed_additive']['harms']) for s,v in splits.items()},flush=True)
 for path,h in {**locks,**model_locks}.items():assert digest(Path(path))==h
 write(OUTPUT/'results.json',dict(states=results,source_only=True,target_read=False,parameters_unchanged=True,extra_source_annotations=True,evaluation_subset='Boxed nonempty plus explicit empty only, excludes unboxed nonempty',input_sha256={**locks,**model_locks}));print('complete',flush=True)
if __name__=='__main__':run()

"""All-image source-only replay of locked local patch class probes."""
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from .terra_local_class_hierarchical import OUTPUT,joint_logprob
from .terra_attention_eda import ROOT,load_models
from .terra_source_object_probe import CACHE
from .terra_object_pool_readout import score
from experiments.sirm_h1_h6.run import ViewTransform
from .shrinkage import write,digest

DEST=OUTPUT/'all_image_replay'

def run():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;DEST.mkdir(exist_ok=True);assert not (DEST/'plan.json').exists();results_path=OUTPUT/'results.json';results=json.loads(results_path.read_text());plan=dict(scope='All1172 source-in and304 source-out images, including66 source-out unboxed nonempty images missing from ROI evaluation.',source_only=True,target_read=False,fitting=False,selector='Top8 by1-p(empty), fixed in initial probe plan',centering='Existing locked train-only center; no re-centering on source-out or full-source images',precision='TF32 disabled',model_sha256={name:v['probe_sha256'] for name,v in results['states'].items()},code_sha256=digest(Path(__file__)),original_results_sha256=digest(results_path));write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());current,_,_,locks=load_models();versions=[p._version for p in current.parameters()];t=ViewTransform(current.config,'original',False);data={}
 with torch.no_grad():
  for split in ['source_in','source_out']:
   original=dict(np.load(CACHE/f'{split}_tokens.npz'));partial=torch.load(OUTPUT/f'{split}_cache.pt',map_location='cpu',weights_only=False);lookup={id:i for i,id in enumerate(partial['ids'])};groups=[]
   for start in range(0,len(original['ids']),32):
    ids=[str(id) for id in original['ids'][start:start+32]];missing=[id for id in ids if id not in lookup];captured=[];found={}
    if missing:
     batch=torch.stack([t(Image.open(ROOT/'domainbed/data/terra_incognita'/id).convert('RGB')) for id in missing]).cuda();hook=current.featurizer.vit.norm.register_forward_hook(lambda m,args,out:captured.append(out))
     try:current.featurizer(batch)
     finally:hook.remove()
     found={id:patches[2:].cpu() for id,patches in zip(missing,captured[0])}
    patches=torch.stack([partial['features'][lookup[id]] if id in lookup else found[id] for id in ids]);assert float((patches.amax(1)-torch.tensor(original['patch_max'][start:start+len(ids)])).abs().max())<1e-4;groups.append(patches)
   data[split]=dict(features=torch.cat(groups),ids=original['ids'].tolist(),labels=torch.tensor(original['y']),domains=torch.tensor(original['domains']),global_logits=torch.tensor(original['global_logits']),boxed_or_empty=torch.tensor([str(id) in lookup for id in original['ids']]))
  states={}
  for name in results['states']:
   path=OUTPUT/f'{name}_probe.pt';assert digest(path)==plan['model_sha256'][name];p=torch.load(path,map_location='cpu',weights_only=False);splits={};replays={};held=None if name=='pooled' else int(name.replace('held_domain',''))
   for split,a in data.items():
    take=torch.ones(len(a['labels']),dtype=torch.bool) if held is None else a['domains'].eq(held);x=a['features'][take];prob=joint_logprob(((x-p['mean'])/p['std'])@p['weight'].T+p['bias']).exp();rank=(1-prob[:,:,5]).topk(8,dim=1).indices;deployed=prob.gather(1,rank[:,:,None].expand(-1,-1,10)).mean(1).clamp_min(1e-8).log();g=a['global_logits'][take];y=a['labels'][take];d=a['domains'][take];f=g+deployed-p['center'];splits[split]=dict(standalone=score(g,deployed,y,d),fixed_additive=score(g,f,y,d),global_control=score(g,g,y,d),n=len(y));replays[split]=dict(ids=[id for id,m in zip(a['ids'],take.tolist()) if m],labels=y,domains=d,global_logits=g,standalone_logits=deployed,additive_logits=f,top8_indices=rank,probabilities=prob,boxed_or_empty=a['boxed_or_empty'][take])
   cache=DEST/f'{name}_replay.pt';torch.save(replays,cache);states[name]=dict(splits=splits,replay_sha256=digest(cache));print(name,{s:dict(n=v['n'],global_acc=v['global_control']['accuracy_equal_domain'],standalone=v['standalone']['accuracy_equal_domain'],additive=v['fixed_additive']['accuracy_equal_domain'],repair=v['fixed_additive']['repairs'],harm=v['fixed_additive']['harms']) for s,v in splits.items()},flush=True)
 assert versions==[p._version for p in current.parameters()] and all(p.grad is None and not p.requires_grad for p in current.parameters())
 for path,h in locks.items():assert digest(Path(path))==h
 write(DEST/'results.json',dict(states=states,source_only=True,target_read=False,fitting=False,parameters_unchanged=True,held_domain_scope='Only auxiliary probe fitting excludes held source domain. Frozen Stage1 ERM backbone/global were trained on all3 sources; not a whole-pipeline unseen-domain test.'))
if __name__=='__main__':run()

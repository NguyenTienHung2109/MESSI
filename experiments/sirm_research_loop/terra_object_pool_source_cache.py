"""Deployable source-only object pooling cache and empty-scene specificity audit."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .terra_attention_eda import ROOT,OUT,load_models
from .terra_source_object_probe import DEST,CACHE
from experiments.sirm_h1_h6.run import ViewTransform
from .shrinkage import digest,write

RESULT=DEST/'full_source_pooling'

def run():
 torch.set_num_threads(4);RESULT.mkdir(exist_ok=True);assert not (RESULT/'plan.json').exists();probe_path=DEST/'pooled_probe.pt';probe=torch.load(probe_path,map_location='cpu',weights_only=False)
 write(RESULT/'plan.json',dict(scope='All1172 source-in and304 source-out images including empty and unboxed; target excluded',probe_sha256=digest(probe_path),fitting=False,threshold_tuning=False,code_sha256=digest(Path(__file__)),empty_diagnostic='Peak/mean uncalibrated objectness score, AUROC for empty-vs-nonempty image labels; pooled feature does not use box/class labels.'))
 (RESULT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());current,historical,funcs,locks=load_models();t=ViewTransform(current.config,'original',False);versions=[q._version for q in current.parameters()];summary={};names=json.loads((OUT/'results.json').read_text())['class_names'];empty=names.index('empty')
 with torch.no_grad():
  for split in ['source_in','source_out']:
   original=dict(np.load(CACHE/f'{split}_tokens.npz'));partial=torch.load(DEST/f'{split}_patches.pt',map_location='cpu',weights_only=False);byid={id:i for i,id in enumerate(partial['ids'])};raws=[];probs=[];cls=[]
   for start in range(0,len(original['ids']),32):
    group=[str(id) for id in original['ids'][start:start+32]];missing=[id for id in group if id not in byid];found={}
    if missing:
     images=torch.stack([t(Image.open(ROOT/'domainbed/data/terra_incognita'/id).convert('RGB')) for id in missing]).cuda();captured=[];hook=current.featurizer.vit.norm.register_forward_hook(lambda m,a,z:captured.append(z))
     try:current.featurizer(images)
     finally:hook.remove()
     found={id:tokens[2:].cpu() for id,tokens in zip(missing,captured[0])}
    patches=torch.stack([partial['features'][byid[id]] if id in byid else found[id] for id in group]);score=(((patches-probe['mean'])/probe['std'])@probe['weight'].T+probe['bias']).squeeze(-1).sigmoid();pooled=(patches*(score/score.sum(1,keepdim=True))[:,:,None]).sum(1);c=torch.tensor(original['cls'][start:start+len(group)]);raws.append(torch.cat([c,pooled],1));probs.append(score);cls.append(c)
   features=torch.cat(raws);prob=torch.cat(probs);y=torch.tensor(original['y']);domains=torch.tensor(original['domains']);global_logits=current.global_head(torch.cat(cls).cuda()).cpu();assert np.allclose(global_logits.numpy(),original['global_logits'],rtol=1e-5,atol=1e-5)
   path=RESULT/f'{split}_features.pt';torch.save(dict(features=features,ids=original['ids'].tolist(),labels=y,domains=domains,objectness=prob,global_logits=global_logits,anchor_sha256=partial['anchor_sha256'],target_read=False,label_conditioned=False,view='original',pooling='Frozen source-in-trained class-agnostic objectness-weighted patch mean; true boxes/classes never inference inputs',probe_sha256=digest(probe_path)),path)
   score=prob.max(1).values.numpy();positive=y.numpy()!=empty
   # Rank AUROC, including exact ties, no fitted threshold.
   pp=score[positive];nn=score[~positive];auc=float(((pp[:,None]>nn[None,:])+.5*(pp[:,None]==nn[None,:])).mean())
   summary[split]=dict(images=len(y),nonempty=int(positive.sum()),empty=int((~positive).sum()),object_presence_peak_score_AUROC=auc,empty_peak_score_median=float(np.median(nn)),nonempty_peak_score_median=float(np.median(pp)),empty_mean_score_median=float(np.median(prob.numpy()[~positive].mean(1))),nonempty_mean_score_median=float(np.median(prob.numpy()[positive].mean(1))),feature_cache_sha256=digest(path))
   print(split,summary[split],flush=True)
 assert versions==[q._version for q in current.parameters()] and all(q.grad is None for q in current.parameters())
 for path,h in locks.items():assert digest(Path(path))==h
 write(RESULT/'results.json',dict(summary=summary,parameters_unchanged=True,target_read=False,fitting=False,probe_sha256=digest(probe_path)))
if __name__=='__main__':run()

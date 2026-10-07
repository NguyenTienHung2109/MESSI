"""Read-only intact-patch pooling using frozen global predictions; source images only."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.terra_10pct_factor_diagnosis.report import data_for
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
OUT=DEST/'coherent_patch_source_cache'

@torch.no_grad()
def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'extraction_audit.json').exists();parent=ROOT/'results/sirm_research_loop/iteration_12';old=parent/'frozen_token_source_diagnosis';prior=json.loads((old/'extraction_audit.json').read_text());anchor=ROOT/'results/terra_10pct_factor_diagnosis/runs/L38/ERM_H2/source_best.pkl';manifestfile=parent/'experiment/subset_manifest.json';assert digest(anchor)==prior['anchor_sha256'];assert digest(manifestfile)==prior['manifest_sha256'];payload=torch.load(anchor,map_location='cpu',weights_only=False);c=payload['config'];manifest=json.loads(manifestfile.read_text())
 plan=dict(source_only=True,target_read=False,fitting=False,bank_updates=False,anchor_sha256=digest(anchor),manifest_sha256=digest(manifestfile),poolings=['predicted_class_top49','token_norm_top49','image_seeded_random49','patch_mean','patch_max'],patches=196,selected_patches=49,selection='Final normalized patch tokens excluding both prefix tokens. Predicted-class score=token dot (frozen global weight[predicted CLS class] minus mean global weight). Select highest49 scores with stable index ties, then average entire vectors. Token-norm and per-image SHA-seeded random49 controls also average entire vectors. No image label or domain used for pooling.',limits='Frozen global head trained on CLS, not patch tokens; score is heuristic, not calibrated localization. Top patches need not be adjacent, foreground, or causal factors. Same dimensionality is not matched pooling information content.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());backbone=DeiTFeaturizer(pretrained=False,model_name=c['backbone']);backbone.load_state_dict({k.removeprefix('featurizer.'):v for k,v in payload['model_dict'].items() if k.startswith('featurizer.')},strict=True);head=torch.nn.Linear(384,10);head.load_state_dict({k.removeprefix('global_head.'):v for k,v in payload['model_dict'].items() if k.startswith('global_head.')},strict=True)
 backbone.requires_grad_(False).cuda().eval();head.requires_grad_(False).cuda().eval();before=[{k:v.cpu().clone() for k,v in m.state_dict().items()} for m in [backbone,head]];captured=[];hook=backbone.vit.norm.register_forward_hook(lambda mod,args,result:captured.append(result));summaries=[];stats=[];locks={str(anchor):digest(anchor),str(manifestfile):digest(manifestfile)}
 try:
  for split in ['in','out']:
   name='source_'+split;refpath=old/(name+'_tokens.npz');ref=dict(np.load(refpath));assert digest(refpath)==next(r['cache_sha256'] for r in prior['rows'] if r['split']==name);locks[str(refpath)]=digest(refpath);lookup={str(v):i for i,v in enumerate(ref['ids'])};arrays={k:[] for k in ['cls','global_logits','predicted_class_top49','token_norm_top49','image_seeded_random49','patch_mean','patch_max','class_patch_indices','norm_patch_indices','random_patch_indices']};ids=[];labels=[];domains=[];maxerr=0.
   for d,ds in sorted(data_for(manifest,[0,2,3],c,split).items()):
    imageids=[manifest['sources'][str(d)]['images'][i]['id'] for i in ds.indices]
    for i in ds.indices:
     r=manifest['sources'][str(d)]['images'][i];assert digest(ROOT/'domainbed/data/terra_incognita'/r['id'])==r['sha256']
    offset=0
    for x,y in DataLoader(ds,batch_size=32,num_workers=0,shuffle=False):
     current=imageids[offset:offset+len(y)];ix=[lookup[id] for id in current];captured.clear();cls=backbone(x.cuda());assert len(captured)==1;t=captured[0];assert t.shape==(len(x),198,384) and torch.equal(cls,t[:,0]);patch=t[:,2:];g=head(cls);pred=g.argmax(1);w=head.weight[pred]-head.weight.mean(0);scores=(patch*w[:,None,:]).sum(2);norm=patch.norm(dim=2);ci=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:49];ni=torch.argsort(norm,dim=1,descending=True,stable=True)[:,:49];random=[]
     for id in current:
      seed=int(hashlib.sha256(id.encode()).hexdigest()[:16],16);random.append(np.sort(np.random.default_rng(seed).choice(196,49,replace=False)))
     ri=torch.tensor(np.array(random),device=patch.device)
     def pool(index):return patch.gather(1,index[:,:,None].expand(-1,-1,384)).mean(1)
     values=dict(cls=cls,global_logits=g,predicted_class_top49=pool(ci),token_norm_top49=pool(ni),image_seeded_random49=pool(ri),patch_mean=patch.mean(1),patch_max=patch.amax(1),class_patch_indices=ci,norm_patch_indices=ni,random_patch_indices=ri)
     for key in ['cls','global_logits','patch_mean','patch_max']:
      error=float(np.max(np.abs(values[key].cpu().numpy()-ref[key][ix])));maxerr=max(maxerr,error);assert error<1e-4,(key,error)
     assert np.array_equal(y.numpy(),ref['y'][ix]) and np.all(ref['domains'][ix]==d)
     maxima=patch.argmax(1);cos=torch.nn.functional.cosine_similarity(patch,patch.amax(1)[:,None,:],dim=2)
     for j,id in enumerate(current):stats.append(dict(split=name,image_id=id,coordinatewise_max_distinct_patch_count=int(maxima[j].unique().numel()),max_vector_nearest_patch_cosine=float(cos[j].max()),class_norm_top49_overlap=int(torch.isin(ci[j],ni[j]).sum()),global_prediction=int(pred[j])))
     for key,v in values.items():arrays[key].append(v.cpu().numpy())
     ids.extend(current);labels.extend(y.tolist());domains.extend([d]*len(y));offset+=len(y)
   packed={k:np.concatenate(v) for k,v in arrays.items()};packed.update(ids=np.array(ids),y=np.array(labels),domains=np.array(domains));assert len(set(ids))==len(ids)==len(ref['ids']);file=OUT/(name+'_tokens.npz');np.savez_compressed(file,**packed);summaries.append(dict(split=name,n=len(ids),cache_sha256=digest(file),old_cache_max_error=maxerr,labels_and_image_ids_match=True));write(OUT/'progress.json',dict(completed_splits=summaries));print(name,len(ids),'maxerr',maxerr,flush=True)
 finally:hook.remove()
 for model,state in zip([backbone,head],before):
  assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in state.items());assert all(not p.requires_grad and p.grad is None for p in model.parameters())
 for file,h in locks.items():assert digest(Path(file))==h
 write(OUT/'pooling_case_geometry.json',stats);write(OUT/'extraction_audit.json',dict(plan=plan,rows=summaries,input_sha256=locks,source_only=True,target_read=False,bank_updates=False,global_backbone_tensors_exactly_unchanged=True))
if __name__=='__main__':run()

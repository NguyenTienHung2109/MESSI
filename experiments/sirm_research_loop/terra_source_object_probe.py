"""Source-only class-agnostic patch objectness diagnostic on a frozen strong ERM."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .terra_attention_eda import ROOT,OUT,load_models,box_mask,localization
from experiments.sirm_h1_h6.run import ViewTransform
from .shrinkage import digest,write

DEST=OUT/'source_object_probe'
CACHE=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis'

def run():
 torch.set_num_threads(4);DEST.mkdir(exist_ok=True);assert not (DEST/'plan.json').exists()
 official=json.loads((OUT/'caltech_bboxes_20200316.json').read_text());sizes={a['id']:(a['width'],a['height']) for a in official['images']};cats={a['id']:a['name'] for a in official['categories']};annotations={}
 for a in official['annotations']:
  if a.get('bbox'):annotations.setdefault(a['image_id'],[]).append((cats[a['category_id']],a['bbox']))
 names=json.loads((OUT/'results.json').read_text())['class_names'];rows={};locks={}
 for split in ['source_in','source_out']:
  path=CACHE/f'{split}_tokens.npz';locks[str(path)]=digest(path);c=dict(np.load(path));rows[split]=[]
  for id,y,d in zip(c['ids'],c['y'],c['domains']):
   stem=Path(str(id)).stem
   if stem not in sizes:continue
   ow,oh=sizes[stem];bb=[[x/ow,z/oh,w/ow,h/oh] for label,(x,z,w,h) in annotations.get(stem,[]) if label==names[int(y)]]
   if not bb:continue
   mask=box_mask(bb);occupancy=mask.reshape(14,16,14,16).mean((1,3)).reshape(196)
   if occupancy.sum()<1e-8 or (1-occupancy).sum()<1e-8:continue
   rows[split].append(dict(id=str(id),label=int(y),domain=int(d),boxes=bb,occupancy=occupancy,mask=mask))
 assert not set(r['id'] for r in rows['source_in'])&set(r['id'] for r in rows['source_out'])
 plan=dict(hypothesis='Frozen patch features retain class-agnostic object localization sufficient to improve pooling; a probe is diagnostic, not new ERM/SIRM training.',
  scope='Source-in fitting and source-out reporting only. Class-matched official boxes supervise foreground, classifier class labels are not model inputs.',
  source_counts={split:len(v) for split,v in rows.items()},steps=400,seed=0,optimizer='Adam lr .01 weight_decay0; L2 .01 coefficient explicit',
  objective='Image/domain-balanced BCE, each image foreground and background each half mass, fractional patch box coverage',
  protocol='Pooled3-source fit and three leave-one-source-domain-out fits; every domain has >=2 distinct train images. Train-only feature mean/std. Last update only; no validation selection/tuning.',
  architecture='One linear384->1 patch classifier; excludes CLS/dist tokens. Frozen anchor unchanged.',control='Fixed train-only average occupancy spatial prior, no image inputs',
  metrics='Box mass/enrichment and peak-center-inside on official source-out boxes; compare fixed spatial prior and raw attention. No target inference or target selection.',
  fitting=True,target_read=False,checkpoint_changes=False,code_sha256=digest(Path(__file__)),input_sha256=locks)
 write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
 current,historical,forwards,model_locks=load_models();versions=[q._version for q in current.parameters()];t=ViewTransform(current.config,'original',False);arrays={}
 for split,rr in rows.items():
  feats=[];occ=[]
  for start in range(0,len(rr),32):
   group=rr[start:start+32];imgs=[]
   for r in group:
    p=ROOT/'domainbed/data/terra_incognita'/r['id'];imgs.append(t(Image.open(p).convert('RGB')))
   capture=[];hook=current.featurizer.vit.norm.register_forward_hook(lambda module,args,out:capture.append(out))
   try:
    with torch.no_grad():current.featurizer(torch.stack(imgs).cuda())
   finally:hook.remove()
   assert len(capture)==1 and capture[0].shape[1:]==(198,384)
   feats.append(capture[0][:,2:].cpu());occ.extend(r['occupancy'] for r in group)
  x=torch.cat(feats);y=torch.tensor(np.array(occ),dtype=torch.float32);d=torch.tensor([r['domain'] for r in rr]);arrays[split]=(x,y,d)
  torch.save(dict(features=x,occupancy=y,ids=[r['id'] for r in rr],domains=d,anchor_sha256=model_locks[str(ROOT/'results/terra_10pct_factor_diagnosis/runs/L38/ERM_H2/source_best.pkl')]),DEST/f'{split}_patches.pt')
  print('cached',split,len(rr),flush=True)
 results={}
 for held in [None,0,2,3]:
  name='pooled' if held is None else f'held_domain{held}';x,y,d=arrays['source_in'];keep=torch.ones(len(d),dtype=torch.bool) if held is None else d.ne(held);train_x=x[keep].cuda();train_y=y[keep].cuda();train_d=d[keep];doms=torch.unique(train_d).tolist();assert all(int(train_d.eq(a).sum())>=2 for a in doms)
  mean=train_x.mean((0,1));std=train_x.std((0,1),unbiased=False).clamp_min(1e-6);train_x=(train_x-mean)/std;torch.manual_seed(0);head=torch.nn.Linear(384,1).cuda();torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias);opt=torch.optim.Adam(head.parameters(),lr=.01);gen=torch.Generator().manual_seed(0);indices={a:torch.nonzero(train_d.eq(a)).flatten() for a in doms};losses=[]
  for step in range(400):
   idx=torch.cat([indices[a][torch.randint(len(indices[a]),(12,),generator=gen)] for a in doms]).cuda();z=head(train_x[idx]).squeeze(-1);o=train_y[idx];pos=-(F.logsigmoid(z)*o).sum(1)/o.sum(1);neg=-(F.logsigmoid(-z)*(1-o)).sum(1)/(1-o).sum(1);loss=.5*(pos+neg).mean()+.01*head.weight.square().sum();opt.zero_grad();loss.backward();opt.step();losses.append(float(loss))
  prior=train_y.mean(0).cpu().numpy();prior/=prior.sum();splits={};pred_cache={}
  with torch.no_grad():
   for split,(xx,yy,dd) in arrays.items():
    take=torch.ones(len(dd),dtype=torch.bool) if held is None else dd.eq(held);pred=head((xx[take].cuda()-mean)/std).squeeze(-1).sigmoid().cpu().numpy();ix=torch.nonzero(take).flatten().tolist();values=[]
    for i,prob in zip(ix,pred):
     record=rows[split][i];heat=prob/prob.sum();v=localization(heat.reshape(14,14),record['mask']);v.update(id=record['id'],class_name=names[record['label']],domain=record['domain'],prior=localization(prior.reshape(14,14),record['mask']));values.append(v)
    def agg(vv):return dict(n=len(vv),peak_inside=sum(v['peak_patch_center_in_box'] for v in vv),peak_inside_fraction=float(np.mean([v['peak_patch_center_in_box'] for v in vv])),median_enrichment=float(np.median([v['area_normalized_enrichment'] for v in vv])))
    splits[split]=dict(probe=agg(values),spatial_prior=agg([v['prior'] for v in values]),cases=values);pred_cache[split]=dict(ids=[rows[split][i]['id'] for i in ix],probabilities=pred)
  path=DEST/f'{name}_probe.pt';torch.save(dict(weight=head.weight.detach().cpu(),bias=head.bias.detach().cpu(),mean=mean.cpu(),std=std.cpu(),held_domain=held),path);np.savez_compressed(DEST/f'{name}_source_out_predictions.npz',ids=np.array(pred_cache['source_out']['ids']),probabilities=pred_cache['source_out']['probabilities'])
  results[name]=dict(train_images=len(train_x),loss_initial=losses[0],loss_final=losses[-1],splits=splits,probe_sha256=digest(path));write(DEST/'progress.json',dict(status='fit_complete',fit=name));print(name,{s:{k:v[k] for k in ['probe','spatial_prior']} for s,v in splits.items()},flush=True)
 assert versions==[q._version for q in current.parameters()] and all(q.grad is None for q in current.parameters())
 for p,h in {**locks,**model_locks}.items():assert digest(Path(p))==h
 write(DEST/'results.json',dict(states=results,parameters_unchanged=True,target_read=False,checkpoint_changes=False,input_sha256={**locks,**model_locks}));write(DEST/'progress.json',dict(status='complete'))
if __name__=='__main__':run()

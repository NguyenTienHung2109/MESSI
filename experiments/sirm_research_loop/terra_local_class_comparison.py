"""Source-only paired EDA for local class aggregation and matched empty loss."""
import json, hashlib
from pathlib import Path
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .terra_attention_eda import ROOT,OUT
from .terra_local_class_probe import OUTPUT as BASE
from .terra_local_class_hard_negative import OUTPUT as HARD
from .terra_object_pool_readout import score
from .shrinkage import write,digest

def main():
 torch.set_num_threads(4)
 # Verify the existing artifact registry before adding new results.
 registry=json.loads((OUT/'verification.json').read_text())
 for rel,h in registry['sha256'].items():
  assert digest(OUT/rel)==h,rel
 names=['bird','bobcat','cat','coyote','dog','empty','opossum','rabbit','raccoon','squirrel']
 audits={}; schedules={}
 for split in ['source_in','source_out']:
  a=torch.load(BASE/f'{split}_cache.pt',map_location='cpu',weights_only=False)
  b=torch.load(HARD/f'{split}_cache.pt',map_location='cpu',weights_only=False)
  audits[split]={k:(a[k]==b[k] if k=='ids' else torch.equal(a[k],b[k])) for k in a};assert all(audits[split].values())
  if split=='source_in':
   for held in [None,0,2,3]:
    keep=torch.ones(len(a['labels']),dtype=torch.bool) if held is None else a['domains'].ne(held);y=a['labels'][keep];d=a['domains'][keep];cohorts={c:{dom:torch.nonzero(d.eq(dom)&y.eq(c)).flatten() for dom in sorted(d.unique().tolist()) if int((d.eq(dom)&y.eq(c)).sum())>=2} for c in range(10)};gen=torch.Generator().manual_seed(0);rows=[]
    for step in range(400):
     row=[]
     for c in range(10):
      ds=sorted(cohorts[c]);ix=cohorts[c][ds[step%len(ds)]];row.append(ix[torch.randint(len(ix),(8,),generator=gen)])
     rows.append(torch.cat(row))
    schedules[str(held)]=hashlib.sha256(torch.stack(rows).numpy().tobytes()).hexdigest()
  del a,b
 result=dict(source_only=True,target_read=False,not_SIRM=True,cache_equality=audits,schedule_sha256=schedules,states={})
 pairs={tag:torch.load(path/'all_image_replay/pooled_replay.pt',map_location='cpu',weights_only=False) for tag,path in [('mean_empty_loss',BASE),('hard_negative_loss',HARD)]}
 for split in ['source_in','source_out']:
  a=pairs['mean_empty_loss'][split];b=pairs['hard_negative_loss'][split];assert a['ids']==b['ids']
  for k in ['labels','domains','global_logits','boxed_or_empty']:assert torch.equal(a[k],b[k])
 for tag,path in [('mean_empty_loss',BASE),('hard_negative_loss',HARD)]:
  splits={}
  for split,p in pairs[tag].items():
   y=p['labels'];d=p['domains'];classes={}
   for c,name in enumerate(names):
    m=y.eq(c);classes[name]=dict(n=int(m.sum()),global_correct=int(p['global_logits'].argmax(1)[m].eq(c).sum()),standalone_correct=int(p['standalone_logits'].argmax(1)[m].eq(c).sum()),additive_correct=int(p['additive_logits'].argmax(1)[m].eq(c).sum()))
   splits[split]=dict(classes=classes,global_score=score(p['global_logits'],p['global_logits'],y,d),standalone_score=score(p['global_logits'],p['standalone_logits'],y,d),additive_score=score(p['global_logits'],p['additive_logits'],y,d))
  p=pairs[tag]['source_out'];empty=p['labels'].eq(5);prob=p['probabilities'][empty];ix=p['top8_indices'][empty];q=prob.gather(1,ix[:,:,None].expand(-1,-1,10)).mean(1);allq=prob.mean(1)
  result['states'][tag]=dict(splits=splits,empty_aggregation=dict(n=int(empty.sum()),all_patch_correct=int(allq.argmax(1).eq(5).sum()),top8_correct=int(q.argmax(1).eq(5).sum()),all_patch_p_empty_median=float(torch.quantile(allq[:,5],.5)),top8_p_empty_median=float(torch.quantile(q[:,5],.5)),mean_patch_empty_CE=float(-prob[:,:,5].clamp_min(1e-8).log().mean()),top8_empty_CE=float(-q[:,5].clamp_min(1e-8).log().mean())))
 for name in ['pooled','held_domain0','held_domain2','held_domain3']:
  a=torch.load(BASE/f'{name}_probe.pt',map_location='cpu',weights_only=False);b=torch.load(HARD/f'{name}_probe.pt',map_location='cpu',weights_only=False)
  assert torch.equal(a['mean'],b['mean']) and torch.equal(a['std'],b['std'])
 result['normalization_exact']=True
 a=pairs['mean_empty_loss']['source_out'];b=pairs['hard_negative_loss']['source_out'];y=a['labels'];g=a['global_logits'].argmax(1);ap=a['additive_logits'].argmax(1);bp=b['additive_logits'].argmax(1);sp=b['standalone_logits'].argmax(1);cases=[]
 for i,id in enumerate(a['ids']):
  if ap[i]!=bp[i] or bp[i]!=y[i]:cases.append(dict(index=i,id=id,label=names[y[i]],global_prediction=names[g[i]],baseline_prediction=names[ap[i]],hard_prediction=names[bp[i]],hard_standalone=names[sp[i]],new_harm=bool(ap[i]==y[i] and bp[i]!=y[i]),new_repair=bool(ap[i]!=y[i] and bp[i]==y[i]),hard_harms_global=bool(g[i]==y[i] and bp[i]!=y[i])))
 result['cases']=cases;write(HARD/'paired_eda.json',result)
 # Exactly all new harms first, then remaining hard-additive global harms sorted by ID.
 new=sorted([c for c in cases if c['new_harm']],key=lambda c:c['id']);other=sorted([c for c in cases if c['hard_harms_global'] and not c['new_harm']],key=lambda c:c['id']);selected=(new+other)[:16];assert len(selected)==16
 boxes={r['image_id']:r['boxes'] for r in json.loads((OUT/'source_object_population/results.json').read_text())['records']};fig,axes=plt.subplots(4,4,figsize=(14,15))
 for j,(ax,c) in enumerate(zip(axes.flat,selected),1):
  ax.imshow(Image.open(ROOT/'domainbed/data/terra_incognita'/c['id']).convert('RGB').resize((224,224),Image.Resampling.BILINEAR))
  for ix in b['top8_indices'][c['index']].tolist():
   row,col=divmod(ix,14);ax.add_patch(Rectangle((col*16,row*16),16,16,fill=False,edgecolor='cyan',linewidth=1))
  for x,z,w,h in boxes.get(c['id'],[]):ax.add_patch(Rectangle((x*224,z*224),w*224,h*224,fill=False,edgecolor='lime',linewidth=1.6))
  ax.set_title(f"{j:02d} GT:{c['label']} | ERM:{c['global_prediction']}\nMean-loss:{c['baseline_prediction']} → hard:{c['hard_prediction']}\nLocal alone:{c['hard_standalone']}",fontsize=8);ax.axis('off')
 fig.suptitle('Local-class hard-negative reader: source-out failed corrections',fontsize=13);fig.text(.5,.015,'Cyan: selected top8 nonempty patches; green: available official boxes.\nSelection overlay, not attention/Grad-CAM. Missing annotations do not imply no object.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.045,1,.97));fig.savefig(HARD/'failed_corrections_4x4.png',dpi=130);plt.close(fig);write(HARD/'grid_manifest.json',dict(cases=selected,selection='New additive harms then remaining hard-additive harms versus frozen ERM, sorted by image ID, first16',source_only=True))
 write(BASE/'aggregation_mismatch.json',dict(source_only=True,target_read=False,quantile_definition='torch.quantile(q,.5), conventional median',states={k:v['empty_aggregation'] for k,v in result['states'].items()}))
 print(json.dumps({k:dict(classes=v['splits']['source_out']['classes'],empty=v['empty_aggregation']) for k,v in result['states'].items()},indent=2))
if __name__=='__main__':main()

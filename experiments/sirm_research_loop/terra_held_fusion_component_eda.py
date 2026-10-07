"""Source EDA of local/calibration components; no new fitting or selection."""
import json
from pathlib import Path
import torch,numpy as np
from .terra_local_fusion_held_domain import DEST,SOURCE
from .terra_local_class_hierarchical_uniform import joint_logprob
from .terra_object_pool_readout import score
from .shrinkage import write,digest

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;data={s:torch.load(DEST/f'{s}_patch_cache.pt',map_location='cpu',weights_only=False) for s in ['source_in','source_out']};states={};combined={n:[] for n in ['full_fusion','global_component_only','local_component_only']}
 for held in [0,2,3]:
  probe=torch.load(SOURCE/f'held_domain{held}_probe.pt',map_location='cpu',weights_only=False);reader=torch.load(DEST/f'held{held}_global_local.pt',map_location='cpu',weights_only=False);features={}
  with torch.no_grad():
   for split,a in data.items():
    qs=[]
    for start in range(0,len(a['features']),128):
     x=a['features'][start:start+128];p=joint_logprob(((x-probe['mean'])/probe['std'])@probe['weight'].T+probe['bias']).exp();ix=(1-p[:,:,5]).topk(8,dim=1).indices;qs.append(p.gather(1,ix[:,:,None].expand(-1,-1,10)).mean(1).clamp_min(1e-8).log())
    raw=torch.cat([a['global_logits'],torch.cat(qs)],1);x=(raw-reader['mean'])/reader['std']/np.sqrt(2);features[split]=dict(global_part=x[:,:10]@reader['weight'][:,:10].T,local_part=x[:,10:]@reader['weight'][:,10:].T)
   d=data['source_in']['domains'];keep=d.ne(held);domains=sorted(d[keep].unique().tolist());center={k:torch.stack([features['source_in'][k][d.eq(dom)].mean(0) for dom in domains]).mean(0) for k in ['global_part','local_part']};states[str(held)]={}
   for split,a in data.items():
    take=a['domains'].ne(held) if split=='source_in' else a['domains'].eq(held);g=a['global_logits'][take];y=a['labels'][take];dd=a['domains'][take];gc=features[split]['global_part'][take]-center['global_part'];lc=features[split]['local_part'][take]-center['local_part'];full=g+gc+lc;assert torch.allclose(full,reader['replays'][split]['logits'],atol=1e-5,rtol=1e-5)
    states[str(held)][split]={}
    for name,f in [('full_fusion',full),('global_component_only',g+gc),('local_component_only',g+lc)]:
     states[str(held)][split][name]=score(g,f,y,dd)
     if split=='source_out':combined[name].append(dict(ids=[id for id,m in zip(a['ids'],take.tolist()) if m],global_logits=g,logits=f,labels=y,domains=dd))
 aggregate={}
 for name,rows in combined.items():
  pooled={k:sum([r[k] for r in rows],[]) if k=='ids' else torch.cat([r[k] for r in rows]) for k in rows[0]};aggregate[name]=score(pooled['global_logits'],pooled['logits'],pooled['labels'],pooled['domains'])
 # Size-stratified errors on available boxes, descriptive and annotation-partial.
 records=json.loads((DEST.parent.parent/'source_object_population/results.json').read_text())['records'];look={id:i for i,id in enumerate(data['source_out']['ids'])};fusion={id:(int(r['labels'][i]),int(r['logits'][i].argmax())) for r in combined['full_fusion'] for i,id in enumerate(r['ids'])};size={}
 for group in ['area_lt_0.05','area_ge_0.05']:
  rr=[r for r in records if (r['box_area']<.05)==(group=='area_lt_0.05')];size[group]=dict(n=len(rr),global_correct=sum(int(data['source_out']['global_logits'][look[r['image_id']]].argmax())==r['label'] for r in rr),fusion_correct=sum(fusion[r['image_id']][1]==r['label'] for r in rr),classes={n:sum(r['class_name']==n for r in rr) for n in sorted({r['class_name'] for r in rr})})
 write(DEST/'component_eda.json',dict(states=states,aggregate=aggregate,size_strata=size,source_only=True,target_read=False,fitting=False,selection=False,scope='Post-hoc additive-component deletion; joint trained coefficients not independently optimal. Not deployable policy chosen here.',code_sha256=digest(Path(__file__))));print(json.dumps({k:{n:v for n,v in s.items() if n in ['accuracy_equal_domain','CE_equal_domain','correct','repairs','harms']} for k,s in aggregate.items()},indent=2));print(json.dumps(size,indent=2))
if __name__=='__main__':main()

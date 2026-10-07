"""Source-only paired EDA and train-fixed correction-amplitude controls."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from .terra_ranked_pool_control import OUTPUT
from .terra_attention_eda import OUT
from .terra_object_pool_readout import score
from .shrinkage import write,digest

def main():
 torch.set_num_threads(4);r=json.loads((OUTPUT/'results.json').read_text());names=json.loads((OUT/'results.json').read_text())['class_names'];probe=json.loads((OUT/'source_object_probe/results.json').read_text());locations={v['id']:v for v in probe['states']['pooled']['splits']['source_out']['cases']};baseline=torch.load(OUTPUT/'all_patch_max_reader.pt',map_location='cpu')['replays'];reference=(baseline['source_in']['logits']-baseline['source_in']['global_logits']).square().mean().sqrt();results={}
 for name in r['states']:
  path=OUTPUT/f'{name}_reader.pt';v=torch.load(path,map_location='cpu')['replays'];train=v['source_in'];out=v['source_out'];g=out['global_logits'];f=out['logits'];y=out['labels'];d=out['domains'];ids=out['ids'];gc=g.argmax(1).eq(y);fc=f.argmax(1).eq(y);u=F.cross_entropy(g,y,reduction='none')-F.cross_entropy(f,y,reduction='none');classes={};cells={};cases=[]
  for cls,c in enumerate(names):
   mask=y.eq(cls);classes[c]=dict(n=int(mask.sum()),global_correct=int(gc[mask].sum()),reader_correct=int(fc[mask].sum()),CE_gain=float(u[mask].mean()),repairs=int((~gc&fc&mask).sum()),harms=int((gc&~fc&mask).sum()))
   for domain in sorted(d.unique().tolist()):
    m=mask&d.eq(domain);count=int(m.sum());cells[f'{domain}/{c}']=dict(n=count,valid=count>=2,CE_gain=float(u[m].mean()) if count>=2 else None)
  for i,id in enumerate(ids):
   if gc[i]!=fc[i] or f[i].argmax()!=baseline['source_out']['logits'][i].argmax():cases.append(dict(id=id,label=names[int(y[i])],global_prediction=names[int(g[i].argmax())],prediction=names[int(f[i].argmax())],allmax_prediction=names[int(baseline['source_out']['logits'][i].argmax())],foreground_peak_inside=locations.get(id,{}).get('peak_patch_center_in_box'),event='repair' if not gc[i] and fc[i] else 'harm' if gc[i] and not fc[i] else 'other'))
  scale=float(reference/(train['logits']-train['global_logits']).square().mean().sqrt());scaled=g+scale*(f-g);results[name]=dict(reader_sha256=digest(path),classes=classes,class_domain_cells=cells,cases=cases,source_train_RMS_match_scale=scale,source_out_train_RMS_matched=score(g,scaled,y,d),source_out_equal_domain_CE_gain=float(torch.stack([u[d.eq(dom)].mean() for dom in sorted(d.unique().tolist())]).mean()))
 write(OUTPUT/'eda.json',dict(source_only=True,target_read=False,fitting=False,minimum_class_domain_samples=2,scope='Single-reader CE gain, not six-expert utility/Q or executed SIRM. Scaling chosen on source-in only.',states=results));print(json.dumps({k:dict(scale=v['source_train_RMS_match_scale'],RMS_matched={x:v['source_out_train_RMS_matched'][x] for x in ['accuracy_equal_domain','repairs','harms']}) for k,v in results.items()},indent=2))
if __name__=='__main__':main()

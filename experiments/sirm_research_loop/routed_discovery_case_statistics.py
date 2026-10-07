"""Quantitative EDA and original-file hash inventory; visual review remains separate."""
import json,hashlib
from pathlib import Path
import numpy as np
from .routed_discovery_run import DEST,ROOT
from .shrinkage import write,digest

def run():
 manifest=json.loads((DEST/'experiment/subset_manifest.json').read_text());names=manifest['class_names'];lookup={r['id']:r for env in manifest['sources'].values() for r in env['images']};result={};review=[]
 for comparison,folder in [('SIRM',DEST/'eda'),('weighting',DEST/'matched_weighting_control/eda')]:
  splits={}
  for split in ['source_out','target_in','target_out']:
   a=np.load(folder/f'{split}_replay.npz');y=a['y'];g=a['B0'];gp=g.argmax(1);wrong=gp!=y;ix=np.arange(len(y));competitor=g.copy();competitor[ix,y]=-np.inf;competitor=competitor.argmax(1);gap=g[ix,y]-g[ix,competitor];arms={}
   for label,arm in [('B1','OFF'),('B2','ON')]:
    full=a[label];pred=full.argmax(1);d=a[f'{arm}_delta'];pi=a[f'{arm}_pi'];single=(g[:,None]+d).argmax(2);capacity=(single==y[:,None]).any(1)&wrong;routed_capacity=((g[:,None]+d*pi[:,:6,None]).argmax(2)==y[:,None]).any(1)&wrong;repairs=wrong&(pred==y);harms=~wrong&(pred!=y);correction=full-g;added=correction[ix,y]-correction[ix,competitor]
    arms[label]={'repairs':int(repairs.sum()),'harms':int(harms.sum()),'global_errors':int(wrong.sum()),'bank_errors':int((pred!=y).sum()),'unscaled_single_expert_capacity':int(capacity.sum()),'routed_single_expert_capacity':int(routed_capacity.sum()),'routed_capacity_without_deployed_repair':int((routed_capacity&~repairs).sum()),'wrong_global_margin_median':float(np.median(gap[wrong])),'wrong_margin_correction_median':float(np.median(added[wrong])),'per_class':[{'class':name,'n':int((y==c).sum()),'global_errors':int(((y==c)&wrong).sum()),'bank_errors':int(((y==c)&(pred!=y)).sum()),'repairs':int(((y==c)&repairs).sum()),'harms':int(((y==c)&harms).sum()),'capacity':int(((y==c)&capacity).sum())} for c,name in enumerate(names)]}
   pairs=[('logits','B1','B2'),('global','OFF_g','ON_g'),('residual','OFF_delta','ON_delta'),('routing','OFF_pi','ON_pi'),('utility','B1_utility','B2_utility')]
   comparisons={name:{'exact':bool(np.array_equal(a[x],a[z])),'max_abs_difference':float(np.max(np.abs(a[x]-a[z])))} for name,x,z in pairs};splits[split]={'n':len(y),'arms':arms,'differences':comparisons,'prediction_disagreement':int((a['B1'].argmax(1)!=a['B2'].argmax(1)).sum()),'npz_sha256':digest(folder/f'{split}_replay.npz')}
  result[comparison]=splits
  for spec in json.loads((folder/'contact_sheet_index.json').read_text()):
   file=folder/f"{spec['split']}_{spec['pattern']}.png";rows=[]
   for row in spec['cases']:
    image=row['image_id'];sha=digest(ROOT/'domainbed/data/terra_incognita'/image);assert sha==lookup[image]['sha256'];rows.append({'image_id':image,'original_sha256':sha})
   review.append({'comparison':comparison,'sheet':str(file.relative_to(DEST)),'sha256':digest(file),'cases':rows,'visually_reviewed':False})
 write(DEST/'eda/all_case_statistics.json',{'scope':'All locked replay rows, label-informed capacity and per-class errors. SIRM comparison is routed-discovery OFF/ON; weighting comparison is iteration14 unscaled OFF/iteration15 routed OFF. No fitting or method selection.','comparisons':result})
 write(DEST/'eda/visual_review_inventory.json',{'all_original_hashes_match_manifest':True,'case_occurrences':sum(len(r['cases']) for r in review),'unique_image_ids':len({c['image_id'] for r in review for c in r['cases']}),'visual_review_complete':False,'sheets':review})
 return result
if __name__=='__main__':print(json.dumps(run(),indent=2))

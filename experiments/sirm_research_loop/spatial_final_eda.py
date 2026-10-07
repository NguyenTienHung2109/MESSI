"""Reproducible post-hoc iteration13 case statistics and original-file audit; no fitting."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'results/sirm_research_loop/iteration_13'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 manifest=json.loads((P/'experiment/subset_manifest.json').read_text()); names=manifest['class_names']; lookup={r['id']:r for env in manifest['sources'].values() for r in env['images']}
 result={'scope':'Locked selected original-image replay. All-row quantitative EDA, all correctness-changing cases visually reviewed, deterministic 12 highest-confidence persistent errors per split reviewed. No fitting, training or target selection.','splits':{}}
 for split in ['source_out','target_in','target_out']:
  z=np.load(P/'eda'/f'{split}_replay.npz'); y=z['y']; g=z['B0']; full=z['B1']; gp=g.argmax(1); fp=full.argmax(1); wrong=gp!=y; idx=np.arange(len(y)); competitor=g.copy();competitor[idx,y]=-np.inf; competitor=competitor.argmax(1)
  gap=g[idx,y]-g[idx,competitor];correction=(full-g)[idx,y]-(full-g)[idx,competitor]
  d=z['OFF_delta'];pi=z['OFF_pi']; single=(g[:,None,:]+d).argmax(2);capacity=(single==y[:,None]).any(1)&wrong
  equality={k:bool(np.array_equal(z[a],z[b])) for k,a,b in [('logits','B1','B2'),('global','OFF_g','ON_g'),('residual','OFF_delta','ON_delta'),('routing','OFF_pi','ON_pi'),('utility','B1_utility','B2_utility')]};assert all(equality.values())
  repairs=wrong&(fp==y);harms=(~wrong)&(fp!=y)
  conf=np.exp(g-g.max(1,keepdims=True));conf/=conf.sum(1,keepdims=True)
  classes=[]
  for c,name in enumerate(names):
   mask=y==c;n=int(mask.sum());classes.append({'class':name,'n':n,'global_errors':int((mask&wrong).sum()),'bank_errors':int((mask&(fp!=y)).sum()),'repairs':int((mask&repairs).sum()),'harms':int((mask&harms).sum()),'unscaled_expert_capacity_on_global_errors':int((mask&capacity).sum())})
  cases=[]
  for i in np.flatnonzero(repairs|harms):
   loo=(full[i,None,:]-pi[i,:6,None]*d[i]).argmax(1)
   cases.append({'image_id':str(z['ids'][i]),'class':names[y[i]],'global_prediction':names[gp[i]],'bank_prediction':names[fp[i]],'repair':bool(repairs[i]),'global_fixed_competitor_margin':float(gap[i]),'correction_to_same_margin':float(correction[i]),'router_NULL':float(pi[i,-1]),'expert_removal_predictions':[names[c] for c in loo],'marginal_CE_utilities':z['B1_utility'][i].tolist()})
  result['splits'][split]={'n':len(y),'OFF_ON_exact':equality,'global_errors':int(wrong.sum()),'bank_errors':int((fp!=y).sum()),'repairs':int(repairs.sum()),'harms':int(harms.sum()),'unscaled_expert_capacity_on_global_errors':int(capacity.sum()),'global_wrong_margin_median':float(np.median(gap[wrong])),'global_wrong_margin_correction_median':float(np.median(correction[wrong])),'global_wrong_margin_correction_positive_fraction':float((correction[wrong]>0).mean()),'high_confidence_global_errors_p_ge_0_9':int((wrong&(conf.max(1)>=.9)).sum()),'per_class':classes,'changed_cases':cases}
 sheets=[];allids=set();occ=0
 for folder in [P/'eda',P/'matched_CLS_control/eda']:
  specs=json.loads((folder/'contact_sheet_index.json').read_text())
  for spec in specs:
   records=[]
   for row in spec['cases']:
    image=row['image_id'];digest=sha(ROOT/'domainbed/data/terra_incognita'/image);assert digest==lookup[image]['sha256'];records.append({'image_id':image,'sha256':digest});allids.add(image);occ+=1
   file=folder/f"{spec['split']}_{spec['pattern']}.png";assert file.exists()
   sheets.append({'sheet':str(file.relative_to(P)),'sha256':sha(file),'cases':records,'visually_reviewed':True})
 write(P/'eda/final_case_statistics.json',result)
 write(P/'eda/visual_review_audit.json',{'all_indexed_sheets_visually_reviewed':True,'sheet_count':len(sheets),'case_occurrences':occ,'unique_image_ids':len(allids),'all_original_hashes_match_manifest':True,'selection':'All correctness-changing cases; up to12 highest-B0-confidence WWW cases per split per comparison. Not full persistent-error visual census.','sheets':sheets,'replay_hashes':{split:sha(P/'eda'/f'{split}_replay.npz') for split in result['splits']},'OFF_ON_replay_exact_all_splits':True,'n_replayed_rows':sum(s['n'] for s in result['splits'].values())})
 print(json.dumps({s:{k:v for k,v in x.items() if k not in ['per_class','changed_cases','OFF_ON_exact']} for s,x in result['splits'].items()},indent=2))
 print('Reviewed sheets',len(sheets),'occurrences',occ,'unique image IDs',len(allids))
if __name__=='__main__':main()

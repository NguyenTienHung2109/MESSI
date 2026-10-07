"""Prespecified source failure grids from completed locked replays, no inference."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from . import failed_case_grids as grids
from .shrinkage import ROOT,digest,write
DEST=ROOT/'results/sirm_research_loop/iteration_19'
def main():
 proof=json.loads((DEST/'source_analysis/OFF_results.json').read_text());assert proof['source_only'] and not proof['target_read'];manifest=json.loads((DEST/'experiment/subset_manifest.json').read_text());names=manifest['class_names'];hashes={r['id']:r['sha256'] for s in manifest['sources'].values() for r in s['images']};grids.OUT=DEST/'OFF_source_failure_grids';grids.OUT.mkdir(exist_ok=True);assert not (grids.OUT/'index.json').exists();items={}
 for arm in ['OFF']:
  for state in ['selected','last']:
   p=DEST/'source_analysis'/f'{arm}_{state}_source_out_replay.npz';a=dict(np.load(p));g=a['global_logits'];f=a['full_logits'];y=a['y'];old=g.argmax(1);new=f.argmax(1);oc=softmax(g,axis=1).max(1);nc=softmax(f,axis=1).max(1);indices=grids.select(a['ids'],y,old,new,nc,True);step=proof['banks'][arm+'/'+state]['step'];title=f'Source validation | iteration19 {arm} {state}, step{step} | source-conditioned confidence router';item=grids.draw(arm+'_'+state+'_4x4',title,a['ids'],y,old,new,oc,nc,names,indices,'Frozen ERM','Residual bank',hashes);classes={};confusion=np.zeros((10,10),dtype=int)
   for label,pred in zip(y,new):confusion[label,pred]+=1
   for label,name in enumerate(names):
    mask=y==label;classes[name]=dict(n=int(mask.sum()),global_correct=int((mask&(old==y)).sum()),full_correct=int((mask&(new==y)).sum()),repairs=int((mask&(old!=y)&(new==y)).sum()),harms=int((mask&(old==y)&(new!=y)).sum()))
   item.update(replay_sha256=digest(p),checkpoint_sha256=proof['banks'][arm+'/'+state]['checkpoint_sha256'],total_images=len(y),failures=int((new!=y).sum()),repairs=int(((old!=y)&(new==y)).sum()),harms=int(((old==y)&(new!=y)).sum()),class_counts=classes,confusion_matrix=confusion.tolist(),class_order=names);items[arm+'/'+state]=item
 write(grids.OUT/'index.json',dict(grids=items,source_only=True,target_read=False,inference=False,training=False,selection='All new harms first; source failed birds next; then class round-robin high-confidence errors. Not a prevalence sample.',code_sha256=digest(Path(__file__))))
 print(json.dumps({k:{x:v[x] for x in ['file','failures','repairs','harms']} for k,v in items.items()},indent=2))
if __name__=='__main__':main()

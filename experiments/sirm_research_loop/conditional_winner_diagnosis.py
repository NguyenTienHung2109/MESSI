"""Source-only fixed-cache winner/utility generalization; no fitting or target read."""
import hashlib,json
from pathlib import Path
import numpy as np
BASE=Path('results/sirm_research_loop/iteration_08')
OUT=BASE/'source_router_diagnosis'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ce(x,y):
 x=x.astype(np.float64); a=x.max(-1);return a+np.log(np.exp(x-a[...,None]).sum(-1))-np.take_along_axis(x,np.broadcast_to(y[:,None,None],x.shape[:-1]+(1,)),axis=-1)[...,0]
def evaluate(a):
 loss=ce(a['g'][:,None,:]+a['delta'],a['y']); ties=loss==loss.min(1,keepdims=True);w=ties/ties.sum(1,keepdims=True)
 return w,loss

def main():
 paths={s:OUT/f'{s}_original.npz' for s in ['source_in','source_out']};a={s:dict(np.load(p)) for s,p in paths.items()};w={s:evaluate(v)[0] for s,v in a.items()}
 results={'scope':'Selected375 fixed source caches only. Winners use true labels and are diagnostic oracles. Exact ties receive fractional mass. No target, training, threshold changes, or method selection. Utilities retain contextual routed definition. Cells require2 images descriptively, not a gate/ESS change.','cache_hashes':{s:digest(p) for s,p in paths.items()},'code_sha256':digest(Path(__file__)),'groups':{},'cells':[],'prototype_controls':{}}
 classes=sorted(set(a['source_in']['y']));ds=sorted(set(a['source_in']['domains']))
 for split,v in a.items():
  groups={};correct=v['g'].argmax(1)==v['y']
  for name,mask in [('all',np.ones(len(correct),bool)),('global_wrong',~correct),('global_correct',correct)]:
   groups[name]={'n':int(mask.sum()),'fractional_winner_load':w[split][mask].mean(0).tolist(),'U':v['marginal'][mask].mean(0).tolist(),'single_utility':v['single'][mask].mean(0).tolist()}
  results['groups'][split]=groups
 for d in ds:
  for c in classes:
   masks={s:(a[s]['domains']==d)&(a[s]['y']==c) for s in a}
   if any(m.sum()<2 for m in masks.values()):continue
   row={'domain':int(d),'class':int(c)}
   for s,m in masks.items():
    row[s]={'n':int(m.sum()),'winner_load':w[s][m].mean(0).tolist(),'U':a[s]['marginal'][m].mean(0).tolist(),'single_utility':a[s]['single'][m].mean(0).tolist(),'global_wrong_n':int((a[s]['g'][m].argmax(1)!=a[s]['y'][m]).sum())}
   results['cells'].append(row)
 # Source-in calibrated winner prototypes. Tied prototype maxima also fractional.
 for kind in ['constant','class','class_domain']:
  scores=[];out=a['source_out']
  for i in range(len(out['y'])):
   mask=np.ones(len(a['source_in']['y']),bool)
   if kind!='constant':mask &= a['source_in']['y']==out['y'][i]
   if kind=='class_domain':mask &= a['source_in']['domains']==out['domains'][i]
   load=w['source_in'][mask].mean(0);p=(load==load.max()).astype(float);p/=p.sum();scores.append(float(p@w['source_out'][i]))
  scores=np.array(scores);results['prototype_controls'][kind]={'pooled_winner_match':float(scores.mean()),'equal_domain_winner_match':float(np.mean([scores[out['domains']==d].mean() for d in ds])),'global_wrong_winner_match':float(scores[out['g'].argmax(1)!=out['y']].mean())}
 train=np.array([r['source_in']['U'] for r in results['cells']]);val=np.array([r['source_out']['U'] for r in results['cells']]);single_train=np.array([r['source_in']['single_utility'] for r in results['cells']]);single_val=np.array([r['source_out']['single_utility'] for r in results['cells']]);
 def signs(t,v):return {'cells':int(t.size),'positive_train':int((t>0).sum()),'positive_both':int(((t>0)&(v>0)).sum()),'train_positive_to_validation_negative':int(((t>0)&(v<0)).sum()),'above_002_train':int((t>.002).sum()),'above_002_both':int(((t>.002)&(v>.002)).sum()),'pearson':float(np.corrcoef(t.ravel(),v.ravel())[0,1])}
 results['matched_class_domain_utility']=signs(train,val);results['matched_class_domain_single_utility']=signs(single_train,single_val)
 results['class_domain_winner_mode_agreement']=float(np.mean([np.argmax(r['source_in']['winner_load'])==np.argmax(r['source_out']['winner_load']) for r in results['cells']]))
 (OUT/'conditional_winner_results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({k:v for k,v in results.items() if k not in ['cells','groups']},indent=2))
if __name__=='__main__':main()

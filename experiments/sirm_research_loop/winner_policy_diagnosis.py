"""Source-only frozen-bank class winner controls. Label policies are oracles."""
import json,hashlib
from pathlib import Path
import numpy as np
from experiments.sirm_research_loop.conditional_winner_diagnosis import evaluate,ce
B=Path('results/sirm_research_loop/iteration_08/source_router_diagnosis')
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a={s:dict(np.load(B/f'{s}_original.npz')) for s in ['source_in','source_out']}
w={s:evaluate(v)[0] for s,v in a.items()};train=a['source_in'];out=a['source_out'];global_ok=out['g'].argmax(1)==out['y'];results={'scope':'Source-in calibrated expert choices; fixed source-out original bank. No target or fit. True-class policies use validation labels and are oracles. g+delta changes strength relative to the deployed routed mixture; routed_single retains existing pi without renormalization. Known-source-domain policy is not a target deployment rule.','cache_hashes':{s:h(B/f'{s}_original.npz') for s in a},'code_sha256':h(Path(__file__)),'policies':{},'error_conditioned_utility':[]}
for class_kind in ['true_class','global_predicted_class']:
 for domain_kind in ['class_only','class_and_source_domain']:
  weights=[]
  for i in range(len(out['y'])):
   c=int(out['y'][i] if class_kind=='true_class' else out['g'][i].argmax());mask=train['y']==c
   if domain_kind=='class_and_source_domain':mask &= train['domains']==out['domains'][i]
   if not mask.any():raise ValueError('Unseen source class/domain')
   load=w['source_in'][mask].mean(0);p=(load==load.max()).astype(float);p/=p.sum();weights.append(p)
  weights=np.array(weights)
  for strength in ['unscaled','routed_single']:
   scale=weights if strength=='unscaled' else weights*out['pi'][:,:6];logits=out['g']+(scale[:,:,None]*out['delta']).sum(1);correct=logits.argmax(1)==out['y'];loss=ce(logits[:,None,:],out['y'])[:,0]
   results['policies'][f'{class_kind}/{domain_kind}/{strength}']={'equal_domain_accuracy':float(np.mean([correct[out['domains']==d].mean() for d in np.unique(out['domains'])])),'equal_domain_CE':float(np.mean([loss[out['domains']==d].mean() for d in np.unique(out['domains'])])),'repairs':int((~global_ok&correct).sum()),'harms':int((global_ok&~correct).sum())}
for d in np.unique(train['domains']):
 for c in np.unique(train['y']):
  m={s:(v['domains']==d)&(v['y']==c)&(v['g'].argmax(1)!=v['y']) for s,v in a.items()}
  if any(x.sum()<2 for x in m.values()):continue
  results['error_conditioned_utility'].append({'domain':int(d),'class':int(c),**{s:{'n':int(mask.sum()),'U':a[s]['marginal'][mask].mean(0).tolist(),'single_U':a[s]['single'][mask].mean(0).tolist()} for s,mask in m.items()}})
rows=results['error_conditioned_utility'];t=np.array([r['source_in']['U'] for r in rows]);v=np.array([r['source_out']['U'] for r in rows]);results['error_conditional_summary']={'class_domain_groups':len(rows),'expert_cells':int(t.size),'train_positive':int((t>0).sum()),'positive_both':int(((t>0)&(v>0)).sum()),'train_positive_to_validation_negative':int(((t>0)&(v<0)).sum()),'pearson':float(np.corrcoef(t.ravel(),v.ravel())[0,1]) if t.size else None}
(B/'winner_policy_results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({k:v for k,v in results.items() if k!='error_conditioned_utility'},indent=2))

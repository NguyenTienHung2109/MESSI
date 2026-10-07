"""Source-only saved-output audit for paired shared-utility discovery candidate."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from .pair_shared_discovery import paired_shared_loss
from experiments.sirm_discover.math import expert_ce
from .shrinkage import digest,write
B=Path('results/sirm_research_loop/iteration_08');out=B/'pair_shared_saved_audit';out.mkdir(exist_ok=True);paths={s:B/f'source_router_diagnosis/{s}_original.npz' for s in ['source_in','source_out']};plan={'code_sha256':digest(Path(__file__)),'candidate_code_sha256':digest(Path('experiments/sirm_research_loop/pair_shared_discovery.py')),'cache_hashes':{s:digest(p) for s,p in paths.items()},'scope':'Saved selected375 corrected logits treated as independent delta variables; derivative norms are output-logit gradients, not parameter gradients/historical updates. No target, bank fitting or primary method changes. Source-in pair/class winners assessed on source-out matching labels/domains as diagnostic, not deployable policy.','hypothesis':'Per-image hard competition admits class/camera-specific winners. Same-expert worst relative CE across a source-domain pair may reward correction with shared positive utility. Thresholds, Q, teacher, lambda and balance remain unchanged in this candidate.'};write(out/'plan.json',plan);raw={s:dict(np.load(p)) for s,p in paths.items()};results={};means={}
for split,a in raw.items():
 g=torch.tensor(a['g']);y=torch.tensor(a['y']);dom=torch.tensor(a['domains']);delta=torch.tensor(a['delta'],requires_grad=True);hard=expert_ce(g[:,None]+delta,y).amin(1).mean();pair,details=paired_shared_loss(g,delta,y,dom);gh=torch.autograd.grad(hard,delta,retain_graph=True)[0];gp=torch.autograd.grad(pair,delta)[0];results[split]={'per_image_hard_CE':float(hard),'pair_relative_loss':float(pair),'pair_logit_gradient_norm':float(gp.norm()),'hard_logit_gradient_norm':float(gh.norm()),'gradient_norm_ratio':float(gp.norm()/gh.norm()),'gradient_cosine':float(torch.nn.functional.cosine_similarity(gp.flatten(),gh.flatten(),dim=0)),'pair_details':details}
 single=a['single'];means[split]={(int(c),int(d)):single[(a['y']==c)&(a['domains']==d)].mean(0) for c in np.unique(a['y']) for d in np.unique(a['domains']) if ((a['y']==c)&(a['domains']==d)).sum()>=2}
rows=[]
for c in np.unique(raw['source_in']['y']):
 for i,j in itertools.combinations(np.unique(raw['source_in']['domains']),2):
  keys=[(int(c),int(i)),(int(c),int(j))]
  if any(k not in m for m in means.values() for k in keys):continue
  train=np.minimum(means['source_in'][keys[0]],means['source_in'][keys[1]]);val=np.minimum(means['source_out'][keys[0]],means['source_out'][keys[1]]);select=(train==train.max()).astype(float);select/=select.sum();rows.append({'class':int(c),'domains':[int(i),int(j)],'train_best_worst_U':float(train@select),'validation_U_for_train_winner':float(val@select),'winner_weights':select.tolist()})
summary={'matched_pair_class_groups':len(rows),'positive_train_best':sum(r['train_best_worst_U']>0 for r in rows),'positive_both':sum(r['train_best_worst_U']>0 and r['validation_U_for_train_winner']>0 for r in rows),'above_002_train':sum(r['train_best_worst_U']>.002 for r in rows),'above_002_both':sum(r['train_best_worst_U']>.002 and r['validation_U_for_train_winner']>.002 for r in rows)}
assert all(digest(p)==plan['cache_hashes'][s] for s,p in paths.items());write(out/'results.json',{'plan':plan,'output_gradient_diagnostics':results,'pair_transfer_summary':summary,'pair_transfer_rows':rows});print(summary);print({s:{k:v for k,v in r.items() if k!='pair_details'} for s,r in results.items()})

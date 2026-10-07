"""Source-only same-error-contrast support and fixed-bank utility diagnosis."""
import json,itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from scipy.special import log_softmax
from .routed_discovery_run import ROOT,DEST,PRIOR
from .routed_conditional_utility import load
from .shrinkage import digest,write
from .source_probe_penalty_direction import outputs
OUT=DEST/'source_error_contrast_support'

def table(g,d,y,dom,ids,metadata,pi):
 pred=g.argmax(1);wrong=pred!=y;full=g+np.einsum('nm,nmc->nc',pi[:,:6],d);prob=np.exp(log_softmax(full,1))-np.eye(10)[y];routed=pi[:,:6,None]*d
 ce=-log_softmax(g,1)[np.arange(len(g)),y];single=-log_softmax(g[:,None]+d,2)[np.arange(len(g))[:,None],np.arange(6)[None],y[:,None]];gain=ce[:,None]-single;G=np.einsum('nc,nmc->nm',prob,routed)
 cells=[]
 for label,guess in sorted(set(zip(y[wrong].tolist(),pred[wrong].tolist()))):
  for domain in sorted(set(dom)):
   k=(y==label)&(pred==guess)&(dom==domain);n=int(k.sum())
   if not n:continue
   event={metadata[str(id)]['official_sequence'] for id in ids[k] if metadata[str(id)]['official_sequence'] is not None};rawpred=(g[k,None]+d[k]).argmax(2);corr=np.array([d[k,i,label]-d[k,i,guess] for i in range(6)]).T
   cells.append(dict(label=label,global_prediction=guess,domain=int(domain),n=n,distinct_images=len(set(map(str,ids[k]))),distinct_official_sequences=len(event),count_eligible=n>=2,distinct_image_eligible=len(set(map(str,ids[k])))>=2,mean_raw_CE_gain=gain[k].mean(0).tolist(),raw_repair_count=(rawpred==label).sum(0).tolist(),mean_error_class_contrast=corr.mean(0).tolist(),mean_native_routed_G=G[k].mean(0).tolist(),ids=ids[k].tolist()))
 def shared(key,distinct=False):
  groups=defaultdict(list)
  for c in cells:
   if (c['distinct_image_eligible'] if distinct else c['count_eligible']):groups[key(c)].append(c['domain'])
  return [dict(key=list(k) if isinstance(k,tuple) else int(k),domains=sorted(set(v))) for k,v in groups.items() if len(set(v))>=2]
 # Class support must aggregate all error confusions before checking count.
 cls=[]
 for label in sorted(set(y[wrong].tolist())):
  dd=[]
  for domain in sorted(set(dom)):
   k=wrong&(y==label)&(dom==domain)
   if int(k.sum())>=2:dd.append(int(domain))
  if len(dd)>=2:cls.append(dict(label=label,domains=dd))
 return dict(n=len(y),global_errors=int(wrong.sum()),error_cells=cells,shared_error_classes=cls,shared_error_contrasts=shared(lambda c:(c['label'],c['global_prediction'])),shared_contrasts_distinct_image_min2=shared(lambda c:(c['label'],c['global_prediction']),True))

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();locks={};rank=json.loads((DEST/'source_residual_rank_diagnosis/results.json').read_text());metadatafile=ROOT/'results/sirm_research_loop/iteration_08/source_annotation_audit/selected_source_results.json';meta=json.loads(metadatafile.read_text());locks[str(metadatafile)]=digest(metadatafile);metadata={x['local_id']:x for x in meta['records']};source={}
 for split in ['source_in','source_out']:
  path=ROOT/f'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/{split}_tokens.npz';assert digest(path)==rank['input_sha256'][str(path)];source[split]=dict(np.load(path));locks[str(path)]=digest(path)
 proofpath=DEST/'source_conditional_H2/extraction_audit.json';proof=json.loads(proofpath.read_text());locks[str(proofpath)]=digest(proofpath)
 plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,minimum_count=2,hypothesis='A class label can mix different frozen-global confusions; same (true class, frozen-global predicted class) populations may have different support/utility. Test this before error-contrast discovery.',scope='Two saved selected banks. Original source-in/out plus40 exact scheduled H2 minibatches replayed at saved weights; not historical parameter gradients or training histories. Source-in chooses diagnostic group experts; source-out is evaluation only.',limits='True-label confusion groups are nondeployable diagnostic cohorts, not an inference gate. Minimum2 images is retained. Distinct image/event counts diagnose reuse, not new training thresholds. H2 views of one image are not independent samples. Raw individual-expert CE gain is separate from native routed scale derivative and target transfer.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());states={}
 for state in rank['states']:
  bp=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';assert digest(bp)==rank['input_sha256'][str(bp)];bank=dict(np.load(bp));locks[str(bp)]=digest(bp);original={}
  for split,b in source.items():
   assert np.array_equal(b['ids'],bank[split+'_ids']);original[split]=table(b['global_logits'].astype(float),bank[split+'_bank_delta'].astype(float),b['y'],b['domains'],b['ids'],metadata,bank[split+'_bank_pi'].astype(float))
  # Choose one raw expert per supported original source-in error contrast, equally across its eligible domains.
  selected=[]
  for group in original['source_in']['shared_error_contrasts']:
   label,guess=group['key'];train=[c for c in original['source_in']['error_cells'] if c['label']==label and c['global_prediction']==guess and c['count_eligible']];u=np.mean([c['mean_raw_CE_gain'] for c in train],0);expert=int(np.argmax(u));tests=[c for c in original['source_out']['error_cells'] if c['label']==label and c['global_prediction']==guess];selected.append(dict(label=label,global_prediction=guess,expert=expert,source_in_equal_eligible_domain_CE_gain=float(u[expert]),source_in_cells=train,source_out_cells=tests,source_out_equal_present_domain_CE_gain=float(np.mean([c['mean_raw_CE_gain'][expert] for c in tests])) if tests else None,source_out_raw_repairs=sum(c['raw_repair_count'][expert] for c in tests),source_out_images=sum(c['n'] for c in tests)))
  m,p,path=load(PRIOR if state.startswith('unscaled') else DEST,'selected');locks[str(path)]=digest(path);before={k:v.clone() for k,v in m.state_dict().items()};assert not any(v.requires_grad for module in [m.featurizer,m.global_head] for v in module.parameters());batches=[];pool=[]
  with torch.no_grad():
   for rec in proof['rows']:
    f=DEST/'source_conditional_H2'/f"H2_step{rec['step']}.pt";assert digest(f)==rec['sha256'];locks[str(f)]=digest(f);b=torch.load(f,map_location='cpu',weights_only=False);g,d,_=outputs(m,b['raw'],torch.zeros((len(b['y']),7)),True);pi=m.router(m.bank_features(b['raw'])).softmax(1);assert pi.shape==(len(b['y']),7) and torch.allclose(pi.sum(1),torch.ones(len(pi)));arrs=[z.numpy().astype(float) for z in [g,d,pi]];tab=table(*arrs[:2],b['y'].numpy(),b['domains'].numpy(),np.array(b['ids']),metadata,arrs[2]);batches.append(dict(step=rec['step'],**tab));pool.append((arrs,b))
  pooledg=np.concatenate([x[0][0] for x in pool]);pooledd=np.concatenate([x[0][1] for x in pool]);pooledpi=np.concatenate([x[0][2] for x in pool]);pooledy=np.concatenate([x[1]['y'].numpy() for x in pool]);pooleddom=np.concatenate([x[1]['domains'].numpy() for x in pool]);pooledids=np.concatenate([np.array(x[1]['ids']) for x in pool]);pooled=table(pooledg,pooledd,pooledy,pooleddom,pooledids,metadata,pooledpi)
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and all(v.grad is None for v in m.parameters());states[state]=dict(original=original,source_in_selected_contrast_experts=selected,H2_batches=batches,H2_pooled=pooled);print(state,'original contrasts',len(selected),'H2 windows with contrasts',sum(bool(x['shared_error_contrasts']) for x in batches),flush=True)
 for f,h in locks.items():assert digest(Path(f))==h
 write(OUT/'results.json',dict(plan=plan,states=states,input_sha256=locks,target_read=False,bank_tensors_unchanged=True,grad_fields_unassigned=True))
if __name__=='__main__':run()

"""Case-grounded source EDA for completed native guard trajectory; no target."""
import json,csv
from pathlib import Path
import numpy as np
import torch
from .routed_conditional_utility import load
from .source_probe_penalty_direction import outputs
from .routed_discovery_run import DEST as PRIOR
from .shrinkage import digest,write
DEST=PRIOR.parent/'iteration_16';OUT=DEST/'source_analysis'
def run(state="selected"):
 source_file=OUT/'results.json' if (OUT/'results.json').exists() else OUT/'OFF_results.json';result=json.loads(source_file.read_text());oldfile=PRIOR/'eda/source_out_replay.npz';mapping=json.loads((PRIOR/'matched_weighting_control/arm_mapping.json').read_text());assert mapping['B2']=='iteration15 detached-routed discovery OFF';old=dict(np.load(oldfile));newfile=OUT/f'OFF_{state}_source_out_replay.npz';new=dict(np.load(newfile));assert np.array_equal(old['ids'],new['ids']) and np.array_equal(old['y'],new['y']) and np.array_equal(old['domains'],new['domains']);m,payload,checkpoint=load(PRIOR,state);cache=PRIOR.parent/'iteration_12/frozen_token_source_diagnosis'
 torch.set_num_threads(4);train=dict(np.load(cache/'source_in_tokens.npz'));validation=dict(np.load(cache/'source_out_tokens.npz'));m.calibrate_features(torch.tensor(np.concatenate([train['cls'],train['patch_max']],1)),torch.tensor(train['domains']));raw=torch.tensor(np.concatenate([validation['cls'],validation['patch_max']],1))
 with torch.no_grad():pi=m.router(m.bank_features(raw)).softmax(1);g,_,f=outputs(m,raw,pi)
 assert np.array_equal(g.numpy(),new['global_logits']);old_cache_replay_difference=float(np.max(np.abs(old['B2']-f.numpy())));old['B2']=f.numpy()

 y=new['y'];global_logits=new['global_logits'];prior=old['B2'];current=new['full_logits'];gc=global_logits.argmax(1)==y;pc=prior.argmax(1)==y;nc=current.argmax(1)==y
 def margin(logits):z=logits.copy();z[np.arange(len(y)),y]=-np.inf;return logits[np.arange(len(y)),y]-z.max(1)
 def ce(logits):mx=logits.max(1);return mx+np.log(np.exp(logits-mx[:,None]).sum(1))-logits[np.arange(len(y)),y]
 gm,pm,nm=map(margin,[global_logits,prior,current]);pce,nce=map(ce,[prior,current]);rows=[]
 for i,id in enumerate(new['ids']):rows.append(dict(id=str(id),domain=int(new['domains'][i]),label=int(y[i]),global_prediction=int(global_logits[i].argmax()),prior_prediction=int(prior[i].argmax()),guard_prediction=int(current[i].argmax()),global_margin=float(gm[i]),prior_margin=float(pm[i]),guard_margin=float(nm[i]),prior_CE=float(pce[i]),guard_CE=float(nce[i]),guard_minus_prior_CE=float(nce[i]-pce[i]),prior_repairs=bool(~gc[i]&pc[i]),guard_repairs=bool(~gc[i]&nc[i]),prior_harms=bool(gc[i]&~pc[i]),guard_harms=bool(gc[i]&~nc[i]),guard_expert_utilities=new['utility'][i].tolist()))
 fields=[k for k in rows[0] if k!='guard_expert_utilities']
 with (OUT/f'source_case_comparison_{state}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in rows])
 classes={}
 for c in np.unique(y):
  mask=y==c;classes[str(int(c))]=dict(n=int(mask.sum()),global_correct=int(gc[mask].sum()),prior_correct=int(pc[mask].sum()),guard_correct=int(nc[mask].sum()),guard_CE_change_mean=float((nce-pce)[mask].mean()),guard_utility_mean=new['utility'][mask].mean(0).tolist())
 changed=[r for i,r in enumerate(rows) if prior[i].argmax()!=current[i].argmax()];sentinels={prefix:next(r for r in rows if prefix in r['id']) for prefix in ['59f2c51e','59bfdfaa','598c5b04','59eab95e','59bfe004','5a096976']}
 o=dict(source_only=True,target_read=False,scope=('Independent source-selected checkpoints; possibly different selected steps.' if state=='selected' else 'Common final step399 checkpoints.')+' Guard-only intervention, frozen anchor and schedules matched; cached B2 explicitly means iteration15 OFF, not ON treatment.',prior_cache_sha256=digest(oldfile),prior_checkpoint_sha256=digest(checkpoint),prior_replayed_same_CPU_source_tokens=True,old_GPU_cache_max_logit_difference=old_cache_replay_difference,current_cache_sha256=digest(newfile),current_selected_step=result['banks']['OFF/'+state]['step'],previous_selected_step=payload.get('selected_step',payload.get('step')),prior_repairs=int((~gc&pc).sum()),guard_repairs=int((~gc&nc).sum()),prior_harms=int((gc&~pc).sum()),guard_harms=int((gc&~nc).sum()),prior_wrong_guard_correct=int((~pc&nc).sum()),prior_correct_guard_wrong=int((pc&~nc).sum()),classes=classes,changed_cases=changed,sentinel_cases=sentinels)
 write(OUT/('case_comparison.json' if state=='selected' else 'case_comparison_last.json'),o);(OUT/'case_comparison_code_snapshot.py').write_bytes(Path(__file__).read_bytes());print('Source case comparison complete')
if __name__=='__main__':run()

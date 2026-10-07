"""Completed-run source-only EDA; target untouched and source locks verified."""
import json,copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .routed_discovery_run import ROOT,DEST as PRIOR
from .routed_conditional_utility import load
from .source_probe_penalty_direction import outputs
from .shrinkage import digest,write
DEST=PRIOR.parent/'iteration_16';OUT=DEST/'source_analysis'
def run(only_off=False):
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);result_file=OUT/('OFF_results.json' if only_off else 'results.json');assert not result_file.exists();assert (DEST/'experiment/runs/L38/OFF/done').exists();
 if not only_off:assert json.loads((DEST/'training_audit.json').read_text())['passed']
 plan=json.loads((DEST/'plan.json').read_text());locks={};data={};cache=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';proof=json.loads((cache/'extraction_audit.json').read_text())
 for split in ['source_in','source_out']:
  path=cache/(split+'_tokens.npz');assert digest(path)==next(r['cache_sha256'] for r in proof['rows'] if r['split']==split);locks[str(path)]=digest(path);a=dict(np.load(path));data[split]=(torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains']),a['ids'])
 summaries={};traces={};content={}
 for arm in (['OFF'] if only_off else ['OFF','ON']):
  bank=DEST/'experiment/runs/L38'/arm;assert (bank/'done').exists();history=[json.loads(l) for l in (bank/'diagnostics.jsonl').read_text().splitlines()];refresh=[json.loads(l) for l in (bank/'q_refresh.jsonl').read_text().splitlines() if json.loads(l)['step']>=200];assert [r['step'] for r in history]==list(range(400));assert [r['step'] for r in refresh]==list(range(200,400,5));ratios=[r['weighted_sirm_grad_norm']/r['base_residual_grad_norm'] for r in refresh if r['sirm_active'] and r['weighted_sirm_grad_norm'] is not None and r['base_residual_grad_norm']>0]
  guards=[r['discovery_gradient_norm_match']['correct_cohort_guard'] for r in history];assert all(r['source_only'] and not r['target_read'] for r in guards if not r.get('zero_discovery'))
  traces[arm]=dict(active_steps=sum(r['sirm_active'] for r in history[200:]),stage3_steps=200,refreshes=len(refresh),shared_refreshes=sum(r['shared_experts']>0 for r in refresh),selected_cell_max=max(r['selected_cells'] for r in refresh),guard_norm_retained_mean=float(np.mean([r['projection']['retained_norm_fraction'] for r in guards if 'projection' in r])),guard_norm_replay_error_max=max(r.get('gradient_max_replay_error',0) for r in guards),gradient_ratio_samples=ratios)
  for state in ['selected','last']:
   if arm=='OFF':m,p,path=load(DEST,state)
   else:
    m,p,_=load(DEST,state);path=bank/('source_best.pkl' if state=='selected' else 'last.pkl');p=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()});m.configure_stage();m.eval()
   locks[str(path)]=digest(path);content[arm+'/'+state]={k:v.clone() for k,v in m.state_dict().items()};before=copy.deepcopy(m.state_dict());raw,y,dom,_=data['source_in'];m.calibrate_features(raw,dom);splits={}
   for split,(raw,y,dom,ids) in data.items():
    with torch.no_grad():pi=m.router(m.bank_features(raw)).softmax(1);g,d,f=outputs(m,raw,pi);ce=F.cross_entropy(f,y,reduction='none');gc=g.argmax(1)==y;fc=f.argmax(1)==y;contribution=pi[:,:6,None]*d;utility=torch.stack([F.cross_entropy(f-contribution[:,e],y,reduction='none')-ce for e in range(6)],1);U=torch.stack([utility[dom==v].mean(0) for v in dom.unique()]);V=U-U.mean(1,keepdim=True)-U.mean(0,keepdim=True)+U.mean();flat=d.permute(1,0,2).reshape(6,-1);cos=F.normalize(flat,dim=1)@F.normalize(flat,dim=1).T;correction=f-g
    np.savez_compressed(OUT/f'{arm}_{state}_{split}_replay.npz',ids=ids,y=y.numpy(),domains=dom.numpy(),global_logits=g.numpy(),residual_logits=d.numpy(),router=pi.numpy(),full_logits=f.numpy(),utility=utility.numpy())
    splits[split]=dict(global_accuracy=float(torch.stack([gc[dom==v].float().mean() for v in dom.unique()]).mean()),full_accuracy=float(torch.stack([fc[dom==v].float().mean() for v in dom.unique()]).mean()),CE=float(torch.stack([ce[dom==v].mean() for v in dom.unique()]).mean()),repairs=int((~gc&fc).sum()),harms=int((gc&~fc).sum()),U=U.tolist(),interaction=V.tolist(),residual_cosine=cos.tolist(),cosine_mean=float(cos[~torch.eye(6,dtype=torch.bool)].mean()),correction_global_RMS=float(correction.square().mean().sqrt()/g.square().mean().sqrt()),router_null_mass=float(pi[:,-1].mean()),residual_norm=float(d.norm(dim=2).mean()),cases=[dict(id=str(ids[i]),domain=int(dom[i]),label=int(y[i]),global_prediction=int(g[i].argmax()),full_prediction=int(f[i].argmax()),global_CE=float(F.cross_entropy(g[i:i+1],y[i:i+1])),full_CE=float(ce[i]),expert_utility=utility[i].tolist()) for i in range(len(y))])
   m.load_state_dict(before);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());assert all(q.grad is None for q in m.parameters());summaries[arm+'/'+state]=dict(step=p.get('selected_step',p.get('step')),checkpoint_sha256=digest(path),splits=splits)
 equal=None if only_off else {state:all(torch.equal(content['OFF/'+state][k],content['ON/'+state][k]) for k in content['OFF/'+state]) for state in ['selected','last']}
 for f,h in locks.items():assert digest(Path(f))==h
 write(result_file,dict(OFF_only=only_off,source_only=True,target_read=False,fitting=False,parameter_updates=False,training=traces,banks=summaries,OFF_ON_state_equal=equal,input_sha256=locks,code_sha256=digest(Path(__file__)),scientific_goal_achieved=False));(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());print('Source analysis complete',flush=True)
if __name__=='__main__':run()

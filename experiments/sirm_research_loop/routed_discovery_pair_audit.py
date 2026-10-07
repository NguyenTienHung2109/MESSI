"""Pair audit allows genuine supported-SIRM divergence; never presumes inactivity."""
import json
from pathlib import Path
import torch
from .routed_discovery_run import DEST,ANCHOR
from .shrinkage import write,digest

def main():
 torch.set_num_threads(4);banks=DEST/'experiment/runs/L38';assert all((banks/a/'done').exists() for a in ['OFF','ON'])
 logs={a:[json.loads(l) for l in (banks/a/'diagnostics.jsonl').read_text().splitlines()] for a in ['OFF','ON']};assert all([r['step'] for r in v]==list(range(400)) for v in logs.values())
 keys=set().union(*(set(r) for v in logs.values() for r in v))-{'effective_lambda_sirm'}
 differences={k:[i for i in range(400) if ((k in logs['OFF'][i])!=(k in logs['ON'][i])) or logs['OFF'][i].get(k)!=logs['ON'][i].get(k)] for k in sorted(keys)};differences={k:v for k,v in differences.items() if v}
 firstactive=next((r['step'] for r in logs['ON'] if r['sirm_active']),None);assert all(i>=200 for v in differences.values() for i in v)
 if firstactive is None:assert not differences
 else:assert all(i>=firstactive for v in differences.values() for i in v)
 anchor=torch.load(ANCHOR,map_location='cpu',weights_only=False)['model_dict'];states={}
 for state,file in [('selected','source_best.pkl'),('last','last.pkl')]:
  payload={a:torch.load(banks/a/file,map_location='cpu',weights_only=False) for a in ['OFF','ON']};sd={a:{k.removeprefix('model.'):v for k,v in p['model_dict'].items()} for a,p in payload.items()};assert sd['OFF'].keys()==sd['ON'].keys();groups={}
  for prefix in ['featurizer.','global_head.','residuals.','router.']:
   names=[k for k in sd['OFF'] if k.startswith(prefix)];neq=0;elements=0;maximum=0.
   for k in names:
    a,b=sd['OFF'][k],sd['ON'][k];elements+=a.numel();neq+=int((a!=b).sum());maximum=max(maximum,float((a.double()-b.double()).abs().max()))
    if prefix in ['featurizer.','global_head.']:assert torch.equal(a,anchor[k]) and torch.equal(b,anchor[k])
   groups[prefix.rstrip('.')]=dict(elements=elements,different_elements=neq,max_abs_difference=maximum)
  steps={a:p.get('selected_step',p.get('step')) for a,p in payload.items()};states[state]={'steps':steps,'same_step':len(set(steps.values()))==1,'all_saved_tensors_exact':all(torch.equal(v,sd['ON'][k]) for k,v in sd['OFF'].items()),'tensor_keys':len(sd['OFF']),'modules':groups,'checkpoint_sha256':{a:digest(banks/a/file) for a in payload}}
  if firstactive is None:assert states[state]['all_saved_tensors_exact']
 result={'passed':True,'source_only':True,'target_read':False,'ON_active_steps':sum(r['sirm_active'] for r in logs['ON']),'stage3_steps':200,'first_active_step':firstactive,'preactivation_exact':True,'trajectory_first_differences':{k:v[0] for k,v in differences.items()},'states':states,'scope':'Direct checkpoint tensors and full400-step logs; same-step399 comparison separates selection-time effects. Nonzero penalty is not assumed to imply nonzero update. Frozen anchor required exact at both saved states.'};write(DEST/'pair_tensor_audit.json',result);return result
if __name__=='__main__':print(json.dumps(main(),indent=2))

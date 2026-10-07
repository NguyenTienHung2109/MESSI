"""Direct saved-state tensor/trajectory audit; no replay, fitting or target read."""
import json
from pathlib import Path
import torch
from .spatial_bank_run import DEST,ANCHOR
from .shrinkage import write,digest


def main():
    torch.set_num_threads(4);banks=DEST/'experiment/runs/L38';assert all((banks/a/'done').exists() for a in ['OFF','ON'])
    logs={a:[json.loads(l) for l in (banks/a/'diagnostics.jsonl').read_text().splitlines()] for a in ['OFF','ON']}
    assert all([r['step'] for r in v]==list(range(400)) for v in logs.values())
    excluded={'effective_lambda_sirm'}
    keys=set().union(*(set(r) for v in logs.values() for r in v))-excluded
    # Logged norm/EMA/objective values are compared directly; intentional
    # lambda ramp metadata is not a prediction/training objective difference.
    differences={k:[i for i in range(400) if ((k in logs['OFF'][i])!=(k in logs['ON'][i])) or logs['OFF'][i].get(k)!=logs['ON'][i].get(k)] for k in sorted(keys)}
    differences={k:v for k,v in differences.items() if v}
    results={};anchor=torch.load(ANCHOR,map_location='cpu',weights_only=False)['model_dict']
    for state,file in [('selected','source_best.pkl'),('last','last.pkl')]:
        p={a:torch.load(banks/a/file,map_location='cpu',weights_only=False) for a in ['OFF','ON']}
        steps={a:v.get('selected_step',v.get('step')) for a,v in p.items()};sd={a:{k.removeprefix('model.'):v for k,v in p[a]['model_dict'].items()} for a in p};assert sd['OFF'].keys()==sd['ON'].keys()
        groups={}
        for prefix in ['featurizer.','global_head.','residuals.','router.']:
            names=[k for k in sd['OFF'] if k.startswith(prefix)];neq=0;total=0;maximum=0.
            for k in names:
                a=sd['OFF'][k];b=sd['ON'][k];total+=a.numel();neq+=int((a!=b).sum());maximum=max(maximum,float((a.double()-b.double()).abs().max()))
                if prefix in ['featurizer.','global_head.']:assert torch.equal(a,anchor[k]) and torch.equal(b,anchor[k])
            groups[prefix.rstrip('.')]=dict(elements=total,different_elements=neq,max_abs_difference=maximum)
        results[state]=dict(steps=steps,same_step=len(set(steps.values()))==1,all_saved_tensors_exact=all(torch.equal(v,sd['ON'][k]) for k,v in sd['OFF'].items()),tensor_keys=len(sd['OFF']),modules=groups,checkpoint_sha256={a:digest(banks/a/file) for a in p})
    active=sum(r['sirm_active'] for r in logs['ON']);assert active==0 and not differences and all(v['all_saved_tensors_exact'] for v in results.values())
    record=dict(passed=True,source_only=True,target_read=False,anchor_sha256=digest(ANCHOR),updates400=True,ON_active_steps=active,compared_trajectory_keys=sorted(keys),trajectory_differences=differences,states=results,scope='Actual saved tensor contents at both states plus400 logged steps. Config/checkpoint-byte differences are not parameter divergence. No target logits compared here. No active SIRM means no mechanism-benefit experiment.')
    write(DEST/'pair_tensor_audit.json',record);print(json.dumps(results,indent=2))

if __name__=='__main__':main()

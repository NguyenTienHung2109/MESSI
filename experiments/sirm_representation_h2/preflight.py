"""Verify new OFF initialization against all reused ON banks before full compute."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import torch
from experiments.sirm_h1_h6 import run as base
from .protocol import config_for,validate_reuse

def main():
    p=argparse.ArgumentParser();p.add_argument('--reuse-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);root=Path(a.reuse_root);checks=[]
    freeze=subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True)
    for seed in range(3):
        base.seed_everything(seed);model=base.build_model(config_for(False)['hparams']);h=hashlib.sha256()
        for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
        initial=h.hexdigest();params=sum(p.numel() for p in model.parameters());del model;torch.cuda.empty_cache()
        for target in range(4):
            pair=[]
            for condition,variant in [('C','native'),('D','balanced_views')]:
                bank=root/variant/f'env{target}'/f'seed{seed}';evidence=validate_reuse(bank,condition,target,seed)
                manifest=json.loads((bank/'manifest.json').read_text())
                if manifest['initialization_sha256']!=initial or manifest['parameters_total']!=params:raise ValueError('new OFF initialization or capacity differs from reused ON')
                if (bank/'environment.freeze.txt').read_text()!=freeze:raise ValueError('reuse environment packages differ')
                pair.append(manifest)
                checks.append(dict(condition=condition,target=target,seed=seed,initialization_matches_new_OFF=True,environment_identical=True,**evidence))
            if pair[0]['schedule_sha256']!=pair[1]['schedule_sha256'] or pair[0]['split_hash']!=pair[1]['split_hash']:raise ValueError('ON augmentation arms not paired')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);base.write(out,dict(passed=True,reused_runs=24,checks=checks,gpu=torch.cuda.get_device_name(0)))
    print('preflight passed: all24 reused banks match OFF initialization, capacity, schedules, splits, environment',flush=True)

if __name__=='__main__':main()

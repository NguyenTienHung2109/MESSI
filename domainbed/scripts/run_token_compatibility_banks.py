"""Follow through source-supported token controls on the complete expert bank."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from domainbed.scripts.analyze_compatibility_failures import load_bank
from domainbed.scripts.run_transfer_compatibility import run_fold,family_events
from domainbed.scripts.audit_predictive_compatibility import write_json
from domainbed.scripts.check_compatibility_mechanisms import save_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('predictive_support_outputs/pacs_compatibility_lodo_seed0'))
    parser.add_argument('--tokens',type=Path,default=Path('predictive_support_outputs/pacs_failure_cycle2_seed0/token_ablation'))
    parser.add_argument('--output',type=Path,default=Path('predictive_support_outputs/pacs_failure_cycle2_seed0/token_banks'))
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use fresh bank output directory')
    args.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    write_json(args.output/'protocol.json',dict(seed=0,variants=['erm','dist_erm','mean_erm'],
        rationale='DIST and mean-token inputs both improved mean strict source-exclusion; retain both, do not choose using targets',
        map_steps=500,head_steps=1000,dimension=16,tau=.12,delta=.05,
        alpha=.05,family_events=family_events(),scope='reference hull only; unchanged contribution gate',
        prospective=False,feature_modes={'erm':'cls','dist_erm':'dist','mean_erm':'mean_tokens'}))
    base=load_bank(args.root);dist={}
    for d in base:
        payload=torch.load(args.tokens/f'dist_env{d}.pt',map_location='cpu',weights_only=True)
        assert torch.equal(payload['labels'],base[d][1]);dist[d]=payload['dist']
    rows=[];metrics=[];selections=[]
    for variant,mode in [('erm','cls'),('dist_erm','dist'),('mean_erm','mean_tokens')]:
        bank={d:((x if mode=='cls' else dist[d] if mode=='dist' else (x+dist[d])/2),y) for d,(x,y) in base.items()}
        for target in range(4):
            for fold in ['main']+[f'loso_env{d}' for d in range(4) if d!=target]:
                write_json(args.output/'status.json',dict(state='running',variant=variant,target=target,fold=fold))
                r,m,s=run_fold(args.root,args.output,target,fold,variant,bank,.05/family_events(),feature_mode=mode)
                rows.extend(r);metrics.extend(m);selections.append(dict(s,target=target,name=fold))
                save_csv(args.output/'certificates.csv',rows);save_csv(args.output/'metrics.csv',metrics)
                write_json(args.output/'selections.json',selections)
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for path in [Path(__file__),Path('domainbed/scripts/run_transfer_compatibility.py'),Path('domainbed/transfer_compatibility.py')]:
        (snapshot/path.name).write_bytes(path.read_bytes())
    write_json(args.output/'status.json',dict(state='completed',seed=0,folds=48))


if __name__=='__main__':main()

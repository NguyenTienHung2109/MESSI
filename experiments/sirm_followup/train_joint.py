"""Isolated normalized H6 training wrapper; never writes legacy source files."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
import experiments.sirm_h1_h6.run as base
from experiments.sirm_followup.run import objective_components,sha,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--banks',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--criterion',choices=['Grad','Regret'],required=True)
    parser.add_argument('--target-env',type=int,choices=range(4),required=True)
    parser.add_argument('--seed',type=int,choices=range(3),required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    prior=Path(args.banks)/'native'/f'env{args.target_env}'/f'seed{args.seed}'
    config=json.loads((prior/'effective_config.json').read_text())['config']
    config['followup_wrapper_sha256']=sha(__file__)
    config['followup_normalization']='risk and compatibility divided by SD of their feasible-support objectives on source-probe; alpha fixed at one; std floor 1e-8'
    original=base.assignment_statistics
    def normalized_statistics(source_in,validation,c,criterion=None):
        stats=original(source_in,validation,c,criterion)
        supports,risk,comp,_,_=objective_components(stats,criterion)
        rs=max(risk.std(unbiased=False).item(),1e-8)
        cs=max(comp.std(unbiased=False).item(),1e-8)
        stats['risk']=stats['risk']/rs
        key='grad' if criterion=='Grad' else 'gap'
        stats[key]=stats[key]/cs
        info=dict(stats.get('grad_info',{}),normalization=dict(risk_objective_sd=rs,compatibility_objective_sd=cs,alpha=1.))
        stats['grad_info']=info
        return stats
    # Runtime injection is limited to this child process; existing files stay unchanged.
    base.assignment_statistics=normalized_statistics
    train_args=argparse.Namespace(output=args.output,data_dir='domainbed/data',seed=args.seed,target_env=args.target_env,
        criterion=args.criterion,assignment_weight=1.,variant='normalized_'+args.criterion,cohort='standard',
        topk=None,steps=None,smoke=False,limit=None,resume=False,stop_after=None)
    base.train(train_args,config)
    output=Path(args.output)
    (output/'source_snapshot/train_joint.py').write_bytes(Path(__file__).read_bytes())
    (output/'source_snapshot/followup_run.py').write_bytes(Path(__file__).with_name('run.py').read_bytes())
    config_path=output/'locked_config.json';write(config_path,config)
    stages=[['cache','--source-only','--bank',str(output),'--output',str(output/'cache')],
            ['fit','--cache',str(output/'cache'),'--output',str(output/'routers.pt')],
            ['cache','--target-only','--bank',str(output),'--routers',str(output/'routers.pt'),'--output',str(output/'cache')],
            ['eval','--cache',str(output/'cache'),'--routers',str(output/'routers.pt'),'--output',str(output/'evaluation')]]
    for i,options in enumerate(stages):
        command=[sys.executable,'-m','experiments.sirm_h1_h6.run','--config',str(config_path),*options]
        with (output/f'followup_stage_{i}.log').open('w') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    write(output/'followup_completed.json',dict(criterion=args.criterion,seed=args.seed,target=args.target_env,status='complete'))


if __name__=='__main__':main()

"""Matched second-stage full training with source-prior locked hard assignments."""
import argparse,json,subprocess,sys
from pathlib import Path
import torch
import experiments.sirm_h1_h6.run as base
from experiments.sirm_h1_h6.statistics import feasible_supports,expert_losses
from experiments.sirm_measurement.run import gradient_audit
from experiments.sirm_measurement.stability import pick
from experiments.sirm_followup.run import sha,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--banks',required=True);p.add_argument('--scores',required=True);p.add_argument('--output',required=True)
    p.add_argument('--criterion',choices=['Risk','Grad'],required=True);p.add_argument('--target-env',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args();torch.set_num_threads(4)
    prior=Path(args.banks)/'native'/f'env{args.target_env}'/f'seed{args.seed}'
    scores_path=Path(args.scores)/f'env{args.target_env}_seed{args.seed}_scores.pt'
    scores=torch.load(scores_path,weights_only=False);r,c=scores['risk'][:80].double(),scores['angular'][:80].double()
    index=int(r.mean(0).argmin()) if args.criterion=='Risk' else pick(r.mean(0),c.mean(0))
    q=feasible_supports()[index].float()/2
    config=json.loads((prior/'effective_config.json').read_text())['config']
    config.update(assignment_strategy='source-prior fixed hard Q; discovery80 only; confirmation80 used only as gate',
        source_prior_sha256=sha(prior/'checkpoints/best.pkl'),score_sha256=sha(scores_path),locked_q=q.tolist(),
        second_stage='5001 updates from same native source-selected checkpoint; optimizer reset in both arms',wrapper_sha256=sha(__file__))
    original_build=base.build_model
    def build(hparams):
        model=original_build(hparams);payload=base.load(prior/'checkpoints/best.pkl')
        if 'hypotheses_q' in payload['model_dict']:model.set_assignment(payload['model_dict']['hypotheses_q'])
        model.load_state_dict(payload['model_dict'],strict=True);return model
    cache=base.load(prior/'cache/source_in_original.pt');losses=expert_losses(cache['expert_logits'],cache['labels'])
    risk=torch.stack([losses[cache['domains']==d].mean(0) for d in range(3)]);audit,angular=gradient_audit(cache)
    def statistics(source_in,validation,c,criterion=None):
        return dict(risk=risk,grad=angular,fit_ids=[],probe_ids=cache['image_ids'],grad_info=dict(locked_source_prior=True,assignment_index=index,autograd_error=audit['autograd_max_error']))
    base.build_model=build;base.assignment_statistics=statistics;base.choose_assignment=lambda *a,**kw:q.clone()
    train_args=argparse.Namespace(output=args.output,data_dir='domainbed/data',seed=args.seed,target_env=args.target_env,
        criterion=args.criterion,assignment_weight=1.,variant='locked_angular_'+args.criterion,cohort='standard',topk=None,
        steps=6 if args.smoke else None,smoke=args.smoke,limit=8 if args.smoke else None,resume=False,stop_after=None)
    base.train(train_args,config);out=Path(args.output);(out/'source_snapshot/train_locked.py').write_bytes(Path(__file__).read_bytes())
    write(out/'locked_config.json',config)
    if args.smoke:return
    stages=[['cache','--source-only','--bank',str(out),'--output',str(out/'cache')],
        ['fit','--skip-h6','--cache',str(out/'cache'),'--output',str(out/'routers.pt')],
        ['cache','--target-only','--bank',str(out),'--routers',str(out/'routers.pt'),'--output',str(out/'cache')],
        ['eval','--cache',str(out/'cache'),'--routers',str(out/'routers.pt'),'--output',str(out/'evaluation')]]
    for i,stage in enumerate(stages):
        with (out/f'stage_{i}.log').open('w') as stream:
            subprocess.run([sys.executable,'-m','experiments.sirm_h1_h6.run','--config',str(out/'locked_config.json'),*stage],stdout=stream,stderr=subprocess.STDOUT,check=True)
    write(out/'completed.json',dict(criterion=args.criterion,status='completed'))

if __name__=='__main__':main()

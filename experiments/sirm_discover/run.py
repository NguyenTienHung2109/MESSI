"""Train and verify matched SIRM-Discover ON/OFF; default suite is smoke only."""
import argparse,copy,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from experiments.sirm_h1_h6 import run as base
from domainbed.scripts.train_subset_irm_pacs import make_schedule,seed_everything,save_checkpoint
from .model import SIRMDiscover
from .math import residual_utilities


def config_for(smoke,arm):
    config=json.loads(Path(__file__).with_name('config.json').read_text())
    if smoke:
        config.update(global_steps=4,discover_steps=8,invariance_steps=8,sirm_ramp_steps=3,
            batch_size=8,checkpoint_freq=4,num_workers=0,utility_interval=1,utility_min_samples=8,
            utility_floor=0.,interaction_floor=0.)
    if arm=='off':config['lambda_sirm']=0.
    validate_config(config)
    return config


def validate_config(c):
    positive=['global_steps','discover_steps','invariance_steps','sirm_ramp_steps','num_experts','residual_hidden','utility_interval','utility_min_samples','batch_size','checkpoint_freq','eval_batch_size','discovery_temperature','router_temperature','q_temperature','min_ess','lr']
    if any(c[k]<=0 for k in positive):raise ValueError('stage budgets, intervals, sizes and temperatures must be positive')
    if not 0<=c['utility_ema_beta']<1:raise ValueError('EMA beta must be in [0,1)')
    if any(c[k]<0 for k in ['lambda_sirm','lambda_discover','lambda_router','lambda_balance','weight_decay','utility_floor','interaction_floor']):raise ValueError('weights and effect floors must be nonnegative')
    if c['invariance_steps']<=c['sirm_ramp_steps']:raise ValueError('invariance stage must extend beyond the ramp')


def build(config):
    from domainbed.deit_transformer import DeiTFeaturizer
    backbone=DeiTFeaturizer(pretrained=config['pretrained'],model_name=config['backbone'])
    return SIRMDiscover(backbone,backbone.n_outputs,7,3,config).cuda()


def weights_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(model,data,config,diagnostics=False):
    model.eval();scores={};global_scores={};utility=[];domain_ids=[];mass=[]
    for d,(env,subset) in enumerate(sorted(data.items())):
        correct=global_correct=count=0
        for x,y in DataLoader(subset,batch_size=config['eval_batch_size'],num_workers=config['num_workers']):
            x=x.cuda();y=y.cuda();f=model(x)
            correct+=int(f['logits'].argmax(1).eq(y).sum());global_correct+=int(f['global_logits'].argmax(1).eq(y).sum());count+=len(y)
            if diagnostics:
                full,contribution,marginal,single=residual_utilities(f['global_logits'],f['delta'],f['pi'],y)
                torch.testing.assert_close(full,f['logits'])
                utility.append(marginal.cpu());domain_ids.append(torch.full((len(y),),d));mass.append(f['pi'].cpu())
        scores[str(env)]=correct/count;global_scores[str(env)]=global_correct/count
    result=dict(source_accuracy=scores,source_mean=sum(scores.values())/len(scores),global_accuracy=global_scores,global_mean=sum(global_scores.values())/len(global_scores))
    if diagnostics:
        u=torch.cat(utility);domains=torch.cat(domain_ids);matrix=torch.stack([u[domains==d].mean(0) for d in range(len(data))]);interaction=matrix-matrix.mean(1,keepdim=True)-matrix.mean(0,keepdim=True)+matrix.mean()
        selected=model.q.cpu()>0;selected_values=[];other_values=[];interaction_selected=[];interaction_other=[]
        for d in range(len(data)):
            if selected[d].any():selected_values.extend(matrix[d,selected[d]].tolist());interaction_selected.extend(interaction[d,selected[d]].tolist())
            if (~selected[d]).any():other_values.extend(matrix[d,~selected[d]].tolist());interaction_other.extend(interaction[d,~selected[d]].tolist())
        avg=lambda values:sum(values)/len(values) if values else None
        result.update(source_domains=sorted(data),marginal_utility=matrix.tolist(),utility_interaction=interaction.tolist(),q=model.q.cpu().tolist(),q_null=model.q_null.cpu().tolist(),router_mass=torch.cat(mass).mean(0).tolist(),selected_utility=avg(selected_values),nonselected_utility=avg(other_values),selected_interaction_utility=avg(interaction_selected),nonselected_interaction_utility=avg(interaction_other),q_fit_split='source-in augmented training minibatches',diagnostic_split='source-out deterministic original view',null_rows=int(model.q.sum(1).eq(0).sum()))
    return result


def train(args):
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for PACS training')
    out=Path(args.output).resolve();config=config_for(args.smoke,args.arm)
    if args.config:
        if args.smoke:raise ValueError('custom config and --smoke cannot be combined')
        config=json.loads(Path(args.config).read_text())
        if args.arm=='off':config['lambda_sirm']=0.
        validate_config(config)
    if not args.resume and out.exists() and any(out.iterdir()):raise FileExistsError('fresh output directory required')
    out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    seed_everything(args.seed);split=base.inventory(args.data_dir,[e for e in range(4) if e!=args.target],args.seed,config)
    effective=dict(config=config,arm=args.arm,smoke=args.smoke,target_env=args.target,seed=args.seed,data_dir=args.data_dir,split_hash=base.canonical_hash(split),target_read_during_training=False,checkpoint_selection='equal-source-out mean accuracy during invariance stage; earliest tie')
    if args.resume:
        assert json.loads((out/'effective_config.json').read_text())==effective,'resume config changed'
        assert json.loads((out/'split_manifest.json').read_text())==split,'resume split changed'
    else:
        base.write(out/'effective_config.json',effective);base.write(out/'split_manifest.json',split)
    model=build(config);steps=config['global_steps']+config['discover_steps']+config['invariance_steps'];start_step=0;best=-1.;best_step=None;state=None
    source_hashes={str(p):base.digest(p) for p in Path(__file__).parent.glob('*.py')}
    source_hashes['experiments/sirm_h1_h6/run.py']=base.digest(Path('experiments/sirm_h1_h6/run.py'))
    if args.resume:
        old=json.loads((out/'manifest.json').read_text());assert old['source_hashes']==source_hashes,'resume source changed'
        assert (out/'environment.freeze.txt').read_text()==subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),'resume environment changed'
        state=base.load(out/'resume.pt');assert state['effective_hash']==base.canonical_hash(effective)
        model.load_state_dict(state['model_dict']);model.configure_stage();model.optimizer.load_state_dict(state['optimizer'])
        start_step=state['next_step'];best=state['best'];best_step=state['best_step']
        assert int(model.step)==start_step
        for name in ['train_log.jsonl','diagnostics.jsonl','q_refresh.jsonl']:
            if not (out/name).exists():continue
            rows=[line for line in (out/name).read_text().splitlines() if json.loads(line)['step']<start_step]
            (out/name).write_text(''.join(x+'\n' for x in rows))
    source_train=base.datasets_for(args.data_dir,split,config,train=True,cohort='balanced_views',limit=32 if args.smoke else None)
    validation=base.source_data(args.data_dir,split,config,'out',cohort='balanced_views',limit=14 if args.smoke else None)
    loaders=[];schedules=[]
    for env,subset in sorted(source_train.items()):
        schedule=make_schedule(list(range(len(subset))),steps,config['batch_size'],args.seed,env);schedules.append(schedule)
        loaders.append(iter(DataLoader(base.ScheduledImages(subset,args.seed,env),batch_sampler=base.ScheduledBatches(schedule,start_step),num_workers=config['num_workers'],generator=torch.Generator().manual_seed(args.seed+env*101),pin_memory=True)))
    schedule_hash=hashlib.sha256(b''.join(s.tobytes() for s in schedules)).hexdigest()
    if state:
        assert old['schedule_sha256']==schedule_hash;base.restore_rng(state['rng']);manifest=old
    else:
        from domainbed.deit_transformer import _DEIT_CONFIGS
        pretrained_name=Path(_DEIT_CONFIGS[config['backbone']][-1]).name
        pretrained=Path(torch.hub.get_dir())/'checkpoints'/pretrained_name
        local=Path('domainbed/pretrained')/pretrained_name
        if local.exists():pretrained=local
        if config['pretrained'] and not pretrained.exists():raise FileNotFoundError('pretrained provenance file unavailable')
        manifest=dict(effective,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes=source_hashes,initialization_sha256=weights_hash(model),schedule_sha256=schedule_hash,parameters_total=sum(p.numel() for p in model.parameters()),pretrained_sha256=base.digest(pretrained) if config['pretrained'] else None,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version())
        base.write(out/'manifest.json',manifest);snapshot=out/'source_snapshot';snapshot.mkdir()
        for p in Path(__file__).parent.glob('*.py'):(snapshot/p.name).write_bytes(p.read_bytes())
        (out/'environment.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    begin=time.time()
    for step in range(start_step,steps):
        minibatches=[(x.cuda(),y.cuda()) for x,y in [next(loader) for loader in loaders]]
        values=model.update(minibatches)
        with (out/'diagnostics.jsonl').open('a') as f:f.write(json.dumps({'step':step,**values,**model.last_diagnostics})+'\n')
        checkpoint=step%config['checkpoint_freq']==0 or step in [config['global_steps']-1,config['global_steps']+config['discover_steps']-1] or step==steps-1 or step in config.get('diagnostic_steps',[])
        refresh_eval=config.get('validate_q_refresh',False) and model.last_diagnostics['q_refreshed']
        if not checkpoint and not refresh_eval:continue
        result=evaluate(model,validation,config);score=result['source_mean'];row=dict(step=step,**values,**result,seconds=time.time()-begin)
        if refresh_eval:
            with (out/'q_refresh.jsonl').open('a') as f:f.write(json.dumps({'step':step,**values,**model.last_diagnostics,'source_validation_accuracy':score,'source_validation_per_domain':result['source_accuracy'],'validation_timing':'post-update; utility/Q/loss/ESS are pre-update'})+'\n')
        if not checkpoint:continue # refresh validation must not change checkpoint selection cadence
        with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if values['stage']=='invariance' and score>best:
            best=score;best_step=step;save_checkpoint(out/'checkpoints/best.pkl',model,dict(step=step,manifest=manifest))
        state=dict(model_dict=model.state_dict(),optimizer=model.optimizer.state_dict(),next_step=step+1,best=best,best_step=best_step,rng=base.capture_rng(),effective_hash=base.canonical_hash(effective))
        torch.save(state,out/'resume.tmp.pt');(out/'resume.tmp.pt').replace(out/'resume.pt')
        print(json.dumps(dict(step=step,stage=values['stage'],source_mean=score,lambda_sirm=values['effective_lambda_sirm'],sirm_pairs=values['sirm_pair_count'],q_null=values['q_null_mean'])),flush=True)
    save_checkpoint(out/'checkpoints/last.pkl',model,dict(step=steps-1,manifest=manifest))
    payload=base.load(out/'checkpoints/best.pkl');model.load_state_dict(payload['model_dict']);model.configure_stage()
    diagnostics=evaluate(model,validation,config,diagnostics=True);base.write(out/'source_diagnostics.json',diagnostics)
    base.write(out/'summary.json',dict(selected_step=best_step,source_mean=best,steps=steps,smoke=args.smoke,target_evaluated=False,seconds=time.time()-begin))


def suite(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    on=config_for(True,'on');off=config_for(True,'off');expected=copy.deepcopy(on);expected['lambda_sirm']=0.;assert expected==off
    hashes={p.name:base.digest(p) for p in Path(__file__).parent.glob('*.py')}
    plan=dict(smoke=True,target=args.target,seed=args.seed,data_dir=args.data_dir,configs=dict(on=on,off=off),arms=['off','on'],steps=sum(on[k] for k in ['global_steps','discover_steps','invariance_steps']),only_objective_difference='lambda_sirm',target_evaluated=False,source_hashes=hashes)
    if (root/'plan.json').exists():assert json.loads((root/'plan.json').read_text())==plan,'locked smoke plan changed'
    else:base.write(root/'plan.json',plan)
    for arm in ['off','on']:
        dest=root/'banks'/arm
        if (dest/'summary.json').exists():
            completed=json.loads((dest/'effective_config.json').read_text())
            assert completed['config']==plan['configs'][arm] and completed['target_env']==args.target and completed['seed']==args.seed and completed['smoke'],'completed smoke identity changed'
            continue
        command=[sys.executable,'-u','-m','experiments.sirm_discover.run','train','--smoke','--output',str(dest),'--arm',arm,'--target',str(args.target),'--seed',str(args.seed),'--data-dir',args.data_dir]
        if (dest/'resume.pt').exists():command.append('--resume')
        base.write(root/'progress.json',dict(stage='training',arm=arm,pid=os.getpid()))
        with (root/(arm+'.log')).open('a') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            base.write(root/'progress.json',dict(stage='failed',arm=arm,returncode=result.returncode));raise RuntimeError((root/(arm+'.log')).read_text()[-4000:])
    manifests=[json.loads((root/'banks'/arm/'manifest.json').read_text()) for arm in ['off','on']]
    for key in ['initialization_sha256','split_hash','schedule_sha256','parameters_total','pretrained_sha256']:assert manifests[0][key]==manifests[1][key],key
    rows={arm:[json.loads(line) for line in (root/'banks'/arm/'diagnostics.jsonl').read_text().splitlines()] for arm in ['off','on']}
    start=on['global_steps']+on['discover_steps']
    for a,b in zip(rows['off'],rows['on']):
        assert a['step']==b['step']
        if a['step']<=start:
            for key in ['loss_cls','loss_discover','loss_router','loss_balance','q','utility_ema']:assert a[key]==b[key],('unmatched pre-penalty trajectory',a['step'],key)
    assert all(r['effective_lambda_sirm']==0 and r['loss_sirm']==0 for r in rows['off'])
    assert all(r['effective_lambda_sirm']==0 for r in rows['on'] if r['stage']!='invariance')
    assert any(r['effective_lambda_sirm']>0 for r in rows['on'])
    audit=dict(initialization_split_schedule_capacity_matched=True,pre_penalty_trajectory_identical=True,OFF_sirm_zero=True,target_read=False,ON_active_pair_steps=sum(r['sirm_active'] for r in rows['on']))
    base.write(root/'paired_audit.json',audit)
    text=['# SIRM-Discover ON/OFF smoke','', 'Two matched 20-step PACS runs completed. Numerical/pipeline check only; no target evaluation and no accuracy/generalization claim. Smoke compresses the three stages, uses 8 samples/source and zero utility/interaction effect floors to exercise the discovery path; defaults require positive effect floors.', '', '| Arm | Selected step | Source accuracy | Penalty-active steps |','|---|---:|---:|---:|']
    for arm in ['off','on']:
        result=json.loads((root/'banks'/arm/'summary.json').read_text());text.append(f'| {arm} | {result["selected_step"]} | {result["source_mean"]:.6f} | {sum(r["sirm_active"] for r in rows[arm])} |')
    text.extend(['','Initialization, source splits, minibatch schedules, capacity and pre-penalty trajectory match exactly. OFF only changes lambda_sirm. Source checkpoints are selected during stage 3; target was never read. Known synthetic utility/gradient tests separately cover active pair and NULL behavior.'])
    (root/'summary.md').write_text('\n'.join(text)+'\n');base.write(root/'progress.json',dict(stage='completed',arms=2,target_evaluated=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['train','smoke']);p.add_argument('--output',required=True);p.add_argument('--arm',choices=['on','off'],default='on');p.add_argument('--target',type=int,choices=range(4),default=1);p.add_argument('--seed',type=int,default=0);p.add_argument('--data-dir',default='domainbed/data');p.add_argument('--smoke',action='store_true');p.add_argument('--config');p.add_argument('--resume',action='store_true');args=p.parse_args()
    train(args) if args.command=='train' else suite(args)

if __name__=='__main__':main()

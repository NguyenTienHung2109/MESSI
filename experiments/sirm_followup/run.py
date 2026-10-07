"""Preregistered, source-selected frozen-bank follow-up; existing code is unchanged."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_h1_h6.controls import InputRouter, route, mixture, metrics
from experiments.sirm_h1_h6.statistics import feasible_supports
from domainbed.subset_irm import per_sample_topk


def read(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def domain_weights(cache,device):
    domains=cache['domains'].to(device)
    weights=torch.zeros(len(domains),device=device)
    for d in domains.unique():
        mask=domains==d;weights[mask]=1/(len(domains.unique())*int(mask.sum()))
    return weights


def validate_sources(fit,val):
    if fit['metadata']['kind']!='source' or val['metadata']['kind']!='source':
        raise ValueError('target caches cannot fit or select')
    if fit['metadata']['split']!='in' or val['metadata']['split']!='out':
        raise ValueError('expected DomainBed source-in/source-out')
    if set(fit['image_ids']) & set(val['image_ids']):
        raise ValueError('source fitting/validation overlap')
    for field in ['checkpoint_sha256','split_hash','seed','target_env']:
        if fit['metadata'][field]!=val['metadata'][field]:raise ValueError('source provenance mismatch')


def fit_component(fit,val,old,kind,epochs=300,lr=.003,device='cuda'):
    validate_sources(fit,val)
    torch.manual_seed(fit['metadata']['seed'])
    x=fit['projected_feature'].to(device);vx=val['projected_feature'].to(device)
    logits=fit['expert_logits'].to(device);vl=val['expert_logits'].to(device)
    y=fit['labels'].to(device);vy=val['labels'].to(device)
    w,vw=domain_weights(fit,device),domain_weights(val,device)
    classes=logits.shape[-1]
    if kind.startswith('ce_'):
        old_state=old['input_sigma0']['state']
        model=InputRouter(old_state['mean'],old_state['std'],6,old['input_sigma0']['hidden']).to(device)
        parameters=list(model.parameters())
    elif kind=='class':
        model=torch.nn.Parameter(torch.zeros(classes,6,device=device));parameters=[model]
        # The source-selected GLOBAL predictor supplies class posteriors.
        # Class labels enter CE only; deployable gating never uses true labels.
        with torch.no_grad():
            p=mixture(fit['expert_logits'],route(old['global'],fit)).softmax(1).to(device)
            vp=mixture(val['expert_logits'],route(old['global'],val)).softmax(1).to(device)
    elif kind=='calibration':
        model=torch.nn.Parameter(torch.zeros(6,device=device));parameters=[model]
    else:raise ValueError(kind)
    optimizer=torch.optim.Adam(parameters,lr=lr)
    best=float('inf');best_state=None;selected_epoch=None
    for epoch in range(epochs):
        optimizer.zero_grad()
        if kind=='calibration':
            t=model.exp()
            per_image=torch.stack([F.cross_entropy(logits[:,m]/t[m],y,reduction='none') for m in range(6)],1).mean(1)
        else:
            pi=p@model.softmax(1) if kind=='class' else model(x)
            if kind!='ce_dense':pi=per_sample_topk(pi,2)
            per_image=F.cross_entropy(mixture(logits,pi),y,reduction='none')
        loss=(per_image*w).sum();loss.backward();optimizer.step()
        if kind=='calibration':
            with torch.no_grad():model.clamp_(np.log(.25),np.log(4.))
        with torch.no_grad():
            if kind=='calibration':
                per_val=torch.stack([F.cross_entropy(vl[:,m]/model.exp()[m],vy,reduction='none') for m in range(6)],1).mean(1)
            else:
                pi=vp@model.softmax(1) if kind=='class' else model(vx)
                if kind!='ce_dense':pi=per_sample_topk(pi,2)
                per_val=F.cross_entropy(mixture(vl,pi),vy,reduction='none')
            score=(per_val*vw).sum().item()
        if score<best:
            best=score;selected_epoch=epoch
            best_state=({k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        if kind.startswith('ce_') else model.detach().cpu().clone())
    return dict(kind=kind,state=best_state,selected_epoch=selected_epoch,validation_nll=best,
                epochs=epochs,lr=lr,hidden=old['input_sigma0']['hidden'])


def predict(component,cache,old,true_class=False):
    kind=component['kind']
    if kind.startswith('ce_'):
        state=component['state']
        model=InputRouter(state['mean'],state['std'],6,component['hidden']);model.load_state_dict(state)
        with torch.no_grad():pi=model(cache['projected_feature'])
        return pi if kind=='ce_dense' else per_sample_topk(pi,2)
    if kind=='class':
        probabilities=(F.one_hot(cache['labels'],cache['expert_logits'].shape[-1]).float() if true_class else
                       mixture(cache['expert_logits'],route(old['global'],cache)).softmax(1))
        return per_sample_topk(probabilities@component['state'].softmax(1),2)
    raise ValueError(kind)


def objective_components(statistics,criterion):
    supports=feasible_supports()
    risk=(supports*statistics['risk'].unsqueeze(0)).sum((1,2))/6
    masks=sum(supports[:,d].long()*(1<<d) for d in range(3))
    active=masks>0;cols=torch.arange(6).expand_as(masks)
    comp=statistics['grad' if criterion=='Grad' else 'gap']
    scores=(comp[masks,cols]*active).sum(1)/active.sum(1)
    return supports,risk,scores,masks,active


def assignment_diagnostic(old):
    result={}
    for criterion in ['Grad','Regret']:
        supports,risk,comp,masks,active=objective_components(old['h6_statistics'],criterion)
        risk_std=risk.std(unbiased=False).item();comp_std=comp.std(unbiased=False).item()
        risk_index=risk.argmin().item()
        normalized_index=((risk-risk.mean())/max(risk_std,1e-8)+(comp-comp.mean())/max(comp_std,1e-8)).argmin().item()
        changed=not torch.equal(supports[risk_index],supports[normalized_index])
        signal_to_se=None
        if criterion=='Regret':
            se=old['h6_statistics']['gap_se']
            cols=torch.arange(6).expand_as(masks)
            typical_se=((se[masks,cols]*active).sum(1)/active.sum(1)).mean().item()
            signal_to_se=comp_std/max(typical_se,1e-12)
        result[criterion]=dict(risk_objective_std=risk_std,compatibility_objective_std=comp_std,
                normalized_alpha=1.,q_differs_from_risk=changed,risk_q=(supports[risk_index].float()/2).tolist(),
                normalized_q=(supports[normalized_index].float()/2).tolist(),signal_to_se=signal_to_se,
                eligible=changed and comp_std>1e-8 and (criterion=='Grad' or signal_to_se>=1.))
    return result


def evaluate_bank(cache,old,components):
    result={}
    baseline_pi=route(old['input_sigma0'],cache)
    result['kl_top2']=metrics(cache,per_sample_topk(baseline_pi,2))
    result['kl_dense']=metrics(cache,baseline_pi)
    result['kl_top1']=metrics(cache,per_sample_topk(baseline_pi,1))
    result['global']=metrics(cache,route(old['global'],cache))
    result['fixed_pair']=metrics(cache,route(old['fixed_pair'],cache))
    for name in ['ce_top2','ce_dense','class']:
        result[name]=metrics(cache,predict(components[name],cache,old))
    result['class_true_label_diagnostic']=metrics(cache,predict(components['class'],cache,old,True))
    calibrated=dict(cache,expert_logits=cache['expert_logits']/components['calibration']['state'].exp()[None,:,None])
    for name,k in [('calibrated_top1',1),('calibrated_top2',2),('calibrated_dense',None)]:
        pi=baseline_pi if k is None else per_sample_topk(baseline_pi,k)
        result[name]=metrics(calibrated,pi)
    if result['calibrated_top1']['accuracy']!=result['kl_top1']['accuracy']:
        raise AssertionError('positive expert temperature must preserve fixed-route Top-1 accuracy')
    return result


def aggregate(output):
    rows=[json.loads(path.read_text()) for path in sorted(output.glob('env*/seed*/results.json'))]
    if not rows:return
    methods=rows[0]['metrics'].keys()
    summary={}
    for method in methods:
        summary[method]={}
        for metric in ['accuracy','nll','routing_regret']:
            values=[np.mean([row['metrics'][method][metric] for row in rows if row['seed']==seed]) for seed in range(3)
                    if sum(row['seed']==seed for row in rows)==4]
            summary[method][metric]=dict(mean=float(np.mean(values)) if values else None,
                seed_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,complete_seeds=len(values))
    comparisons={}
    for name,first,second in [('objective','ce_top2','kl_top2'),('ce_vs_global','ce_top2','global'),
                              ('ce_vs_fixed','ce_top2','fixed_pair'),('class_vs_input','class','kl_top2'),
                              ('calibration','calibrated_top2','kl_top2')]:
        comparisons[name]={}
        for metric in ['accuracy','nll']:
            deltas=[np.mean([row['metrics'][first][metric]-row['metrics'][second][metric] for row in rows if row['seed']==seed])
                    for seed in range(3) if sum(row['seed']==seed for row in rows)==4]
            mean=float(np.mean(deltas)) if deltas else None
            sd=float(np.std(deltas,ddof=1)) if len(deltas)>1 else None
            # 95% paired t interval across THREE training seeds, NOT across images.
            half=4.302652729911275*sd/np.sqrt(3) if len(deltas)==3 else None
            comparisons[name][metric]=dict(mean_delta=mean,seed_sd=sd,ci95=[mean-half,mean+half] if half is not None else None)
    write(output/'report.json',dict(completed_banks=len(rows),methods=summary,paired=comparisons,
          inference_scope='exploratory; previously observed PACS targets; 3 seed means, 4 targets per seed',
          practical_threshold_accuracy=.005,practical_threshold_nll=.02,class_equivalence_margin=.005,
          decision_note='Wide intervals mean inconclusive, not disproven; fixed-bank conclusions do not establish joint-training or causal mechanism claims.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--banks',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=300);p.add_argument('--device',default='cuda')
    p.add_argument('--launch-joint-on-source-gate',action='store_true')
    args=p.parse_args();torch.set_num_threads(4)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    source=Path(args.banks)
    plan=dict(existing_banks=12,targets=list(range(4)),seeds=list(range(3)),new_fits_per_bank=['ce_top2','ce_dense','class','calibration'],
        frozen_fit_total=48,epochs=args.epochs,router_lr=.003,teacher_baseline='reuse source-selected KL router',
        primary_objective='ce_top2 vs kl_top2, same architecture/init/features/Top-2/source selection',
        source_selection='equal-domain source-out NLL; earliest epoch on ties',
        h6_gate='each criterion must change Q at alpha=1 and be nondegenerate in >=8/12 source banks; Regret also needs objective spread >= typical probe SE',
        conditional_full_training_max=24 if args.launch_joint_on_source_gate else 0,
        statistical_scope='3-seed paired exploratory screen, not proof; minimum detectable effect depends on observed paired variance',
        preregistered_thresholds=dict(accuracy=.005,nll=.02,class_equivalence_margin=.005),
        source_hashes={path.name:sha(path) for path in Path(__file__).parent.glob('*.py')})
    write(output/'plan.json',plan)
    # Source-only phase for ALL 12 banks before any new target scoring.
    fitted=[]
    for env in range(4):
        for seed in range(3):
            bank=source/'native'/f'env{env}'/f'seed{seed}'
            fit=read(bank/'cache/source_in_original.pt');val=read(bank/'cache/source_out_original.pt')
            old=read(bank/'routers.pt');validate_sources(fit,val)
            for split in ['in','out']:
                if sha(bank/f'cache/source_{split}_original.pt')!=old['metadata']['source_cache_sha256'][split]:raise ValueError('cache changed')
            dest=output/f'env{env}'/f'seed{seed}';dest.mkdir(parents=True)
            components={}
            for name in ['ce_top2','ce_dense','class','calibration']:
                start=time.time()
                components[name]=fit_component(fit,val,old,name,args.epochs,device=args.device)
                print(f'source fit env{env}/seed{seed}/{name}: {time.time()-start:.1f}s valNLL={components[name]["validation_nll"]:.5f}',flush=True)
            torch.save(components,dest/'components.pt')
            diagnostic=assignment_diagnostic(old);write(dest/'assignment_diagnostic.json',diagnostic)
            write(dest/'source_selection.json',{name:{k:v for k,v in component.items() if k!='state'} for name,component in components.items()})
            fitted.append((env,seed,bank,dest))
    # This gate is decided before target scores are read by this follow-up.
    gate={criterion:sum(json.loads((dest/'assignment_diagnostic.json').read_text())[criterion]['eligible'] for _,_,_,dest in fitted)
          for criterion in ['Grad','Regret']}
    write(output/'source_gate.json',dict(eligible_banks=gate,required=8,alpha=1.,target_used=False))
    for env,seed,bank,dest in fitted:
        target=read(bank/'cache/target_all_original.pt');old=read(bank/'routers.pt')
        if target['metadata']['routers_sha256']!=sha(bank/'routers.pt'):raise ValueError('target provenance mismatch')
        components=read(dest/'components.pt');result=evaluate_bank(target,old,components)
        write(dest/'results.json',dict(target=env,seed=seed,bank_checkpoint=target['metadata']['checkpoint_sha256'],metrics=result))
        aggregate(output)
        print(f'evaluated env{env}/seed{seed}',flush=True)
    if args.launch_joint_on_source_gate:
        selected=[criterion for criterion,count in gate.items() if count>=8]
        write(output/'joint_decision.json',dict(criteria=selected,new_full_runs=12*len(selected),reason='preregistered source-only discrimination gate'))
        for criterion in selected:
            for env,seed,bank,dest in fitted:
                joint=output/'joint'/criterion/f'env{env}'/f'seed{seed}'
                cmd=[sys.executable,'-u','-m','experiments.sirm_followup.train_joint','--banks',str(source),'--criterion',criterion,
                     '--target-env',str(env),'--seed',str(seed),'--output',str(joint)]
                joint.parent.mkdir(parents=True,exist_ok=True)
                with joint.with_suffix('.log').open('w') as stream:
                    subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
    write(output/'completed.json',dict(frozen_banks=12,new_frozen_fits=48,status='complete'))
    print('completed frozen follow-up and any source-gated joint runs',flush=True)


if __name__=='__main__':main()

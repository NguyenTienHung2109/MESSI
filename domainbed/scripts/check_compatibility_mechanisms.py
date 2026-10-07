"""Four fixed, seed-0 mechanism checks; preserve existing models and selection.

This is exploratory analysis on an already studied dataset, not new external
validation. Held domains are excluded from all adaptation in each fold.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from domainbed.predictive_compatibility import (
    CandidateMap, make_cell, solve_head, solve_common, uniform_risk_bound, entropy_lower)
from domainbed.compatibility_diagnostics import prediction_metrics
from domainbed.scripts.audit_predictive_compatibility import train_map, write_json


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def covering_terms(n, classes=7, dimension=16, radius=1., alpha=.05/228):
    loss_bound=math.log(classes)+2*radius
    ball=radius*math.sqrt(classes)
    choices=[]
    for r in np.geomspace(1e-6, 2*ball, 160):
        complexity=classes*dimension*math.log1p(2*ball/r)
        confidence=math.log(2/alpha)
        sampling=loss_bound*math.sqrt((complexity+confidence)/(2*n))
        net=2*math.sqrt(2)*r
        choices.append(dict(n=n, net_radius=float(r), log_cover=complexity,
                            log_confidence=confidence, sampling_term=sampling,
                            net_term=float(net), uncapped_epsilon=float(sampling+net)))
    best=min(choices, key=lambda v:v['uncapped_epsilon'])
    return dict(best, epsilon=min(loss_bound,best['uncapped_epsilon']))


def first_feasible_n(rule, tau=.12):
    lo,hi=1,1
    while rule(hi)>tau:
        hi*=2
        if hi>10**10:
            return None
    while lo<hi:
        mid=(lo+hi)//2
        if rule(mid)<=tau: hi=mid
        else: lo=mid+1
    return lo


def bound_check(root, output):
    rows=[]
    manifest=json.loads((root/'env0_seed0/features.json').read_text())
    labels={d:torch.load(Path('predictive_support_outputs/compatibility_cache')/
        f'env{d}_{manifest[str(d)]["fingerprint"][:16]}.pt',map_location='cpu',weights_only=True)['y'] for d in range(4)}
    # Sharper deployment-head estimate adds a separate risk event per cell:
    # 96*(uniform risk + fixed-head risk + entropy) + 36 selection events.
    alpha_new=.05/324
    loss_bound=math.log(7)+2
    for target in range(4):
        for path in (root/f'env{target}_seed0').glob('*/compatibility.json'):
            splits=json.loads((path.parent/'split_manifest.json').read_text())
            for expert in json.loads(path.read_text()):
                for domain,d in zip(expert['support'],expert['audit']['domains']):
                    n=d['n']; terms=covering_terms(n)
                    uniform_new=uniform_risk_bound(n,7,16,1.,alpha_new)
                    fixed=loss_bound*math.sqrt(math.log(2/alpha_new)/(2*n))
                    counts=torch.bincount(labels[domain][splits[str(domain)]['audit']],minlength=7).numpy()
                    _,new_entropy_lower=entropy_lower(counts,alpha_new)
                    rows.append(dict(target=target,fold=path.parent.name,expert=expert['expert'],domain=domain,
                        **terms, empirical_regret_upper=d['empirical_regret_upper'],
                        audit_solver_gap=d['optimum']['gap'],calibration_solver_gap=expert['calibration']['gap'],
                        old_regret_upper=d['regret_upper'], fixed_head_epsilon=fixed,
                        refined_regret_upper=d['empirical_regret_upper']+uniform_new+fixed,
                        old_predictive_margin=d['entropy_lower']-d['risk_upper'],
                        fixed_head_predictive_margin=new_entropy_lower-min(loss_bound,d['risk']+fixed)))
    save_csv(output/'bound_decomposition.csv',rows)
    curve=[]
    for dimension in [1,4,16,64,384]:
        for n in [334,410,469,786,1000,10000,100000,1000000,10000000]:
            t=covering_terms(n,dimension=dimension)
            curve.append(dict(dimension=dimension,**t))
    save_csv(output/'bound_sample_complexity.csv',curve)
    old_n=first_feasible_n(lambda n:2*uniform_risk_bound(n,7,16,1.,.05/228))
    refined_n=first_feasible_n(lambda n:uniform_risk_bound(n,7,16,1.,alpha_new)
        +loss_bound*math.sqrt(math.log(2/alpha_new)/(2*n)))
    result=dict(optimistic_n_per_domain_zero_empirical_regret=old_n,
                refined_optimistic_n_per_domain_zero_empirical_regret=refined_n,
                old_min_regret_upper=min(r['old_regret_upper'] for r in rows),
                refined_min_regret_upper=min(r['refined_regret_upper'] for r in rows),
                tau=.12, refined_rule='empirical regret upper + uniform epsilon + fixed-head Hoeffding epsilon',
                note='Different risk-event allocation .05/324. No original acceptance decisions changed. Required n assumes iid data, fixed dimension and zero empirical regret.')
    write_json(output/'bound_summary.json',result)
    return result


def fit_constant_router(logits,y,steps=300,lr=.001):
    parameter=nn.Parameter(torch.zeros(logits.shape[1]))
    optimizer=torch.optim.Adam([parameter],lr=lr)
    generator=torch.Generator().manual_seed(0)
    for _ in range(steps):
        ix=torch.randint(len(y),(64,),generator=generator)
        loss=F.cross_entropy((parameter.softmax(0)[None,:,None]*logits[ix]).sum(1),y[ix])
        optimizer.zero_grad();loss.backward();optimizer.step()
    return parameter.detach().softmax(0)


def merged_cell(cells):
    return make_cell(torch.cat([c[0] for c in cells]),torch.cat([c[1] for c in cells]),
                     torch.cat([c[2]/len(cells) for c in cells]))


def paired_stats(a,b,y):
    # Positive CE delta means the first model is worse.
    gains=(F.cross_entropy(a,y,reduction='none')-F.cross_entropy(b,y,reduction='none')).numpy()
    rng=np.random.default_rng(0)
    means=np.array([rng.choice(gains,len(gains),replace=True).mean() for _ in range(500)])
    return dict(ce_delta=float(gains.mean()), bootstrap_low=float(np.quantile(means,.025)),
                bootstrap_high=float(np.quantile(means,.975)))


def check_fold(root, target, name, bank, output, cache):
    folder=root/f'env{target}_seed0'/name
    payload=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    config=payload['config']; args=SimpleNamespace(**config)
    splits={int(d):{k:np.array(v) for k,v in s.items()} for d,s in json.loads((folder/'split_manifest.json').read_text()).items()}
    sources=sorted(splits)
    held=target if name=='main' else int(name.removeprefix('loso_env'))
    assert held not in sources and target not in sources
    destination=output/f'env{target}_{name}';destination.mkdir()
    maps=[]
    for state in payload['maps']:
        m=CandidateMap(384,args.dimension,args.hidden);m.load_state_dict(state);maps.append(m.eval().requires_grad_(False))
    router=nn.Linear(384,len(maps));router.load_state_dict(payload['all_router']);router.eval().requires_grad_(False)
    @torch.no_grad()
    def logits_for(x):
        return torch.stack([m(x)@w.T for m,w in zip(maps,payload['heads'])],1)
    max_n=max(len(splits[d]['train']) for d in sources)
    train_x=torch.cat([bank[d][0][np.resize(splits[d]['train'],max_n)] for d in sources])
    train_y=torch.cat([bank[d][1][np.resize(splits[d]['train'],max_n)] for d in sources])
    constant=fit_constant_router(logits_for(train_x),train_y)

    # Fit additional probes and a single wider representation intervention on
    # adaptation sources only. Cache identical source sets across exclusion runs.
    key=tuple(sources)
    if key not in cache:
        torch.manual_seed(0)
        wide_args=SimpleNamespace(**dict(config,dimension=64))
        wide,trace=train_map({d:bank[d] for d in sources},splits,sources,wide_args,7,0,
                            destination/'wide_training.jsonl',0)
        fits={}
        with torch.no_grad():
            representations={'backbone384':{d:F.normalize(bank[d][0],dim=1) for d in sources},
                             'mlp16':{d:maps[0](bank[d][0]) for d in sources},
                             'mlp64':{d:wide(bank[d][0]) for d in sources}}
        for representation,z in representations.items():
            cells=[make_cell(z[d][splits[d]['calibration']],bank[d][1][splits[d]['calibration']]) for d in sources]
            conditions=[('mean',1.),('minimax',1.)] if representation!='mlp16' else [('mean',1.)]
            if representation in ['backbone384','mlp16']:conditions.append(('mean',4.))
            for objective,radius in conditions:
                label=f'{representation}_{objective}_r{int(radius)}'
                print(f'  fit {sources}: {label}',flush=True)
                w,certificate=(solve_common(cells,7,radius,args.solver_steps,args.solver_tolerance)
                    if objective=='minimax' else solve_head(merged_cell(cells),7,radius,args.solver_steps,args.solver_tolerance))
                fits[label]=(w.float(),certificate)
        cache[key]=(wide,fits,trace,{k:v.clone() for k,v in maps[0].state_dict().items()})
    wide,fits,trace,original_state=cache[key]
    assert all(torch.equal(v,original_state[k]) for k,v in maps[0].state_dict().items()), 'Cached fold map mismatch'
    write_json(destination/'probe_certificates.json',{k:v[1] for k,v in fits.items()})
    write_json(destination/'adaptation.json',dict(sources=sources,held=held,original_target=target,
        stages={str(d):{k:v.tolist() for k,v in sp.items()} for d,sp in splits.items()},
        constant_router_mass=constant.tolist(),wide_training_trace=trace,
        probe_supervision='calibration labels only; original/wide MLPs additionally learn maps on train labels',
        cached_by_adaptation_domain_set=list(key)))
    torch.save(dict(probe_heads={k:v[0] for k,v in fits.items()},wide_map=wide.state_dict(),
                    constant_router_mass=constant),destination/'diagnostic_models.pt')
    rows, comparisons=[],[]
    eval_sets=[('held',held,*bank[held])]+[('source_selection',d,bank[d][0][splits[d]['selection']],
                                          bank[d][1][splits[d]['selection']]) for d in sources]
    for split,domain,x,y in eval_sets:
        with torch.no_grad():
            logits=logits_for(x);weights=router(x).softmax(1)
            learned=(weights[:,:,None]*logits).sum(1)
            uniform=logits.mean(1)
            predictions={'global_original':logits[:,0], 'router_learned':learned,
                         'router_uniform':uniform,'router_half_uniform':.5*learned+.5*uniform,
                         'router_constant':(constant[None,:,None]*logits).sum(1)}
            z={'backbone384':F.normalize(x,dim=1),'mlp16':maps[0](x),'mlp64':wide(x)}
            predictions.update({label:z[label.split('_')[0]]@w.T for label,(w,_) in fits.items()})
            for label,prediction in predictions.items():
                metrics=prediction_metrics(prediction,y)
                rows.append(dict(target=target,fold=name,split=split,domain=domain,method=label,
                    n=len(y),accuracy=metrics['accuracy'],ce=metrics['cross_entropy'],
                    confidence=metrics['mean_confidence'],ece=metrics['ece_15']))
            for a,b in [('router_learned','router_uniform'),('router_constant','router_uniform'),
                        ('router_half_uniform','router_uniform'),('backbone384_mean_r1','mlp16_mean_r1'),
                        ('mlp16_mean_r1','global_original'),('mlp64_minimax_r1','global_original'),
                        ('backbone384_mean_r4','backbone384_mean_r1'),('mlp16_mean_r4','mlp16_mean_r1')]:
                comparisons.append(dict(target=target,fold=name,split=split,domain=domain,
                                        first=a,reference=b,**paired_stats(predictions[a],predictions[b],y)))
            if split=='held':
                np.savez_compressed(destination/'held_logits.npz',labels=y.numpy(),
                                    **{k:v.numpy() for k,v in predictions.items()})
    return rows,comparisons


def report(output,rows,comparisons,bounds):
    names=['Art','Cartoon','Photo','Sketch']
    methods=list(dict.fromkeys(r['method'] for r in rows))
    lines=['# Four mechanism checks — PACS, seed 0','',
           'Prespecified diagnostic controls; original models, supports and deployment choices remain unchanged.',
           'This is exploratory analysis on previously inspected PACS data. No target scores select a method.', '',
           '## 1. Statistical bound','',
           f'Original best regret upper: {bounds["old_min_regret_upper"]:.3f}; fixed-head refinement: '
           f'{bounds["refined_min_regret_upper"]:.3f}; tolerance: .12.',
           f'Even with zero empirical regret, the existing bound needs about '
           f'{bounds["optimistic_n_per_domain_zero_empirical_regret"]:,} audit samples/domain; '
           f'the elementary fixed-head refinement needs {bounds["refined_optimistic_n_per_domain_zero_empirical_regret"]:,}.',
           'These are sample requirements of these sufficient bounds, not information-theoretic lower limits.',
           'The refinement bounds the already fixed deployed head pointwise; the audit optimum still requires a uniform bound.',
           'Full decomposition: [bound_decomposition.csv](bound_decomposition.csv). Dimension/sample-size sensitivity: '
           '[bound_sample_complexity.csv](bound_sample_complexity.csv). No threshold was relaxed.', '',
           '## 2. Router intervention on strictly excluded sources','',
           '| Control | Mean accuracy | Mean CE | Folds with lower CE than uniform |',
           '|---|---:|---:|---:|']
    for method in ['router_uniform','router_learned','router_constant','router_half_uniform']:
        r=[v for v in rows if v['fold']!='main' and v['split']=='held' and v['method']==method]
        c=[v for v in comparisons if v['fold']!='main' and v['split']=='held' and v['first']==method and v['reference']=='router_uniform']
        lines.append(f'| {method} | {100*np.mean([v["accuracy"] for v in r]):.2f}% | {np.mean([v["ce"] for v in r]):.4f} | '
                     f'{sum(v["ce_delta"]<0 for v in c)}/12 |' if method!='router_uniform' else
                     f'| {method} | {100*np.mean([v["accuracy"] for v in r]):.2f}% | {np.mean([v["ce"] for v in r]):.4f} | reference |')
    lines += ['', 'Constant routing learns only a single mixture vector on source-training data. Half-uniform is a fixed 50/50 mixture, without tuning.',
              'Paired image bootstrap intervals in comparisons.csv are descriptive; folds overlap and are not 12 independent datasets.', '',
              '## 3–4. Direct backbone probes and wider representation','',
              '| Method | Art | Cartoon | Photo | Sketch | Mean | Excluded-source mean |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for method in methods:
        accuracy=[next(r['accuracy'] for r in rows if r['target']==t and r['fold']=='main' and r['split']=='held' and r['method']==method) for t in range(4)]
        transfer=[r['accuracy'] for r in rows if r['fold']!='main' and r['split']=='held' and r['method']==method]
        lines.append('| '+method+' | '+' | '.join(f'{100*v:.2f}%' for v in [*accuracy,np.mean(accuracy),np.mean(transfer)])+' |')
    lines += ['', 'Direct probe = L2-normalized frozen 384-D DeiT features. MLP16 = existing trained map, with refitted heads.',
              'MLP64 = the same 384→64→output MLP training protocol with output dimension enlarged from 16 to 64.',
              'All maps remain frozen during convex head fitting. Calibration samples and per-domain averaging weights are identical.',
              'Both pooled mean-CE and minimax-regret heads are reported to separate head-objective effects from representation effects.',
              'r4 changes only the head row-radius for fixed maps; it is a scale sensitivity diagnostic, not a certified candidate.',
              'Larger dimensions/radii also weaken the current uniform bounds. MLP64 is not asserted to preserve certification.',
              'Direct probes use calibration labels only; MLPs additionally learned maps from train labels. This gives the direct probe less supervised fitting, not a perfectly matched total training procedure.',
              'Budget-limited convex solver gaps are retained in each fold/probe_certificates.json; small performance differences need caution when convergence is weak.', '',
              '## Scope of the fourth check','',
              'This checks a fixed representation-capacity intervention before deciding whether backbone fine-tuning is warranted.',
              'No backbone fine-tuning is performed here. A useful direct probe would show that frozen features retain exploitable signal;',
              'a poor probe would not alone prove that fine-tuning is necessary. Conclusions must consider excluded-source evidence.', '',
              '[All metrics](metrics.csv) · [Paired comparisons](comparisons.csv) · [Fixed protocol](protocol.json)']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def matched_probe_followup(root,output):
    """Additional control: match the union of labels used by map+head learning.

    This change addresses a known supervision imbalance, without selecting any
    hyperparameter or model from held-domain scores. Original rows are kept.
    """
    marker=output/'matched_probe_protocol.json'
    if marker.exists():raise FileExistsError('Matched-probe follow-up already started')
    write_json(marker,dict(seed=0,head_fit_splits=['train','calibration'],
        excluded_splits=['audit','selection','held'],representations=['backbone384','mlp16'],
        radius=[1.,4.],objective='equal-domain mean CE',solver_steps=1000,
        reason='Match supervised sample pool; all controls reported, no target-based selection'))
    manifest=json.loads((root/'env0_seed0/features.json').read_text())
    bank={}
    for d in range(4):
        data=torch.load(Path('predictive_support_outputs/compatibility_cache')/
            f'env{d}_{manifest[str(d)]["fingerprint"][:16]}.pt',map_location='cpu',weights_only=True)
        bank[d]=(data['x'],data['y'])
    rows=[];fitted={}
    for target in range(4):
        for name in ['main']+[f'loso_env{d}' for d in range(4) if d!=target]:
            folder=root/f'env{target}_seed0'/name
            payload=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            splits={int(d):{k:np.array(v) for k,v in sp.items()}
                    for d,sp in json.loads((folder/'split_manifest.json').read_text()).items()}
            sources=tuple(sorted(splits));held=target if name=='main' else int(name.removeprefix('loso_env'))
            assert target not in sources and held not in sources
            model=CandidateMap(384,16,64);model.load_state_dict(payload['maps'][0]);model.eval().requires_grad_(False)
            if sources not in fitted:
                heads={}
                for representation in ['backbone384','mlp16']:
                    cells=[]
                    for d in sources:
                        indices=np.r_[splits[d]['train'],splits[d]['calibration']]
                        x,y=bank[d][0][indices],bank[d][1][indices]
                        with torch.no_grad():z=F.normalize(x,dim=1) if representation=='backbone384' else model(x)
                        cells.append(make_cell(z,y))
                    for radius in [1.,4.]:
                        label=f'{representation}_traincal_mean_r{int(radius)}'
                        print(f'MATCHED {sources}: {label}',flush=True)
                        w,certificate=solve_head(merged_cell(cells),7,radius,1000,1e-5)
                        heads[label]=(w.float(),certificate)
                fitted[sources]=(heads,{k:v.clone() for k,v in model.state_dict().items()})
            heads,original=fitted[sources]
            assert all(torch.equal(v,original[k]) for k,v in model.state_dict().items())
            destination=output/f'env{target}_{name}'
            write_json(destination/'matched_probe_certificates.json',{k:v[1] for k,v in heads.items()})
            torch.save({k:v[0] for k,v in heads.items()},destination/'matched_probe_heads.pt')
            eval_sets=[('held',held,*bank[held])]+[('source_selection',d,bank[d][0][splits[d]['selection']],
                bank[d][1][splits[d]['selection']]) for d in sources]
            for split,domain,x,y in eval_sets:
                with torch.no_grad():
                    z={'backbone384':F.normalize(x,dim=1),'mlp16':model(x)}
                    for label,(w,_) in heads.items():
                        pred=z[label.split('_')[0]]@w.T
                        metrics=prediction_metrics(pred,y)
                        rows.append(dict(target=target,fold=name,split=split,domain=domain,method=label,n=len(y),
                            accuracy=metrics['accuracy'],ce=metrics['cross_entropy'],
                            confidence=metrics['mean_confidence'],ece=metrics['ece_15']))
    save_csv(output/'matched_probe_metrics.csv',rows)
    text=['','## Additional control: matched supervised sample pool','',
          'Both heads fit the same union of train+calibration samples with equal domain weighting.',
          'The MLP also used train labels to learn its map; the direct backbone is unchanged.',
          'No audit, selection or held-domain labels enter fitting. This supplements rather than replaces the initial probes.','',
          '| Method | Art | Cartoon | Photo | Sketch | Mean | Excluded-source mean |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for method in dict.fromkeys(r['method'] for r in rows):
        accuracy=[next(r['accuracy'] for r in rows if r['target']==t and r['fold']=='main' and r['split']=='held' and r['method']==method) for t in range(4)]
        transfer=[r['accuracy'] for r in rows if r['fold']!='main' and r['split']=='held' and r['method']==method]
        text.append('| '+method+' | '+' | '.join(f'{100*v:.2f}%' for v in [*accuracy,np.mean(accuracy),np.mean(transfer)])+' |')
    with (output/'report.md').open('a') as stream:stream.write('\n'.join(text)+'\n')
    (output/'matched_probe_source_snapshot.py').write_bytes(Path(__file__).read_bytes())


def standard_probe_check(root,output):
    """Conventional linear probe to isolate restrictive normalization/head class.

    Fixed C=1, intercept enabled; no grid search. This is not a certified head
    in the method's original class and cannot be substituted into its bound.
    """
    from sklearn.linear_model import LogisticRegression
    marker=output/'standard_probe_protocol.json'
    if marker.exists():raise FileExistsError('Standard probe check already started')
    write_json(marker,dict(seed=0,C=1.,solver='lbfgs',max_iter=2000,tol=1e-6,intercept=True,
        representations=['raw_frozen_CLS384','learned_unit_MLP16'],fit_splits=['train','calibration'],
        sample_weight='equal domains, weights sum to number of training examples',
        no_search=True,certification_applies=False,
        purpose='Separate representation quality from the normalized, row-bounded head restriction'))
    manifest=json.loads((root/'env0_seed0/features.json').read_text());bank={}
    for d in range(4):
        p=torch.load(Path('predictive_support_outputs/compatibility_cache')/
            f'env{d}_{manifest[str(d)]["fingerprint"][:16]}.pt',map_location='cpu',weights_only=True)
        bank[d]=(p['x'],p['y'])
    cached={};rows=[];model_info=[]
    for target in range(4):
        for name in ['main']+[f'loso_env{d}' for d in range(4) if d!=target]:
            folder=root/f'env{target}_seed0'/name
            payload=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            splits={int(d):{k:np.array(v) for k,v in s.items()}
                for d,s in json.loads((folder/'split_manifest.json').read_text()).items()}
            sources=tuple(sorted(splits));held=target if name=='main' else int(name.removeprefix('loso_env'))
            assert target not in sources and held not in sources
            model=CandidateMap(384,16,64);model.load_state_dict(payload['maps'][0]);model.eval().requires_grad_(False)
            if sources not in cached:
                ids={d:np.r_[splits[d]['train'],splits[d]['calibration']] for d in sources}
                x=torch.cat([bank[d][0][ids[d]] for d in sources]);y=torch.cat([bank[d][1][ids[d]] for d in sources])
                weight=np.concatenate([np.full(len(ids[d]),len(y)/(len(sources)*len(ids[d]))) for d in sources])
                with torch.no_grad():reps={'backbone_raw_logreg':x.numpy(),'mlp16_logreg':model(x).numpy()}
                classifiers={}
                for label,z in reps.items():
                    print(f'STANDARD {sources}: {label}',flush=True)
                    lr=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,tol=1e-6,fit_intercept=True,random_state=0)
                    lr.fit(z,y.numpy(),sample_weight=weight)
                    assert np.array_equal(lr.classes_,np.arange(7))
                    classifiers[label]=lr
                cached[sources]=(classifiers,{k:v.clone() for k,v in model.state_dict().items()})
            classifiers,state=cached[sources]
            assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
            destination=output/f'env{target}_{name}'
            write_json(destination/'standard_probe_heads.json',{k:dict(coef=v.coef_.tolist(),bias=v.intercept_.tolist(),
                iterations=v.n_iter_.tolist(),classes=v.classes_.tolist()) for k,v in classifiers.items()})
            for label,lr in classifiers.items():model_info.append(dict(target=target,fold=name,method=label,
                iterations=int(max(lr.n_iter_)),hit_iteration_limit=bool(max(lr.n_iter_)>=2000)))
            eval_sets=[('held',held,*bank[held])]+[('source_selection',d,bank[d][0][splits[d]['selection']],
                bank[d][1][splits[d]['selection']]) for d in sources]
            for split,domain,x,y in eval_sets:
                with torch.no_grad():reps={'backbone_raw_logreg':x.numpy(),'mlp16_logreg':model(x).numpy()}
                for label,z in reps.items():
                    logits=torch.from_numpy(classifiers[label].decision_function(z)).float()
                    m=prediction_metrics(logits,y)
                    rows.append(dict(target=target,fold=name,split=split,domain=domain,method=label,n=len(y),
                        accuracy=m['accuracy'],ce=m['cross_entropy'],confidence=m['mean_confidence'],ece=m['ece_15']))
    save_csv(output/'standard_probe_metrics.csv',rows);save_csv(output/'standard_probe_optimization.csv',model_info)
    text=['','## Conventional linear-probe control (outside certified head class)','',
          'Fixed multinomial logistic regression C=1, intercept enabled, raw frozen CLS or learned unit MLP16 features.',
          'Both use train+calibration labels, equal-domain weighting, no parameter search and no held-domain fitting.',
          'This changes feature scaling and classifier restrictions together; it tests recoverable linear signal, not a matched certified-method replacement.','',
          '| Method | Art | Cartoon | Photo | Sketch | Mean | Excluded-source mean |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for method in dict.fromkeys(r['method'] for r in rows):
        accuracy=[next(r['accuracy'] for r in rows if r['target']==t and r['fold']=='main' and r['split']=='held' and r['method']==method) for t in range(4)]
        transfer=[r['accuracy'] for r in rows if r['fold']!='main' and r['split']=='held' and r['method']==method]
        text.append('| '+method+' | '+' | '.join(f'{100*v:.2f}%' for v in [*accuracy,np.mean(accuracy),np.mean(transfer)])+' |')
    with (output/'report.md').open('a') as stream:stream.write('\n'.join(text)+'\n')
    (output/'standard_probe_source_snapshot.py').write_bytes(Path(__file__).read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('predictive_support_outputs/pacs_compatibility_lodo_seed0'))
    parser.add_argument('--output',type=Path,default=Path('predictive_support_outputs/pacs_mechanism_checks_seed0'))
    parser.add_argument('--matched-probe-followup',action='store_true')
    parser.add_argument('--standard-probe-check',action='store_true')
    args=parser.parse_args()
    if args.standard_probe_check:
        torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
        standard_probe_check(args.root,args.output)
        return
    if args.matched_probe_followup:
        torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
        matched_probe_followup(args.root,args.output)
        return
    if args.output.exists():raise FileExistsError('Use fresh output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    write_json(args.output/'protocol.json',dict(seed=0,targets=[0,1,2,3],loso=True,
        router_controls=['original','uniform','constant_train300_lr.001','fixed_half_uniform'],
        feature_controls=['existing_mlp16','normalized_backbone384','mlp_output64_train500'],
        head_conditions=['mean_r1','minimax_r1','mean_r4_for_existing_maps'],
        solver_steps=1000,solver_tolerance=1e-5,bootstrap_replicates=500,
        selection='none; diagnostic comparisons only',fine_tuning=False))
    (args.output/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    bounds=bound_check(args.root,args.output)
    manifest=json.loads((args.root/'env0_seed0/features.json').read_text())
    bank={}
    for d in range(4):
        f=Path('predictive_support_outputs/compatibility_cache')/f'env{d}_{manifest[str(d)]["fingerprint"][:16]}.pt'
        payload=torch.load(f,map_location='cpu',weights_only=True)
        assert payload['fingerprint']==manifest[str(d)]['fingerprint']
        bank[d]=(payload['x'],payload['y'])
    rows,comparisons,cache=[],[],{}
    for target in range(4):
        for name in ['main']+[f'loso_env{d}' for d in range(4) if d!=target]:
            print(f'CHECK target={target} fold={name}',flush=True)
            write_json(args.output/'status.json',dict(state='running',target=target,fold=name))
            r,c=check_fold(args.root,target,name,bank,args.output,cache)
            rows.extend(r);comparisons.extend(c)
            save_csv(args.output/'metrics.csv',rows);save_csv(args.output/'comparisons.csv',comparisons)
    report(args.output,rows,comparisons,bounds)
    write_json(args.output/'status.json',dict(state='completed',seed=0,folds=16))


if __name__=='__main__':main()

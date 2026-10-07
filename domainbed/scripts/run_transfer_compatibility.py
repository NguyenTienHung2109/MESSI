"""Seed-0 matched tests of hull certification and episodic feature learning."""
import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F

from domainbed.predictive_compatibility import CandidateMap,make_cell,solve_head,solve_common,audit_head
from domainbed.transfer_compatibility import train_transfer_map,hull_audit,bernstein_upper,ce_interval
from domainbed.compatibility_diagnostics import prediction_metrics,bank_behavior
from domainbed.scripts.audit_predictive_compatibility import candidate_supports,write_json,train_map
from domainbed.scripts.check_compatibility_mechanisms import merged_cell,save_csv

VARIANTS=['erm','within_domain','cross_domain']


def family_events():
    total=0
    for target in range(4):
        sources=[d for d in range(4) if d!=target]
        for fold in [sources]+[[d for d in sources if d!=held] for held in sources]:
            # vertices=|support|+2; hull directions + risk + entropy;
            # additionally original full-class risk/entropy audit, and selection.
            total+=sum(len(a)*(len(a)+6) for a in candidate_supports(fold))+len(fold)
    return total*len(VARIANTS)


def mixture_ce_interval(heads):
    distances=torch.stack([torch.cdist(w.double(),w.double()) for w in heads]).mean(0)
    return float(torch.logsumexp(-distances,1).min()),float(torch.logsumexp(distances,1).max())


def run_fold(root,output,target,name,variant,bank,alpha_event,feature_mode='cls'):
    expected={'erm':'cls','dist_erm':'dist','mean_erm':'mean_tokens'}
    if variant in expected and feature_mode!=expected[variant]:
        raise ValueError(f'{variant} requires feature_mode={expected[variant]}')
    original=root/f'env{target}_seed0'/name
    old=torch.load(original/'model.pt',map_location='cpu',weights_only=True)
    args=SimpleNamespace(**dict(old['config'],ridge_lambda=.01,transfer_weight=1.,feature_mode=feature_mode))
    splits={int(d):{k:np.asarray(v) for k,v in s.items()}
            for d,s in json.loads((original/'split_manifest.json').read_text()).items()}
    sources=sorted(splits);held=target if name=='main' else int(name.removeprefix('loso_env'))
    assert held not in sources and target not in sources
    out=output/variant/f'env{target}_{name}';out.mkdir(parents=True)
    supports=candidate_supports(sources)
    maps=[];heads=[];reports=[];rows=[];saved_vertices=[]
    for expert,support in enumerate(supports):
        print(f'{variant} target={target} {name} expert={expert} support={support}',flush=True)
        if variant=='erm':
            model=CandidateMap(384,16,64);model.load_state_dict(old['maps'][expert]);model.eval().requires_grad_(False)
            trace=[]
        elif variant in ['dist_erm','mean_erm']:
            model,trace=train_map({d:bank[d] for d in sources},splits,support,args,7,expert,
                                   out/'training.jsonl',expert)
        else:
            model,trace=train_transfer_map({d:bank[d] for d in sources},splits,support,args,7,expert,
                                          out/'training.jsonl',expert,variant)
        with torch.no_grad():z={d:model(bank[d][0]) for d in support}
        def cells(stage):
            return [make_cell(z[d][splits[d][stage]],bank[d][1][splits[d][stage]]) for d in support]
        calibration=cells('calibration')
        w,common=solve_common(calibration,7,args.radius,args.solver_steps,args.solver_tolerance)
        # Every calibrated domain head is retained, never selected using audit.
        references=[solve_head(c,7,args.radius,args.solver_steps,args.solver_tolerance) for c in calibration]
        pooled,pooled_certificate=solve_head(merged_cell(calibration),7,args.radius,args.solver_steps,args.solver_tolerance)
        vertices=torch.stack([w,*[r[0] for r in references],pooled])
        counts=[torch.bincount(bank[d][1][splits[d]['audit']],minlength=7).tolist() for d in support]
        full=audit_head(w,cells('audit'),counts,args.radius,args.tau,args.delta,alpha_event,
                        args.solver_steps,args.solver_tolerance)
        local=[]
        for domain,full_row in zip(support,full['domains']):
            ix=splits[domain]['audit']
            audit=hull_audit(w,vertices,z[domain][ix],bank[domain][1][ix],args.tau,args.delta,alpha_event)
            local.append(audit)
            rows.append(dict(variant=variant,target=target,fold=name,expert=expert,domain=domain,
                n=audit['n'],full_class_accepted=full['accepted'],full_class_regret_upper=full_row['regret_upper'],
                full_class_empirical_regret_upper=full_row['empirical_regret_upper'],
                hull_regret_upper=audit['regret_upper'],hull_tangent_empirical=audit['empirical_tangent_gap'],
                risk_upper=audit['risk']['upper'],entropy_lower=audit['entropy']['lower'],
                predictive_margin=audit['entropy']['lower']-audit['risk']['upper'],
                compatible=audit['compatible'],predictive=audit['predictive'],domain_accepted=audit['accepted']))
        accepted=all(r['accepted'] for r in local)
        reports.append(dict(expert=expert,support=list(support),hull_accepted=accepted,
                            full_class_audit=full,hull_audits=local,common_solver=common,
                            reference_solvers=[r[1] for r in references],pooled_solver=pooled_certificate,
                            training_trace=trace,head_class='calibration-fixed convex hull'))
        maps.append(model);heads.append(w);saved_vertices.append(vertices)
        write_json(out/'certificates.json',reports)
    accepted=[0]+[i for i,r in enumerate(reports) if i>0 and r['hull_accepted']]
    @torch.no_grad()
    def predict(x):
        z=torch.stack([m(x) for m in maps],1)
        logits=torch.stack([z[:,i].double()@w.T for i,w in enumerate(heads)],1)
        return z,logits
    selection=[]
    low0,high0=ce_interval(heads[0]);low_mix,high_mix=mixture_ce_interval([heads[i] for i in accepted])
    for d in sources:
        ix=splits[d]['selection'];_,logits=predict(bank[d][0][ix]);y=bank[d][1][ix]
        gain=(F.cross_entropy(logits[:,0],y,reduction='none')-
              F.cross_entropy(logits[:,accepted].mean(1),y,reduction='none')).numpy()
        lower,upper=(0.,0.) if len(accepted)==1 else (low0-high_mix,high0-low_mix)
        bound=bernstein_upper(-gain,-upper,-lower,alpha_event)
        selection.append(dict(domain=d,gain=float(gain.mean()),gain_lower=-bound['upper'],bound_on_negative_gain=bound))
    use_mixture=len(accepted)>1 and np.mean([r['gain_lower'] for r in selection])>0
    summary=dict(variant=variant,feature_mode=feature_mode,sources=sources,held=held,accepted_experts=accepted[1:],
        selected='hull_uniform' if use_mixture else 'global',selection=selection,
        certificate_scope='reference hull only; full row-ball guarantee is not established',
        family_events=family_events(),alpha_per_event=alpha_event)
    write_json(out/'selection.json',summary)
    write_json(out/'splits.json',{str(d):{k:v.tolist() for k,v in s.items()} for d,s in splits.items()})
    torch.save(dict(maps=[m.state_dict() for m in maps],heads=heads,reference_vertices=saved_vertices,
                    accepted=accepted,selected=summary['selected'],config=vars(args)),out/'model.pt')
    metric_rows=[];behavior={}
    for stage,d,x,y in [('held',held,*bank[held])]+[('source_selection',d,bank[d][0][splits[d]['selection']],bank[d][1][splits[d]['selection']]) for d in sources]:
        z,logits=predict(x)
        composed=logits[:,accepted].mean(1)
        controls=dict(global_predictor=logits[:,0],uniform_all=logits.mean(1),hull_uniform=composed,
                      selected=composed if use_mixture else logits[:,0])
        scores={}
        for label,value in controls.items():
            scores[label]=prediction_metrics(value,y)
            metric_rows.append(dict(variant=variant,target=target,fold=name,stage=stage,domain=d,control=label,
                n=len(y),accuracy=scores[label]['accuracy'],ce=scores[label]['cross_entropy'],
                ece=scores[label]['ece_15'],confidence=scores[label]['mean_confidence']))
        weights=torch.full((len(y),len(maps)),1/len(maps),dtype=logits.dtype)
        behavior[f'{stage}_{d}']=dict(controls=scores,experts=[prediction_metrics(logits[:,i],y) for i in range(len(maps))],
            diagnostics=bank_behavior(z,logits,weights,y))
        if stage=='held':np.savez_compressed(out/'held_predictions.npz',labels=y.numpy(),features=z.numpy(),logits=logits.numpy())
    write_json(out/'behavior.json',behavior)
    return rows,metric_rows,summary


def report(output,rows,metrics,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines=['# Transfer-oriented compatibility revision — seed 0','',
           '**Scope:** certification is now relative to a fixed reference-head convex hull. It is not a full-row-ball certificate.',
           'Both old and new audit bounds are logged for each identical map/head. All settings were fixed before this run.',
           'The dataset has been inspected repeatedly: results are exploratory, not prospective confirmation of nominal coverage.', '',
           '| Feature objective | Accepted extra hull candidates (main / 16 total) | Main folds selecting mixture | Full-class accepted extra candidates |',
           '|---|---:|---:|---:|']
    for variant in VARIANTS:
        s=[x for x in summaries if x['variant']==variant and x['name']=='main']
        full={(r['target'],r['expert']) for r in rows if r['variant']==variant and r['fold']=='main' and r['expert']>0 and r['full_class_accepted']}
        lines.append(f'| {variant} | {sum(len(x["accepted_experts"]) for x in s)}/16 | {sum(x["selected"]=="hull_uniform" for x in s)}/4 | {len(full)} |')
    lines+=['','| Variant / control | Art | Cartoon | Photo | Sketch | Mean | Excluded-source mean |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        for control in ['global_predictor','uniform_all','selected']:
            main=[next(r['accuracy'] for r in metrics if r['variant']==variant and r['fold']=='main' and r['stage']=='held' and r['target']==t and r['control']==control) for t in range(4)]
            transfer=[r['accuracy'] for r in metrics if r['variant']==variant and r['fold']!='main' and r['stage']=='held' and r['control']==control]
            lines.append('| '+variant+' / '+control+' | '+' | '.join(f'{100*x:.2f}%' for x in [*main,np.mean(main),np.mean(transfer)])+' |')
    lines+=['','## Interpreting the interventions','',
        '- ERM retains the exact original maps. Its new certificate isolates the certification change.',
        '- Within-domain episodes control for extra ridge prediction supervision and compute. Query and inner examples are disjoint.',
        '- Cross-domain episodes fit the auxiliary ridge head on other source-training domains. Both episode variants use the same number of inner samples, same ERM minibatches, dimension, budget and initialization.',
        '- The additional loss is squared query prediction error through a differentiable ridge solve, not feature diversity or forced alignment.',
        '- Strict held domains enter neither feature learning, head calibration, certification nor composition selection.',
        '- Uniform mixing replaces the input-dependent router. Extra candidates are deployed only if paired source-selection CE gain has a positive lower confidence bound.',
        '- A hull certificate does not bound regret relative to every row-bounded head. The gap between the two optimum risks remains an additional approximation term.',
        '- Higher feature rank alone is not a success criterion; transfer performance and the compute-matched within-domain control determine whether the new objective has empirical support.', '',
        '[Certificates](certificates.csv) · [Metrics](metrics.csv) · [Protocol](protocol.json) · [Math](../../docs/transfer_compatibility.md)', '',
        '![Matched controls](comparison.png)']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for i,variant in enumerate(VARIANTS):
        selected=[next(r['accuracy'] for r in metrics if r['variant']==variant and r['control']=='selected' and r['fold']=='main' and r['stage']=='held' and r['target']==t) for t in range(4)]
        axes[0].bar(np.arange(4)+(i-1)*.25,np.array(selected)*100,.25,label=variant)
        held=[r for r in rows if r['variant']==variant and r['fold']=='main']
        axes[1].scatter(np.full(len(held),i),[r['hull_regret_upper'] for r in held],alpha=.65,label=variant)
    axes[0].set(xticks=range(4),xticklabels=['Art','Cartoon','Photo','Sketch'],ylim=(0,100),ylabel='Selected target accuracy (%)')
    axes[0].legend();axes[1].axhline(.12,color='black',linestyle='--',label='tau=.12')
    axes[1].set(xticks=range(3),xticklabels=VARIANTS,ylabel='Hull regret upper (nats)',title='Different reference class; not full-class regret')
    fig.savefig(output/'comparison.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('predictive_support_outputs/pacs_compatibility_lodo_seed0'))
    parser.add_argument('--output',type=Path,default=Path('predictive_support_outputs/pacs_transfer_compatibility_seed0'))
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a fresh output directory')
    args.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    protocol=dict(seed=0,variants=VARIANTS,targets=[0,1,2,3],strict_loso=True,alpha=.05,
        family_events=family_events(),tau=.12,delta=.05,ridge_lambda=.01,transfer_weight=1.,
        map_steps=500,head_steps=1000,dimension=16,router='uniform',
        hull='deployed common + all per-domain calibrated optima + pooled calibrated head',
        selection='independent source-selection paired empirical Bernstein gain',
        scope='fixed calibration reference hull, not full row-bounded class',
        coverage_note='iid and fixed-procedure assumptions; reused PACS is exploratory, not fresh validation')
    write_json(args.output/'protocol.json',protocol)
    for file in ['domainbed/transfer_compatibility.py',__file__]:
        dest=args.output/'source_snapshot'/Path(file).name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(Path(file).read_bytes())
    manifest=json.loads((args.root/'env0_seed0/features.json').read_text());bank={}
    for d in range(4):
        p=torch.load(Path('predictive_support_outputs/compatibility_cache')/f'env{d}_{manifest[str(d)]["fingerprint"][:16]}.pt',map_location='cpu',weights_only=True)
        assert p['fingerprint']==manifest[str(d)]['fingerprint'];bank[d]=(p['x'],p['y'])
    rows=[];metrics=[];summaries=[]
    for variant in VARIANTS:
        for target in range(4):
            for name in ['main']+[f'loso_env{d}' for d in range(4) if d!=target]:
                write_json(args.output/'status.json',dict(state='running',variant=variant,target=target,fold=name))
                r,m,s=run_fold(args.root,args.output,target,name,variant,bank,.05/family_events())
                rows.extend(r);metrics.extend(m);summaries.append(dict(s,name=name,target=target))
                save_csv(args.output/'certificates.csv',rows);save_csv(args.output/'metrics.csv',metrics)
                write_json(args.output/'selections.json',summaries)
    report(args.output,rows,metrics,summaries)
    write_json(args.output/'status.json',dict(state='completed',seed=0,folds=48))


if __name__=='__main__':main()

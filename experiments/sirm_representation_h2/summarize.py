"""Consolidated CSVs and explicitly paired factorial effects; missing cells stay missing."""
import argparse,csv,json,statistics,math
from pathlib import Path
import numpy as np
from .protocol import CONDITIONS

PRIMARY=['acc_original_native','worst_view_native','worst_view_global','acc_gray_blur_native','acc_gray_blur_global']
EFFECTS={'augmentation_without_SIRM':[-1,1,0,0],'augmentation_with_SIRM':[0,0,-1,1],
         'SIRM_baseline':[-1,0,1,0],'SIRM_H2':[0,-1,0,1],'interaction':[1,-1,-1,1]}

def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or sorted({k for row in rows for k in row}) or ['status']
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def describe(values):
    return dict(n=len(values),mean=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else '')

def paired_bootstrap(values,units,seed=321,repeats=5000):
    """Resample paired target×seed cells; also report fixed-target seed-block intervals."""
    a=np.asarray(values);gen=np.random.default_rng(seed)
    estimates=a[gen.integers(0,len(a),(repeats,len(a)))].mean(1)
    ci=np.quantile(estimates,[.025,.975]);seeds=sorted(set(s for _,s in units));blocks=[]
    for s in seeds:blocks.append(np.mean([v for v,(_,seed_id) in zip(values,units) if seed_id==s]))
    block=np.asarray(blocks);block_est=block[gen.integers(0,len(block),(repeats,len(block)))].mean(1)
    bci=np.quantile(block_est,[.025,.975])
    t_low=t_high=''
    if len(block) in [2,3]:
        critical={2:12.706204736432095,3:4.302652729911275}[len(block)]
        margin=critical*statistics.stdev(block.tolist())/math.sqrt(len(block))
        t_low=float(block.mean()-margin);t_high=float(block.mean()+margin)
    return dict(ci95_cell_low=float(ci[0]),ci95_cell_high=float(ci[1]),ci95_seed_block_low=float(bci[0]),ci95_seed_block_high=float(bci[1]),
                ci95_seed_t_low=t_low,ci95_seed_t_high=t_high,seed_blocks=len(block),
                bootstrap_unit='paired target×seed cell; dependencies across overlapping source domains limit cell CI',
                block_unit='seed block retaining all observed fixed targets; only3 blocks in full experiment')

def summarize(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());raw=[];diagnostics=[];lookup={};runs=[]
    for job in plan['jobs']:
        path=root/'runs'/job['condition']/f'env{job["target"]}'/f'seed{job["seed"]}';file=path/'metrics.json'
        row=dict(condition=job['condition'],target_env=job['target'],seed=job['seed'],augmentation=CONDITIONS[job['condition']][0],sirm_enabled=CONDITIONS[job['condition']][1],status='planned',reused=job['reused'])
        if (path/'failure.json').exists():row.update(status='failed',error=json.loads((path/'failure.json').read_text())['error'])
        if file.exists():
            result=json.loads(file.read_text());p=result['provenance'];raw.append(result);lookup[job['condition'],job['target'],job['seed']]=result
            row.update(result['metrics']);row.update(status='completed',selected_step=p['selected_step'],selected_source_accuracy=p['selected_source_accuracy'],
                checkpoint_sha256=p['checkpoint_sha256'],effective_config_hash=p['effective_config_hash'],split_hash=p['split_hash'],git_sha=p['git_sha'],
                backbone=p['hparams']['model'],pretrained_sha256=p['pretrained_sha256'],num_experts=p['hparams']['num_experts'],topk=p['hparams']['subset_irm_router_topk'],
                router_config=json.dumps(p.get('router_config',{'type':'Linear','in_features':384,'out_features':6,'bias':True})),optimizer=p['optimizer'],lr=p['hparams']['lr'],weight_decay=p['hparams']['weight_decay'],
                batch_size=p['hparams']['batch_size'],steps=p['total_steps'],source_split_seed=p['source_split_seed'],all_lambda_values=json.dumps(p['all_lambda_values']),
                global_weights=json.dumps(p['global_weights']),smoke=p['smoke'])
            diagnostics.append((p,json.loads((path/'source_diagnostics.json').read_text())))
        runs.append(row)
    write_csv(root/'runs.csv',runs)
    aggregate=[]
    metrics=sorted({k for r in raw for k in r['metrics']})
    for condition in CONDITIONS:
        for metric in metrics:
            cells=[r for r in raw if r['provenance']['condition']==condition]
            for target in plan['targets']:
                v=[r['metrics'][metric] for r in cells if r['provenance']['target_env']==target]
                if v:aggregate.append(dict(condition=condition,metric=metric,scope='per_target_across_seeds',target_env=target,**describe(v)))
            seedmeans=[]
            for seed in plan['seeds']:
                vals=[r['metrics'][metric] for r in cells if r['provenance']['seed']==seed]
                if len(vals)==len(plan['targets']):seedmeans.append(statistics.mean(vals))
            if seedmeans:aggregate.append(dict(condition=condition,metric=metric,scope='equal_target_mean_then_seed_SD',target_env='all',**describe(seedmeans)))
    write_csv(root/'aggregate.csv',aggregate)
    robustness=[];router=[];rep=[];probe=[]
    for p,d in diagnostics:
        identity=dict(condition=p['condition'],target_env=p['target_env'],seed=p['seed'])
        result=lookup[p['condition'],p['target_env'],p['seed']]
        for view,v in result['views'].items():
            for mode,met in v.items():robustness.append(dict(**identity,view=view,mode=mode,**met))
        router.extend(dict(**identity,data_split='source_out',**r) for r in d['router_stability'])
        rep.extend(dict(**identity,data_split='source_out',**r) for r in d['representation_stability'])
        for view,v in result['linear_probe'].items():probe.append(dict(**identity,view=view,scope='individual_checkpoint',**v,regularization=p['probe']['regularization']))
        target=root/'competence_matrices'/f'{p["condition"]}_env{p["target_env"]}_seed{p["seed"]}.json';target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(d['competence_matrix'],indent=2)+'\n')
        write_csv(target.with_suffix('.csv'),d['competence_rows'])
        q_path=root/'q_matrices'/target.name;q_path.parent.mkdir(exist_ok=True);q_path.write_text(json.dumps(d['q'],indent=2)+'\n')
    for enabled,first,second in [(False,'A','B'),(True,'C','D')]:
        for target in plan['targets']:
            for seed in plan['seeds']:
                a,b=lookup.get((first,target,seed)),lookup.get((second,target,seed))
                if a and b:
                    for view in a['linear_probe']:
                        probe.append(dict(condition=f'{second}-{first}',target_env=target,seed=seed,view=view,scope='paired_H2_probe_delta',sirm_enabled=enabled,
                            probe_acc_baseline=a['linear_probe'][view]['accuracy'],probe_acc_h2=b['linear_probe'][view]['accuracy'],
                            delta_probe=b['linear_probe'][view]['accuracy']-a['linear_probe'][view]['accuracy']))
    write_csv(root/'view_robustness.csv',robustness);write_csv(root/'router_stability.csv',router)
    write_csv(root/'representation_stability.csv',rep);write_csv(root/'linear_probe.csv',probe)
    factorial=[]
    all_metrics=sorted(set(PRIMARY+['acc_original_global','mean_view_native','mean_view_global','probe_acc_original']+[m for m in metrics if m.startswith('source_')]))
    units=[(t,s) for t in plan['targets'] for s in plan['seeds'] if all((c,t,s) in lookup for c in CONDITIONS)]
    for metric in all_metrics:
        for effect,coeff in EFFECTS.items():
            values=[]
            for t,s in units:
                delta=sum(coeff[i]*lookup[c,t,s]['metrics'][metric] for i,c in enumerate(CONDITIONS));values.append(delta)
                factorial.append(dict(effect=effect,metric=metric,scope='paired_cell',target_env=t,seed=s,delta=delta))
            for t in plan['targets']:
                v=[x for x,(target,_) in zip(values,units) if target==t]
                if v:factorial.append(dict(effect=effect,metric=metric,scope='per_target_across_seeds',target_env=t,**describe(v)))
            if values:
                factorial.append(dict(effect=effect,metric=metric,scope='overall_paired_cells',target_env='all',
                    complete_factorial=len(units)==len(plan['targets'])*len(plan['seeds']),**describe(values),**paired_bootstrap(values,units)))
    write_csv(root/'factorial_effects.csv',factorial)
    diagnostic_aggregate=[]
    for kind,rows in [('representation',rep),('router',router)]:
        measures=(['paired_cosine','cosine_margin','feature_variance','native_accuracy'] if kind=='representation' else ['router_cosine','router_js','topk_jaccard','accuracy'])
        groups={}
        for r in rows:
            key=(r['condition'],r['view'],r.get('representation','router'),r.get('expert',-1),r['target_env'],r['seed'])
            groups.setdefault(key,[]).append(r)
        # Equal source-domain averaging within each checkpoint, then preserve target×seed units.
        for measure in measures:
            units_by={}
            for key,part in groups.items():units_by.setdefault(key[:4],[]).append((key[4],key[5],statistics.mean(x[measure] for x in part)))
            for key,vals in units_by.items():
                for target in plan['targets']:
                    v=[x for t,s,x in vals if t==target]
                    if v:diagnostic_aggregate.append(dict(kind=kind,condition=key[0],view=key[1],representation=key[2],expert=key[3],metric=measure,target_env=target,scope='per_target_across_seeds',**describe(v)))
                means=[statistics.mean(x for t,s,x in vals if s==seed) for seed in plan['seeds'] if sum(s==seed for t,s,x in vals)==len(plan['targets'])]
                if means:diagnostic_aggregate.append(dict(kind=kind,condition=key[0],view=key[1],representation=key[2],expert=key[3],metric=measure,target_env='all',scope='equal_target_mean_then_seed_SD',**describe(means)))
    write_csv(root/'diagnostic_aggregate.csv',diagnostic_aggregate)
    status=dict(completed=len(raw),failed=sum(r['status']=='failed' for r in runs),planned=len(runs),paired_cells=len(units))
    summary=['# SIRM × H2 representation factorial experiment','',f'Completed {len(raw)}/{len(runs)} runs; failed {status["failed"]}.',
        '','OFF changes only `subset_irm_lambda_sirm` from 0.1 to 0. All architecture, router/expert losses and training settings are held fixed.',
        '','Native routing uses the selected model router, not a separately teacher-fitted input router. Worst view is the minimum of three view accuracies, not the previous weighted-mixture metric.',
        '','## Main table','', '| Condition | Aug | SIRM | Original native | Worst native | Worst global | Gray+blur native |', '|---|---|---|---:|---:|---:|---:|']
    def overall(condition,metric):
        rows=[r for r in aggregate if r['condition']==condition and r['metric']==metric and r['target_env']=='all']
        if not rows:return 'pending'
        row=rows[0];sd=row['sample_sd'];return f'{row["mean"]*100:.2f} ± {sd*100:.2f}' if sd!='' else f'{row["mean"]*100:.2f} (n=1 seed)'
    for c,(aug,enabled) in CONDITIONS.items():summary.append(f'| {c} | {aug} | {"ON" if enabled else "OFF"} | '+' | '.join(overall(c,m) for m in PRIMARY[:4])+' |')
    summary+=['','Values are percentage accuracy; SD is over complete per-seed equal-target means. Missing results are not imputed.',
        '','## Factorial effects','', '| Quantity | Original | Worst native | Worst global |', '|---|---:|---:|---:|']
    for effect in EFFECTS:
        cells=[]
        for metric in PRIMARY[:3]:
            matches=[r for r in factorial if r['effect']==effect and r['metric']==metric and r['scope']=='overall_paired_cells']
            if not matches or not matches[0]['complete_factorial']:cells.append('pending')
            else:
                row=matches[0];cells.append(f'{row["mean"]*100:+.2f} [{row["ci95_seed_block_low"]*100:+.2f}, {row["ci95_seed_block_high"]*100:+.2f}]')
        summary.append('| '+effect+' | '+' | '.join(cells)+' |')
    summary+=['','Effects are percentage points; brackets use seed-block paired bootstrap. `factorial_effects.csv` also contains cell bootstrap and all raw paired differences.',
        '','## Per-target main results','', '| Target | Condition | Original native | Worst native | Worst global | Gray+blur native |', '|---|---|---:|---:|---:|---:|']
    for target in plan['targets']:
        for c in CONDITIONS:
            entries=[]
            for metric in PRIMARY[:4]:
                rows=[r for r in aggregate if r['target_env']==target and r['condition']==c and r['metric']==metric]
                entries.append(f'{rows[0]["mean"]*100:.2f} (n={rows[0]["n"]})' if rows else 'pending')
            summary.append(f'| {target} | {c} | '+' | '.join(entries)+' |')
    summary+=['','## Representation and competence diagnostics','',
        '| Condition | Backbone cosine gray | Backbone cosine blur | Probe target original | Expert loss variability |',
        '|---|---:|---:|---:|---:|']
    def source_overall(c,m):
        matches=[r for r in aggregate if r['condition']==c and r['metric']==m and r['target_env']=='all']
        if not matches:return 'pending'
        r=matches[0];return f'{r["mean"]:.4f}'
    for c in CONDITIONS:
        summary.append('| '+c+' | '+' | '.join(source_overall(c,m) for m in ['source_backbone_cos_grayscale','source_backbone_cos_blur','probe_acc_original','source_competence_loss_std'])+' |')
    summary+=['','Probe values in this diagnostic table are accuracy fractions. Class cosine margins and paired-view classification accuracies are available in the CSVs; stability alone is not evidence of quality.',
        '','## Interpretation and limits','',
        'Full scientific interpretation is pending until all factorial cells and corresponding diagnostics are complete.' if len(raw)<len(runs) else 'The estimates below refer to this fixed PACS configuration; uncertainty and representation diagnostics must be considered together.',
        '', 'The intervention is augmentation × invariance-loss activation. A positive interaction demonstrates synergy of these interventions; it does not isolate representation robustness as the sole causal mediator.',
        '', 'Previously observed PACS targets make this exploratory. Three seeds limit uncertainty estimation. Cell bootstrap treats 12 paired target×seed cells as independent despite overlapping source data; seed-block intervals use only three blocks and must also be treated cautiously.',
        '', 'A confidence interval containing zero is not evidence of equivalence. Practical equivalence margin is ±0.5 percentage points, specified before OFF target evaluation.',
        '', 'Representation cosine is paired on source validation; class cosine margins guard against collapse. Frozen backbone probes fit source-in and select regularization on source-out. Target probe accuracy is diagnostic only.',
        '', 'Gray+blur is evaluation-only, not a dedicated training branch. Baseline RandomGrayscale(p=0.1) can combine with the H2 blur branch in approximately3.3% of training draws; combined perturbation is therefore not entirely unseen. Complementary-cue interpretations remain tentative.',
        '', 'No claim of semantic decomposition, causal feature identification, or performance on unseen datasets follows from this experiment.']
    if len(raw)==len(runs) and not plan['smoke']:
        interaction=next(r for r in factorial if r['effect']=='interaction' and r['metric']=='acc_original_native' and r['scope']=='overall_paired_cells')
        cells=[r['delta'] for r in factorial if r['effect']=='interaction' and r['metric']=='acc_original_native' and r['scope']=='paired_cell']
        def effect_value(effect,metric):
            return next(r['mean'] for r in factorial if r['effect']==effect and r['metric']==metric and r['scope']=='overall_paired_cells')
        probe_delta=effect_value('augmentation_with_SIRM','probe_acc_original')
        cosine_gray=effect_value('augmentation_with_SIRM','source_backbone_cos_grayscale')
        cosine_blur=effect_value('augmentation_with_SIRM','source_backbone_cos_blur')
        competence_delta=effect_value('augmentation_with_SIRM','source_competence_loss_std')
        summary+=['',f'Original/native interaction: {interaction["mean"]*100:+.3f}pp; positive in {sum(v>0 for v in cells)}/{len(cells)} paired cells.',
            f'H2-with-SIRM probe delta: {probe_delta*100:+.3f}pp; backbone paired cosine changes gray={cosine_gray:+.4f}, blur={cosine_blur:+.4f}; expert loss-variability change={competence_delta:+.5f}.']
        low,high=interaction['ci95_seed_t_low'],interaction['ci95_seed_t_high']
        competence_accuracy_delta=effect_value('augmentation_with_SIRM','source_expert_accuracy_original')
        competence_gray_delta=effect_value('augmentation_with_SIRM','source_expert_accuracy_grayscale')
        competence_blur_delta=effect_value('augmentation_with_SIRM','source_expert_accuracy_blur')
        robust_quality=(probe_delta>0 and cosine_gray>0 and cosine_blur>0) or (competence_delta<0 and min(competence_accuracy_delta,competence_gray_delta,competence_blur_delta)>=0)
        if low!='' and low>0 and sum(v>0 for v in cells)>=9 and robust_quality:
            interpretation='Evidence supports augmentation/SIRM synergy, consistent with representation or competence robustness enabling the subset mechanism. This does not identify representation as the sole causal mediator.'
        elif low!='' and low>=-.005 and high<=.005:
            interpretation=('H2 improves linearly transferable backbone features, but interaction is practically equivalent to zero within the preregistered margin; no larger marginal utility of SIRM is established.' if probe_delta>0 else 'Interaction is practically equivalent to zero within the preregistered margin; H2 benefits are consistent with generic augmentation/regularization.')
        elif high!='' and high<0:
            interpretation='H2 reduces the marginal contribution of the SIRM term in this configuration; mechanistic diagnostics may suggest redundancy, but do not establish its cause.'
        else:
            interpretation='The interaction remains inconclusive at three seeds. A near-zero point estimate or an interval containing zero does not establish equivalence or absence of a SIRM-specific benefit.'
        summary+=['',interpretation,'','Seed-block t intervals are supplied as an additional uncertainty check because bootstrap intervals with only three seed blocks can be optimistic. No statistical-significance claim is made from the 12 cell bootstrap.']
    (root/'summary.md').write_text('\n'.join(summary)+'\n')
    return status

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();print(json.dumps(summarize(args.output)))

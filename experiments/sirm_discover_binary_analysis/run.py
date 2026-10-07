"""Read-only monitoring and five-part report for the locked binary experiment."""
import argparse,itertools,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from experiments.sirm_h1_h6 import run as base
from experiments.sirm_discover_binary.run import restore
from experiments.sirm_discover_gate.data import Adapter
from experiments.sirm_discover_gate.metrics import evaluate
from experiments.sirm_representation_h2.summarize import write_csv
from domainbed.lib.query import Q
from domainbed.model_selection import OracleSelectionMethod


def read(p):return json.loads(p.read_text())
def rows(p):
    if not p.exists():return []
    result=[]
    for line in p.read_text().splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:pass # a writer may be midway through the final row
    return result

def describe(values):
    v=np.asarray(values,dtype=float)
    if not len(v):return dict(n=0,median=None,mean=None,p90=None,max=None)
    return dict(n=len(v),median=float(np.median(v)),mean=float(v.mean()),p90=float(np.quantile(v,.9)),max=float(v.max()),fraction_0_01_to_0_3=float(((v>=.01)&(v<=.3)).mean()),fraction_over_1=float((v>1).mean()),fraction_over_10=float((v>10).mean()))

def ratio_records(refresh):
    result=[]
    for r in refresh:
        if r['stage']!='invariance':continue
        grad=r.get('weighted_sirm_grad_norm');baseline=r.get('base_residual_grad_norm')
        ratio=grad/baseline if grad is not None and baseline is not None and baseline>0 else (0. if not r['sirm_active'] else None)
        result.append(dict(step=r['step'],active=r['sirm_active'],gradient_ratio=ratio,weighted_SIRM_grad_norm=grad,base_residual_grad_norm=baseline,eligible_terms=r['eligible_pair_expert_count'],lambda_sirm=r['effective_lambda_sirm'],source_validation_accuracy=r['source_validation_accuracy']))
    return result

def support(q):
    q=np.asarray(q);s=q>0;edges={(i,j,m) for i,j in itertools.combinations(range(len(q)),2) for m in range(q.shape[1]) if q[i,m]*q[j,m]>0}
    return s,edges

def evolution_stats(refresh):
    previous=None;changes=0;support_changes=0;distinct=set();counts=[];first_shared=None
    for r in refresh:
        q=np.asarray(r['q']);selected,edges=support(q)
        distinct.add(q.tobytes());counts.append(int(selected.sum()))
        if edges and first_shared is None:first_shared=r['step']
        if previous is not None:
            changes+=int(not np.array_equal(q,previous))
            support_changes+=int(not np.array_equal(selected,previous>0))
        previous=q
    return dict(refreshes=len(refresh),distinct_Q_matrices=len(distinct),Q_weight_changes=changes,Q_support_changes=support_changes,selected_cells_min=min(counts,default=None),selected_cells_max=max(counts,default=None),first_shared_support_step=first_shared)

def domainbed_scores(root):
    records=rows(root/'banks/R1/results.jsonl');target=read(root/'plan.json')['target']
    test_key=f'env{target}_in_acc';val_key=f'env{target}_out_acc'
    full=sorted([r for r in records if test_key in r and val_key in r],key=lambda r:r['step'])
    if not full:return None
    latest=full[-1];stage3=[r for r in full if r['step']>=2000]
    best=max(full,key=lambda r:r[val_key]);best3=max(stage3,key=lambda r:r[val_key]) if stage3 else None
    source=max(stage3,key=lambda r:r['source_val_mean']) if stage3 else None
    score=lambda r:dict(step=r['step'],target_test_acc=r[test_key],target_validation_acc=r[val_key],source_validation_acc=r['source_val_mean']) if r else None
    result=dict(oracle_method='local DomainBed OracleSelectionMethod: last evaluated checkpoint, no early stopping',oracle=OracleSelectionMethod.run_acc(Q(full)),oracle_step=latest['step'],latest=score(latest),best_target_validation_all_steps=score(best),best_target_validation_stage3=score(best3),current_source_selected=score(source),complete_target_score_rows=len(full),extra_source_only_rows=len(records)-len(full),per_domain_at_latest={k:float(v) for k,v in latest.items() if k.startswith('env') and k.endswith('_acc')},notes='Monitoring/post-hoc only: fixed configuration, one seed, no hparam sweep. Source-only extra Q-refresh rows lack target scores and are excluded. Target in/out are DomainBed 80/20 splits, distinct from full-target source-selected accuracy. Best-target-validation across steps is reported separately and does not change checkpoint selection.')
    dest=root/'analysis';dest.mkdir(exist_ok=True);base.write(dest/'DomainBed_scores.json',result)
    lines=['# R1 DomainBed target and per-domain scores','',f'Latest complete per-domain evaluation: step **{latest["step"]}**. Uses the repository OracleSelectionMethod, which takes the last checkpoint rather than early stopping.','', '| Report | Step | Target test (in) | Target validation (out) |','|---|---:|---:|---:|']
    for label,s in [('DomainBed oracle / latest',score(latest)),('Current source-selected Stage 3',score(source)),('Best target validation, Stage 3',score(best3)),('Best target validation, all stages',score(best))]:
        if s:lines.append(f'| {label} | {s["step"]} | {100*s["target_test_acc"]:.6f}% | {100*s["target_validation_acc"]:.6f}% |')
    lines.extend(['', '| Environment | Domain | in accuracy | out accuracy |','|---|---|---:|---:|'])
    for env,name in enumerate(['location_100','location_38 (target)','location_43','location_46']):
        lines.append(f'| {env} | {name} | {100*latest[f"env{env}_in_acc"]:.6f}% | {100*latest[f"env{env}_out_acc"]:.6f}% |')
    lines.extend(['',result['notes'],'', 'The step-0 maximum, when present, is an initial training checkpoint; it is not a Stage-3 SIRM result.','', 'All complete original DomainBed rows: [DomainBed_results.jsonl](DomainBed_results.jsonl). Source-only refresh records remain in the original training results.jsonl.'])
    (dest/'DomainBed_scores.md').write_text('\n'.join(lines)+'\n');(dest/'DomainBed_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in full))
    return result

def monitoring(root):
    dest=root/'analysis';dest.mkdir(exist_ok=True);logs={a:rows(root/'banks'/a/'diagnostics.jsonl') for a in ['R0','R1']};refs={a:[r for r in rows(root/'banks'/a/'q_refresh.jsonl') if r['stage']=='invariance'] for a in logs}
    history=ratio_records(refs['R1']);write_csv(dest/'gradient_ratio_history.csv',history)
    active=[r['gradient_ratio'] for r in history if r['active'] and r['gradient_ratio'] is not None];all_refresh=[r['gradient_ratio'] for r in history if r['gradient_ratio'] is not None]
    ratio=dict(scope='residual parameters; measurements at active Q refreshes, not all optimizer steps',active_refreshes=describe(active),all_refreshes_with_inactive_zero=describe(all_refresh),unknown_active_refreshes=sum(r['active'] and r['gradient_ratio'] is None for r in history))
    evolution=[]
    for a,rs in refs.items():
        for r in rs:
            q=np.array(r['q']);s,e=support(q)
            evolution.append(dict(arm=a,step=r['step'],selected_cells=int(s.sum()),shared_experts=int((s.sum(0)>=2).sum()),structural_pairs=len(e),eligible_pairs=r['eligible_pair_expert_count'],executed_pairs=r['sirm_pair_count'],q_NULL_mean=float(np.mean(r['q_null'])),router_NULL_mean=r['router_null_mass'],source_accuracy=r['source_validation_accuracy'],Q=r['q'],utility_EMA=r['utility_ema']))
    write_csv(dest/'Q_evolution.csv',evolution)
    paired=[];by0={r['step']:r for r in refs['R0']};first_q=first_u=first_s=None
    for r in refs['R1']:
        if r['step'] not in by0:continue
        b=by0[r['step']];q=np.array(r['q']);q0=np.array(b['q']);s,e=support(q);s0,e0=support(q0)
        dq=float(np.abs(q-q0).max());du=float(np.abs(np.array(r['utility_ema'])-np.array(b['utility_ema'])).max());support_diff=int((s!=s0).sum())
        if dq>0 and first_q is None:first_q=r['step']
        if du>0 and first_u is None:first_u=r['step']
        if support_diff>0 and first_s is None:first_s=r['step']
        paired.append(dict(step=r['step'],Q_max_abs_difference=dq,utility_EMA_max_abs_difference=du,changed_selected_cells=support_diff,support_jaccard=float((s&s0).sum()/(s|s0).sum()) if (s|s0).any() else 1.,pair_jaccard=len(e&e0)/len(e|e0) if e|e0 else 1.,R0_pairs=len(e0),R1_pairs=len(e)))
    write_csv(dest/'Q_feedback_comparison.csv',paired)
    fractions={}
    for a,rs in logs.items():
        stage=[r for r in rs if r['stage']=='invariance'];n=sum(r['sirm_active'] for r in stage)
        fractions[a]=dict(stage3_steps_seen=len(stage),active_steps=n,active_fraction=n/len(stage) if stage else None)
    previous=read(Path('results/sirm_discover_dynamic/terra_L38_seed0/behavior.json'))['active_fraction']
    stats=dict(gradient_ratio=ratio,active_fraction=fractions,previous_confidence_ON_active_fraction=previous,Q_evolution={a:evolution_stats(rs) for a,rs in refs.items()},Q_feedback=dict(first_utility_EMA_difference_step=first_u,first_Q_weight_difference_step=first_q,first_Q_support_difference_step=first_s,paired_refreshes=len(paired)),progress=read(root/'progress.json'))
    base.write(dest/'monitoring.json',stats)
    text=['# Binary-SIRM: five requested results','', 'Live report; all final accuracy and decomposition values remain pending until both runs finish. Training code/configuration are untouched. Gradient ratio is measured on residual parameters at active Q refreshes. Statistics below are sampled refresh statistics, not all-step extrema or averages. Inactive refreshes have ratio zero; missing active measurements are never imputed as zero.','', '## 1. Accuracy','', 'Pending source-selected full-target evaluation.' if not (root/'behavior.json').exists() else json.dumps(read(root/'behavior.json')['target_accuracy']), '', '## 2. Gradient ratio history','', '| Sample set | n | Median | Mean | p90 | Max |','|---|---:|---:|---:|---:|---:|']
    for label,k in [('Active refreshes','active_refreshes'),('All refreshes; inactive=0','all_refreshes_with_inactive_zero')]:
        d=ratio[k];text.append('| '+label+' | '+str(d['n'])+' | '+' | '.join(f'{d[z]:.7g}' if d[z] is not None else 'pending' for z in ['median','mean','p90','max'])+' |')
    text.extend(['',f'Active refresh ratio >1 fraction: {ratio["active_refreshes"].get("fraction_over_1")}; >10 fraction: {ratio["active_refreshes"].get("fraction_over_10")}. [Full sampled history](gradient_ratio_history.csv). A large ratio with lower accuracy flags excessive regularization; it is not by itself evidence against binary selection or IRMv1.', '', '## 3. SIRM active fraction','',json.dumps(fractions,indent=2), '', f'Previous confidence-weighted ON: {100*previous:.4f}% of all Stage-3 optimizer steps. OFF executes no penalty even when support is eligible. Compare Q eligibility separately from execution.', '', '## 4. Q evolution and feedback','',json.dumps(stats['Q_feedback'],indent=2), '', 'Full matrices/EMA/support/NULL: [Q_evolution.csv](Q_evolution.csv). Paired assignment divergence: [Q_feedback_comparison.csv](Q_feedback_comparison.csv). Empty-vs-empty Jaccard is defined as 1. Global backbone/head remain frozen; feedback passes through residual/router composition and utility EMA.', '', '## 5. Residual decomposition','', 'Pending both source-selected checkpoints. Will report pooled residual cosine, residual norms, correction/global RMS, router NULL, Q NULL and source-out G under common masks for both arms. Low cosine alone does not establish useful factors; high NULL or small composed corrections alone does not establish zero residuals.'])
    (dest/'summary.md').write_text('\n'.join(text)+'\n')
    scores=domainbed_scores(root)
    if scores:
        oracle=scores['oracle'];note=f'DomainBed target monitoring at step **{scores["oracle_step"]}**: OracleSelectionMethod test **{100*oracle["test_acc"]:.6f}%**, validation **{100*oracle["val_acc"]:.6f}%**. This repo uses the last evaluated checkpoint for oracle; final full-target source-selected accuracy is a separate result. [Per-domain scores and checkpoint history](DomainBed_scores.md).\n\n'
        saved=(dest/'summary.md').read_text();(dest/'summary.md').write_text(saved.replace('## 1. Accuracy\n\n','## 1. Accuracy\n\n'+note,1))
    if (root/'R0_reuse_plan.json').exists():
        provenance=read(root/'R0_reuse_plan.json')
        archive=provenance.get('archived_partial_bank','banks/R0_domainbed_partial/')
        note=f'R0 is the completed historical OFF control, reused after exact audit through step {provenance["stopped_duplicate_last_step"]}; only R1 trains fresh. Historical source-selected checkpoints and all tensor contents are preserved. The original stopped duplicate is archived at {archive}. Final results are pending completion of R1.\n\n'
        saved=(dest/'summary.md').read_text()
        (dest/'summary.md').write_text(saved.replace('\n\n','\n\n'+note,1))
    return stats

@torch.no_grad()
def pooled_cosine(model,data,c):
    gram=None;count=0
    for subset in data.values():
        for x,y in DataLoader(subset,batch_size=c['eval_batch_size'],num_workers=2):
            f=model(x.cuda());d=f['delta'].double().permute(1,0,2).reshape(model.num_experts,-1);part=(d@d.T).cpu();gram=part if gram is None else gram+part;count+=len(y)
    diag=gram.diag();cos=gram/(diag[:,None]*diag[None,:]).sqrt().clamp_min(1e-16);off=cos.numpy()[np.triu_indices(model.num_experts,1)]
    return dict(samples=count,matrix=cos.tolist(),pair_min=float(off.min()),pair_mean=float(off.mean()),pair_max=float(off.max()),scope='class-centered residual output cosine pooled over samples/classes')

def final(root,data_dir,stats):
    dest=root/'analysis';original=read(root/'behavior.json');split=read(root/'source_split_manifest.json');diags={};summary_rows=[]
    for a in ['R0','R1']:
        bank=root/'banks'/a;sha=base.digest(bank/'source_best.pkl');model,c,payload=restore(bank);adapter=Adapter('TerraIncognita');adapter.classes=split['class_names']
        source=adapter.source_data(data_dir,split,c,'out');target={1:ImageFolder(adapter.folder(data_dir,1),base.ViewTransform(c,'original',False,1,'balanced_views'))}
        ds=evaluate(model,source,c,True);dt=read(root/'target_metrics.json')[a]
        cs=pooled_cosine(model,source,c);ct=pooled_cosine(model,target,c)
        diags[a]=dict(selected_step=payload['selected_step'],source=ds,target=dt,source_cosine=cs,target_cosine=ct,checkpoint_sha256=sha)
        base.write(dest/(a+'_decomposition.json'),diags[a]);assert base.digest(bank/'source_best.pkl')==sha;del model;torch.cuda.empty_cache()
        for name,d,cos in [('source-out',ds,cs),('target',dt,ct)]:summary_rows.append(dict(arm=a,split=name,selected_step=payload['selected_step'],accuracy=d['source_mean'],global_accuracy=d['global_mean'],residual_cosine_mean=cos['pair_mean'],residual_cosine_min=cos['pair_min'],residual_cosine_max=cos['pair_max'],correction_global_RMS=d['correction_global_norm_ratio_rms'],router_NULL=d['router_mass'][-1],Q_NULL_mean=float(np.mean(d['q_null'])),residual_norm_mean=float(np.mean(d['residual_norm_per_expert'])),G_selected=d['G_selected']))
    write_csv(dest/'decomposition.csv',summary_rows)
    fixed_mask=np.array(diags['R1']['source']['q'])>0;exposed=np.zeros_like(fixed_mask)
    for r in rows(root/'banks/R1/diagnostics.jsonl'):
        for key in r['G']:
            d,m=map(int,key.split(','));exposed[d,m]=True
    mask_stats={}
    for name,mask in [('R1_selected_Q',fixed_mask),('R1_union_of_executed_cells',exposed)]:
        mask_stats[name]=dict(mask=mask.tolist(),G={a:float(np.array(diags[a]['source']['G_matrix'])[mask].mean()) if mask.any() else None for a in diags})
    base.write(dest/'common_mask_G.json',mask_stats)
    acc=original['target_accuracy'];rat=stats['gradient_ratio']['active_refreshes'];above=rat.get('fraction_over_1');warning=bool(above and above>0)
    outcome=dict(accuracy=acc,delta_R1_minus_R0_percentage_points=100*(acc['R1']-acc['R0']),gradient_ratio=stats['gradient_ratio'],active_fraction=stats['active_fraction'],Q_evolution=stats['Q_evolution'],Q_feedback=stats['Q_feedback'],decomposition=summary_rows,common_mask_G=mask_stats,ratio_over_1_observed_at_sampled_refresh=warning,interpretation='A ratio above 1 flags possible excessive strength. Do not treat a lower accuracy under dominant gradients as evidence that binary selection is intrinsically bad. No lambda/ramp changes or target tuning are applied.',limits='One seed; gradient statistics are refresh samples; source-out selected checkpoints; selected times can differ; cosine is output similarity, not proof of causal factor recovery.')
    base.write(dest/'five_results.json',outcome)
    text=['# Binary-SIRM: five requested results','', 'Completed post-hoc report. No training/configuration changes. All target accuracies use locked source-selected checkpoints.','', '## 1. Accuracy','', '| R0 OFF | R1 binary ON | R1−R0 |','|---:|---:|---:|',f'| {100*acc["R0"]:.6f}% | {100*acc["R1"]:.6f}% | {outcome["delta_R1_minus_R0_percentage_points"]:+.6f} pp |','', '## 2. Gradient ratio history','', 'Norm scope: residual parameters, matching the previous diagnostic. Measurements at active Q refreshes (every 50 steps); these are not all-step extrema/averages. Inactive refreshes are zero. No missing active measurement is silently replaced with zero.','', '| Sample set | n | Median | Mean | p90 | Max |','|---|---:|---:|---:|---:|---:|']
    for label,k in [('Active refreshes','active_refreshes'),('All refreshes; inactive=0','all_refreshes_with_inactive_zero')]:
        d=stats['gradient_ratio'][k];text.append('| '+label+' | '+str(d['n'])+' | '+' | '.join(f'{d[z]:.7g}' if d[z] is not None else 'N/A' for z in ['median','mean','p90','max'])+' |')
    text.extend(['',f'Sampled active refreshes with r>1: {rat.get("fraction_over_1")}; r>10: {rat.get("fraction_over_10")}. [History](gradient_ratio_history.csv).', '', '## 3. SIRM active fraction',''])
    for a,s in stats['active_fraction'].items():text.append(f'- {a}: {s["active_steps"]}/{s["stage3_steps_seen"]} = {100*s["active_fraction"]:.4f}%.')
    text.extend(['',f'Previous confidence ON: {100*stats["previous_confidence_ON_active_fraction"]:.4f}%. OFF fraction measures execution, not eligibility.', '', '## 4. Q evolution','',json.dumps(stats['Q_feedback'],indent=2),'', '[Matrices/support/EMA](Q_evolution.csv); [R0/R1 Q-feedback differences](Q_feedback_comparison.csv). Global backbone/head are frozen; the feedback acts through residuals/router and their predictive utilities.', '', '## 5. Residual decomposition','', '| Arm / split | Cosine mean | Correction/global RMS | Router NULL | Q NULL mean | Mean ‖δ‖ |','|---|---:|---:|---:|---:|---:|'])
    index=text.index('## 5. Residual decomposition')
    text[index:index]=['Temporal Q statistics: '+json.dumps(stats['Q_evolution']), '',f'Learned parameter divergence: {original["parameter_divergence"]}; first baseline loss-trajectory difference: {original["first_baseline_trajectory_difference"]}; target logit max difference: {original["target_max_abs_logit_difference"]}; changed target predictions: {original["different_target_predictions"]}.', '']
    for r in summary_rows:text.append(f'| {r["arm"]} / {r["split"]} | {r["residual_cosine_mean"]:.6f} | {r["correction_global_RMS"]:.6f} | {r["router_NULL"]:.6f} | {r["Q_NULL_mean"]:.6f} | {r["residual_norm_mean"]:.6f} |')
    text.extend(['', 'Output cosine pools class-centered residual logits over images/classes. Full pairwise matrices and expert norms are in R0/R1_decomposition.json. Q NULL is a source assignment statistic, including in target rows; router NULL is measured on that split. A lower output cosine alone is not proof of unique useful factors, and contraction alone is not proof of collapse.', '', '### Source-out G under common masks',''])
    for name,d in mask_stats.items():text.append(f'- {name}: R0={d["G"]["R0"]}, R1={d["G"]["R1"]}.')
    text.extend(['', 'Masks are fixed from R1 for both models; the executed-cell union tracks all cells actually exposed to IRM. These are domain-mean scale derivatives at independently source-selected checkpoints, which may be at different steps; source-out was used for selection.','', outcome['interpretation'],'', outcome['limits']])
    (dest/'summary.md').write_text('\n'.join(text)+'\n')
    scores=domainbed_scores(root)
    if scores:
        oracle=scores['oracle'];note=f'DomainBed oracle at final complete evaluation step **{scores["oracle_step"]}**: target test **{100*oracle["test_acc"]:.6f}%**, target validation **{100*oracle["val_acc"]:.6f}%**. Target in/out split scores differ from the full-target accuracies above. [Per-domain scores/history](DomainBed_scores.md).\n\n'
        saved=(dest/'summary.md').read_text();(dest/'summary.md').write_text(saved.replace('## 2. Gradient ratio history',''+note+'## 2. Gradient ratio history',1))
    if (root/'R0_reuse_plan.json').exists():
        reuse=read(root/'R0_reuse_plan.json')
        saved=(dest/'summary.md').read_text()
        (dest/'summary.md').write_text(saved.replace('\n\n',f'\n\nR0 reuses the completed historical source-selected OFF control after exact reproduction through step {reuse["stopped_duplicate_last_step"]}. Only R1 trains fresh; provenance is in ../R0_reuse_plan.json.\n\n',1))
    main=root/'summary.md';s=main.read_text();plan=read(root/'plan.json')
    if 'requested_lambda' in plan:
        s=s.replace('Only active-term weighting/normalization changed: each eligible pair weight=1 and loss is its mean.',f'Relative to the previous binary run, only lambda changes to {plan["requested_lambda"]}; each eligible pair still has weight=1 and the loss remains its mean.')
    marker='\n## Five-part analysis\n';main.write_text(s.split(marker)[0]+marker+'\n[Accuracy, gradient-ratio distribution/history, active fraction, Q feedback, residual decomposition and common-mask G](analysis/summary.md).\n')
    base.write(dest/'progress.json',dict(stage='completed'));print(json.dumps({k:outcome[k] for k in ['accuracy','delta_R1_minus_R0_percentage_points','ratio_over_1_observed_at_sampled_refresh']}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='results/sirm_discover_binary/terra_L38_seed0');p.add_argument('--data-dir',default='domainbed/data');p.add_argument('--watch',action='store_true');a=p.parse_args();torch.set_num_threads(4);root=Path(a.root)
    while True:
        stats=monitoring(root);progress=stats['progress'];base.write(root/'analysis/progress.json',dict(stage='waiting_for_training',pid=os.getpid(),training_stage=progress['stage']))
        if progress['stage']=='completed':final(root,a.data_dir,stats);break
        if not a.watch:break
        if progress['stage']=='failed':raise RuntimeError('training failed; report remains pending')
        pid=progress.get('pid')
        if pid:
            try:os.kill(pid,0)
            except ProcessLookupError:raise RuntimeError('training process exited before completion')
        time.sleep(30)

if __name__=='__main__':main()

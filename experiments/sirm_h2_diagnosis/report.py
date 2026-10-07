"""Package measured diagnostics and their interpretation limits."""
import csv,json,statistics,itertools,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.sirm_representation_h2.summarize import write_csv
from experiments.sirm_h1_h6 import run as base
from experiments.sirm_representation_h2.diagnostics import json_write

ROOT=Path('results/sirm_representation_h2/diagnosis')
def read(name):return list(csv.DictReader((ROOT/name).open()))
def num(x):return float(x) if x not in ['',None] else None
def avg(rows,key):
    values=[num(r[key]) for r in rows if num(r[key]) is not None]
    return statistics.mean(values) if values else None

def main():
    margin=read('competence_margins.csv');curves=read('training_curves.csv');grad=read('gradient_interaction.csv');direction=read('classifier_directions.csv')
    assert len(grad)==48*3*3*5 and len(direction)==48*6*3
    batch_audit=[];signatures={}
    for item in json.loads((ROOT/'inventory.json').read_text()):
        bank=Path(item['bank']);assert base.digest(bank/'checkpoints/best.pkl')==item['checkpoint_sha256']
        split=json.loads((bank/'split_manifest.json').read_text());ids=[]
        assert str(item['target_env']) not in split['sources']
        for env,entry in sorted(split['sources'].items()):
            groups={c:[i for i in entry['out'] if entry['images'][i]['class']==c] for c in range(7)}
            selected=[i for pos in range(max(map(len,groups.values()))) for c in range(7) for i in groups[c][pos:pos+1]][:48]
            ids.append([entry['images'][i]['id'] for i in selected])
        signature=base.canonical_hash(ids);key=(item['target_env'],item['seed'])
        if key in signatures:assert signatures[key]==signature
        signatures[key]=signature
        batch_audit.append(dict(condition=item['condition'],target_env=item['target_env'],seed=item['seed'],
                               source_sample_sha256=signature,source_ids=ids,checkpoint_unchanged=True))
    json_write(ROOT/'paired_gradient_batches_audit.json',batch_audit)
    for r in grad:
        for key in ['norm_cls','norm_sirm_raw']:assert float(r[key])>=0
        if r['cosine']!='':assert -1.00001<=float(r['cosine'])<=1.00001
        if r['condition'] in 'AB' and r['weighted_ratio_actual']!='':assert float(r['weighted_ratio_actual'])==0
        if r['group']=='router':assert float(r['norm_sirm_raw'])==0
    history=[]
    for r in curves:
        for group in ['backbone','router','experts','expert_heads']:
            cls=num(r.get(f'grad_raw_cls_{group}'));sirm=num(r.get(f'grad_raw_sirm_{group}'))
            if cls is not None and cls>0 and sirm is not None:
                history.append(dict(condition=r['condition'],target_env=r['target_env'],seed=r['seed'],step=r['step'],group=group,
                    cls_norm=cls,sirm_raw_norm=sirm,raw_ratio=sirm/cls,weighted_ratio=num(r['effective_lambda_sirm'])*sirm/cls,
                    actual_loss_active=r['sirm_active'],measurement='logged training minibatch; cosine was not stored; OFF raw SIRM skipped'))
    write_csv(ROOT/'gradient_history_norms.csv',history)
    viewstability=[];qstability=[];temporal=[]
    snapshots=json.loads((ROOT/'q_selected_last.json').read_text())
    for condition,target,seed in itertools.product('ABCD',range(4),range(3)):
        rows=[r for r in margin if r['condition']==condition and int(r['target_env'])==target and int(r['seed'])==seed]
        for domain in {r['source_domain'] for r in rows}:
            entry={r['view']:r for r in rows if r['source_domain']==domain};original=set(np.argsort(json.loads(entry['original']['risk_values']))[:2])
            for view in ['grayscale','blur']:
                other=set(np.argsort(json.loads(entry[view]['risk_values']))[:2])
                viewstability.append(dict(condition=condition,target_env=target,seed=seed,source_domain=domain,view=view,
                    risk_top2_jaccard=len(original&other)/len(original|other),measurement='source-val risk reranking diagnostic; saved Q is unchanged'))
        snaps={r['snapshot']:r for r in snapshots if r['condition']==condition and r['target_env']==target and r['seed']==seed}
        for d,domain in enumerate(snaps['best']['source_domains']):
            a=set(np.flatnonzero(np.array(snaps['best']['q'][d])>0));b=set(np.flatnonzero(np.array(snaps['last']['q'][d])>0))
            temporal.append(dict(condition=condition,target_env=target,seed=seed,source_domain=domain,selected_step=snaps['best']['step'],last_step=snaps['last']['step'],
                                 jaccard=len(a&b)/len(a|b)))
    for on,off in [('A','B'),('C','D')]:
        for target,seed in itertools.product(range(4),range(3)):
            a=json.loads((ROOT/'q_matrices'/f'{on}_env{target}_seed{seed}.json').read_text());b=json.loads((ROOT/'q_matrices'/f'{off}_env{target}_seed{seed}.json').read_text())
            for d,domain in enumerate(a['source_domains']):
                sa=set(np.flatnonzero(np.array(a['q'][d])>0));sb=set(np.flatnonzero(np.array(b['q'][d])>0))
                qstability.append(dict(comparison=f'{off}-{on}',target_env=target,seed=seed,source_domain=domain,jaccard=len(sa&sb)/len(sa|sb)))
    write_csv(ROOT/'risk_top2_view_stability.csv',viewstability);write_csv(ROOT/'q_augmentation_support.csv',qstability);write_csv(ROOT/'q_selected_last_stability.csv',temporal)
    gaps=[]
    for r in read('runs.csv'):
        for view in ['original','gray','blur','gray_blur']:
            gaps.append(dict(condition=r['condition'],target_env=r['target_env'],seed=r['seed'],view=view,
                native_accuracy=r[f'acc_{view}_native'],global_accuracy=r[f'acc_{view}_global'],native_minus_global=float(r[f'acc_{view}_native'])-float(r[f'acc_{view}_global'])))
    write_csv(ROOT/'native_global_gap.csv',gaps)
    summary=[]
    for c in 'ABCD':
        m=[r for r in margin if r['condition']==c and r['view']=='original'];g=[r for r in grad if r['condition']==c and r['view']=='original' and r['group']=='backbone']
        h=[r for r in history if r['condition']==c and r['group']=='backbone' and int(r['step'])>=500];d=[r for r in direction if r['condition']==c and r['q_shared']=='True']
        summary.append(dict(condition=c,best_second_margin=avg(m,'best_second_margin'),top2_boundary_margin=avg(m,'top2_boundary_margin'),
            q_selected_margin=avg(m,'q_selected_vs_other_margin'),risk_sd=avg(m,'expert_risk_sd'),
            backbone_posthoc_cosine=avg(g,'cosine'),backbone_posthoc_ratio_at_point1=avg(g,'weighted_ratio_at_lambda_point1'),
            backbone_training_weighted_ratio=avg(h,'weighted_ratio'),q_shared_direction_cosine=avg(d,'coefficient_cosine'),
            q_shared_cross_domain_val_acc=avg(d,'cross_domain_val_acc'),q_shared_pairs=len(d)))
    write_csv(ROOT/'diagnosis_overall.csv',summary)
    lines=['# H2 / SIRM causal diagnosis evidence','', 'All 48 factorial checkpoints; no retraining, model changes, or target-selected hyperparameters. Expert numbers are zero-based.',
        '', '## Important correction','', 'The active SIRM loss is already Q-gated IRMv1 on predictive risk: sum of Q-weighted squared derivatives of responsibility-weighted CE with respect to a scalar logit scale at scale=1, restricted to experts with at least two Q-assigned domains and adequate effective samples. It is not feature-distribution alignment. SSI alignment is disabled. This loss is not full classifier-gradient equality.',
        '', '## Original-view source competence','', '| Condition | Best/second gap | Top2 boundary gap | Q-selected/nonselected gap | Expert risk SD |', '|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["condition"]} | {r["best_second_margin"]:.5f} | {r["top2_boundary_margin"]:.5f} | {r["q_selected_margin"]:.5f} | {r["risk_sd"]:.5f} |')
    lines+=['', 'Gaps use CE loss, not percentage points. Best/second=R(second)-R(best); top2 boundary=R(third)-R(second); Q gap=mean(nonselected)-mean(selected). Every expert sees identical source-validation examples. Means give equal weight to source domains and target×seed cells.',
        '', '## Gradient interaction','', '| Condition | Posthoc backbone cosine | Posthoc 0.1 × norm ratio | Historical weighted norm ratio |', '|---|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["condition"]} | {r["backbone_posthoc_cosine"]:.4f} | {r["backbone_posthoc_ratio_at_point1"]:.4f} | {r["backbone_training_weighted_ratio"]:.4f} |')
    lines+=['', 'Posthoc measurements use eval mode at selected checkpoints, three identical class-interleaved source-out batches of16 examples/domain per view, original/gray/blur. Raw SIRM is explicitly recomputed even for OFF; its actual weighted gradient remains zero. Zero-norm cosine is undefined and exported blank. Historical norm measurements are training-minibatch snapshots every500 steps, not epoch averages; historical cosine was never recorded. Posthoc gradients cannot reconstruct earlier feedback loops or prove mediation.',
        '', '## Per-domain classifier directions','', '| Condition | Q-shared coefficient cosine | Cross-domain validation accuracy | Pair count |', '|---|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["condition"]} | {r["q_shared_direction_cosine"]:.4f} | {100*r["q_shared_cross_domain_val_acc"]:.2f}% | {r["q_shared_pairs"]} |')
    lines+=['', 'Each expert/domain classifier fits all source-in features with shared source-in standardization and fixed L2=.01, frozen features, LBFGS checked for convergence. Coefficients are centered across classes to remove common-logit-shift ambiguity. Each source domain is evaluated with its own and another domain classifier on source-out. Full pair-level rows include Q-shared and non-shared pairs; aggregates above involve changing subsets and are descriptive, not controlled causal contrasts. Similar direction alone does not establish transferability; cross-domain prediction accuracy is supplied.',
        '', '## Missing evidence and limits','', 'Only selected/best and last Q states were saved; no epoch-by-epoch Q trajectory exists. Best-to-last differences do not identify switching frequency. Expert indices cannot be compared across random seeds as semantic identities without permutation alignment. Risk-top2 reranking across views is a diagnostic, not a new trained Q. Historical gradients cannot be recreated from scalar logs; selected-checkpoint gradients are a separate measurement. No source-only lambda sweep or intervention freezing Q has been run.',
        '', 'The earlier claim that H2 has proven representation quality was too strong: probe/cosine results support improvement in this protocol; they do not prove causal semantic factors or identify the mediator of target gains.',
        '', 'Commands: `python -m experiments.sirm_h2_diagnosis.run --mode extract`, `--mode gradients`, `--mode directions`, then `python -m experiments.sirm_h2_diagnosis.report`. Environment: existing dg; OMP/MKL threads4. Failures, if any, are retained as failure JSON rather than hidden.']
    lines+=['','## Diagnosis against the five hypotheses','',
        '1. Partial support for an ambiguous top2 boundary: C→D boundary gap drops .02835→.00977, but Q-selected/nonselected mean gap remains .26097→.24793 and expert risk SD does not shrink. This does not support a globally equal-competence bank. Risk-top2 paired-view Jaccard improves C→D .7917→.8333, contrary to the proposed increase in view instability.',
        '', '2. No pervasive negative gradient conflict is observed at selected checkpoints: backbone cosine over three views averages C=.6599 and D=.6774; negative in11/108 vs12/108 batch×view measurements. Historical weighted norm ratio medians are .00376 and .00456. D has one important onset spike: target Sketch, seed0, step500, weighted SIRM/backbone-classification norm ratio1.7163 (raw ratio17.1632); this makes activation strength/transient optimization a plausible follow-up, not an established cause. Historical cosine at that spike is missing.',
        '', '3. Q-selected pairs have high shared-classifier direction similarity, but this measures source structure, not transferable causal mechanisms. Q-shared coefficients C=.9592 vs D=.9611; cross-domain source-val accuracy stays near97.9%. Non-Q-shared pairs also have near97.8% cross-domain source accuracy after refitting heads. Predictive structure may be widespread; this is not evidence that selected experts alone carry transferable factors.',
        '', '4. No direct evidence that SIRM suppresses specialization is provided by these measurements. Active SIRM is predictive IRMv1, so the proposed feature-alignment-to-IRM replacement does not describe the current code. Scalar-scale IRMv1 can still be an inadequate proxy for shared prediction, but that requires an intervention or stronger diagnostic.',
        '', '5. Best/last Q support is stable, especially D (mean Jaccard1.0;11/12 runs have distinct selected and last steps). This does not support persistent late switching, but does not rule out early feedback because intermediate Q trajectories are absent.',
        '', 'Suggested next diagnostic: source-only replay around step500 to log raw/weighted gradient cosine, Q margins, effective sample counts, and Q switching every update; then a source-selected smaller-lambda or gradual-activation control. A residual-competence Q redesign is not yet justified solely by these observations. No new augmentation or contrastive-loss experiment was launched.']
    (ROOT/'summary.md').write_text('\n'.join(lines)+'\n')
    for target in [0,3]:
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for c in 'ABCD':
            rows=sorted([r for r in curves if r['condition']==c and int(r['target_env'])==target and int(r['seed'])==0],key=lambda r:int(r['step']));steps=[int(r['step']) for r in rows]
            for ax,metric in zip(axes.flat,['loss_cls','loss_route','loss_sirm','source_mean','grad_raw_cls_backbone','grad_raw_sirm_backbone']):
                ax.plot(steps,[float(r[metric]) for r in rows],label=c);ax.set_title(metric);ax.set_xlabel('Step');ax.grid(alpha=.2)
        axes.flat[0].legend();fig.suptitle(f'Target {target}, seed0; logged minibatch losses/norms, source-val selection');fig.tight_layout();fig.savefig(ROOT/f'training_env{target}_seed0.png',dpi=160);plt.close(fig)
    with zipfile.ZipFile(ROOT.parent/'diagnosis_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in ROOT.rglob('*'):
            if p.is_file():archive.write(p,p.relative_to(ROOT))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

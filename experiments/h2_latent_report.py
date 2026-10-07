"""Final interpretation and paired plots for the locked latent analysis."""
import argparse
import csv
import json
from pathlib import Path


def finish(root):
    effects=list(csv.DictReader((root/'paired_effects.csv').open()))
    aggregates=list(csv.DictReader((root/'aggregate_metrics.csv').open()))
    progress=json.loads((root/'progress.json').read_text())
    assert progress['completed']==48 and progress['status']=='completed'
    def matches(rows, **where):
        return [r for r in rows if all(r[k]==str(v) for k,v in where.items())]
    def delta(file,metric,split,view,effect):
        found=matches(effects,file=file,metric=metric,data_split=split,view=view,effect=effect,scope='overall_paired_mean')
        assert len(found)==1
        return found[0]
    def absolute(file,metric,split,view,condition):
        found=matches(aggregates,file=file,metric=metric,data_split=split,view=view,condition=condition,target_env='all',scope='cell_mean')
        assert len(found)==1
        return float(found[0]['mean'])
    def show(file,metric,split='source_out',view='original',scale=1,unit=''):
        out=[]
        for effect,first,second in [('OFF','A','B'),('ON','C','D')]:
            d=delta(file,metric,split,view,effect)
            a=absolute(file,metric,split,view,first)*scale
            b=absolute(file,metric,split,view,second)*scale
            change=float(d['mean'])*scale
            trend='decreases' if change<0 else 'increases' if change>0 else 'is unchanged'
            difference_unit=' pp' if unit=='%' else unit
            out.append(f"{effect}: {a:.5f}→{b:.5f}{unit}, Δ={change:+.5f}{difference_unit}; mean {trend}, {d['decreased_cells']}/12 cells decrease")
        return '; '.join(out)+'.'
    findings=['## Final interpretation (all 48 checkpoints)', '',
        '**1. Class-conditional domain separation: no consistent source-domain reduction.** '+show('class_cond_domain_distance','euclidean_distance'),
        'Scale-normalized source-out distance: '+show('class_cond_domain_distance','scale_normalized_distance'),
        'Raw-feature distances must be interpreted with feature scale and cosine distances. Centroid convergence is not proof that domain information has been removed.', '',
        '**2. Class-conditioned domain predictability: little mean change; domain information remains highly linearly accessible.** '+show('domain_probe','domain_accuracy',scale=100,unit='%'),
        'Chance is 33.33%. These are class/domain-balanced accuracies from source-out, also used for regularization selection, not an independent probe test set. A lower score measures reduced accessibility to this linear probe, not conditional mutual information.', '',
        '**3. Nuisance displacement: observed mean reductions on both source-out and target.**',
        'Source gray: '+show('nuisance_displacement','euclidean_displacement',view='grayscale'),
        'Source blur: '+show('nuisance_displacement','euclidean_displacement',view='blur'),
        'Target gray: '+show('nuisance_displacement','euclidean_displacement',split='target_all',view='grayscale'),
        'Target blur: '+show('nuisance_displacement','euclidean_displacement',split='target_all',view='blur'),
        'Shorter displacements support reduced variation under the measured transforms; they do not establish elimination of domain/style information.', '',
        '**4. Alignment of domain shift with nuisance directions: mean fractions decrease, with cell heterogeneity.**',
        'Source gray fraction: '+show('nuisance_projection','projection_fraction',view='grayscale'),
        'Source blur fraction: '+show('nuisance_projection','projection_fraction',view='blur'),
        'Target-source gray fraction: '+show('nuisance_projection','projection_fraction',split='target_vs_source_out',view='grayscale'),
        'Target-source blur fraction: '+show('nuisance_projection','projection_fraction',split='target_vs_source_out',view='blur'),
        'These are absolute cosine-like projections onto separately estimated source-out axes; they are not explained variance or a complete style subspace.', '',
        '**5. Classifier reliance: finite sensitivity decreases on average, including unit-axis controls.**',
        'Source full gray-axis shift: '+show('classifier_nuisance_reliance','logit_L2_change',view='grayscale'),
        'Source full blur-axis shift: '+show('classifier_nuisance_reliance','logit_L2_change',view='blur'),
        'Target full gray-axis shift: '+show('classifier_nuisance_reliance','logit_L2_change',split='target_all',view='grayscale'),
        'Target full blur-axis shift: '+show('classifier_nuisance_reliance','logit_L2_change',split='target_all',view='blur'),
        'Source unit gray axis, class-centered logits: '+show('classifier_nuisance_reliance','unit_axis_centered_logit_L2_change',view='grayscale'),
        'Source unit blur axis, class-centered logits: '+show('classifier_nuisance_reliance','unit_axis_centered_logit_L2_change',view='blur'),
        'Target unit gray axis, class-centered logits: '+show('classifier_nuisance_reliance','unit_axis_centered_logit_L2_change',split='target_all',view='grayscale'),
        'Target unit blur axis, class-centered logits: '+show('classifier_nuisance_reliance','unit_axis_centered_logit_L2_change',split='target_all',view='blur'),
        'Full-axis sensitivity combines nuisance-axis magnitude and predictor response. Unit-axis sensitivity controls displacement length, while class-centering excludes common-logit offsets. Neither measures a causal shortcut effect or a globally invariant Jacobian; each finite intervention recomputes native routing.', '',
        '**6. Label information: existing target label probes improve on average.** '+show('domain_probe','label_probe_accuracy',split='target_all',scale=100,unit='%'),
        'Existing frozen probes improve on average; this demonstrates preserved/improved linearly accessible predictive performance in this protocol, not equality of all label information in Z.', '',
        '**Strongest supported claim:** in these selected PACS representations, H2 reduces mean latent variation induced by grayscale/blur on source-out and target, with decreases in all source-out paired cells, and lowers the predictor’s mean finite sensitivity along the measured nuisance directions, while frozen label-probe target accuracy improves on average. The normalized direction controls also decrease, so shorter displacement alone does not explain the measured sensitivity reduction. Conditional domain identity nevertheless remains highly linearly recoverable (~97.2% balanced accuracy versus 33.3% chance), and source-domain centroid separation does not decrease consistently. The evidence therefore supports nuisance variation being reduced and these particular directions being used less sensitively; it does not support general removal of domain information from Z. Each model uses its own class-conditioned source-out axes; this is not a comparison of responses to one shared cross-model coordinate vector. These are exploratory behavior proxies, not proof of a causal OOD mechanism. No claim of spurious-feature disappearance, shape learning, or direct measurement of I(Z;D|Y) is warranted.', '',
        'The baseline/H2 intervention still confounds increased grayscale frequency with added blur. Three seeds, repeated overlapping source domains and previously inspected targets make this exploratory; cell SD is not an independence-adjusted confidence interval.', '',
        '[Paired source-out diagnostic plot](paired_source_metrics.png). Each dot is one target×seed pair; large bars mark means, not confidence intervals.']
    summary=(root/'summary.md').read_text()
    summary=summary.replace('Interpretation is descriptive until all 48 cells complete.', 'All 48 cells are complete; the interpretation remains descriptive.')
    summary=summary.split('## Final interpretation (all 48 checkpoints)')[0].rstrip()
    (root/'summary.md').write_text(summary+'\n\n'+'\n\n'.join(x for x in findings if x)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=[('class_cond_domain_distance','euclidean_distance','original','Domain centroid distance','feature units'),
        ('domain_probe','domain_accuracy','original','Conditional domain probe','percentage points'),
        ('nuisance_displacement','euclidean_displacement','grayscale','Gray displacement','feature units'),
        ('nuisance_displacement','euclidean_displacement','blur','Blur displacement','feature units'),
        ('nuisance_projection','projection_fraction','grayscale','Gray projection fraction','fraction'),
        ('nuisance_projection','projection_fraction','blur','Blur projection fraction','fraction'),
        ('classifier_nuisance_reliance','logit_L2_change','grayscale','Full gray-axis logit sensitivity','logit units'),
        ('classifier_nuisance_reliance','logit_L2_change','blur','Full blur-axis logit sensitivity','logit units')]
    fig,axes=plt.subplots(4,2,figsize=(11,12),constrained_layout=True)
    for ax,(file,metric,view,title,unit) in zip(axes.flat,metrics):
        scale=100 if metric=='domain_accuracy' else 1
        for xpos,effect,color in [(0,'OFF','#4477AA'),(1,'ON','#EE7733')]:
            part=matches(effects,file=file,metric=metric,data_split='source_out',view=view,effect=effect,scope='paired_cell')
            assert len(part)==12
            for r in part:
                jitter=(int(r['target_env'])-1.5)*.045+(int(r['seed'])-1)*.012
                ax.scatter(xpos+jitter,float(r['delta'])*scale,c=color,marker=['o','s','^','D'][int(r['target_env'])],s=28,alpha=.7)
            d=delta(file,metric,'source_out',view,effect)
            ax.plot([xpos-.16,xpos+.16],[float(d['mean'])*scale]*2,color='black',linewidth=2)
        ax.axhline(0,color='gray',linestyle='--',linewidth=.8)
        ax.set_xticks([0,1],['SIRM OFF','SIRM ON'])
        ax.set_xlim(-.35,1.35)
        ax.set_ylabel('H2 − baseline ('+unit+')')
        ax.set_title(title)
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('PACS source-out paired differences: 4 targets × 3 seeds\nDots are paired cells; black lines are means; no independence-based intervals')
    fig.savefig(root/'paired_source_metrics.png',dpi=170)
    fig.savefig(root/'paired_source_metrics.svg')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('results/h2_latent_diagnosis'))
    args=parser.parse_args()
    finish(args.root)

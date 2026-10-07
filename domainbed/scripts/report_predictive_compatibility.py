"""Render saved audit evidence without refitting or changing any selection."""
import argparse
import csv
import json
from pathlib import Path


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = json.loads((root/'results.json').read_text())
    config = json.loads((root/'config.json').read_text())
    rows = []
    for path in sorted(root.glob('*/compatibility.json')):
        for expert in json.loads(path.read_text()):
            for domain, audit in zip(expert['support'], expert['audit']['domains']):
                rows.append(dict(fold=path.parent.name, expert=expert['expert'], domain=domain,
                                 support=','.join(map(str, expert['support'])),
                                 accepted=expert['audit']['accepted'],
                                 calibration_J_lower=expert['calibration']['lower'],
                                 calibration_J_upper=expert['calibration']['upper'],
                                 calibration_solve_gap=expert['calibration']['gap'],
                                 calibration_converged=expert['calibration']['converged'],
                                 n=audit['n'], audit_ce=audit['risk'],
                                 audit_optimum_lower=audit['optimum']['lower'],
                                 audit_optimum_upper=audit['optimum']['upper'],
                                 audit_solve_gap=audit['optimum']['gap'],
                                 empirical_regret_upper=audit['empirical_regret_upper'],
                                 epsilon=audit['epsilon'], regret_upper=audit['regret_upper'],
                                 entropy=audit['entropy'], entropy_lower=audit['entropy_lower'],
                                 risk_upper=audit['risk_upper'], compatible=audit['compatible'],
                                 predictive=audit['predictive']))
    with (root/'certificate_table.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    controls = []
    for name, value in result['target_evaluation']['controls'].items():
        controls.append(dict(evaluation='target', domain=result['target_domain'],
                             control=name, accuracy=value['accuracy'], cross_entropy=value['cross_entropy']))
    for held, fold in result['transfer'].items():
        for name, value in fold['evaluation']['controls'].items():
            controls.append(dict(evaluation='excluded_source', domain=held,
                                 control=name, accuracy=value['accuracy'], cross_entropy=value['cross_entropy']))
    with (root/'matched_controls.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(controls[0]))
        writer.writeheader()
        writer.writerows(controls)
    main = [r for r in rows if r['fold'] == 'main']
    experts = sorted(set(r['expert'] for r in main))
    empirical = [max(r['empirical_regret_upper'] for r in main if r['expert'] == i) for i in experts]
    confidence = [max(r['regret_upper'] for r in main if r['expert'] == i) for i in experts]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(experts, empirical, 'o-', label='Empirical deployed-head regret upper')
    axes[0].plot(experts, confidence, 's-', label='Population regret confidence upper')
    axes[0].axhline(config['tau'], color='black', linestyle='--', label=f"Tolerance {config['tau']}")
    axes[0].set(yscale='log', xlabel='Expert (0 = baseline)', ylabel='Excess CE (nats)',
                title='Small empirical regret does not certify compatibility', xticks=experts)
    axes[0].legend(fontsize=7)
    names = ['global_predictor', 'uniform', 'learned', 'certified']
    target = result['target_evaluation']['controls']
    axes[1].bar(range(4), [100*target[n]['accuracy'] for n in names], color=['#444444', '#3182bd', '#6baed6', '#9ecae1'])
    axes[1].set(xticks=range(4), xticklabels=['Global', 'Uniform', 'Router', 'Certified'],
                ylabel='Accuracy (%)', ylim=(0, 100), title='Held target: diagnostics only')
    fig.savefig(root/'audit_diagnostics.png', dpi=170)
    plt.close(fig)
    lines = ['# Predictive compatibility audit — seed 0', '',
             f'Final selection: **{result["selection"]["selected"]}**; reason: '
             f'`{result["selection"]["fallback_reason"]}`.', '',
             'No candidate was forced into the composition. Compatibility and predictivity',
             'refer to the declared normalized map and bounded linear-head class.', '',
             '![Audit diagnostics](audit_diagnostics.png)', '',
             '## Main-fold certificates', '',
             '| Expert | Support | Calibration J upper | Solve gap | Max audit empirical regret upper | Max population regret upper | Accepted |',
             '|---|---|---:|---:|---:|---:|---|']
    for i in experts:
        r = next(r for r in main if r['expert'] == i)
        lines.append(f'| {i} | {r["support"]} | {r["calibration_J_upper"]:.6f} | '
                     f'{r["calibration_solve_gap"]:.6f} | {empirical[i]:.6f} | '
                     f'{confidence[i]:.4f} | {r["accepted"]} |')
    lines += ['', 'Full domain-wise CE, entropy bounds, numerical gaps and confidence terms:',
              '[certificate_table.csv](certificate_table.csv).', '',
              '## Matched controls and strict domain exclusion', '',
              '| Evaluation | Held domain | Control | Accuracy | CE |', '|---|---:|---|---:|---:|']
    for r in controls:
        lines.append(f'| {r["evaluation"]} | {r["domain"]} | {r["control"]} | '
                     f'{100*r["accuracy"]:.2f}% | {r["cross_entropy"]:.4f} |')
    lines += ['', 'Domain IDs: 0 Art, 1 Cartoon, 2 Photo, 3 Sketch.', '',
              'All pseudo-unseen domains are excluded from maps, calibration, audit, router and',
              'selection. With two remaining domains, the candidate bank has only global supports.',
              'The target is evaluated after all source-only decisions. Oracle scores and',
              'pairwise expert disagreement in results.json are diagnostic, not selection rules.', '',
              '## Interpretation', '',
              'Small empirical regret is not a certificate when statistical uncertainty is large.',
              'Failure to certify means insufficient evidence in this tested family; it does not',
              'prove that PACS lacks useful subsets. Frozen-feature target performance is poor',
              'in this first run, so this does not demonstrate an accuracy improvement.',
              'The protocol differs from fine-tuned V1/V2; those accuracies are not matched controls.', '',
              'Configuration, exact split indices, solver diagnostics and source snapshots are',
              'stored beside this report. The baseline is always available, not assumed certified.',
              'Statistical claims require iid within-domain sampling and fresh audit data; source',
              'compatibility does not establish target coverage or persistent mechanisms.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    render(args.directory)


if __name__ == '__main__':
    main()

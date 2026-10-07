"""Fixed four-target PACS audit with one seed and detailed behavior summaries."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from domainbed.scripts.audit_predictive_compatibility import write_json
from domainbed.scripts.report_predictive_compatibility import render


def summarize(root):
    names = ['Art', 'Cartoon', 'Photo', 'Sketch']
    rows, behavior, certificates = [], [], []
    for target in range(4):
        folder = root/f'env{target}_seed0'
        result = json.loads((folder/'results.json').read_text())
        evaluations = [('target', target, 'held', result['target_evaluation'])]
        sources = json.loads((folder/'main/source_behavior.json').read_text())
        for domain, stages in sources.items():
            evaluations.extend(('source', int(domain), stage, v) for stage, v in stages.items())
        evaluations.extend(('excluded_source', int(d), 'held', v['evaluation']) for d, v in result['transfer'].items())
        for kind, domain, stage, evaluation in evaluations:
            for control, m in evaluation['controls'].items():
                rows.append(dict(target=target, kind=kind, domain=domain, stage=stage, control=control,
                                 **{k: m[k] for k in ['n', 'accuracy', 'cross_entropy', 'ece_15', 'brier', 'mean_confidence']}))
            b = evaluation['behavior']
            for i, expert in enumerate(evaluation['experts']):
                behavior.append(dict(target=target, kind=kind, domain=domain, stage=stage, expert=i,
                    accuracy=expert['accuracy'], ce=expert['cross_entropy'],
                    router_mass=evaluation['router_mass'][i],
                    router_top1=b['router_top1_fraction'][i], router_entropy=b['router_entropy'],
                    removal_delta_learned_ce=b['removal_delta_learned_ce'][i],
                    learned_minus_uniform_ce=b['learned_minus_uniform_ce'],
                    oracle_accuracy=evaluation['oracle_single_expert_accuracy'],
                    **b['feature_statistics'][i]))
        for path in folder.glob('*/compatibility.json'):
            for candidate in json.loads(path.read_text()):
                for domain, d in zip(candidate['support'], candidate['audit']['domains']):
                    certificates.append(dict(target=target, fold=path.parent.name, expert=candidate['expert'],
                        domain=domain, accepted=candidate['audit']['accepted'],
                        empirical_regret_upper=d['empirical_regret_upper'],
                        regret_upper=d['regret_upper'], epsilon=d['epsilon'],
                        empirical_predictive_margin=d['entropy']-d['risk'],
                        certified_predictive_margin=d['entropy_lower']-d['risk_upper'],
                        calibration_solver_gap=candidate['calibration']['gap'],
                        audit_solver_gap=d['optimum']['gap']))
    for filename, data in [('metrics.csv', rows), ('expert_behavior.csv', behavior), ('hypothesis_audit.csv', certificates)]:
        with (root/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    scores = {name: [next(r['accuracy'] for r in rows if r['kind']=='target' and r['target']==t and r['control']==name)
                     for t in range(4)] for name in ['global_predictor', 'uniform', 'learned', 'certified', 'selected']}
    write_json(root/'summary.json', dict(seed=0, target_accuracies=scores,
        mean_accuracy={k: float(np.mean(v)) for k, v in scores.items()},
        accepted_main_candidates=len({(r['target'],r['expert']) for r in certificates
                                      if r['fold']=='main' and r['expert']>0 and r['accepted']})))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for i, name in enumerate(['global_predictor', 'uniform', 'learned']):
        axes[0].bar(np.arange(4)+(i-1)*.25, np.array(scores[name])*100, width=.25, label=name)
    axes[0].set(xticks=range(4), xticklabels=names, ylabel='Target accuracy (%)', ylim=(0,100))
    axes[0].legend(fontsize=8)
    main = [r for r in certificates if r['fold']=='main']
    for t in range(4):
        values = [r for r in main if r['target']==t]
        axes[1].scatter([t]*len(values), [r['empirical_regret_upper'] for r in values], color='tab:blue', alpha=.5)
        axes[1].scatter([t+.15]*len(values), [r['regret_upper'] for r in values], color='tab:orange', alpha=.5)
    axes[1].axhline(.12, linestyle='--', color='black', label='tau=.12')
    axes[1].set(yscale='log', xticks=range(4), xticklabels=names, ylabel='Regret upper (nats)',
                title='Blue: empirical; orange: confidence upper')
    axes[1].legend()
    fig.savefig(root/'overview.png', dpi=170)
    plt.close(fig)
    lines = ['# PACS predictive compatibility — four targets, seed 0', '',
             'Frozen pretrained DeiT; fixed thresholds and budgets; 12 strict excluded-source folds.',
             'All four runs share α=.05 across 228 prespecified events. Results are exploratory:',
             'PACS has been examined in earlier iterations; this is not a fresh external test dataset.', '',
             '![Overview](overview.png)', '', '| Predictor | Art | Cartoon | Photo | Sketch | Mean |',
             '|---|---:|---:|---:|---:|---:|']
    for k,v in scores.items():
        lines.append('| '+k+' | '+' | '.join(f'{100*x:.2f}%' for x in [*v,np.mean(v)])+' |')
    main_certificates = [r for r in certificates if r['fold']=='main']
    lines += ['', '## Observed behavior in this fixed run', '',
              f'Accepted extra candidates: {len({(r["target"], r["expert"]) for r in main_certificates if r["accepted"] and r["expert"] > 0})}.',
              f'Empirical deployed-head regret upper spans {min(r["empirical_regret_upper"] for r in main_certificates):.5f}–'
              f'{max(r["empirical_regret_upper"] for r in main_certificates):.5f} nats; '
              f'epsilon spans {min(r["epsilon"] for r in main_certificates):.3f}–{max(r["epsilon"] for r in main_certificates):.3f}.',
              'If 2×epsilon exceeds .12, the compatibility rule cannot certify even zero empirical regret.', '',
              '| Target | Source selection global acc. (domain mean) | Target global acc. | Router normalized entropy | Max routing mass | Oracle expert acc. |',
              '|---|---:|---:|---:|---:|---:|']
    for t, name in enumerate(names):
        result = json.loads((root/f'env{t}_seed0/results.json').read_text())
        evaluation=result['target_evaluation']
        source_acc = np.mean([r['accuracy'] for r in rows if r['target']==t and r['kind']=='source'
                              and r['stage']=='selection' and r['control']=='global_predictor'])
        lines.append(f'| {name} | {100*source_acc:.2f}% | {100*scores["global_predictor"][t]:.2f}% | '
                     f'{evaluation["behavior"]["router_entropy_normalized"]:.3f} | {max(evaluation["router_mass"]):.3f} | '
                     f'{100*evaluation["oracle_single_expert_accuracy"]:.2f}% |')
    lines += ['', 'The gap to the label-informed expert oracle suggests prediction differences worth investigating;',
              'it does not show that a source-trained router can realize those gains on an unseen domain.',
              'Inspect negative removal deltas for harmful routing, and check the same pattern on source-exclusion folds',
              'before choosing an intervention. Preserve these runs as the fixed baseline for later changes.']
    lines += ['', '## Hypothesis diagnostics', '',
              '- **Predictive compatibility:** inspect empirical regret, population upper bound and solver gap separately in [hypothesis_audit.csv](hypothesis_audit.csv). Small empirical regret is not population certification.',
              '- **Within-domain predictivity:** empirical and confidence-corrected entropy-minus-CE margins are logged separately. The required margin is .05 nats.',
              '- **Feature quality and collapse:** [expert_behavior.csv](expert_behavior.csv) records accuracy/CE, variance trace, effective rank and mean feature norm across source stages and held domains.',
              '- **Complementarity:** full JSON includes disagreement, probability JS, loss correlation and oracle expert scores. Oracle scores use labels and are diagnostics only.',
              '- **Router contribution:** compare learned versus uniform CE, routing entropy/mass/top-1 fractions and the CE change when each expert is removed with other weights renormalized. Positive removal delta means that expert helped the fixed router on that split; this is not a retrained ablation.',
              '- **Generalization:** [metrics.csv](metrics.csv) separates train, calibration, audit, selection, target and strictly excluded-source evaluations. No held domain participates in its exclusion fold adaptation.', '',
              '## Evidence limits', '',
              'No thresholds, feature dimensions or training budgets were selected from these target scores.',
              'Source certification does not prove target mechanism persistence. One seed gives no estimate of seed variability.',
              'The whole-bank learned router is a diagnostic control; it is not substituted for the certified deployment rule.',
              'Sample-level logits, labels, features and routing weights are saved in NPZ for targets and excluded sources.', '',
              '## Logs and detailed results', '']
    for t,name in enumerate(names):
        lines.append(f'- {name}: [report](env{t}_seed0/report.md), [console log](env{t}_seed0.log), '
                     f'[training events](env{t}_seed0/main/training.jsonl), [source behavior](env{t}_seed0/main/source_behavior.json).')
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('predictive_support_outputs/pacs_compatibility_lodo_seed0'))
    args=parser.parse_args()
    root=args.output_dir
    if root.exists():
        raise FileExistsError('Use a fresh suite directory')
    root.mkdir(parents=True)
    write_json(root/'protocol.json', dict(seed=0, targets=[0,1,2,3], loso=True, family_runs=4,
        alpha=.05, tau=.12, delta=.05, map_steps=500, router_steps=300, solver_steps=1000,
        purpose='Prespecified full PACS behavior audit; no target-driven tuning'))
    environment=dict(os.environ, OMP_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONWARNINGS='ignore')
    for target in range(4):
        write_json(root/'status.json', dict(state='running', target=target, completed=list(range(target))))
        command=[sys.executable, '-u', '-m', 'domainbed.scripts.audit_predictive_compatibility',
                 '--target', str(target), '--seed','0','--loso','--family-runs','4',
                 '--output-dir', str(root/f'env{target}_seed0')]
        print(f'Start target {target}, seed 0', flush=True)
        with (root/f'env{target}_seed0.log').open('w') as stream:
            result=subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        if result.returncode:
            write_json(root/'status.json', dict(state='failed', target=target, exit_code=result.returncode))
            raise RuntimeError(f'Target {target} failed; see console log')
        render(root/f'env{target}_seed0')
        print(f'Completed target {target}', flush=True)
    summarize(root)
    write_json(root/'status.json', dict(state='completed', completed=[0,1,2,3], seed=0))


if __name__=='__main__':
    main()

"""Regenerate EDA figures and a readable report from a running or completed run."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def build_report(directory):
    directory = Path(directory)
    rows = []
    for line in (directory / 'metrics.jsonl').read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a running writer can have an incomplete final record
    if not rows:
        return
    config = json.loads((directory / 'config.json').read_text())
    dest = directory / 'eda'
    dest.mkdir(exist_ok=True)
    steps = [r['step'] for r in rows]
    structures = [r for r in rows if 'structure_eda' in r]
    validations = [r for r in rows if 'source_validation' in r]
    names = config.get('environment_names', [str(i) for i in range(4)])
    sources = [names[i] for i in config['source_envs']]
    figures = []
    def save(fig, name):
        fig.tight_layout(); fig.savefig(dest / name, dpi=140); plt.close(fig)
        figures.append(name)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for field in ['weighted_predictive' if 'weighted_predictive' in rows[0] else 'local_risk',
                  'weighted_admissibility', 'weighted_mmd', 'mixture_ce']:
        axes[0, 0].plot(steps, [r.get(field, np.nan) for r in rows], label=field, alpha=.8)
    axes[0, 0].set_title('Objective components (raw per-step values)'); axes[0, 0].legend()
    for field in ['prediction_bound', 'mixture_ce']:
        axes[0, 1].plot(steps, [r.get(field, np.nan) for r in rows], label=field)
    axes[0, 1].set_title('Prediction bound and mixture CE'); axes[0, 1].legend()
    axes[1, 0].plot(steps, [r['inadmissible_mass'] for r in rows]); axes[1, 0].set_title('Train inadmissible mass')
    if validations:
        for k, name in enumerate(sources):
            axes[1, 1].plot([r['step'] for r in validations], [r['source_validation'][k]['accuracy'] for r in validations], label=name)
        axes[1, 1].legend()
    axes[1, 1].set_title('Source validation accuracy (selection only)')
    for ax in axes.flat:
        ax.set_xlabel('Step'); ax.grid(alpha=.2)
    save(fig, 'training.png')
    if structures:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        history = np.array([r['support'] for r in structures])
        axes[0, 0].imshow(history.reshape(len(history), -1).T, aspect='auto', vmin=0, vmax=1)
        axes[0, 0].set_title('Support membership history (domain-major rows)')
        axes[0, 0].set_xlabel('Structure refresh index')
        ss = [r['step'] for r in structures]
        for key in ['active_experts', 'support_changed_bits']:
            axes[0, 1].plot(ss, [r['structure_eda'][key] for r in structures], label=key)
        axes[0, 1].legend()
        axes[1, 0].plot(ss, [r['structure_eda']['empirical_runner_up_gap'] for r in structures])
        axes[1, 0].set_title('Empirical score gap (not a recovery certificate)')
        a = np.array(structures[-1]['structure_eda']['pair_expert_mmd'])
        im = axes[1, 1].imshow(a, aspect='auto', cmap='coolwarm'); fig.colorbar(im, ax=axes[1, 1])
        axes[1, 1].set_yticks(range(len(a)), [f'{sources[i]} / {sources[j]}' for i,j in structures[-1]['structure_eda']['pair_indices']])
        axes[1, 1].set_title('Latest source-training MMD U-statistic'); axes[1, 1].set_xlabel('Expert')
        save(fig, 'structure.png')
    results = json.loads((directory / 'results.json').read_text()) if (directory / 'results.json').exists() else None
    reports = results['source'] if results else (validations[-1]['source_validation'] if validations else [])
    if reports:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, field, eda in zip(axes.flat, ['expert_accuracy','expert_risk','routing_mean','routing_top1_fraction'], [False,False,True,True]):
            arr = np.array([(r['eda'] if eda else r)[field] for r in reports])
            im = ax.imshow(arr, aspect='auto'); fig.colorbar(im, ax=ax)
            ax.set_yticks(range(len(sources)), sources); ax.set_xlabel('Expert'); ax.set_title(field)
        save(fig, 'expert_domain.png')
        all_reports = reports + (results['target'] if results else [])
        all_names = sources + ([names[config['target_env']]] if results else [])
        fig, axes = plt.subplots(len(all_reports), 3, figsize=(16, 4 * len(all_reports)), squeeze=False)
        for k, (r, name) in enumerate(zip(all_reports, all_names)):
            e = r['eda']
            cm = np.array(e['confusion_matrix']); normalized = cm / np.maximum(cm.sum(1, keepdims=True),1)
            axes[k, 0].imshow(normalized, vmin=0, vmax=1); axes[k, 0].set_title(name + ': row-normalized confusion')
            axes[k, 0].set_ylabel('True class'); axes[k, 0].set_xlabel('Predicted class')
            routing = [c['routing_mean'] or [np.nan]*len(e['routing_mean']) for c in e['per_class']]
            axes[k, 1].imshow(routing, aspect='auto', vmin=0, vmax=1); axes[k, 1].set_title('Routing per class'); axes[k, 1].set_xlabel('Expert')
            bins = [b for b in e['calibration_bins'] if b['count']]
            axes[k, 2].plot([b['confidence'] for b in bins], [b['accuracy'] for b in bins], 'o-')
            axes[k, 2].plot([0,1],[0,1],'--'); axes[k, 2].set_title(f'Calibration, ECE={e["ece_10bins"]:.3f}')
            axes[k, 2].set_xlabel('Confidence'); axes[k, 2].set_ylabel('Accuracy')
        save(fig, 'class_calibration.png')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for name,r in zip(sources,reports):
            axes[0].plot(r['eda']['feature_variance_trace'], 'o-', label=name)
            axes[1].plot(r['eda']['feature_mean_norm'], 'o-', label=name)
        axes[0].set_title('Feature variance trace per expert'); axes[1].set_title('Feature mean norm per expert')
        for ax in axes:
            ax.set_xlabel('Expert'); ax.legend()
        save(fig, 'feature_collapse.png')
    if validations and 'eda' in validations[-1]['source_validation'][0]:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for ax, field, nested in zip(axes.flat,
                ['inadmissible_mass','routing_entropy','bound_gap','ece_10bins','brier_score','routing_effective_experts'],
                [False,True,False,True,True,True]):
            for k,name in enumerate(sources):
                ax.plot([r['step'] for r in validations],
                        [(r['source_validation'][k]['eda'] if nested else r['source_validation'][k])[field]
                         for r in validations],label=name)
            ax.set_title(field); ax.set_xlabel('Step'); ax.legend(); ax.grid(alpha=.2)
        save(fig, 'validation_diagnostics.png')
    gradients = [r for r in rows if 'gradient_norms' in r]
    if gradients:
        fig, ax = plt.subplots(figsize=(12, 5))
        for name in gradients[0]['gradient_norms']:
            ax.plot([r['step'] for r in gradients], [r['gradient_norms'][name] for r in gradients],label=name)
        ax.set_yscale('symlog',linthresh=1e-5); ax.set_xlabel('Step'); ax.set_ylabel('Gradient L2 norm')
        ax.set_title('Total objective gradient by parameter group'); ax.legend(ncol=3); ax.grid(alpha=.2)
        save(fig,'gradients.png')
    if structures and 'gap_predictive' in structures[-1]['structure_eda']:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for field in ['gap_predictive', 'gap_mmd', 'gap_router']:
            axes[0].plot([r['step'] for r in structures], [r['structure_eda'][field] for r in structures], label=field)
        axes[0].set_title('Runner-up minus best: signed score components'); axes[0].legend()
        constraints = [r for r in rows if r.get('source_predictive_constraint')]
        if constraints:
            for m in range(len(constraints[0]['source_predictive_constraint']['risk'])):
                # Exclude inactive experts: their zero violation is not competence.
                axes[1].plot([r['step'] for r in constraints],
                             [r['source_predictive_constraint']['violation'][m]
                              if r['source_predictive_constraint']['active'][m] else np.nan
                              for r in constraints], label=f'E{m}')
            axes[1].legend()
        axes[1].set_title('Source-validation threshold violations (active experts)')
        for ax in axes: ax.set_xlabel('Step'); ax.grid(alpha=.2)
        save(fig, 'structure_information.png')
    scalar_keys = sorted({k for r in rows for k,v in r.items() if isinstance(v,(int,float,bool))})
    with (dest / 'training_scalars.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=scalar_keys,lineterminator='\n'); writer.writeheader()
        writer.writerows({k:r.get(k) for k in scalar_keys} for r in rows)
    if reports:
        with (dest / 'expert_domain.csv').open('w') as f:
            writer=csv.writer(f,lineterminator='\n'); writer.writerow(['domain','expert','assigned','accuracy','risk','routing_mean','top1_fraction','feature_variance'])
            for name,r in zip(sources,reports):
                for m in range(len(r['expert_risk'])):
                    writer.writerow([name,m,r['assigned_experts'][m],r['expert_accuracy'][m],r['expert_risk'][m],r['eda']['routing_mean'][m],r['eda']['routing_top1_fraction'][m],r['eda']['feature_variance_trace'][m]])
    text = [f'# PACS support EDA — target {names[config["target_env"]]}', '',
            f'Status: {"completed" if results else "running"}; latest step {steps[-1] + 1}/{config["steps"]}; seed {config["seed"]}.', '',
            'Source domains: ' + ', '.join(sources) + '.',
            'Class indices: ' + ', '.join(f'{i}={v}' for i,v in enumerate(config.get('class_names', []))) + '.', '',
            'Checkpoint selection uses mean source-validation accuracy. Target statistics appear only after selection.', '',
            'MMD estimates may be negative. Empirical candidate gaps are probe-specific, not population separation guarantees. Feature statistics and routing utilization are diagnostics, not new losses.', '']
    if config['hparams'].get('support_predictive_objective') == 'information_hinge':
        text += ['This run uses mixture CE and a support-conditional predictive hinge. The old max-CE prediction bound remains a diagnostic only. Entropy thresholds are fixed from source-training class counts; an empirical hinge of zero does not certify population mutual information.', '']
    if results:
        text += [f'Best source accuracy: {results["best_source_accuracy"]:.4f}; selected step: {results["best_step"] + 1}.',
                 f'Target accuracy: {results["target"][0]["accuracy"]:.4f}.', '']
    for filename in figures:
        text += [f'![{filename}]({filename})', '']
    (dest / 'report.md').write_text('\n'.join(text))
    if (directory.parent / 'queue.json').exists():
        build_queue_report(directory.parent)


def build_queue_report(root):
    root = Path(root)
    queue = json.loads((root / 'queue.json').read_text())
    lines = ['# PACS experiment queue', '',
             'Target metrics are scored only after source-based checkpoint selection. Compare arm/seed labels before aggregating results.', '',
             '| Arm | Seed | Target | Progress | Source validation (latest) | Target accuracy (selected checkpoint) | EDA |',
             '|---|---:|---|---:|---:|---:|---|']
    for job in queue['jobs']:
        folder = Path(job['output'])
        cfg = json.loads((folder/'config.json').read_text()) if (folder/'config.json').exists() else {}
        name = cfg.get('environment_names',['art_painting','cartoon','photo','sketch'])[job['target_env']]
        rows = []
        if (folder/'metrics.jsonl').exists():
            for line in (folder/'metrics.jsonl').read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        validations = [r for r in rows if 'source_validation' in r]
        source = np.mean([r['accuracy'] for r in validations[-1]['source_validation']]) if validations else None
        result = json.loads((folder/'results.json').read_text()) if (folder/'results.json').exists() else None
        target = result['target'][0]['accuracy'] if result else None
        progress = str(rows[-1]['step']+1) + '/' + str(cfg.get('steps',5000)) if rows else job['state']
        source_text = f'{source:.4f}' if source is not None else '—'
        target_text = f'{target:.4f}' if target is not None else 'pending'
        link = f'[{name}]({folder.name}/eda/report.md)' if (folder/'eda/report.md').exists() else 'pending'
        lines.append(f'| {job.get("arm", "original")} | {job.get("seed", 0)} | {name} | {progress} | {source_text} | {target_text} | {link} |')
    lines += ['', 'Reports refresh at source-validation checkpoints. Raw logs and queue status are in this directory. Controls with one seed do not establish variance across seeds.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    build_report(parser.parse_args().directory)

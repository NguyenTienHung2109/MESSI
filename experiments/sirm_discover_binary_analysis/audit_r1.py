"""Audit the live binary R1 trajectory without changing training or selection."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .run import describe, evolution_stats, rows, support


def read(path):
    return json.loads(path.read_text())


def audit(root):
    dest = root / 'analysis'
    on = rows(root / 'banks/R1/diagnostics.jsonl')
    off = rows(root / 'banks/R0/diagnostics.jsonl')
    refresh = rows(root / 'banks/R1/q_refresh.jsonl')
    old_refresh = {r['step']: r for r in rows(root / 'banks/R0/q_refresh.jsonl')}
    scores = rows(root / 'banks/R1/results.jsonl')
    config = scores[-1]['hparams']['sirm_discover_config']
    expected_config = read(root / 'plan.json')['config']['R1']['sirm_discover_config']
    last = on[-1]['step']
    start = config['global_steps'] + config['discover_steps']
    stage3 = [r for r in on if r['stage'] == 'invariance']
    active = [r for r in stage3 if r['sirm_active']]
    qrows = [r for r in stage3 if r['q_refreshed']]
    measured = [r for r in qrows if r['weighted_sirm_grad_norm'] is not None]
    ratios = [r['weighted_sirm_grad_norm'] / r['base_residual_grad_norm']
              for r in measured if r['base_residual_grad_norm'] > 0]
    base_keys = ['loss_cls', 'loss_discover', 'loss_router', 'loss_balance',
                 'residual_rms', 'router_null_mass']
    pre_keys = base_keys + ['loss', 'utility_ema', 'utility_interaction', 'q', 'q_null']
    first_difference = {k: next((r['step'] for r in on
                                if r[k] != off[r['step']][k]), None)
                        for k in pre_keys}
    shared = [r for r in qrows if r['shared_experts'] > 0]
    eligible = [r for r in qrows if r['eligible_pair_expert_count'] > 0]
    first_support_change = next((r['step'] for r in qrows
                                if not np.array_equal(np.asarray(r['q']) > 0,
                                                      np.asarray(off[r['step']]['q']) > 0)), None)
    normalized_losses = []
    for r in active:
        grads = {tuple(map(int, k.split(','))): value for k, value in r['G'].items()}
        q = np.asarray(r['q'])
        terms = [grads[i, m] ** 2 + grads[j, m] ** 2
                 for i, j in itertools.combinations(range(len(q)), 2)
                 for m in range(q.shape[1])
                 if q[i, m] * q[j, m] > 0 and (i, m) in grads and (j, m) in grads]
        normalized_losses.append(len(terms) == r['sirm_pair_count'] and
                                 np.isclose(r['loss_sirm'], np.mean(terms), rtol=1e-6, atol=1e-12))
    checks = dict(
        actual_config_matches_locked_R1=config == expected_config,
        binary_weighting=config['sirm_weighting'] == 'binary',
        lambda_config_unchanged=config['lambda_sirm'] == expected_config['lambda_sirm'],
        initialization_matches_R0=read(root / 'banks/R1/initialization.json') == read(root / 'banks/R0/initialization.json'),
        source_schedules_match_R0=read(root / 'banks/R1/source_schedules.json') == read(root / 'banks/R0/source_schedules.json'),
        contiguous_steps=[r['step'] for r in on] == list(range(len(on))),
        stage_boundaries_correct=all(r['stage'] == ('global' if r['step'] < config['global_steps']
                                    else 'discover' if r['step'] < start else 'invariance') for r in on),
        all_objectives_finite=all(np.isfinite(r[k]) for r in on
                                  for k in ['loss', 'loss_cls', 'loss_discover', 'loss_router', 'loss_balance', 'loss_sirm']),
        preactivation_trajectory_exactly_matches_R0=all(r[k] == off[r['step']][k]
                                                      for r in on if r['step'] <= start for k in pre_keys),
        penalty_zero_before_activation=all(r['effective_lambda_sirm'] == 0 and r['loss_sirm'] == 0
                                            and not r['sirm_active'] for r in on if r['step'] <= start),
        ramp_matches_original_schedule=all(abs(r['effective_lambda_sirm'] - config['lambda_sirm'] *
                                              min(max((r['step'] - start) / config['sirm_ramp_steps'], 0.), 1.)) < 1e-15
                                           for r in on),
        Q_refresh_cadence_correct=[r['step'] for r in on if r['q_refreshed']] == list(range(config['global_steps'], last + 1, config['utility_interval'])),
        Q_not_locked=all(not r['assignment_locked'] for r in on),
        Q_held_between_refreshes=all(on[i]['q'] == on[i-1]['q'] for i in range(1, len(on)) if not on[i]['q_refreshed']),
        preactivation_source_validation_matches_R0=all(r['source_validation_accuracy'] == old_refresh[r['step']]['source_validation_accuracy']
                                                      and r['source_validation_per_domain'] == old_refresh[r['step']]['source_validation_per_domain']
                                                      for r in refresh if r['step'] <= start),
        binary_loss_matches_mean_of_eligible_raw_terms=all(normalized_losses),
        inactive_loss_zero=all(r['loss_sirm'] == 0 for r in on if not r['sirm_active']),
        active_losses_nonzero_and_finite=all(r['loss_sirm'] > 0 and np.isfinite(r['loss_sirm']) for r in active),
        measured_gradients_finite_and_nonzero=all(np.isfinite(r['weighted_sirm_grad_norm']) and r['weighted_sirm_grad_norm'] > 0 for r in measured),
    )
    first_active = active[0] if active else None
    first_measurement = measured[0] if measured else None
    latest_val = refresh[-1] if refresh else None
    result = dict(
        snapshot_last_step=last, stage=on[-1]['stage'], steps_checked=len(on),
        checks=checks, all_checks_pass=all(bool(v) for v in checks.values()),
        first_shared_support_step=shared[0]['step'] if shared else None,
        first_eligible_support_step=eligible[0]['step'] if eligible else None,
        first_nonzero_SIRM_step=first_active['step'] if first_active else None,
        first_total_loss_difference_step=first_difference['loss'],
        first_baseline_trajectory_difference_step=min((first_difference[k] for k in base_keys if first_difference[k] is not None), default=None),
        first_difference_by_quantity=first_difference,
        first_Q_support_difference_step=first_support_change,
        first_active_loss=first_active['loss_sirm'] if first_active else None,
        first_active_lambda=first_active['effective_lambda_sirm'] if first_active else None,
        first_measurement=({k: first_measurement[k] for k in
                            ['step', 'weighted_sirm_grad_norm', 'base_residual_grad_norm', 'sirm_changes_gradient_elements',
                             'selected_cells', 'shared_experts', 'eligible_pair_expert_count', 'sirm_pair_count', 'G']}
                           if first_measurement else None),
        gradient_ratio_active_refreshes=describe(ratios),
        stage3_steps=len(stage3), active_steps=len(active),
        active_fraction_so_far=len(active)/len(stage3) if stage3 else None,
        Q_evolution=evolution_stats(qrows),
        refresh_blockers=dict(selected_but_no_shared=sum(r['selected_cells'] > 0 and r['shared_experts'] == 0 for r in qrows),
                              shared_but_no_eligible=sum(r['shared_experts'] > 0 and r['eligible_pair_expert_count'] == 0 for r in qrows)),
        latest_training_batch=dict(residual_RMS=on[-1]['residual_rms'], router_NULL=on[-1]['router_null_mass']),
        latest_completed_source_validation=(dict(step=latest_val['step'], accuracy=latest_val['source_validation_accuracy'],
                                                per_domain=latest_val['source_validation_per_domain']) if latest_val else None),
        limits='Live snapshot, one seed. Ratios use residual-parameter norms at active Q refreshes, not all optimizer steps. '
               'Active fraction is a partial Stage-3 fraction. Direct final checkpoint/logit comparison and target accuracy remain pending. '
               'Loss/output-statistic divergence establishes trajectory change, not a direct tensor comparison at every step. '
               'Q-weight differences are distinct from support differences. No new training, target inference, tuning or selection changes.',
    )
    comparison_path = dest / 'R1_equal_step_parameter_comparison.json'
    if comparison_path.exists():
        result['equal_step_parameter_comparison'] = read(comparison_path)
    (dest / 'R1_behavior_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    text = ['# R1 binary-ON behavior audit', '',
            f'Live snapshot through step **{last}**, stage **{on[-1]["stage"]}**. All checks pass: **{result["all_checks_pass"]}**.', '',
            '| Behavior | Value |', '|---|---:|']
    for key in ['first_shared_support_step', 'first_eligible_support_step', 'first_nonzero_SIRM_step',
                'first_total_loss_difference_step', 'first_baseline_trajectory_difference_step',
                'first_Q_support_difference_step', 'first_active_loss', 'first_active_lambda', 'active_steps', 'stage3_steps', 'active_fraction_so_far']:
        text.append(f'| {key} | {result[key]} |')
    text.extend(['', 'Gradient ratio at active refreshes: ' + json.dumps(result['gradient_ratio_active_refreshes']), '',
                 'First measured penalty: ' + json.dumps(result['first_measurement']), '',
                 'First R1−R0 difference by quantity: ' + json.dumps(first_difference), '',
                 'Q temporal evolution: ' + json.dumps(result['Q_evolution']), '',
                 'Refresh blockers: ' + json.dumps(result['refresh_blockers']), '',
                 'Latest completed source validation: ' + json.dumps(result['latest_completed_source_validation']), '',
                 '| Check | Pass |', '|---|---:|'])
    text.extend(f'| {k} | {value} |' for k, value in checks.items())
    if comparison_path.exists():
        text.extend(['', 'Equal-step parameter comparison: ' + json.dumps(result['equal_step_parameter_comparison']), '',
                     '[Direct checkpoint tensor comparison](R1_equal_step_parameter_comparison.json).'])
    text.extend(['', result['limits']])
    (dest / 'R1_behavior_audit.md').write_text('\n'.join(text) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['checks', 'limits']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/sirm_discover_binary/terra_L38_seed0')
    args = parser.parse_args()
    audit(Path(args.root))


if __name__ == '__main__':
    main()

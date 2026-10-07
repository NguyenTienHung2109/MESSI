"""Actual hard-discovery/gate trajectory audit, partial or complete."""
import argparse
import json
from pathlib import Path
import numpy as np
from .shrinkage import ROOT, digest, write


def run(dest, arm):
    plan = json.loads((dest / 'plan.json').read_text())
    for key in ['base_code_hashes', 'variant_code_hashes']:
        assert all(digest(ROOT / p) == h for p, h in plan[key].items())
    c = plan['hparams'][arm]['sirm_discover_config']
    bank = dest / 'experiment/runs/L38' / arm
    rows = [json.loads(l) for l in (bank / 'diagnostics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows] == list(range(len(rows)))
    assert all(r['discovery_assignment'] == 'hard_winner' and
               r['balance_assignment'] == 'original_soft' for r in rows)
    loss_errors = [abs(r['loss_discover'] - r['discovery_hard_loss_replay']) for r in rows]
    assert max(loss_errors) < 1e-6
    assert all(abs(sum(r['discovery_effective_winner_load']) - 1) < 1e-6 for r in rows)
    assert all(abs(sum(r['discovery_soft_balance_load']) - 1) < 1e-6 for r in rows)
    refs = [r for r in rows if r['q_refreshed'] and r['step'] >= 200]
    blockers = {k: 0 for k in ['invalid_U', 'no_U_floor_pass', 'U_but_no_joint_U_V',
                              'selected_but_no_shared', 'shared_but_ESS_ineligible', 'eligible']}
    records = []
    for r in refs:
        u, v = np.array(r['utility_ema']), np.array(r['utility_interaction'])
        up = u > c['utility_floor']; joint = up & (v > c['interaction_floor'])
        key = ('invalid_U' if not np.array(r['utility_valid']).all() else
               'no_U_floor_pass' if not up.any() else
               'U_but_no_joint_U_V' if not joint.any() else
               'selected_but_no_shared' if not r['shared_experts'] else
               'shared_but_ESS_ineligible' if not r['eligible_pair_expert_count'] else 'eligible')
        blockers[key] += 1
        records.append(dict(step=r['step'], blocker=key, max_U=float(u.max()),
                            max_V=float(v.max()), selected_cells=r['selected_cells'],
                            shared_experts=r['shared_experts'], eligible_pairs=r['eligible_pair_expert_count']))
    result = dict(code_sha256=digest(Path(__file__)), arm=arm, through_step=rows[-1]['step'],
        updates=len(rows), complete=len(rows)==400 and (bank/'done').exists(),
        hard_loss_replay_max_abs_error=max(loss_errors), hard_formula_and_soft_balance_checks_pass=True,
        locked_code_unchanged=True, invariance_refreshes=len(refs), blockers=blockers,
        active_steps=sum(r['sirm_active'] for r in rows),
        first_shared=next((r['step'] for r in refs if r['shared_experts']), None),
        first_active=next((r['step'] for r in rows if r['sirm_active']), None),
        records=records, scientific_goal_achieved=False)
    write(dest / f'{arm}_hard_gate_audit.json', result)
    print({k: v for k, v in result.items() if k != 'records'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('destination', type=Path)
    p.add_argument('arm', choices=['OFF', 'ON']); a=p.parse_args(); run(a.destination,a.arm)

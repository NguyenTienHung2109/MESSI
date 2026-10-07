"""Read-only partial/final audit of the actual logged cap and frozen protocol."""
import argparse
import json
import math
from pathlib import Path
from .shrinkage import digest, write, ROOT


def run(dest, arm):
    plan = json.loads((dest / 'plan.json').read_text())
    for key in ['base_code_hashes', 'variant_code_hashes']:
        assert all(digest(ROOT / p) == h for p, h in plan[key].items())
    bank = dest / 'experiment/runs/L38' / arm
    rows = [json.loads(l) for l in (bank / 'diagnostics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows] == list(range(len(rows)))
    errors = []; ratios = []; pre_ratios = []
    for r in rows:
        cn = r['router_cls_parameter_grad_norm']; an = r['router_aux_pre_cap_parameter_grad_norm']
        bn = r['router_aux_post_cap_parameter_grad_norm']; scale = r['router_aux_detached_scale']
        assert all(math.isfinite(x) for x in [cn, an, bn, scale])
        assert all(x >= 0 for x in [cn, an, bn]) and 0 <= scale <= 1
        assert abs(scale - (min(1, cn / an) if an > 0 else 1)) < 1e-6
        assert bn <= cn * 1.00001 + 1e-10
        assert abs(r['loss_router'] - scale * r['router_aux_raw_loss']) < 1e-5
        if cn > 0:
            ratios.append(bn / cn); pre_ratios.append(an / cn)
        else:
            assert bn == 0
        if scale < 1:
            errors.append(abs(bn - cn) / max(cn, 1e-20))
    frozen = [json.loads(l) for l in (bank / 'frozen_anchor_checks.jsonl').read_text().splitlines()]
    assert all(r['requires_grad_false'] and r['gradient_absent'] and r['parameter_versions_unchanged'] for r in frozen)
    refreshes = [r for r in rows if r['q_refreshed'] and r['step'] >= 200]
    result = {'code_sha256': digest(Path(__file__)), 'arm': arm,
              'through_step': rows[-1]['step'], 'updates': len(rows),
              'complete_400_updates': len(rows) == 400 and (bank / 'done').exists(),
              'all_norm_scale_and_loss_checks_pass': True, 'locked_code_unchanged': True,
              'frozen_checks_pass': True, 'frozen_checks_through_step': frozen[-1]['step'],
              'capped_updates': sum(r['router_aux_capped'] for r in rows),
              'zero_classification_norm_updates': sum(r['router_cls_parameter_grad_norm'] == 0 for r in rows),
              'scale_min': min(r['router_aux_detached_scale'] for r in rows),
              'scale_max': max(r['router_aux_detached_scale'] for r in rows),
              'max_post_cap_ratio_nonzero_reference': max(ratios) if ratios else None,
              'max_pre_cap_ratio_nonzero_reference': max(pre_ratios) if pre_ratios else None,
              'norm_match_error_when_capped_max': max(errors) if errors else None,
              'invariance_refreshes': len(refreshes),
              'shared_refreshes': sum(r['shared_experts'] > 0 for r in refreshes),
              'eligible_refreshes': sum(r['eligible_pair_expert_count'] > 0 for r in refreshes),
              'active_steps': sum(r['sirm_active'] for r in rows),
              'scope': 'Observed actual logged same-state router gradients, not saved-state approximation. '
                       'Classification-only reference; excludes SIRM and does not bound optimizer updates. '
                       'Zero reference is checked directly rather than assigned a zero ratio.',
              'scientific_goal_achieved': False}
    write(dest / f'{arm}_router_cap_audit.json', result)
    print(result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('destination', type=Path)
    p.add_argument('arm', choices=['OFF', 'ON']); a = p.parse_args(); run(a.destination, a.arm)

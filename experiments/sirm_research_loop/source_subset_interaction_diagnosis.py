"""Exact source-only subset capacity and contextual interaction audit."""
import json, math
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .routed_discovery_run import DEST, ROOT
from .source_composition_calibration import STATES
from .shrinkage import digest, write

OUT = DEST / 'source_subset_interaction'
MASKS = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(float)

def analyze(g, t, y, domains, ids):
    logits = g[:, None, :] + np.einsum('nmj,sm->nsj', t, MASKS)
    ce = logsumexp(logits, axis=2) - logits[np.arange(len(y)), :, y]
    correct = logits.argmax(2) == y[:, None]
    comp = logits.copy()
    comp[np.arange(len(y)), :, y] = -np.inf
    margin = logits[np.arange(len(y)), :, y] - comp.max(2)
    native, global_correct = correct[:, 63], correct[:, 0]
    best_ce = ce.argmin(1)
    # This capacity maximizes true-label margin, so it is deliberately not deployable.
    best_margin = margin.argmax(1)
    any_correct = correct.any(1)
    shapley = np.zeros((len(y), 6))
    for expert in range(6):
        for subset in range(64):
            if subset & (1 << expert):
                continue
            k = int(MASKS[subset].sum())
            weight = math.factorial(k) * math.factorial(5-k) / math.factorial(6)
            shapley[:, expert] += weight * (ce[:, subset] - ce[:, subset | (1 << expert)])
    assert np.allclose(shapley.sum(1), ce[:, 0] - ce[:, 63], atol=1e-10)
    standalone = np.stack([ce[:, 0] - ce[:, 1 << m] for m in range(6)], 1)
    marginal = np.stack([ce[:, 63 ^ (1 << m)] - ce[:, 63] for m in range(6)], 1)
    pair_interactions = {}
    for i in range(6):
        for j in range(i+1, 6):
            # Positive is complementary CE gain beyond the two standalone gains.
            pair_interactions[f'{i},{j}'] = ce[:, 1 << i] + ce[:, 1 << j] - ce[:, 0] - ce[:, (1 << i) | (1 << j)]
    def group(mask):
        n = int(mask.sum())
        if not n:
            return {'n': 0}
        return dict(n=n, global_correct=int(global_correct[mask].sum()), native_correct=int(native[mask].sum()),
                    any_subset_correct=int(any_correct[mask].sum()),
                    best_CE_subset_correct=int(correct[np.arange(len(y)), best_ce][mask].sum()),
                    mean_global_CE=float(ce[mask, 0].mean()), mean_native_CE=float(ce[mask, 63].mean()),
                    mean_best_subset_CE=float(ce[np.arange(len(y)), best_ce][mask].mean()),
                    standalone_U=standalone[mask].mean(0).tolist(), contextual_U=marginal[mask].mean(0).tolist(),
                    shapley_CE_gain=shapley[mask].mean(0).tolist(),
                    standalone_context_sign_disagreement=((standalone[mask] > 0) != (marginal[mask] > 0)).mean(0).tolist(),
                    pair_synergy_mean={k: float(v[mask].mean()) for k,v in pair_interactions.items()})
    groups = {'all': group(np.ones(len(y), bool)), 'global_errors': group(~global_correct),
              'global_correct': group(global_correct)}
    for d in sorted(set(domains)):
        for c in sorted(set(y)):
            groups[f'domain{d}/class{c}'] = group((domains == d) & (y == c))
    cases = []
    for i in np.flatnonzero(~global_correct | (global_correct != native)):
        candidates = np.flatnonzero(correct[i])
        minimal = min(candidates, key=lambda s: (int(MASKS[s].sum()), int(s))) if len(candidates) else None
        cases.append(dict(id=str(ids[i]), label=int(y[i]), domain=int(domains[i]),
                          global_correct=bool(global_correct[i]), native_correct=bool(native[i]),
                          global_margin=float(margin[i, 0]), native_margin=float(margin[i, 63]),
                          best_margin=float(margin[i, best_margin[i]]), best_margin_subset=int(best_margin[i]),
                          best_CE_subset=int(best_ce[i]), best_CE_correct=bool(correct[i, best_ce[i]]),
                          correct_subset_count=len(candidates), minimal_correct_subset=None if minimal is None else int(minimal),
                          standalone_U=standalone[i].tolist(), contextual_U=marginal[i].tolist(),
                          shapley_CE_gain=shapley[i].tolist()))
    summary = dict(n=len(y), repairs=int((~global_correct & native).sum()), harms=int((global_correct & ~native).sum()),
                   additional_errors_repairable_by_subset=int((~global_correct & ~native & any_correct).sum()),
                   errors_unrepairable_by_any_subset=int((~global_correct & ~any_correct).sum()),
                   bird_errors=int(((y == 0) & ~global_correct).sum()),
                   bird_errors_repairable_by_subset=int(((y == 0) & ~global_correct & any_correct).sum()),
                   native_harms_restored_by_nonempty_subset=int(sum(np.any(correct[i, 1:63]) for i in np.flatnonzero(global_correct & ~native))),
                   global_equal_domain_accuracy=float(np.mean([global_correct[domains == d].mean() for d in sorted(set(domains))])),
                   native_equal_domain_accuracy=float(np.mean([native[domains == d].mean() for d in sorted(set(domains))])),
                   capacity_equal_domain_accuracy=float(np.mean([any_correct[domains == d].mean() for d in sorted(set(domains))])))
    return dict(summary=summary, groups=groups, cases=cases), logits[:,63]

def run():
    OUT.mkdir(exist_ok=True)
    plan = dict(source_only=True, target_read=False, fitting=False, primary_training=False,
                states=STATES, subsets=64, masks=MASKS.astype(int).tolist(),
                hypothesis='Case-level useful corrections are canceled by other deployed experts; quantify recoverable errors without amplifying or rerouting before proposing another objective.',
                protocol='Enumerate all subsets of the six fixed routed contributions, without renormalizing pi. Source-in/out only, same calibrated saved bank. Empty subset is frozen ERM. Both selected and equal399 states are included.',
                interpretation='Best label-informed subset is retrospective capacity, not a learned gate, selection rule or accuracy claim. CE-minimizing and margin-maximizing subsets are separate. Shapley CE contributions average exact subset contexts, not causal effects or unique factors.',
                code_sha256=digest(Path(__file__)))
    write(OUT/'plan.json', plan)
    (OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    previous = json.loads((DEST/'source_composition_calibration/results.json').read_text())
    token = ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis'
    audit = json.loads((token/'extraction_audit.json').read_text())
    output = {}
    for state in STATES:
        path = DEST/'source_spatial_reliability'/f'{state}_source_replay.npz'
        assert digest(path) == previous['cache_sha256'][state]
        a = np.load(path)
        output[state] = {}
        for split in ['source_in', 'source_out']:
            p = token/f'{split}_tokens.npz'
            assert digest(p) == next(r['cache_sha256'] for r in audit['rows'] if r['split'] == split)
            b = np.load(p)
            assert np.array_equal(a[split+'_ids'], b['ids']) and np.array_equal(a[split+'_y'], b['y'])
            t = a[split+'_bank_delta'].astype(float) * a[split+'_bank_pi'][:,:6,None].astype(float)
            result, full = analyze(b['global_logits'].astype(float), t, b['y'], b['domains'], b['ids'])
            assert np.array_equal(full.argmax(1), a[split+'_bank_full'].argmax(1))
            output[state][split] = result
            print(state, split, result['summary'], flush=True)
        assert not set(a['source_in_ids']) & set(a['source_out_ids'])
        assert digest(path) == previous['cache_sha256'][state]
    write(OUT/'results.json', dict(plan=plan, states=output, cache_sha256=previous['cache_sha256'],
                                  source_only=True, target_read=False, bank_updates=False, fitting=False,
                                  masks_full_native_predictions_match=True, shapley_efficiency_verified=True))

if __name__ == '__main__':
    run()

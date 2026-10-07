"""Source-only specialist/routing diagnosis of completed iteration 16 states."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .routed_discovery_run import ROOT
from .routed_conditional_utility import load
from .source_probe_penalty_direction import outputs
from .shrinkage import digest, write

DEST = ROOT / 'results/sirm_research_loop/iteration_16'
OUT = DEST / 'source_specialist_routing'

@torch.no_grad()
def replay(m, raw, y, training=False):
    pi = m.router(m.bank_features(raw)).softmax(1)
    g, delta, full = outputs(m, raw, pi, training)
    ce = F.cross_entropy(full, y, reduction='none')
    contribution = pi[:, :6, None] * delta
    removed = full[:, None] - contribution
    u = torch.stack([F.cross_entropy(removed[:, e], y, reduction='none') - ce for e in range(6)], 1)
    # Conditional teacher uses the original unscaled expert risk, not LOO utility.
    raw_ce = torch.stack([F.cross_entropy(g + delta[:, e], y, reduction='none') for e in range(6)], 1)
    raw_gain = F.cross_entropy(g, y, reduction='none')[:, None] - raw_ce
    wrong = full.argmax(1)
    gt_gap = contribution[torch.arange(len(y)), :, y] - contribution[torch.arange(len(y)), :, wrong]
    return dict(pi=pi, g=g, full=full, u=u, raw_gain=raw_gain, gt_gap=gt_gap,
                removal_predictions=removed.argmax(2), global_correct=g.argmax(1).eq(y),
                full_correct=full.argmax(1).eq(y), ce=ce)

def group(r, mask):
    n = int(mask.sum())
    if not n: return dict(n=0)
    gc, fc = r['global_correct'][mask], r['full_correct'][mask]
    pred = r['full'][mask].argmax(1)
    return dict(n=n, global_accuracy=float(gc.float().mean()), full_accuracy=float(fc.float().mean()),
        repairs=int((~gc & fc).sum()), harms=int((gc & ~fc).sum()),
        predicted_class_counts=torch.bincount(pred, minlength=10).tolist(),
        router_mean=r['pi'][mask].mean(0).tolist(),
        marginal_utility_mean=r['u'][mask].mean(0).tolist(),
        unscaled_single_expert_gain=r['raw_gain'][mask].mean(0).tolist(),
        removal_accuracy=[float(r['removal_predictions'][mask, e].eq(Y[mask]).float().mean()) for e in range(6)])

def summaries(r, y, dom, names):
    global Y
    Y = y
    groups = {'all': group(r, torch.ones(len(y), dtype=torch.bool))}
    for c, name in enumerate(names):
        for suffix, extra in [('all', torch.ones(len(y), dtype=torch.bool)),
                              ('global_correct', r['global_correct']), ('global_wrong', ~r['global_correct'])]:
            groups[name + '/' + suffix] = group(r, y.eq(c) & extra)
        for d in [0, 2, 3]: groups[f'{name}/domain{d}'] = group(r, y.eq(c) & dom.eq(d))
    return groups

def run():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'results.json').exists()
    audit = json.loads((DEST / 'final_analysis_audit.json').read_text())
    for rel, expected in audit['sha256'].items(): assert digest(DEST / rel) == expected, rel
    manifest = json.loads((DEST / 'experiment/subset_manifest.json').read_text())
    names = manifest['class_names']
    cache = ROOT / 'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis'
    data, locks = {}, {}
    for split in ['source_in', 'source_out']:
        path = cache / (split + '_tokens.npz')
        a = dict(np.load(path)); locks[str(path)] = digest(path)
        data[split] = (torch.tensor(np.concatenate([a['cls'], a['patch_max']], 1)),
                       torch.tensor(a['y']), torch.tensor(a['domains']), a['ids'])
    assert not set(data['source_in'][3]) & set(data['source_out'][3])
    raw, y, dom, ids = data['source_in']
    id_index = {str(v): i for i, v in enumerate(ids)}
    h2dir = ROOT / 'results/sirm_research_loop/iteration_15/source_conditional_H2'
    h2audit = json.loads((h2dir / 'extraction_audit.json').read_text())
    results = {}
    for state in ['selected', 'last']:
        m, payload, checkpoint = load(DEST, state)
        locks[str(checkpoint)] = digest(checkpoint)
        original = copy.deepcopy(m.state_dict())
        m.calibrate_features(raw, dom)
        state_result = {'checkpoint_step': payload.get('selected_step', payload.get('step')), 'original': {}}
        for split, (x, sy, sd, sids) in data.items():
            r = replay(m, x, sy)
            # Verify CPU replay agrees with previously locked source analysis.
            previous = np.load(DEST / f'source_analysis/OFF_{state}_{split}_replay.npz')
            assert np.array_equal(previous['ids'], sids)
            assert np.array_equal(previous['full_logits'], r['full'].numpy())
            state_result['original'][split] = summaries(r, sy, sd, names)
        paired = {k: [] for k in ['y', 'domains', 'original_pi', 'H2_pi', 'original_u', 'H2_u',
                                  'original_correct', 'H2_correct', 'original_global_correct', 'H2_global_correct']}
        sentinel = []
        for record in h2audit['rows']:
            path = h2dir / f"H2_step{record['step']}.pt"
            assert digest(path) == record['sha256']; locks[str(path)] = record['sha256']
            batch = torch.load(path, map_location='cpu', weights_only=False)
            idx = torch.tensor([id_index[str(v)] for v in batch['ids']])
            assert torch.equal(y[idx], batch['y']) and torch.equal(dom[idx], batch['domains'])
            orig = replay(m, raw[idx], y[idx])
            h2 = replay(m, batch['raw'], batch['y'])
            for k, v in dict(y=batch['y'], domains=batch['domains'], original_pi=orig['pi'], H2_pi=h2['pi'],
                             original_u=orig['u'], H2_u=h2['u'], original_correct=orig['full_correct'],
                             H2_correct=h2['full_correct'], original_global_correct=orig['global_correct'],
                             H2_global_correct=h2['global_correct']).items(): paired[k].append(v)
        p = {k: torch.cat(v) for k, v in paired.items()}
        class_pairs = {}
        for c, name in enumerate(names):
            mask = p['y'].eq(c)
            class_pairs[name] = dict(occurrences=int(mask.sum()),
                original_router_mean=p['original_pi'][mask].mean(0).tolist(), H2_router_mean=p['H2_pi'][mask].mean(0).tolist(),
                paired_router_L1=float((p['original_pi'][mask] - p['H2_pi'][mask]).abs().sum(1).mean()),
                original_utility=p['original_u'][mask].mean(0).tolist(), H2_utility=p['H2_u'][mask].mean(0).tolist(),
                original_accuracy=float(p['original_correct'][mask].float().mean()), H2_accuracy=float(p['H2_correct'][mask].float().mean()),
                original_global_accuracy=float(p['original_global_correct'][mask].float().mean()), H2_global_accuracy=float(p['H2_global_correct'][mask].float().mean()))
        state_result['paired_H2'] = class_pairs
        # Direct source harm decomposition, including all expert removal controls.
        vx, vy, vd, vids = data['source_out']; r = replay(m, vx, vy)
        cases = []
        for i in torch.nonzero(r['global_correct'] & ~r['full_correct']).flatten().tolist():
            cases.append(dict(id=str(vids[i]), label=names[int(vy[i])], prediction=names[int(r['full'][i].argmax())],
                router=r['pi'][i].tolist(), CE_utility=r['u'][i].tolist(),
                GT_minus_wrong_contributions=r['gt_gap'][i].tolist(),
                removal_predictions=[names[int(v)] for v in r['removal_predictions'][i]],
                unscaled_expert_gain=r['raw_gain'][i].tolist()))
        state_result['source_harms'] = cases
        results[state] = state_result
        m.load_state_dict(original)
        assert all(torch.equal(v, m.state_dict()[k]) for k, v in original.items())
        assert all(q.grad is None for q in m.parameters())
    for path, expected in locks.items(): assert digest(Path(path)) == expected
    write(OUT / 'results.json', dict(source_only=True, target_read=False, training=False, fitting=False,
        parameter_updates=False, input_sha256=locks, states=results, class_names=names,
        centering='Source-in calibrated eval centering for BOTH paired original/H2 views; isolates view change from batch centering.',
        limitations='Saved selected/last banks, not historical states. 3840 repeated scheduled occurrences, not independent images. H2 mixes crop/jitter/gray/blur; cannot attribute individual transforms. Class-conditioned measures are diagnostic, not deployable label-aware routing.',
        code_sha256=digest(Path(__file__))))
    (OUT / 'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for path in OUT.iterdir(): audit['sha256'][str(path.relative_to(DEST))] = digest(path)
    write(DEST / 'final_analysis_audit.json', audit)
    print('Source specialist audit complete: two states, 40 paired H2 batches each; target untouched.')

if __name__ == '__main__': run()

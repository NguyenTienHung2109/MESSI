"""Render 4x4 current failures from locked replays; no inference or training."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from . import failed_case_grids as grids
from .shrinkage import digest, write

def run():
    root = grids.ROOT / 'results/sirm_research_loop/iteration_16'
    audit_path = root / 'final_analysis_audit.json'
    audit = json.loads(audit_path.read_text())
    for rel, expected in audit['sha256'].items():
        assert digest(root / rel) == expected, rel
    grids.OUT = root / 'failed_case_grids'
    grids.OUT.mkdir(exist_ok=True)
    manifest = json.loads((root / 'experiment/subset_manifest.json').read_text())
    names = manifest['class_names']
    hashes = {r['id']: r['sha256'] for s in manifest['sources'].values() for r in s['images']}
    index = {}
    for split, label in [('source_out', 'Source validation'), ('target_in', 'Target L38 test')]:
        path = root / 'eda' / (split + '_replay.npz')
        a = dict(np.load(path))
        assert np.array_equal(a['B1'], a['B2'])
        old, new = a['B0'].argmax(1), a['B2'].argmax(1)
        oc, nc = softmax(a['B0'], axis=1).max(1), softmax(a['B2'], axis=1).max(1)
        selected = grids.select(a['ids'], a['y'], old, new, nc, split == 'source_out')
        item = grids.draw(split + '_4x4', label + ' | iteration 16, 10% data | OFF = ON; SIRM inactive',
            a['ids'], a['y'], old, new, oc, nc, names, selected, 'ERM global', 'Residual bank', hashes)
        item.update(replay_sha256=digest(path), total_images=len(old),
                    failures=int((new != a['y']).sum()),
                    harms=int(((old == a['y']) & (new != a['y'])).sum()),
                    repairs=int(((old != a['y']) & (new == a['y'])).sum()))
        index[split] = item
    write(grids.OUT / 'index.json', dict(grids=index, training=False, inference=False,
        selection='All new harms first; source failed birds next; then class round-robin highest-confidence failures. Not a prevalence sample.',
        code_sha256=digest(Path(__file__))))
    (grids.OUT / 'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for p in grids.OUT.iterdir():
        if p.is_file(): audit['sha256'][str(p.relative_to(root))] = digest(p)
    audit['reason'] = 'Replay, source/target case interpretation and requested 4x4 failure grids complete; scientific goal remains unachieved.'
    write(audit_path, audit)
    print(json.dumps({k: {x: v[x] for x in ['file', 'failures', 'harms', 'repairs']} for k, v in index.items()}, indent=2))

if __name__ == '__main__':
    run()

"""Source-only class-block domain coverage from the exact training schedule."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from domainbed.scripts.train_subset_irm_pacs import make_schedule
from .shrinkage import ROOT, digest, write


def run(dest, arm):
    plan=json.loads((dest/'plan.json').read_text()); bank=dest/'experiment/runs/L38'/arm
    for group in ['base_code_hashes','variant_code_hashes']:
        assert all(digest(ROOT/p)==h for p,h in plan[group].items())
    manifest_path=dest/'experiment/subset_manifest.json'
    manifest=json.loads(manifest_path.read_text()); lock=json.loads((bank/'source_schedules.json').read_text())
    schedules={d:make_schedule(list(range(len(lock[str(d)]['split_keys']))),400,32,0,d) for d in [0,2,3]}
    assert all(hashlib.sha256(schedules[d].tobytes()).hexdigest()==lock[str(d)]['schedule_sha256'] for d in schedules)
    # Read complete lines only if the training writer is currently appending.
    data=(bank/'diagnostics.jsonl').read_bytes(); lines=data[:data.rfind(b'\n')+1].splitlines()
    logs={r['step']:r for r in map(json.loads,lines)}
    rows=[]
    for step in range(400):
        counts={}
        for d in [0,2,3]:
            labels=[manifest['sources'][str(d)]['images'][lock[str(d)]['split_keys'][int(i)]]['class'] for i in schedules[d][step]]
            counts[d]=Counter(labels)
        labels=sorted(set().union(*(set(v) for v in counts.values())))
        observed={b['label']:b for b in logs[step]['discovery_class_blocks']} if step in logs else None
        if observed is not None: assert set(observed)==set(labels)
        for label in labels:
            ns=[counts[d][label] for d in [0,2,3]]; n=sum(ns)
            if observed is not None: assert observed[label]['count']==n
            rows.append(dict(step=step,stage='discover' if step<200 else 'invariance',label=label,
                             source_counts=dict(zip([0,2,3],ns)),n=n,
                             represented_domains=sum(v>0 for v in ns),
                             domains_with_minimum_two=sum(v>=2 for v in ns),
                             maximum_domain_fraction=max(ns)/n,
                             observed=observed is not None,
                             winners=observed[label]['winners'] if observed is not None else None))
    summaries={}
    for stage in ['all','discover','invariance']:
        group=[r for r in rows if stage=='all' or r['stage']==stage]
        summary=dict(blocks=len(group),image_occurrences=sum(r['n'] for r in group))
        for name,condition in [('at_least_two_domains',lambda r:r['represented_domains']>=2),
                               ('minimum_two_images_in_two_domains',lambda r:r['domains_with_minimum_two']>=2),
                               ('single_domain_only',lambda r:r['represented_domains']==1)]:
            summary[name+'_blocks']=sum(condition(r) for r in group)
            summary[name+'_block_fraction']=sum(condition(r) for r in group)/len(group)
            summary[name+'_image_weight_fraction']=sum(r['n'] for r in group if condition(r))/summary['image_occurrences']
        summary['mean_maximum_domain_fraction']=float(np.mean([r['maximum_domain_fraction'] for r in group]))
        summaries[stage]=summary
    result=dict(arm=arm,code_sha256=digest(Path(__file__)),schedule_lock_sha256=digest(bank/'source_schedules.json'),
                manifest_sha256=digest(manifest_path),target_read=False,training=False,fitting=False,
                observed_through_step=max(logs),all_observed_class_labels_and_counts_match=True,
                scope='Exact fixed400 source schedule. Future rows describe only planned source label/domain coverage, not future learned assignments. '
                      'Cross-domain class grouping is not equal-domain weighting within each class; majority domains can dominate.',
                summaries=summaries,rows=rows,scientific_goal_achieved=False)
    write(dest/f'{arm}_class_coverage_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);p.add_argument('arm',choices=['OFF','ON'])
    a=p.parse_args();run(a.destination,a.arm)

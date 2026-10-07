"""Fixed per-class subsets of the original DomainBed splits."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from domainbed.lib import misc
from experiments.sirm_discover_gate.data import Adapter

ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'subsets/terra_10pct_seed0/manifest.json'
RESULT=ROOT/'results/terra_10pct_factor_diagnosis'

def build():
    adapter=Adapter('TerraIncognita')
    full=adapter.inventory(ROOT/'domainbed/data',range(4),0,{})
    manifest=dict(full,protocol='DomainBed80_20_fixed_stratified_10pct',fraction=.1,rounding='Python round; min(n, max(2, round(0.1*n))) for nonempty classes',roles='in = source train OR held-out target test; out = source validation OR held-out target validation. Roles change by target; IDs never resample.')
    counts=[]
    for env,entry in manifest['sources'].items():
        for split in ['in','out']:
            original=list(entry[split]);entry['full_'+split]=original;chosen=[]
            rng=np.random.RandomState(misc.seed_hash(0,int(env),split,'terra_10pct'))
            for c,name in enumerate(manifest['class_names']):
                ids=[i for i in original if entry['images'][i]['class']==c]
                k=min(len(ids),max(2,round(.1*len(ids)))) if ids else 0
                chosen.extend(rng.choice(ids,k,replace=False).tolist() if k else [])
                counts.append(dict(environment=adapter.spec['domains'][int(env)],split=split,**{'class':name},full_count=len(ids),subset_count=k,subset_fraction=k/len(ids) if ids else 0,tiny_class=len(ids)<20,rounding_delta=k-.1*len(ids)))
            entry[split]=sorted(chosen)
            entry[split+'_ids']=[entry['images'][i]['id'] for i in entry[split]]
    verify(manifest)
    MANIFEST.parent.mkdir(parents=True,exist_ok=True);RESULT.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(manifest,indent=2,sort_keys=True)+'\n';MANIFEST.write_text(raw);(RESULT/'subset_manifest.json').write_text(raw)
    for path in [MANIFEST.with_name('counts.csv'),RESULT/'subset_counts.csv']:
        with path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(counts[0]));w.writeheader();w.writerows(counts)
    report=dict(verified=True,sha256=hashlib.sha256(raw.encode()).hexdigest(),full_total=sum(r['full_count'] for r in counts),subset_total=sum(r['subset_count'] for r in counts),tiny_classes=[r for r in counts if r['tiny_class']],per_environment={e:{s:len(v[s]) for s in ['in','out']} for e,v in manifest['sources'].items()},all_targets_and_methods_share_manifest=True,target_training_overlap=0)
    (RESULT/'subset_verification.json').write_text(json.dumps(report,indent=2)+'\n');return report

def verify(m):
    all_ids=[]
    for env,e in m['sources'].items():
        assert not set(e['in'])&set(e['out'])
        for s in ['in','out']:
            assert len(e[s])==len(set(e[s])) and set(e[s])<=set(e['full_'+s])
            assert e[s+'_ids']==[e['images'][i]['id'] for i in e[s]]
            for c in range(len(m['class_names'])):
                n=sum(e['images'][i]['class']==c for i in e['full_'+s]);k=sum(e['images'][i]['class']==c for i in e[s])
                assert k==(min(n,max(2,round(.1*n))) if n else 0)
            all_ids+=e[s+'_ids']
    assert len(all_ids)==len(set(all_ids))

def apply(env,index,args,hparams):
    m=json.loads(Path(hparams['diagnostic_subset_manifest']).read_text());verify(m)
    assert args.seed==args.trial_seed==0 and args.holdout_fraction==.2 and args.uda_holdout_fraction==0
    e=m['sources'][str(index)]
    original_out,original_in=misc.split_dataset(env,int(len(env)*.2),misc.seed_hash(0,index))
    assert set(original_in.keys)==set(e['full_in']) and set(original_out.keys)==set(e['full_out'])
    samples=env.samples
    actual=[(str(Path(p).relative_to(ROOT/'domainbed/data/terra_incognita')),y) for p,y in samples]
    assert actual==[(r['id'],r['class']) for r in e['images']]
    return misc._SplitDataset(env,e['out']),misc._SplitDataset(env,e['in'])

if __name__=='__main__':print(json.dumps(build(),indent=2))

"""Class/domain and official-event stratified source correction EDA; no fitting."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .shrinkage import ROOT,digest,write


def measure(a,name,mask):
    y=torch.tensor(a['y'][mask]);g=torch.tensor(a[name+'_g'][mask]);full=torch.tensor(a[name+'_full'][mask]);delta=torch.tensor(a[name+'_delta'][mask]);pi=torch.tensor(a[name+'_pi'][mask]);n=len(y)
    if not n:return None
    ce=F.cross_entropy(full,y,reduction='none');base=F.cross_entropy(g,y,reduction='none');gp=g.argmax(1);fp=full.argmax(1)
    contrib=pi[:,:6,None]*delta;u=torch.stack([F.cross_entropy(full-contrib[:,j],y,reduction='none')-ce for j in range(6)],1)
    competitor=g.masked_fill(F.one_hot(y,10).bool(),-torch.inf).argmax(1);idx=torch.arange(n);correction=full-g
    return dict(n=n,global_correct=int(gp.eq(y).sum()),full_correct=int(fp.eq(y).sum()),repairs=int((gp.ne(y)&fp.eq(y)).sum()),harms=int((gp.eq(y)&fp.ne(y)).sum()),
        global_CE=float(base.mean()),full_CE=float(ce.mean()),CE_gain=float((base-ce).mean()),marginal_U=u.mean(0).tolist(),
        mean_true_vs_fixed_competitor_margin_correction=float((correction[idx,y]-correction[idx,competitor]).mean()),
        expert_mean_routing=pi[:,:6].mean(0).tolist(),NULL=float(pi[:,-1].mean()),correctness_gain_images=int(fp.eq(y).sum()-gp.eq(y).sum()))


def main():
    torch.set_num_threads(4);root=ROOT/'results/sirm_research_loop/iteration_12';out=root/'source_scene_utility_diagnosis';out.mkdir(exist_ok=True)
    mp=ROOT/'results/sirm_research_loop/iteration_08/source_annotation_audit/selected_source_results.json';metadata=json.loads(mp.read_text());lookup={r['local_id']:r for r in metadata['records']}
    cache={s:dict(np.load(root/'source_comparison'/f'{s}_replay.npz')) for s in ['source_in','source_out']}
    assert not set(cache['source_in']['ids'])&set(cache['source_out']['ids'])
    for s,a in cache.items():
        for image,label,domain in zip(a['ids'],a['y'],a['domains']):
            r=lookup[str(image)];assert r['split']==s and r['environment']==int(domain) and r['found'] and r['label_matches'] and r['domain_matches'] and r['image_content_matches']
    train_sequences={lookup[str(i)]['official_sequence'] for i in cache['source_in']['ids']};assert None not in train_sequences
    rows=[];changed=[]
    names=['11_selected','12_selected','12_last']
    for split,a in cache.items():
        sequences=np.array([lookup[str(i)]['official_sequence'] for i in a['ids']]);assert not any(v is None for v in sequences)
        seen=np.array([v in train_sequences for v in sequences])
        groups=[('all',None,np.ones(len(a['y']),dtype=bool))]
        for d in [0,2,3]:
            for y in np.unique(a['y'][a['domains']==d]):groups.append(('class_domain',{'domain':d,'label':int(y)},(a['domains']==d)&(a['y']==y)))
        if split=='source_out':
            for value in [True,False]:
                groups.append(('source_in_sequence_overlap',{'seen_in_source_in':value},seen==value))
                for d in [0,2,3]:
                    for y in np.unique(a['y'][(a['domains']==d)&(seen==value)]):groups.append(('class_domain_sequence_overlap',{'domain':d,'label':int(y),'seen_in_source_in':value},(a['domains']==d)&(a['y']==y)&(seen==value)))
        for scope,key,mask in groups:
            rows.append(dict(split=split,scope=scope,key=key,unique_sequences=len(set(sequences[mask])),metrics={name:measure(a,name,mask) for name in names}))
        for name in names:
            gp=a[name+'_g'].argmax(1);fp=a[name+'_full'].argmax(1);change=(gp==a['y'])!=(fp==a['y'])
            for i in np.flatnonzero(change):
                changed.append(dict(split=split,checkpoint=name,image_id=str(a['ids'][i]),label=int(a['y'][i]),domain=int(a['domains'][i]),
                                    global_pred=int(gp[i]),full_pred=int(fp[i]),repair=bool(fp[i]==a['y'][i]),
                                    official_sequence=sequences[i],source_in_sequence_seen=bool(seen[i]),official_datetime=lookup[str(a['ids'][i])]['official_datetime']))
    sourceoutseen=sum(lookup[str(i)]['official_sequence'] in train_sequences for i in cache['source_out']['ids'])
    result=dict(training=False,fitting=False,target_read=False,selection_changes=False,relabeling=False,metadata_sha256=digest(mp),
        source_replay_hashes={s:digest(root/'source_comparison'/f'{s}_replay.npz') for s in cache},source_out_sequence_seen_images=sourceoutseen,
        scope='Fixed official source-in/out image split. Event-overlap strata describe a previously trained bank; removing overlap at evaluation is not sequence-held-out training. Class-domain groups below n2 are retained explicitly and must not support reliability claims. Sequence is official event ID, not an independent physical camera ID or causal background intervention.',groups=rows,changed_cases=changed)
    assert digest(mp)==result['metadata_sha256'];write(out/'results.json',result)
    print(json.dumps({'source_out_sequence_seen_images':sourceoutseen,'sourceout_groups':[r for r in rows if r['split']=='source_out' and r['scope'] in ['all','source_in_sequence_overlap']],'changed_source_out':[r for r in changed if r['split']=='source_out']},indent=2))

if __name__=='__main__':main()

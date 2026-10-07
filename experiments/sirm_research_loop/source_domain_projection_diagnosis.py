"""Source-in conditional-domain centroid projection diagnostic on fixed bank."""
import json
from pathlib import Path
import numpy as np
import torch
from .fixed_teacher_probe import score
from .shrinkage import digest,write
B=Path('results/sirm_research_loop/iteration_08');out=B/'source_domain_projection_diagnosis';out.mkdir(exist_ok=True);paths={s:B/f'source_router_diagnosis/{s}_original.npz' for s in ['source_in','source_out']};plan={'code_sha256':digest(Path(__file__)),'cache_hashes':{s:digest(p) for s,p in paths.items()},'hypothesis':'Source nearest-neighbor geometry is strongly domain-associated even conditional on class. Removing source-in within-class cross-domain centroid span may improve label-free teacher transfer.','recipe':'Source-in only: center features by pooled train mean. Stack class-domain means minus equal-domain class mean, all classes with>=2 images in all3domains. SVD full numerical rank via standard eps*max(shape)*s_max tolerance. Remove entire nonzero span; no rank/hparam selection. Compare raw cosine, centered cosine, centered projected cosine. Same fixed1NN teacher policy.','scope':'Diagnostic feature map used only for retrieval, not changing global/backbone/residuals. Projection may remove useful class information; no causal/domain-information-removal claim. Source-out labels evaluate only. No target, grid or primary training.'};write(out/'plan.json',plan);a={s:dict(np.load(p)) for s,p in paths.items()};t=a['source_in'];v=a['source_out'];x=t['z'].astype(float);mean=x.mean(0);ds=np.unique(t['domains']);vectors=[];included=[]
for c in np.unique(t['y']):
 masks=[(t['domains']==d)&(t['y']==c) for d in ds]
 if any(m.sum()<2 for m in masks):continue
 means=np.stack([x[m].mean(0) for m in masks]);vectors.extend(means-means.mean(0));included.append(int(c))
_,sing,V=np.linalg.svd(np.array(vectors),full_matrices=False);tol=np.finfo(float).eps*max(np.array(vectors).shape)*sing[0];rank=int((sing>tol).sum());basis=V[:rank];np.savez_compressed(out/'source_projection.npz',source_mean=mean,basis=basis,singular_values=sing);results={};arrays={k:torch.tensor(v[k]) for k in ['g','delta','y','domains','pi','teacher']}
for mode in ['raw','centered','centered_projected']:
 z={s:r['z'].astype(float).copy() for s,r in a.items()}
 if mode!='raw':z={s:r-mean for s,r in z.items()}
 if mode=='centered_projected':z={s:r-(r@basis.T)@basis for s,r in z.items()}
 assert all(np.linalg.norm(r,axis=1).min()>1e-8 for r in z.values());z={s:r/np.linalg.norm(r,axis=1,keepdims=True) for s,r in z.items()};sim=z['source_out']@z['source_in'].T;idx=sim.argmax(1);pi=torch.tensor(t['teacher'][idx]);r=score(arrays,pi);wrong=v['g'].argmax(1)!=v['y'];same=t['y'][idx]==v['y'];r['neighbor_geometry']={'same_class':float(same.mean()),'same_class_global_wrong':float(same[wrong].mean()),'same_domain':float((t['domains'][idx]==v['domains']).mean())};results[mode]=r;np.savez_compressed(out/f'{mode}_policies.npz',pi=pi.numpy(),neighbor_index=idx,ids=v['ids'],y=v['y'],domains=v['domains']);print(mode,r['equal_domain'],r['pooled'],r['neighbor_geometry'],flush=True)
assert all(digest(p)==plan['cache_hashes'][s] for s,p in paths.items());write(out/'results.json',{'plan':plan,'projection_rank':rank,'included_classes':included,'numerical_rank_tolerance':tol,'policies':results});print('rank',rank)

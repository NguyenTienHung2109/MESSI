"""Fixed source-only learned local/global fusion, plus same-sized logit controls."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .terra_attention_eda import OUT
from .terra_object_pool_readout import score
from .shrinkage import digest,write
DEST=OUT/'source_object_probe/local_fusion_control'
SOURCE=OUT/'source_object_probe/local_class_hierarchical_uniform'

def run():
 torch.set_num_threads(4);torch.manual_seed(0);DEST.mkdir(exist_ok=True);assert not (DEST/'plan.json').exists();path=SOURCE/'all_image_replay/pooled_replay.pt';data=torch.load(path,map_location='cpu',weights_only=False)
 plan=dict(hypothesis='Hard-negative factorized local evidence remains harmful under direct addition. Learned source-only fusion may retain complementary local predictions while preserving correct global decisions.',source_only=True,target_read=False,not_SIRM=True,annotation_assisted=True,extra_probe_training='Local head already fitted400 updates; fusion adds400, not a total400-update SIRM experiment.',source_input_sha256=digest(path),training='Linear20->10 residual, zero initialization, frozen global/local logits,400 fixed updates, Adam.001,L2.01, no validation selection, train-only feature normalization/sqrt2',controls='Global+local; Global repeated; Global+image-hash-permuted-local. Equal parameters and same32/domain/update replacement schedule. Permutation preserves within-domain local marginal scores but destroys image pairing.',evaluation='Center correction by equal-domain source-in mean, report all304 source-out and repairs/harms. No inference boxes/labels.',code_sha256=digest(Path(__file__)));write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());raw={};permutations={}
 for split,a in data.items():
  g=a['global_logits'];q=a['standalone_logits'];perm=torch.arange(len(g));d=a['domains']
  for dom in sorted(d.unique().tolist()):
   ix=torch.nonzero(d.eq(dom)).flatten().tolist();ordered=sorted(ix,key=lambda i:hashlib.sha256(a['ids'][i].encode()).hexdigest());shifted=ordered[1:]+ordered[:1]
   for i,j in zip(ordered,shifted):perm[i]=j
  assert not perm.eq(torch.arange(len(perm))).any();permutations[split]=perm.tolist();raw[split]={'global_local':torch.cat([g,q],1),'global_repeat':torch.cat([g,g],1),'global_permuted_local':torch.cat([g,q[perm]],1)}
 gen=torch.Generator().manual_seed(0);d=data['source_in']['domains'];domains=sorted(d.unique().tolist());cohorts={dom:torch.nonzero(d.eq(dom)).flatten() for dom in domains};schedule=[torch.cat([cohorts[dom][torch.randint(len(cohorts[dom]),(32,),generator=gen)] for dom in domains]) for _ in range(400)];schedule_hash=hashlib.sha256(torch.stack(schedule).numpy().tobytes()).hexdigest();states={}
 for name in raw['source_in']:
  train=data['source_in'];r=raw['source_in'][name];mean=r.mean(0);std=r.std(0,unbiased=False).clamp_min(1e-6);x=(r-mean)/std/np.sqrt(2);head=torch.nn.Linear(20,10);head.weight.data.zero_();head.bias.data.zero_();opt=torch.optim.Adam(head.parameters(),lr=.001)
  for ix in schedule:
   f=train['global_logits'][ix]+head(x[ix]);loss=F.cross_entropy(f,train['labels'][ix])+.01*head.weight.square().sum();opt.zero_grad();loss.backward();opt.step()
  with torch.no_grad():
   z=head(x);center=torch.stack([z[d.eq(dom)].mean(0) for dom in domains]).mean(0);splits={};replays={}
   for split,a in data.items():
    f=a['global_logits']+head((raw[split][name]-mean)/std/np.sqrt(2))-center;y=a['labels'];dd=a['domains'];s=score(a['global_logits'],f,y,dd);splits[split]=s;replays[split]=dict(ids=a['ids'],labels=y,domains=dd,global_logits=a['global_logits'],logits=f)
   p=DEST/f'{name}.pt';torch.save(dict(weight=head.weight,bias=head.bias,mean=mean,std=std,center=center,replays=replays),p);states[name]=dict(splits=splits,model_sha256=digest(p));print(name,{split:{k:s[k] for k in ['accuracy_equal_domain','CE_equal_domain','correct','repairs','harms']} for split,s in splits.items()},flush=True)
 assert digest(path)==plan['source_input_sha256'];write(DEST/'results.json',dict(states=states,source_only=True,target_read=False,not_SIRM=True,schedule_sha256=schedule_hash,permutations=permutations));print('complete',flush=True)
if __name__=='__main__':run()

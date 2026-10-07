"""Matched source-only linear residual controls: CLS, max, objectness pooling."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from .terra_attention_eda import OUT
from .terra_source_object_probe import CACHE,DEST
from .terra_object_pool_source_cache import RESULT
from .shrinkage import digest,write

OUTDIR=DEST/'source_readout_controls'

def score(g,f,y,dom):
 gc=g.argmax(1).eq(y);fc=f.argmax(1).eq(y);ds=sorted(dom.unique().tolist());return dict(accuracy_equal_domain=float(torch.stack([fc[dom.eq(d)].float().mean() for d in ds]).mean()),CE_equal_domain=float(torch.stack([F.cross_entropy(f[dom.eq(d)],y[dom.eq(d)]) for d in ds]).mean()),correct=int(fc.sum()),repairs=int((~gc&fc).sum()),harms=int((gc&~fc).sum()),global_correct=int(gc.sum()),predictions=f.argmax(1).tolist(),global_predictions=g.argmax(1).tolist())

def run():
 torch.set_num_threads(4);OUTDIR.mkdir(exist_ok=True);assert not (OUTDIR/'plan.json').exists();data={};locks={}
 for s in ['source_in','source_out']:
  path=RESULT/f'{s}_features.pt';locks[str(path)]=digest(path);p=torch.load(path,map_location='cpu',weights_only=False);orig=dict(np.load(CACHE/f'{s}_tokens.npz'));assert np.array_equal(np.array(p['ids']),orig['ids']);assert np.array_equal(p['labels'].numpy(),orig['y']);cls=p['features'][:,:384];data[s]=dict(raw={'CLS_repeat':torch.cat([cls,cls],1),'CLS_patch_max':torch.cat([cls,torch.tensor(orig['patch_max'])],1),'CLS_object_pool':p['features']},g=p['global_logits'],y=p['labels'],d=p['domains'],ids=p['ids']);assert p['anchor_sha256']=='53ab066e3eb0cc1d0f59d7b3dad68f295bf0ce6b69439f1b5197ad0dc30d5bb8'
 write(OUTDIR/'plan.json',dict(hypothesis='Deployable annotation-trained class-agnostic pooling improves source residual repair versus CLS and max pooling, beyond localization metrics.',diagnostic=True,not_SIRM=True,target_read=False,training_global=False,initialization='All residual heads zero, exact frozen global logits',architecture='Linear768->10 residual; fixed train-only normalization each feature dimension and /sqrt2',optimizer='Adam lr .001, 400 updates, explicit L2 .01, fixed last update, no validation/hyperparameter selection',schedule='Same seed0 replacement source-in schedule32 images per domain per update for all3 controls',centering='Eval residual logit mean over source-in, equal-domain average; excludes classifier prior from correction',source_in_object_probe_used=True,extra_box_supervision=True,input_sha256=locks,code_sha256=digest(Path(__file__))))
 (OUTDIR/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());results={};torch.manual_seed(0);gen=torch.Generator().manual_seed(0);d=data['source_in']['d'];domains=sorted(d.unique().tolist());ix={a:torch.nonzero(d.eq(a)).flatten() for a in domains};schedules=[torch.cat([ix[a][torch.randint(len(ix[a]),(32,),generator=gen)] for a in domains]) for _ in range(400)]
 for name in data['source_in']['raw']:
  raw=data['source_in']['raw'][name];mean=raw.mean(0);std=raw.std(0,unbiased=False).clamp_min(1e-8);z=(raw-mean)/std/(2**.5);head=torch.nn.Linear(768,10);torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias);opt=torch.optim.Adam(head.parameters(),lr=.001);initial=None
  for step,batch in enumerate(schedules):
   f=data['source_in']['g'][batch]+head(z[batch]);loss=F.cross_entropy(f,data['source_in']['y'][batch])+.01*head.weight.square().sum();opt.zero_grad();loss.backward();opt.step()
   if initial is None:initial=float(loss)
  splits={};replays={}
  with torch.no_grad():
   center=torch.stack([head(z[d.eq(a)]).mean(0) for a in domains]).mean(0)
   for s,a in data.items():
    zz=(a['raw'][name]-mean)/std/(2**.5);f=a['g']+head(zz)-center;splits[s]=score(a['g'],f,a['y'],a['d']);splits[s]['global']=score(a['g'],a['g'],a['y'],a['d']);replays[s]=dict(logits=f,global_logits=a['g'],ids=a['ids'],labels=a['y'],domains=a['d'])
  path=OUTDIR/f'{name}.pt';torch.save(dict(weight=head.weight.detach(),bias=head.bias.detach(),center=center,feature_mean=mean,feature_std=std,replays=replays),path);results[name]=dict(loss_initial=initial,loss_last=float(loss),splits=splits,reader_sha256=digest(path));print(name,{s:{k:v[k] for k in ['accuracy_equal_domain','correct','repairs','harms']} for s,v in splits.items()},flush=True)
 for p,h in locks.items():assert digest(Path(p))==h
 write(OUTDIR/'results.json',dict(states=results,source_only=True,target_read=False,global_logits_unchanged=True,not_SIRM=True,extra_source_box_supervision=True,input_sha256=locks))
if __name__=='__main__':run()

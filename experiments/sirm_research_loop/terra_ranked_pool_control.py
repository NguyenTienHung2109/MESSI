"""Fixed-K source-only foreground-ranked patch readout controls."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .terra_attention_eda import ROOT,OUT,load_models
from .terra_source_object_probe import DEST,CACHE
from .terra_object_pool_source_cache import RESULT
from .terra_object_pool_readout import score
from experiments.sirm_h1_h6.run import ViewTransform
from .shrinkage import digest,write

OUTPUT=DEST/'foreground_ranked_max'
K=8

def run():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;OUTPUT.mkdir(exist_ok=True);assert not (OUTPUT/'plan.json').exists();probe_path=DEST/'pooled_probe.pt';probe=torch.load(probe_path,map_location='cpu',weights_only=False);old_path=DEST/'source_readout_controls/results.json';old=json.loads(old_path.read_text());locks={str(probe_path):digest(probe_path),str(old_path):digest(old_path)}
 write(OUTPUT/'plan.json',dict(hypothesis='Foreground-weighted averaging may discard sparse discriminative cues. Fixed top8 patch max tests sparse content-preserving readout.',K=K,K_reason='Fixed before results; about4% of196 patches, no K search',controls=['all_patch_max reproduction','foreground_top8_max','foreground_top8_mean','spatial_prior_top8_max','image_hash_random8_max'],reader='Identical zero linear768->10, train-only mean/std /sqrt2; Adam .001,400 updates,L2 .01; equal-source-domain source-in mean-logit centering at evaluation',source_schedule='Seed0 replacement32 per domain, exact prior400 minibatches',foreground_probe='Frozen source-in box-supervised linear384->1; no class/box input at inference',spatial_prior='Train-only mean bbox occupancy; fixed8 positions independent of image',random='Per-image SHA256-seeded fixed8 uniformly chosen positions',fitting=True,target_read=False,checkpoint_changes=False,global_frozen=True,source_only=True,precision='TF32 disabled; extracted patch maxima must agree with locked original cache within1e-4; old max reader must reproduce exactly',selection='Fixed last update; no model/epoch/hparam selection',input_sha256=locks,code_sha256=digest(Path(__file__))))
 (OUTPUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());current,_,_,model_locks=load_models();before=[p._version for p in current.parameters()];transform=ViewTransform(current.config,'original',False);partial_train=torch.load(DEST/'source_in_patches.pt',map_location='cpu',weights_only=False);prior_ix=partial_train['occupancy'].mean(0).topk(K).indices;data={};spatial={}
 with torch.no_grad():
  for split in ['source_in','source_out']:
   orig_path=CACHE/f'{split}_tokens.npz';locks[str(orig_path)]=digest(orig_path);orig=dict(np.load(orig_path));pool=torch.load(RESULT/f'{split}_features.pt',map_location='cpu',weights_only=False);partial=torch.load(DEST/f'{split}_patches.pt',map_location='cpu',weights_only=False);byid={id:i for i,id in enumerate(partial['ids'])};raw={k:[] for k in ['all_patch_max','foreground_top8_max','foreground_top8_mean','spatial_prior_top8_max','image_hash_random8_max']};indices=[]
   for start in range(0,len(orig['ids']),32):
    ids=[str(x) for x in orig['ids'][start:start+32]];missing=[id for id in ids if id not in byid];found={}
    if missing:
     images=torch.stack([transform(Image.open(ROOT/'domainbed/data/terra_incognita'/id).convert('RGB')) for id in missing]).cuda();cap=[];hook=current.featurizer.vit.norm.register_forward_hook(lambda m,a,z:cap.append(z))
     try:current.featurizer(images)
     finally:hook.remove()
     found={id:z[2:].cpu() for id,z in zip(missing,cap[0])}
    patches=torch.stack([partial['features'][byid[id]] if id in byid else found[id] for id in ids]);score_logits=(((patches-probe['mean'])/probe['std'])@probe['weight'].T+probe['bias']).squeeze(-1);rank=score_logits.topk(K,dim=1).indices;selected=patches.gather(1,rank[:,:,None].expand(-1,-1,384));cls=torch.tensor(orig['cls'][start:start+len(ids)]);allmax=torch.tensor(orig['patch_max'][start:start+len(ids)]);assert float((patches.amax(1)-allmax).abs().max())<1e-4
    rand_ix=torch.tensor([np.random.default_rng(int(hashlib.sha256(id.encode()).hexdigest()[:16],16)).choice(196,K,replace=False) for id in ids]);random=patches.gather(1,rand_ix[:,:,None].expand(-1,-1,384)).amax(1)
    pieces=dict(all_patch_max=allmax,foreground_top8_max=selected.amax(1),foreground_top8_mean=selected.mean(1),spatial_prior_top8_max=patches[:,prior_ix].amax(1),image_hash_random8_max=random)
    for name,features in pieces.items():raw[name].append(torch.cat([cls,features],1))
    indices.append(rank)
   data[split]=dict(raw={k:torch.cat(v) for k,v in raw.items()},g=pool['global_logits'],y=pool['labels'],d=pool['domains'],ids=pool['ids']);spatial[split]=torch.cat(indices)
   torch.save(dict(raw=data[split]['raw'],ids=pool['ids'],labels=pool['labels'],domains=pool['domains'],global_logits=pool['global_logits'],top8_indices=spatial[split],prior_indices=prior_ix,anchor_sha256=pool['anchor_sha256'],probe_sha256=digest(probe_path),target_read=False,label_conditioned=False,view='original'),OUTPUT/f'{split}_features.pt');print('cache',split,len(pool['ids']),flush=True)
 assert before==[p._version for p in current.parameters()] and all(p.grad is None for p in current.parameters());assert all(not p.requires_grad for p in current.parameters())
 torch.manual_seed(0);gen=torch.Generator().manual_seed(0);d=data['source_in']['d'];domains=sorted(d.unique().tolist());ix={a:torch.nonzero(d.eq(a)).flatten() for a in domains};assert all(len(v)>=2 for v in ix.values());schedule=[torch.cat([ix[a][torch.randint(len(ix[a]),(32,),generator=gen)] for a in domains]) for _ in range(400)];schedule_sha256=hashlib.sha256(torch.stack(schedule).numpy().tobytes()).hexdigest();results={}
 for name,raw in data['source_in']['raw'].items():
  mean=raw.mean(0);std=raw.std(0,unbiased=False).clamp_min(1e-8);z=(raw-mean)/std/(2**.5);head=torch.nn.Linear(768,10);torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias);opt=torch.optim.Adam(head.parameters(),lr=.001)
  for batch in schedule:
   f=data['source_in']['g'][batch]+head(z[batch]);loss=F.cross_entropy(f,data['source_in']['y'][batch])+.01*head.weight.square().sum();opt.zero_grad();loss.backward();opt.step()
  splits={};replays={}
  with torch.no_grad():
   center=torch.stack([head(z[d.eq(a)]).mean(0) for a in domains]).mean(0)
   for s,a in data.items():
    f=a['g']+head((a['raw'][name]-mean)/std/(2**.5))-center;splits[s]=score(a['g'],f,a['y'],a['d']);replays[s]=dict(logits=f,global_logits=a['g'],ids=a['ids'],labels=a['y'],domains=a['d'])
   if name=='all_patch_max':
    for s in splits:
     for key in ['accuracy_equal_domain','CE_equal_domain','correct','repairs','harms','predictions']:assert splits[s][key]==old['states']['CLS_patch_max']['splits'][s][key],(s,key)
  path=OUTPUT/f'{name}_reader.pt';torch.save(dict(weight=head.weight.detach(),bias=head.bias.detach(),center=center,mean=mean,std=std,replays=replays),path);results[name]=dict(splits=splits,reader_sha256=digest(path),loss_final=float(loss));print(name,{s:{k:v[k] for k in ['accuracy_equal_domain','correct','repairs','harms']} for s,v in splits.items()},flush=True)
 for path,h in {**locks,**model_locks}.items():assert digest(Path(path))==h
 write(OUTPUT/'results.json',dict(states=results,source_only=True,target_read=False,global_parameters_unchanged=True,prior_max_control_exactly_reproduced=True,not_SIRM=True,input_sha256={**locks,**model_locks},schedule_sha256=schedule_sha256));print('complete',flush=True)
if __name__=='__main__':run()

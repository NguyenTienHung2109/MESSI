"""Source-only paired full-image blur residual readout, no inference boxes."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .terra_attention_eda import ROOT,OUT,load_models
from .terra_source_object_probe import CACHE
from .terra_object_pool_readout import score
from .shrinkage import write,digest
from experiments.sirm_h1_h6.run import ViewTransform
DEST=OUT/'source_object_probe/blur_residual_control'

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;DEST.mkdir(exist_ok=True);assert not (DEST/'plan.json').exists();write(DEST/'plan.json',dict(hypothesis='Fixed ROI readers have blur-sensitive classification errors, but oracle crop benefits do not transfer. Full-image paired residual training may discourage residual dependence on blur-sensitive directions while keeping global predictions fixed.',source_only=True,target_read=False,not_SIRM=True,extra_annotations=False,training='Zero linear768->10 residual,400 updates Adam.001/L2.01. Same32/source-domain minibatches, source-original mean/std and sqrt2. Equal-domain source-in correction centering; final fixed update. No validation selection.',controls=['original_only','paired_CE','paired_CE_consistency'],paired_CE='Mean original/blur CE using identical fixed original global logits for both residual feature views',consistency='Coefficient1 times class-centered residual original/blur squared difference averaged images/classes. Explicit new hypothesis condition, not SIRM penalty.',blur_kernel=5,blur_sigma=1,scope='Full1172 source-in and304 source-out. Original-view accuracy primary. Residual-feature blur evaluation keeps original global fixed and is not full-image corruption accuracy.',code_sha256=digest(Path(__file__))));(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());current,historical,fw,locks=load_models();del historical;version=[p._version for p in current.parameters()];t=ViewTransform({**current.config,'blur_kernel':5,'blur_sigma':1},'blur',False);data={}
 with torch.no_grad():
  for split in ['source_in','source_out']:
   orig=dict(np.load(CACHE/f'{split}_tokens.npz'));groups=[]
   for start in range(0,len(orig['ids']),32):
    ids=[str(id) for id in orig['ids'][start:start+32]];images=torch.stack([t(Image.open(ROOT/'domainbed/data/terra_incognita'/id).convert('RGB')) for id in ids]).cuda();captured=[];hook=current.featurizer.vit.norm.register_forward_hook(lambda m,args,out:captured.append(out))
    try:z=current.featurizer(images)
    finally:hook.remove()
    groups.append(torch.cat([z,captured[0][:,2:].amax(1)],1).cpu())
   data[split]=dict(original=torch.cat([torch.tensor(orig['cls']),torch.tensor(orig['patch_max'])],1),blur=torch.cat(groups),ids=orig['ids'].tolist(),labels=torch.tensor(orig['y']),domains=torch.tensor(orig['domains']),global_logits=torch.tensor(orig['global_logits']));torch.save(data[split],DEST/f'{split}_features.pt');print('cache',split,len(orig['ids']),flush=True)
 assert version==[p._version for p in current.parameters()] and all(not p.requires_grad and p.grad is None for p in current.parameters());train=data['source_in'];mean=train['original'].mean(0);std=train['original'].std(0,unbiased=False).clamp_min(1e-6);x=(train['original']-mean)/std/np.sqrt(2);b=(train['blur']-mean)/std/np.sqrt(2);d=train['domains'];doms=sorted(d.unique().tolist());indices={dom:torch.nonzero(d.eq(dom)).flatten() for dom in doms};gen=torch.Generator().manual_seed(0);schedule=[torch.cat([indices[dom][torch.randint(len(indices[dom]),(32,),generator=gen)] for dom in doms]) for _ in range(400)];states={};source_RMS={};readers={}
 for name in ['original_only','paired_CE','paired_CE_consistency']:
  head=torch.nn.Linear(768,10);head.weight.data.zero_();head.bias.data.zero_();opt=torch.optim.Adam(head.parameters(),lr=.001)
  for ix in schedule:
   delta=head(x[ix]);loss=F.cross_entropy(train['global_logits'][ix]+delta,train['labels'][ix])
   if name!='original_only':
    bd=head(b[ix]);loss=.5*(loss+F.cross_entropy(train['global_logits'][ix]+bd,train['labels'][ix]))
    if name=='paired_CE_consistency':
     cd=delta-delta.mean(1,keepdim=True);cb=bd-bd.mean(1,keepdim=True);loss=loss+(cd-cb).square().mean()
   loss=loss+.01*head.weight.square().sum();opt.zero_grad();loss.backward();opt.step()
  with torch.no_grad():
   z=head(x);center=torch.stack([z[d.eq(dom)].mean(0) for dom in doms]).mean(0);splits={};replays={}
   for split,a in data.items():
    g=a['global_logits'];y=a['labels'];dd=a['domains'];od=head((a['original']-mean)/std/np.sqrt(2))-center;bd=head((a['blur']-mean)/std/np.sqrt(2))-center;f=g+od;bf=g+bd;split_score=score(g,f,y,dd);split_score['blur_feature_score']=score(g,bf,y,dd);split_score['correction_RMS']=float(od.square().mean().sqrt());split_score['paired_residual_MSE']=float(((od-od.mean(1,keepdim=True))-(bd-bd.mean(1,keepdim=True))).square().mean());splits[split]=split_score;replays[split]=dict(ids=a['ids'],labels=y,domains=dd,global_logits=g,logits=f,blur_logits=bf)
   p=DEST/f'{name}.pt';torch.save(dict(weight=head.weight,bias=head.bias,mean=mean,std=std,center=center,replays=replays),p);states[name]=dict(splits=splits,model_sha256=digest(p));source_RMS[name]=splits['source_in']['correction_RMS'];readers[name]=replays;print(name,{s:{k:v[k] for k in ['accuracy_equal_domain','CE_equal_domain','correct','repairs','harms','correction_RMS','paired_residual_MSE']} for s,v in splits.items()},flush=True)
 for name in ['paired_CE','paired_CE_consistency']:
  scale=source_RMS['original_only']/source_RMS[name];a=readers[name]['source_out'];states[name]['source_train_RMS_matched']=dict(scale=scale,source_out=score(a['global_logits'],a['global_logits']+scale*(a['logits']-a['global_logits']),a['labels'],a['domains']))
 for path,h in locks.items():assert digest(Path(path))==h
 write(DEST/'results.json',dict(states=states,source_only=True,target_read=False,not_SIRM=True,extra_annotations=False,global_parameters_unchanged=True,schedule_sha256=hashlib.sha256(torch.stack(schedule).numpy().tobytes()).hexdigest(),input_sha256=locks));print('complete',flush=True)
if __name__=='__main__':main()

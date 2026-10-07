"""Matched oracle ROI native-detail versus resize-first class readout probes."""
import json,hashlib
from pathlib import Path
import torch,numpy as np
import torch.nn.functional as F
from PIL import Image
from .terra_attention_eda import ROOT,OUT,load_models
from .terra_local_fusion_control import SOURCE
from .terra_object_pool_readout import score
from experiments.sirm_h1_h6.run import ViewTransform
from .shrinkage import write,digest
DEST=OUT/'source_object_probe/roi_species_probe'

def run():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;DEST.mkdir(exist_ok=True);assert not (DEST/'plan.json').exists();metadata=OUT/'caltech_bboxes_20200316.json';official=json.loads(metadata.read_text());sizes={a['id']:(a['width'],a['height']) for a in official['images']};names=json.loads((OUT/'results.json').read_text())['class_names'];cats={a['id']:a['name'] for a in official['categories']};boxes={}
 for a in official['annotations']:
  if a.get('bbox'):boxes.setdefault(a['image_id'],[]).append((cats[a['category_id']],a['bbox']))
 plan=dict(hypothesis='Small object detail may be lost before spatial readout at224. Matched bbox/context crops from native versus resized-first inputs isolate this spatial-detail issue; refitted class reader avoids old global classifier ROI mismatch.',source_only=True,target_read=False,oracle_box_diagnostic=True,deployment_accuracy=False,evaluation='Same969 source-in and238 source-out cohort:858/206 boxed positives plus111/32 empty images. Unboxed positives excluded; do not report this as all304-source accuracy.',controls=['original_CLS','native_ROI_pad1_CLS','resize_first_ROI_pad1_CLS'],crop='Union of class-matched official boxes, one bbox width/height padding clipped to image. Empty images use whole image in every condition.',training='Pooled and3 auxiliary held-domain fits per view, zero linear384->10,400 updates, Adam.01/L2.01, exact same equal-class/cycled-domain8/class minibatches, minimum2/cohort, train-only standardization, final update, no validation selection',scope='Additional source annotations and oracle label-matched evaluation boxes. Frozen backbone/global remain unchanged; held out only from auxiliary fitting.',code_sha256=digest(Path(__file__)),metadata_sha256=digest(metadata));write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());current,historical,fw,locks=load_models();del historical;version=[p._version for p in current.parameters()];t=ViewTransform(current.config,'original',False);data={}
 with torch.no_grad():
  for split in ['source_in','source_out']:
   original=torch.load(SOURCE/f'{split}_cache.pt',map_location='cpu',weights_only=False);arrays={n:[] for n in plan['controls']};rects=[]
   for start in range(0,len(original['ids']),16):
    imgs={n:[] for n in arrays}
    for i in range(start,min(start+16,len(original['ids']))):
     id=original['ids'][i];label=int(original['labels'][i]);img=Image.open(ROOT/'domainbed/data/terra_incognita'/id).convert('RGB');rect=None
     if label!=5:
      ow,oh=sizes[Path(id).stem];bb=[b for cls,b in boxes[Path(id).stem] if cls==names[label]];assert bb;x0=min(b[0] for b in bb)/ow;y0=min(b[1] for b in bb)/oh;x1=max(b[0]+b[2] for b in bb)/ow;y1=max(b[1]+b[3] for b in bb)/oh;bw=x1-x0;bh=y1-y0;rect=(max(0,x0-bw),max(0,y0-bh),min(1,x1+bw),min(1,y1+bh));w,h=img.size;native=img.crop(tuple(round(v*s) for v,s in zip(rect,[w,h,w,h])));resized=img.resize((224,224),Image.Resampling.BILINEAR).crop(tuple(round(v*224) for v in rect))
     else:native=resized=img
     rects.append(rect);imgs['original_CLS'].append(t(img));imgs['native_ROI_pad1_CLS'].append(t(native));imgs['resize_first_ROI_pad1_CLS'].append(t(resized))
    for name,images in imgs.items():arrays[name].append(current.featurizer(torch.stack(images).cuda()).cpu())
   data[split]=dict(ids=original['ids'],labels=original['labels'],domains=original['domains'],global_logits=original['global_logits'],is_boxed=original['is_boxed'],features={n:torch.cat(v) for n,v in arrays.items()},rectangles=rects);torch.save(data[split],DEST/f'{split}_features.pt');print('cache',split,len(original['ids']),flush=True)
 assert version==[p._version for p in current.parameters()] and all(not p.requires_grad and p.grad is None for p in current.parameters());states={}
 for held in [None,0,2,3]:
  train=data['source_in'];keep=torch.ones(len(train['labels']),dtype=torch.bool) if held is None else train['domains'].ne(held);y=train['labels'][keep];d=train['domains'][keep];cohorts={c:{dom:torch.nonzero(d.eq(dom)&y.eq(c)).flatten() for dom in sorted(d.unique().tolist()) if int((d.eq(dom)&y.eq(c)).sum())>=2} for c in range(10)};assert all(cohorts.values());gen=torch.Generator().manual_seed(0);schedule=[]
  for step in range(400):
   rows=[]
   for c in range(10):ds=sorted(cohorts[c]);ix=cohorts[c][ds[step%len(ds)]];rows.append(ix[torch.randint(len(ix),(8,),generator=gen)])
   schedule.append(torch.cat(rows))
  models={}
  for name in plan['controls']:
   r=train['features'][name][keep];mean=r.mean(0);std=r.std(0,unbiased=False).clamp_min(1e-6);x=(r-mean)/std;head=torch.nn.Linear(384,10);head.weight.data.zero_();head.bias.data.zero_();opt=torch.optim.Adam(head.parameters(),lr=.01)
   for ix in schedule:
    loss=F.cross_entropy(head(x[ix]),y[ix])+.01*head.weight.square().sum();opt.zero_grad();loss.backward();opt.step()
   with torch.no_grad():
    splits={};replays={}
    for split,a in data.items():
     take=torch.ones(len(a['labels']),dtype=torch.bool) if held is None else (a['domains'].ne(held) if split=='source_in' else a['domains'].eq(held));f=head((a['features'][name][take]-mean)/std);yy=a['labels'][take];dd=a['domains'][take];g=a['global_logits'][take];boxed=a['is_boxed'][take];splits[split]=dict(score=score(g,f,yy,dd),boxed_correct=int(f.argmax(1)[boxed].eq(yy[boxed]).sum()),boxed_n=int(boxed.sum()),empty_correct=int(f.argmax(1)[~boxed].eq(5).sum()),empty_n=int((~boxed).sum()));replays[split]=dict(ids=[id for id,m in zip(a['ids'],take.tolist()) if m],labels=yy,domains=dd,global_logits=g,logits=f,is_boxed=boxed)
    path=DEST/f'{held}_{name}.pt';torch.save(dict(weight=head.weight,bias=head.bias,mean=mean,std=std,replays=replays),path);models[name]=dict(splits=splits,model_sha256=digest(path));print(held,name,{s:dict(accuracy=v['score']['accuracy_equal_domain'],boxed=f"{v['boxed_correct']}/{v['boxed_n']}",empty=f"{v['empty_correct']}/{v['empty_n']}") for s,v in splits.items()},flush=True)
  states[str(held)]=dict(models=models,schedule_sha256=hashlib.sha256(torch.stack(schedule).numpy().tobytes()).hexdigest())
 for path,h in locks.items():assert digest(Path(path))==h
 write(DEST/'results.json',dict(states=states,oracle_box_diagnostic=True,source_only=True,target_read=False,global_parameters_unchanged=True,input_sha256=locks));print('complete',flush=True)
if __name__=='__main__':run()

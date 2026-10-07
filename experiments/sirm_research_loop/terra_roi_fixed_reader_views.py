"""Fixed native ROI readers under paired input interventions, no fitting."""
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import rgb_to_grayscale
from .terra_roi_species_probe import DEST as SOURCE
from .terra_attention_eda import ROOT,OUT,load_models
from .terra_object_pool_readout import score
from experiments.sirm_h1_h6.run import ViewTransform
from .shrinkage import write,digest
DEST=SOURCE/'fixed_reader_views'

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;DEST.mkdir(exist_ok=True);assert not (DEST/'plan.json').exists();source_path=SOURCE/'source_out_features.pt';a=torch.load(source_path,map_location='cpu',weights_only=False);paths=[SOURCE/f'{h}_native_ROI_pad1_CLS.pt' for h in [None,0,2,3]];locks={str(p):digest(p) for p in [source_path,*paths]};write(DEST/'plan.json',dict(hypothesis='Grayscale/IR raccoon mistakes under native crops may stem from fine-detail input sensitivity rather than the separately refitted crop readers.',source_only=True,target_read=False,training=False,selection=False,oracle_box_diagnostic=True,deployment_accuracy=False,scope='All238 boxed-positive/empty source-out images; identical locked native readers at native,gray,blur5sigma1,gray+blur andresize-first inputs. Domainsheldout from auxiliary only; frozen anchor saw all3.',blur_kernel=5,blur_sigma=1,input_sha256=locks,code_sha256=digest(Path(__file__))));(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());current,historical,fw,mlocks=load_models();del historical;version=[p._version for p in current.parameters()];cfg={**current.config,'blur_kernel':5,'blur_sigma':1};gray=ViewTransform(cfg,'grayscale',False);blur=ViewTransform(cfg,'blur',False);features={'native':a['features']['native_ROI_pad1_CLS'],'resize_first':a['features']['resize_first_ROI_pad1_CLS']};groups={k:[] for k in ['gray','blur','gray_blur']}
 with torch.no_grad():
  for start in range(0,len(a['ids']),16):
   images={k:[] for k in groups}
   for i in range(start,min(start+16,len(a['ids']))):
    img=Image.open(ROOT/'domainbed/data/terra_incognita'/a['ids'][i]).convert('RGB');rect=a['rectangles'][i]
    if rect:
     w,h=img.size;img=img.crop(tuple(round(v*s) for v,s in zip(rect,[w,h,w,h])))
    images['gray'].append(gray(img));images['blur'].append(blur(img));images['gray_blur'].append(blur(rgb_to_grayscale(img,num_output_channels=3)))
   for view,x in images.items():groups[view].append(current.featurizer(torch.stack(x).cuda()).cpu())
 for view,x in groups.items():features[view]=torch.cat(x)
 torch.save(dict(features=features,ids=a['ids'],labels=a['labels'],domains=a['domains']),DEST/'view_features.pt');states={};combined={v:[] for v in features};prior=json.loads((SOURCE/'eda.json').read_text());index={id:i for i,id in enumerate(a['ids'])};cases=prior['states']['0']['cases'];failure_ids={c['id'] for c in cases if c['event']=='harm_vs_original'}
 with torch.no_grad():
  for held in [None,0,2,3]:
   reader=torch.load(SOURCE/f'{held}_native_ROI_pad1_CLS.pt',map_location='cpu',weights_only=False);take=torch.ones(len(a['ids']),dtype=torch.bool) if held is None else a['domains'].eq(held);y=a['labels'][take];d=a['domains'][take];ids=[id for id,m in zip(a['ids'],take.tolist()) if m];native=((features['native'][take]-reader['mean'])/reader['std'])@reader['weight'].T+reader['bias'];assert torch.allclose(native,reader['replays']['source_out']['logits'],atol=1e-4,rtol=1e-5);states[str(held)]={};replays={}
   for view,x in features.items():
    f=((x[take]-reader['mean'])/reader['std'])@reader['weight'].T+reader['bias'];m=a['is_boxed'][take];s=score(native,f,y,d);s.update(boxed_correct=int(f.argmax(1)[m].eq(y[m]).sum()),boxed_n=int(m.sum()),empty_correct=int(f.argmax(1)[~m].eq(5).sum()),empty_n=int((~m).sum()));states[str(held)][view]=s;replays[view]=dict(ids=ids,labels=y,domains=d,logits=f)
    if held is not None:combined[view].append(dict(ids=ids,labels=y,domains=d,global_logits=native,logits=f))
   if held==0:
    failure=[]
    for i,id in enumerate(ids):
     if id in failure_ids:failure.append(dict(id=id,label=int(y[i]),views={v:dict(prediction=int(r['logits'][i].argmax()),correct=bool(r['logits'][i].argmax()==y[i]),GT_logit=float(r['logits'][i,y[i]])) for v,r in replays.items()}))
    write(DEST/'location100_failed_cases.json',dict(cases=failure,n=len(failure),source_only=True))
   torch.save(replays,DEST/f'{held}_replay.pt');print(held,{v:{k:x[k] for k in ['accuracy_equal_domain','boxed_correct','empty_correct','repairs','harms']} for v,x in states[str(held)].items()},flush=True)
 aggregate={}
 for view,rows in combined.items():
  p={k:sum([r[k] for r in rows],[]) if k=='ids' else torch.cat([r[k] for r in rows]) for k in rows[0]};aggregate[view]=score(p['global_logits'],p['logits'],p['labels'],p['domains'])
 assert version==[p._version for p in current.parameters()] and all(not p.requires_grad and p.grad is None for p in current.parameters())
 for path,h in {**locks,**mlocks}.items():assert digest(Path(path))==h
 write(DEST/'results.json',dict(states=states,aggregate_auxiliary_held_domain=aggregate,source_only=True,target_read=False,training=False,parameters_unchanged=True,oracle_box_diagnostic=True,input_sha256={**locks,**mlocks}));print('complete',flush=True)
if __name__=='__main__':main()

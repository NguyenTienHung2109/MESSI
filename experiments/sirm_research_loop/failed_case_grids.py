"""Requested 4x4 failed-case grids from locked prediction replays, without inference."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import softmax
from torchvision.datasets import ImageFolder
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
OUT=DEST/'failed_case_grids'

def select(ids,y,old,new,conf,source=False):
 wrong=new!=y;harms=np.flatnonzero(wrong&(old==y)).tolist();chosen=[];seen=set()
 def add(i):
  event=Path(str(ids[i])).stem
  if i not in chosen and event not in seen and len(chosen)<16:chosen.append(i);seen.add(event)
 for i in sorted(harms,key=lambda i:(-conf[i],str(ids[i]))):add(i)
 if source:
  for i in np.flatnonzero(wrong&(y==0)):add(int(i))
 groups={int(c):sorted(np.flatnonzero(wrong&(y==c)).tolist(),key=lambda i:(-conf[i],str(ids[i]))) for c in np.unique(y)}
 pos=0
 while len(chosen)<16:
  changed=False
  for c in sorted(groups):
   if pos<len(groups[c]):before=len(chosen);add(groups[c][pos]);changed|=len(chosen)>before
  pos+=1
  if pos>max(map(len,groups.values())):break
 assert len(chosen)==16 and all(wrong[i] for i in chosen)
 return chosen

def draw(name,title,ids,y,old,new,oldconf,newconf,names,selected,oldlabel,newlabel,expected_hashes=None):
 fig,axes=plt.subplots(4,4,figsize=(12,16));records=[]
 for cell,(ax,i) in enumerate(zip(axes.flat,selected),1):
  id=str(ids[i]);path=ROOT/'domainbed/data/terra_incognita'/id;h=digest(path)
  if expected_hashes is not None:assert h==expected_hashes[id]
  image=Image.open(path).convert('RGB').resize((224,224),Image.Resampling.BILINEAR);ax.imshow(image);ax.set_xticks([]);ax.set_yticks([]);harm=int(old[i])==int(y[i]);color='#c62828' if harm else '#b26800';status='NEW HARM' if harm else 'STILL WRONG'
  for spine in ax.spines.values():spine.set_linewidth(2);spine.set_edgecolor(color)
  ax.set_title(f"{cell:02d} | GT: {names[int(y[i])]} | {status}\n{oldlabel}: {names[int(old[i])]} ({oldconf[i]*100:.0f}%)\n{newlabel}: {names[int(new[i])]} ({newconf[i]*100:.0f}%)",fontsize=9,color=color,pad=5)
  ax.set_xlabel(f"{id.split('/')[0]} | {Path(id).stem[:8]}",fontsize=8)
  records.append(dict(cell=cell,image_id=id,image_sha256=h,true_label=names[int(y[i])],comparison_prediction=names[int(old[i])],comparison_confidence=float(oldconf[i]),model_prediction=names[int(new[i])],model_confidence=float(newconf[i]),new_harm=harm))
 fig.suptitle(title,fontsize=12,fontweight='bold',y=.978);fig.text(.5,.015,'16 selected failures, not a random sample. Full image resized to224x224 as in evaluation.\nGT is the dataset label; appearance alone does not establish annotation error.',ha='center',fontsize=9);fig.subplots_adjust(left=.03,right=.98,bottom=.07,top=.91,wspace=.12,hspace=.50);f=OUT/(name+'.png');fig.savefig(f,dpi=150);plt.close(fig)
 return dict(file=str(f.relative_to(ROOT)),sha256=digest(f),rows=records)

def run():
 OUT.mkdir(exist_ok=True);locks={};eda=DEST/'eda';checkpoint_locks=json.loads((eda/'checkpoint_locks.json').read_text());manifestfile=ROOT/'results/sirm_research_loop/iteration_12/experiment/subset_manifest.json';manifest=json.loads(manifestfile.read_text());names=manifest['class_names'];hashes={r['id']:r['sha256'] for source in manifest['sources'].values() for r in source['images']};locks[str(manifestfile)]=digest(manifestfile);locks[str(eda/'checkpoint_locks.json')]=digest(eda/'checkpoint_locks.json');index={}
 # B2 in these EDA arrays is iteration15 routed OFF, not the old ON bank.
 mapping=json.loads((DEST/'matched_weighting_control/arm_mapping.json').read_text());assert 'iteration15' in mapping['B2']
 for split,label in [('source_out','Source validation'),('target_in','Target L38 test split')]:
  f=eda/(split+'_replay.npz');a=dict(np.load(f));locks[str(f)]=digest(f);old=a['B0'].argmax(1);new=a['B2'].argmax(1);oc=softmax(a['B0'],axis=1).max(1);nc=softmax(a['B2'],axis=1).max(1);selected=select(a['ids'],a['y'],old,new,nc,split=='source_out')
  index[split]=draw(split+'_current_4x4',f'{label} | Current10% bank, iteration15 | global versus residual composition',a['ids'],a['y'],old,new,oc,nc,names,selected,'ERM global','SIRM bank',hashes)
  index[split].update(total_images=len(old),total_model_failures=int((new!=a['y']).sum()),total_new_harms=int(((old==a['y'])&(new!=a['y'])).sum()),total_repairs=int(((old!=a['y'])&(new==a['y'])).sum()),SIRM_penalty_active=False,OFF_ON_identical=True)
 # Historical full-data bank, kept explicitly separate from current10% protocol.
 full=ROOT/'results/sirm_discover_binary/terra_L38_seed0_lambda1';ds=ImageFolder(ROOT/'domainbed/data/terra_incognita/location_38');assert ds.classes==names;ids=np.array([str(Path(p).relative_to(ROOT/'domainbed/data/terra_incognita')) for p,y in ds.samples]);y=np.array(ds.targets);arrays={}
 for arm in ['R0','R1']:
  f=full/f'target_logits_{arm}.npy';locks[str(f)]=digest(f);arrays[arm]=np.load(f);assert len(arrays[arm])==len(y)==9736
 assert int((arrays['R0'].argmax(1)==y).sum())==3830 and int((arrays['R1'].argmax(1)==y).sum())==3834
 old=arrays['R0'].argmax(1);new=arrays['R1'].argmax(1);oc=softmax(arrays['R0'],axis=1).max(1);nc=softmax(arrays['R1'],axis=1).max(1);selected=select(ids,y,old,new,nc)
 index['historical_full_lambda1']=draw('historical_full_lambda1_4x4','Historical FULL data | L38 | R0 OFF versus R1 binary SIRM lambda=1',ids,y,old,new,oc,nc,names,selected,'R0 OFF','R1 ON')
 index['historical_full_lambda1'].update(total_images=len(y),total_model_failures=int((new!=y).sum()),total_new_harms=int(((old==y)&(new!=y)).sum()),total_repairs=int(((old!=y)&(new==y)).sum()),separate_protocol=True)
 for f,h in locks.items():assert digest(Path(f))==h
 write(OUT/'index.json',dict(selection_rule='Include all comparison-correct/model-wrong cases first; current source adds all failed birds; fill by class round-robin of highest-confidence remaining failures. Deduplicate image stem across label folders. Selected examples are descriptive, not prevalence estimates or method-selection data.',grids=index,input_sha256=locks,target_inference=False,training=False,checkpoint_changes=False,code_sha256=digest(Path(__file__))))
 (OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:{n:v[n] for n in ['file','total_model_failures','total_new_harms','total_repairs']} for k,v in index.items()},indent=2))
if __name__=='__main__':run()

"""Descriptive source-only factor-score association with image appearance."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
OUT=DEST/'source_factor_pixel_association'
METRICS=['luminance_mean','luminance_std','color_chroma','dark_fraction','neighbor_gradient']

def run():
 OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists()
 rankpath=DEST/'source_residual_rank_diagnosis/results.json';rank=json.loads(rankpath.read_text());token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';data={};locks={};features=[]
 plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,basis_refitting=False,metrics=METRICS,pixels='RGB images; crop top/bottom5pct to exclude timestamp/logo strips, retain full width, resize64x64 bilinear. Luminance=.2126R+.7152G+.0722B, values/255. Chroma=maxRGB-minRGB; dark means luma<.1; neighbor gradient mean absolute horizontal/vertical luma difference.',conditioning='Domain/true-class groups minimum2. n2 correlations saved descriptive only; pooled within-group summaries exclude groups withn<3. Equal-group weights and group-mean centering. No p-values or feature selection.',limits='Image appearance proxies correlate with many semantic/background/image-acquisition variables. Association cannot identify shortcut causality, object morphology or removal of information.',code_sha256=digest(Path(__file__)),rank_results_sha256=digest(rankpath))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
 for split in ['source_in','source_out']:
  p=token/f'{split}_tokens.npz';assert digest(p)==rank['input_sha256'][str(p)];locks[str(p)]=digest(p);b=dict(np.load(p));X=[]
  for image_id in b['ids']:
   path=ROOT/'domainbed/data/terra_incognita'/str(image_id);assert str(image_id).startswith(('location_100/','location_43/','location_46/'));im=Image.open(path).convert('RGB');w,h=im.size;arr=np.asarray(im.crop((0,round(.05*h),w,round(.95*h))).resize((64,64),Image.Resampling.BILINEAR),dtype=float)/255
   lum=arr@np.array([.2126,.7152,.0722]);v=np.array([lum.mean(),lum.std(),(arr.max(2)-arr.min(2)).mean(),(lum<.1).mean(),.5*(np.abs(np.diff(lum,axis=0)).mean()+np.abs(np.diff(lum,axis=1)).mean())]);X.append(v);features.append(dict(split=split,image_id=str(image_id),domain=int(b['domains'][len(X)-1]),label=int(b['y'][len(X)-1]),pixel_sha256=digest(path),metrics=dict(zip(METRICS,map(float,v)))))
  data[split]=dict(X=np.stack(X),y=b['y'],domains=b['domains'],ids=b['ids'])
 rows=[];pooled=[];cases=[]
 for state,info in rank['states'].items():
  path=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';assert digest(path)==rank['input_sha256'][str(path)];bank=dict(np.load(path));axispath=DEST/'source_residual_rank_diagnosis'/info['source_in_basis_file'];assert digest(axispath)==info['source_in_basis_sha256'];axes=np.load(axispath)['axes']
  original=next(x for x in info['rows'] if x['variant']=='full' and x['split']=='source_out' and x['routing']=='original_posterior_recomputed')['metrics'];watch=set(original['repair_ids']+original['harm_ids']);watch|={str(i) for i in data['source_out']['ids'] if '59eab95e' in str(i)}
  for split,b in data.items():
   assert np.array_equal(b['ids'],bank[split+'_ids']);score=np.einsum('nmc,mc->nm',bank[split+'_bank_delta'].astype(float),axes)
   for e in [2,5]:
    residual_s=[];residual_x=[];weights=[]
    for domain in [0,2,3]:
     for label in sorted(set(b['y'][b['domains']==domain])):
      mask=(b['domains']==domain)&(b['y']==label);n=int(mask.sum())
      if n<2:continue
      s=score[mask,e];x=b['X'][mask]
      for k,name in enumerate(METRICS):
       corr=float(spearmanr(s,x[:,k]).statistic) if np.std(s)>1e-12 and np.std(x[:,k])>1e-12 else None
       rows.append(dict(state=state,split=split,expert=e,domain=domain,label=int(label),n=n,metric=name,Spearman=corr,n2_descriptive_only=n==2,score_mean=float(s.mean()),metric_mean=float(x[:,k].mean())))
      if n>=3:residual_s.extend(s-s.mean());residual_x.extend(x-x.mean(0));weights.extend([1/n]*n)
    ss=np.array(residual_s);xx=np.array(residual_x);w=np.array(weights);w/=w.sum()
    for k,name in enumerate(METRICS):
     denom=np.sqrt(np.sum(w*ss**2)*np.sum(w*xx[:,k]**2));corr=float(np.sum(w*ss*xx[:,k])/denom) if denom>1e-12 else None
     pooled.append(dict(state=state,split=split,expert=e,metric=name,conditional_equal_group_Pearson=corr,rows=len(ss),groups=sum(1 for x in rows if x['state']==state and x['split']==split and x['expert']==e and x['metric']==name and x['n']>=3)))
   if split=='source_out':
    for i,image_id in enumerate(b['ids']):
     if str(image_id) in watch:cases.append(dict(state=state,image_id=str(image_id),label=int(b['y'][i]),E2_score=float(score[i,2]),E5_score=float(score[i,5]),pixel_metrics=dict(zip(METRICS,map(float,b['X'][i]))),original_posterior_repair=str(image_id) in original['repair_ids'],original_posterior_harm=str(image_id) in original['harm_ids']))
 for p,h in locks.items():assert digest(Path(p))==h
 write(OUT/'results.json',dict(plan=plan,source_only=True,target_read=False,bank_updates=False,images=features,group_correlations=rows,conditional_pooled=pooled,cases=cases,input_sha256=locks))
if __name__=='__main__':run()

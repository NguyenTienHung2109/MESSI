"""Replay all conditional patch heads; per-class utility and deterministic case EDA."""
import json
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .source_conditional_patch_readout import DEST,CACHE
from .shrinkage import ROOT,write,digest
def main():
 torch.set_num_threads(4);r=json.loads((DEST/'results.json').read_text());data={s:dict(np.load(CACHE/f'{s}_tokens.npz')) for s in ['source_in','source_out']};names=json.loads((DEST.parent/'experiment/subset_manifest.json').read_text())['class_names'];rows=[];cases=[];locks={};maxerr=0;count=0
 for fit,x in r['fits'].items():
  held=None if fit=='pooled' else int(fit[-1]);predictions={}
  for family in x['families']:
   for policy in ['learned','fixed']:
    path=DEST/f'{fit}_{family}_{policy}.pt';locks[str(path)]=digest(path);a=torch.load(path,map_location='cpu',weights_only=False)
    for split,d in data.items():
     mask=np.ones(len(d['y']),bool) if held is None else (d['domains']!=held if split=='source_in' else d['domains']==held);c,p=[(torch.tensor(d[k][mask])-a['input_stats'][k][0])/a['input_stats'][k][1] for k in ['cls','patch_max']];raw={'CLS':c,'patch_raw':p,'patch_conditional':p-c@a['ridge'],'patch_sham':p-(c@a['ridge'])@a['rotation']}[family];z=(raw-a['family_stats'][0])/a['family_stats'][1]/2**.5;g=torch.tensor(d['global_logits'][mask]);y=torch.tensor(d['y'][mask]);dom=torch.tensor(d['domains'][mask]);f=g+F.linear(z,(a['b']-a['b'].mean(0,keepdim=True))@a['a']);utility=F.cross_entropy(g,y,reduction='none')-F.cross_entropy(f,y,reduction='none');gp=g.argmax(1);fp=f.argmax(1)
     if split=='source_out':
      stored=np.load(DEST/f'{fit}_{family}_{policy}_source_out.npz');error=float((f-torch.tensor(stored['full_logits'])).abs().max());maxerr=max(maxerr,error);assert error<2e-5;assert int((fp!=torch.tensor(stored['full_logits']).argmax(1)).sum())==0;count+=1
      if policy=='learned':predictions[family]=fp
     for dd in dom.unique():
      for cc in y[dom==dd].unique():
       take=(dom==dd)&(y==cc);gc=gp.eq(y);fc=fp.eq(y);rows.append(dict(fit=fit,family=family,policy=policy,split=split,domain=int(dd),label=int(cc),n=int(take.sum()),utility_mean=float(utility[take].mean()),repairs=int((take&~gc&fc).sum()),harms=int((take&gc&~fc).sum())))
     if family=='patch_conditional' and policy=='learned' and split=='source_out':
      for i,id in enumerate(d['ids'][mask]):
       if fp[i]!=gp[i]:cases.append(dict(fit=fit,id=str(id),label=int(y[i]),global_prediction=int(gp[i]),conditional_prediction=int(fp[i]),repair=bool(gp[i]!=y[i] and fp[i]==y[i]),harm=bool(gp[i]==y[i] and fp[i]!=y[i])))
  ids=data['source_out']['ids'] if held is None else data['source_out']['ids'][data['source_out']['domains']==held];lookup={str(id):i for i,id in enumerate(ids)}
  for case in cases:
   if case['fit']==fit:case['other_predictions']={k:int(v[lookup[case['id']]]) for k,v in predictions.items()}
 write(DEST/'class_domain_payoff.json',rows);write(DEST/'changed_cases.json',cases);write(DEST/'replay_audit.json',dict(saved_replays=count,max_abs_error=maxerr,prediction_disagreement=0,head_sha256=locks,target_read=False))
 # held0 new harms prioritized to prevent cherry-picking pooled gain; remaining pooled changes.
 selected=sorted([c for c in cases if c['fit']=='held0' and c['harm']],key=lambda c:c['id'])+sorted([c for c in cases if c['fit']=='pooled'],key=lambda c:(not c['harm'],c['id']));seen=set();unique=[]
 for c in selected:
  if c['id'] not in seen:unique.append(c);seen.add(c['id'])
 for c in sorted(cases,key=lambda c:(c['fit'],c['id'])):
  if c['id'] not in seen:unique.append(c);seen.add(c['id'])
 selected=unique[:16];fig,axs=plt.subplots(4,4,figsize=(16,16))
 for ax in axs.flat:ax.axis('off')
 for ax,c in zip(axs.flat,selected):
  with Image.open(ROOT/'domainbed/data/terra_incognita'/c['id']) as im:ax.imshow(im.convert('RGB'))
  o=c['other_predictions'];ax.set_title(f"{c['fit']} | GT {names[c['label']]} | {Path(c['id']).stem[:8]}\nERM {names[c['global_prediction']]} / conditional {names[c['conditional_prediction']]}\nraw {names[o['patch_raw']]} / CLS {names[o['CLS']]} / sham {names[o['patch_sham']]}",fontsize=9)
 fig.suptitle('Source-only conditional patch reader: held0 harms first, then pooled changes; no target');fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(DEST/'cases_4x4.png',dpi=120);plt.close(fig);write(DEST/'grid_manifest.json',selected)
 fig,axs=plt.subplots(1,2,figsize=(13,5))
 for ax,metric,title in [(axs[0],'accuracy','Source accuracy (%)'),(axs[1],'CE','Source CE')]:
  for family in ['CLS','patch_raw','patch_conditional','patch_sham']:
   vals=[x['families'][family]['learned']['source_out'][metric]*(100 if metric=='accuracy' else 1) for x in r['fits'].values()];ax.plot(range(4),vals,'o-',label=family)
  ax.plot(range(4),[x['global'][metric]*(100 if metric=='accuracy' else 1) for x in r['fits'].values()],'k--',label='ERM');ax.set_xticks(range(4),list(r['fits']));ax.set_ylabel(title);ax.legend(fontsize=8)
 fig.suptitle('Prespecified rank3 learned readers; pooled and auxiliary-held cohorts differ');fig.tight_layout();fig.savefig(DEST/'source_payoff.png',dpi=150)
 print('Replayed',count,'heads maxerror',maxerr)
 for fit in r['fits']:
  cs=[c for c in cases if c['fit']==fit];print(fit,[(Path(c['id']).stem[:8],names[c['label']],names[c['conditional_prediction']],c['repair'],c['harm']) for c in cs])
if __name__=='__main__':main()

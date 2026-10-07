"""Source-only replay, class payoff and EDA for domain-crossfit readers."""
import json
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .source_crossfit_patch_readout import DEST,predict
from .source_low_rank_readout import CACHE
from .shrinkage import ROOT,write,digest
def main():
 torch.set_num_threads(4);data={s:dict(np.load(CACHE/f'{s}_tokens.npz')) for s in ['source_in','source_out']};names=json.loads((DEST.parent/'experiment/subset_manifest.json').read_text())['class_names'];results=json.loads((DEST/'results.json').read_text());rows=[];cases=[];locks={};maxerror=0;replays=0
 for fit,x in results['fits'].items():
  held=None if fit=='pooled' else int(fit[-1])
  for family in ['crossfit_conditional','crossfit_sham']:
   for policy in ['learned','fixed']:
    path=DEST/f'{fit}_{family}_{policy}.pt';locks[str(path)]=digest(path);a=torch.load(path,map_location='cpu',weights_only=False)
    for split,d in data.items():
     mask=np.ones(len(d['y']),bool) if held is None else (d['domains']!=held if split=='source_in' else d['domains']==held);c=torch.tensor(d['cls'][mask]);p=torch.tensor(d['patch_max'][mask]);dom=torch.tensor(d['domains'][mask]);y=torch.tensor(d['y'][mask]);g=torch.tensor(d['global_logits'][mask]);pred=torch.zeros_like(p)
     if split=='source_in':
      for dd,model in a['regressors'].items():
       take=dom.eq(dd);pred[take]=predict(c[take],model)
     else:pred=torch.stack([predict(c,m) for m in a['regressors'].values()]).mean(0)
     pp=(p-a['outer_patch_mean'])/a['outer_patch_std'];pred=(pred-a['outer_patch_mean'])/a['outer_patch_std'];raw=pp-(pred if family=='crossfit_conditional' else pred@a['rotation']);z=(raw-a['family_stats'][0])/a['family_stats'][1]/2**.5;f=g+F.linear(z,(a['b']-a['b'].mean(0,keepdim=True))@a['a']);gp=g.argmax(1);fp=f.argmax(1);utility=F.cross_entropy(g,y,reduction='none')-F.cross_entropy(f,y,reduction='none')
     for dd in dom.unique():
      for cc in y[dom.eq(dd)].unique():
       take=dom.eq(dd)&y.eq(cc);gc=gp.eq(y);fc=fp.eq(y);rows.append(dict(fit=fit,family=family,policy=policy,split=split,domain=int(dd),label=int(cc),n=int(take.sum()),utility_mean=float(utility[take].mean()),repairs=int((take&~gc&fc).sum()),harms=int((take&gc&~fc).sum())))
     if split=='source_out':
      stored=np.load(DEST/f'{fit}_{family}_{policy}_source_out.npz');err=float((f-torch.tensor(stored['full_logits'])).abs().max());maxerror=max(maxerror,err);assert err<2e-5;assert (fp!=torch.tensor(stored['full_logits']).argmax(1)).sum()==0;replays+=1
      if family=='crossfit_conditional' and policy=='learned':
       other={}
       for name in ['patch_raw','patch_conditional']:
        old=np.load(DEST.parent/'source_conditional_patch_readout'/f'{fit}_{name}_learned_source_out.npz');assert np.array_equal(old['ids'],stored['ids']);other[name]=old['full_logits'].argmax(1)
       sham=np.load(DEST/f'{fit}_crossfit_sham_learned_source_out.npz')['full_logits'].argmax(1)
       for i,id in enumerate(d['ids'][mask]):
        if fp[i]!=gp[i] or int(other['patch_conditional'][i])!=int(fp[i]):cases.append(dict(fit=fit,id=str(id),label=int(y[i]),global_prediction=int(gp[i]),crossfit_prediction=int(fp[i]),self_fit_prediction=int(other['patch_conditional'][i]),raw_patch_prediction=int(other['patch_raw'][i]),sham_prediction=int(sham[i]),repair=bool(gp[i]!=y[i] and fp[i]==y[i]),harm=bool(gp[i]==y[i] and fp[i]!=y[i])))
 write(DEST/'class_domain_payoff.json',rows);write(DEST/'changed_cases.json',cases);write(DEST/'replay_audit.json',dict(replays=replays,max_abs_logit_error=maxerror,prediction_disagreement=0,head_sha256=locks,target_read=False))
 ranked=sorted(cases,key=lambda c:(not(c['fit']=='held2' and c['self_fit_prediction']==c['label'] and c['crossfit_prediction']!=c['label']),not c['harm'],c['fit'],c['id']));selected=[];seen=set()
 for c in ranked:
  if c['id'] not in seen:selected.append(c);seen.add(c['id'])
  if len(selected)==16:break
 fig,axs=plt.subplots(4,4,figsize=(16,16))
 for ax in axs.flat:ax.axis('off')
 for ax,c in zip(axs.flat,selected):
  with Image.open(ROOT/'domainbed/data/terra_incognita'/c['id']) as im:ax.imshow(im.convert('RGB'))
  ax.set_title(f"{c['fit']} | GT {names[c['label']]} | {Path(c['id']).stem[:8]}\nERM {names[c['global_prediction']]} / crossfit {names[c['crossfit_prediction']]}\nself-fit {names[c['self_fit_prediction']]} / raw {names[c['raw_patch_prediction']]} / sham {names[c['sham_prediction']]}",fontsize=9)
 fig.suptitle('Crossfit conditional patch: held2 lost repairs first, then harms and other differences; source only');fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(DEST/'cases_4x4.png',dpi=120);plt.close(fig);write(DEST/'grid_manifest.json',selected)
 print('replays',replays,'maxerror',maxerror)
 for fit in results['fits']:print(fit,[(Path(c['id']).stem[:8],names[c['label']],names[c['crossfit_prediction']],names[c['self_fit_prediction']],c['repair'],c['harm']) for c in cases if c['fit']==fit])
if __name__=='__main__':main()

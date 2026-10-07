"""Replay fixed-pair banks and audit component identity, payoff and cases."""
import json
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .source_fixed_pair_experts import DEST,Bank
from .source_low_rank_readout import CACHE
from .shrinkage import ROOT,write,digest
def main():
 torch.set_num_threads(4);r=json.loads((DEST/'results.json').read_text());source=dict(np.load(CACHE/'source_out_tokens.npz'));names=json.loads((DEST.parent/'experiment/subset_manifest.json').read_text())['class_names'];cases=[];payoff=[];locks={};maxerror=0;inverse_error=0;norms={}
 for fit,x in r['fits'].items():
  held=None if fit=='pooled' else int(fit[-1]);mask=np.ones(len(source['y']),bool) if held is None else source['domains']==held;other={}
  for family in x['families']:
   path=DEST/f'{fit}_{family}.pt';locks[str(path)]=digest(path);a=torch.load(path,map_location='cpu',weights_only=False);h=Bank(a['state_dict']['axes']);h.load_state_dict(a['state_dict']);h.eval();t=torch.load(a['transform_path'],map_location='cpu',weights_only=False);c,p=[(torch.tensor(source[k][mask])-t['input_stats'][k][0])/t['input_stats'][k][1] for k in ['cls','patch_max']];res=p-c@t['ridge'];z=(res-t['family_stats'][0])/t['family_stats'][1]/2**.5;g=torch.tensor(source['global_logits'][mask]);y=torch.tensor(source['y'][mask]);dom=torch.tensor(source['domains'][mask]);stored=np.load(DEST/f'{fit}_{family}_source_out.npz')
   with torch.no_grad():comps=h.components(z);full=g+comps.sum(1);scalar=(h.scalar(z)-h.mean)/6**.5;recovered=comps.sum(1)@torch.linalg.pinv(h.axes);inverse_error=max(inverse_error,float((scalar-recovered).abs().max()));error=float((full-torch.tensor(stored['full_logits'])).abs().max());maxerror=max(maxerror,error);assert error<2e-5;assert (full.argmax(1)!=torch.tensor(stored['full_logits']).argmax(1)).sum()==0
   other[family]=full.argmax(1);ce=F.cross_entropy(full,y,reduction='none');u=torch.stack([F.cross_entropy(full-comps[:,m],y,reduction='none')-ce for m in range(6)],1)
   for dd in dom.unique():
    for cc in y[dom.eq(dd)].unique():
     take=dom.eq(dd)&y.eq(cc);payoff.append(dict(fit=fit,family=family,domain=int(dd),label=int(cc),n=int(take.sum()),U=u[take].mean(0).tolist()))
   prior=r['fits'][fit]['families'];common=np.mean([v['source_in']['correction_RMS'] for v in prior.values()]);scale=common/prior[family]['source_in']['correction_RMS'];scaled=g+(full-g)*scale;fc=scaled.argmax(1).eq(y);gc=g.argmax(1).eq(y);norms.setdefault(fit,{})[family]=dict(scale=float(scale),accuracy=float(torch.stack([fc[dom.eq(dd)].float().mean() for dd in dom.unique()]).mean()),CE=float(torch.stack([F.cross_entropy(scaled,y,reduction='none')[dom.eq(dd)].mean() for dd in dom.unique()]).mean()),repairs=int((~gc&fc).sum()),harms=int((gc&~fc).sum()))
   if family=='fixed_pairs':
    for i,id in enumerate(source['ids'][mask]):
     gp=int(g[i].argmax());fp=int(full[i].argmax());yy=int(y[i])
     if gp!=fp:cases.append(dict(fit=fit,id=str(id),label=yy,global_prediction=gp,pair_prediction=fp,repair=gp!=yy and fp==yy,harm=gp==yy and fp!=yy,component_U=u[i].tolist(),without_expert_predictions=[int((full[i]-comps[i,m]).argmax()) for m in range(6)]))
  ids=source['ids'][mask];index={str(id):i for i,id in enumerate(ids)}
  for case in cases:
   if case['fit']==fit:case['controls']={family:int(pred[index[case['id']]]) for family,pred in other.items()}
 write(DEST/'class_domain_payoff.json',payoff);write(DEST/'changed_cases.json',cases);write(DEST/'amplitude_controls.json',norms);write(DEST/'replay_audit.json',dict(replays=12,max_abs_logit_error=maxerror,prediction_disagreement=0,max_scalar_inverse_error=inverse_error,interpretation='Full rank fixed output axes identify six coefficients uniquely within this prescribed label-contrast model, not discovered physical factors.',head_sha256=locks,target_read=False))
 selected=[];seen=set()
 for case in sorted(cases,key=lambda c:(not(c['fit']=='held2' and c['harm']),not c['harm'],c['fit'],c['id'])):
  if case['id'] not in seen:selected.append(case);seen.add(case['id'])
  if len(selected)==16:break
 # Fill with pooled original errors with no prediction change if necessary.
 for i,id in enumerate(source['ids']):
  if len(selected)>=16:break
  if str(id) not in seen and int(source['global_logits'][i].argmax())!=int(source['y'][i]):
   full=np.load(DEST/'pooled_fixed_pairs_source_out.npz')['full_logits'];selected.append(dict(fit='pooled',id=str(id),label=int(source['y'][i]),global_prediction=int(source['global_logits'][i].argmax()),pair_prediction=int(full[i].argmax()),controls={family:int(np.load(DEST/f'pooled_{family}_source_out.npz')['full_logits'][i].argmax()) for family in ['same_span_SVD','random_span']}));seen.add(str(id))
 fig,axs=plt.subplots(4,4,figsize=(16,16))
 for ax in axs.flat:ax.axis('off')
 for ax,c in zip(axs.flat,selected):
  with Image.open(ROOT/'domainbed/data/terra_incognita'/c['id']) as im:ax.imshow(im.convert('RGB'))
  ax.set_title(f"{c['fit']} | GT {names[c['label']]} | {Path(c['id']).stem[:8]}\nERM {names[c['global_prediction']]} / fixed-pair {names[c['pair_prediction']]}\nSVD {names[c['controls']['same_span_SVD']]} / random {names[c['controls']['random_span']]}",fontsize=9)
 fig.suptitle('Independent class-pair experts: held2 harms first; source-only diagnostic, no SIRM');fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(DEST/'cases_4x4.png',dpi=120);plt.close(fig);write(DEST/'grid_manifest.json',selected)
 print('maxreplay',maxerror,'inverse',inverse_error)
 for fit,x in norms.items():print(fit,x)
 for c in cases:
  if c['harm']:print(c['fit'],'harm',Path(c['id']).stem[:8],names[c['label']],'->',names[c['pair_prediction']])
if __name__=='__main__':main()

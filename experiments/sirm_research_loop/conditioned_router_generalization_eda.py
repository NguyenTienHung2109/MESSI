"""Read-only source utility generalization and worst-CE cases."""
import json
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr
from .shrinkage import ROOT,write,digest
from pathlib import Path
DEST=ROOT/'results/sirm_research_loop/iteration_19'
def main():
 a=json.loads((DEST/'source_analysis/OFF_results.json').read_text());names=json.loads((DEST/'experiment/subset_manifest.json').read_text())['class_names'];out={}
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from PIL import Image
 for state in ['selected','last']:
  s=a['banks']['OFF/'+state]['splits'];u=np.array(s['source_in']['U']);v=np.array(s['source_out']['U']);p=np.load(DEST/f'source_analysis/OFF_{state}_source_out_replay.npz');g=p['global_logits'];f=p['full_logits'];y=p['y'];ids=p['ids'];gc=g.argmax(1)==y;ce=lambda z:logsumexp(z,axis=1)-z[np.arange(len(y)),y];change=ce(f)-ce(g);ix=np.argsort(-change)[:16]
  cases=[dict(id=str(ids[i]),label=int(y[i]),global_prediction=int(g[i].argmax()),full_prediction=int(f[i].argmax()),global_correct=bool(gc[i]),CE_increase=float(change[i])) for i in ix]
  out[state]=dict(source_in_mean_expert_U=u.mean(0).tolist(),source_out_mean_expert_U=v.mean(0).tolist(),source_in_out_expert_spearman=float(spearmanr(u.mean(0),v.mean(0)).statistic),positive_U_train_cells=int((u>0).sum()),positive_train_negative_validation_cells=int(((u>0)&(v<0)).sum()),U_floor_train_cells=int((u>.002).sum()),U_floor_train_positive_validation_cells=int(((u>.002)&(v>0)).sum()),U_floor_train_validation_floor_passing_cells=int(((u>.002)&(v>.002)).sum()),CE_change_global_correct_mean=float(change[gc].mean()),CE_change_global_wrong_mean=float(change[~gc].mean()),worst_CE_changes=cases)
  fig,axes=plt.subplots(4,4,figsize=(16,16))
  for ax,x in zip(axes.flat,cases):
   ax.axis('off')
   with Image.open(ROOT/'domainbed/data/terra_incognita'/x['id']) as im:ax.imshow(im.convert('RGB'))
   ax.set_title(f"GT {names[x['label']]} | CE increase {x['CE_increase']:.3f}\nERM {names[x['global_prediction']]} -> OFF {names[x['full_prediction']]}\n{x['id'].split('/')[-1][:8]}",fontsize=9)
  fig.suptitle(f'Iteration19 OFF {state}:16 largest source-out CE increases; ranked examples, not prevalence');fig.tight_layout();fig.savefig(DEST/f'OFF_source_{state}_CE_failures_4x4.png',dpi=120);plt.close(fig)
  fig,ax=plt.subplots(figsize=(7,4));ax.hist(change[gc],bins=30,alpha=.65,label=f'Global correct (n={gc.sum()})');ax.hist(change[~gc],bins=30,alpha=.65,label=f'Global wrong (n={(~gc).sum()})');ax.axvline(0,color='black',linewidth=.8);ax.set(xlabel='CE(full) - CE(global); positive is worse',ylabel='Images',title=f'Iteration19 OFF {state}: source-out correction utility');ax.legend();fig.tight_layout();fig.savefig(DEST/f'OFF_source_{state}_CE_change.png',dpi=140);plt.close(fig)
 write(DEST/'OFF_source_generalization_gap.json',dict(source_only=True,target_read=False,fitting=False,states=out,code_sha256=digest(Path(__file__)),interpretation='Original source-in/out utility with same calibrated bank. Sign reversal is source generalization failure; cannot itself identify background shortcuts. Per-imageCE cohorts/image histograms unweighted; headline sourceaccuracy/CE equal-domain.'))
 print('Source generalization EDA complete',flush=True)
if __name__=='__main__':main()

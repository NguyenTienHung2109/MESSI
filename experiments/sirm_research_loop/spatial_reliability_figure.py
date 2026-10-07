"""Source-only utility heatmap and class-matrix coverage audit."""
import json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .spatial_bank_run import DEST
from .shrinkage import write,digest

def main():
 out=DEST/'source_spatial_reliability';path=out/'results.json';v=json.loads(path.read_text());coverage={}
 for k,s in v['states'].items():
  valid=[c for c in range(10) if all(x[str(c)]['complete_minimum_two'] for x in [s['splits'][sp]['class_conditioned'] for sp in ['source_in','source_out']]+s['class_halves'])]
  coverage[k]=dict(complete_class_matrices_in_all_four=valid,stable_shared_for_complete_classes={str(c):s['stable_class_shared'][str(c)] for c in valid},incomplete_classes=[c for c in range(10) if c not in valid])
 write(out/'class_coverage_audit.json',dict(source_only=True,target_read=False,results_sha256=digest(path),coverage=coverage,scope='Only three classes have complete n>=2 domain matrices in train/out/both halves. Other empty shared sets are not evidence of absence. Halves are descriptive, source-selected and not independent validation.'))
 keys=list(v['states']);fig,axes=plt.subplots(2,2,figsize=(15,7),squeeze=False);limit=max(abs(x) for k in keys for row in v['states'][k]['splits']['source_out']['U'] for x in row)
 for ax,k in zip(axes.flat,keys):
  s=v['states'][k];U=np.array(s['splits']['source_out']['U']);Q=np.array(s['splits']['source_out']['Q']);image=ax.imshow(U,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
  for d in range(3):
   for e in range(6):ax.text(e,d,f'{U[d,e]:.4f}'+(' *' if Q[d,e]>0 else ''),ha='center',va='center',fontsize=8,color='black')
  ax.set_xticks(range(6),[f'E{e}' for e in range(6)]);ax.set_yticks(range(3),['location100','location43','location46']);ax.set_title(k+f" (step{s['step']})")
 fig.subplots_adjust(right=.88,hspace=.35,wspace=.45);bar=fig.add_axes([.91,.18,.015,.6]);fig.colorbar(image,cax=bar,label='Marginal CE utility (loss units)');fig.suptitle('Source-out utility: frozen spatial versus parameter-matched CLS\n* = saved-state original-Q selected cell; not historical EMA or shared support',fontsize=11)
 fig.savefig(out/'source_utility_heatmap.png',dpi=150);plt.close(fig)
 (out/'interpretation.md').write_text('''# Source utility reliability: selected and common399

Saved-state original-view replay only, no target/fitting/training/Q change. Fixed U/V floors.002 and temperature.05. The four sets are source-in, source-out, and two deterministic class/domain-stratified source-out halves. These are descriptive reliability proxies at source-selected states, not independent validation or historical EMA.

Spatial bank has5/18 cells positive in all four sets, versus2/18 for matched CLS; only1/18 spatial cell exceeds U>.002 in all four, versus0/18 CLS. That spatial cell is location43/E5. It does not establish shared usefulness. Train has E5 U>.002 in location43 and46; source-out has E5>.002 only in location43 (selected .00718 vs .00088 in46). No expert crosses U floor in two source-out domains at either selected or399 state, even before V. Thus merely removing the V gate would not produce validated shared utility on this source-out diagnostic. This is a saved-state finding, not a proof that the current thresholds are globally optimal.

Class-conditioned counterfactual replay keeps minimum2 and complete-matrix validity. Only empty/opossum/raccoon (labels5/6/8) have complete domain matrices across train/out/both halves; none establish an expert-class shared subset in all four. Incomplete classes cannot be counted as demonstrated absence of support. The small halves, overlapping events and source-selected checkpoints limit reliability claims.

Spatial features may modestly stabilize positive utility signs, while effects remain weak and domain-specific. The source EDA explains why mean positivity is insufficient: E5 improves one raccoon case yet harms one opossum case in the same domain. No class-specific policy is proposed solely from this exploratory test, and no floor, Q formula or primary objective is changed before the full OFF/ON analysis.

[Raw matrices and class groups](results.json), [matrix coverage](class_coverage_audit.json), [utility heatmap](source_utility_heatmap.png).
''')

if __name__=='__main__':main()

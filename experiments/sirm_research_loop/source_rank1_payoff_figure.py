"""Read-only visualization of class utility and rank1 feasibility failures."""
import json
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .source_rank1_class_payoff import DEST
from .shrinkage import ROOT,write

def main():
 a=json.loads((DEST/'results.json').read_text());f=json.loads((DEST/'rank1_feasibility.json').read_text());names=json.loads((DEST.parent/'experiment/subset_manifest.json').read_text())['class_names'];cells=a['families']['risk_learned']['pooled']['cells']['source_out'];mat=np.full((3,10),np.nan);counts=np.zeros((3,10),int)
 for i,d in enumerate([0,2,3]):
  for c in range(10):
   v=cells.get(f'{d},{c}')
   if v:mat[i,c]=v['U'];counts[i,c]=v['n']
 fig,ax=plt.subplots(figsize=(14,4));im=ax.imshow(mat,cmap='RdBu',vmin=-.4,vmax=.4)
 for i in range(3):
  for c in range(10):
   if counts[i,c]:ax.text(c,i,f'{mat[i,c]:+.3f}\nn={counts[i,c]}',ha='center',va='center',fontsize=9)
 ax.set_xticks(range(10),names,rotation=25,ha='right');ax.set_yticks(range(3),['location100','location43','location46']);ax.set_title('Saved pooled risk-axis correction utility: CE(global)−CE(full), source-out\nCell means descriptive; n<2 insufficient; positive CE utility need not repair classification');fig.colorbar(im,ax=ax,label='CE utility');fig.tight_layout();fig.savefig(DEST/'class_domain_utility.png',dpi=140);plt.close(fig)
 rows=[x for x in f['families']['risk_learned']['cases'] if x['global_prediction']!=x['label'] and x['prediction']!=x['label']];unreachable=sorted([x for x in rows if not x['reachable_with_any_scalar']],key=lambda x:(x['label'],x['id']));reachable=sorted([x for x in rows if x['reachable_with_any_scalar']],key=lambda x:(x['label'],x['id']));selected=unreachable[:8]+reachable[:8];assert len(selected)==16
 fig,axs=plt.subplots(4,4,figsize=(16,16))
 for ax,x in zip(axs.flat,selected):
  ax.axis('off')
  with Image.open(ROOT/'domainbed/data/terra_incognita'/x['id']) as image:ax.imshow(image.convert('RGB'))
  status='GT unreachable on fixed rank1 axis' if not x['reachable_with_any_scalar'] else 'GT reachable, reader did not repair'
  ax.set_title(f"GT {names[x['label']]} {Path(x['id']).stem[:8]}\nERM {names[x['global_prediction']]} | rank1 {names[x['prediction']]}\n{status}",fontsize=9)
 fig.suptitle('Source rank1 failures: 8 output-axis infeasible + 8 feasible but missed; label-aware diagnostic');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(DEST/'feasibility_cases_4x4.png',dpi=120);plt.close(fig);write(DEST/'grid_manifest.json',selected)
 print('saved two source-only figures')
if __name__=='__main__':main()

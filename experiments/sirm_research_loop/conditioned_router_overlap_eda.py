"""Iteration18 saved-replay leave-one-out overlap and actual 4x4 cases; no fitting."""
import csv,json
import numpy as np
from .shrinkage import ROOT,write

def main():
 r=ROOT/'results/sirm_research_loop/iteration_19/eda'; rows=list(csv.DictReader((r/'cases.csv').open())); result={}
 for split in ['source_out','target_in','target_out']:
  s=[x for x in rows if x['split']==split]; experts=[]
  for e in range(6):
   harms=[x['image_id'] for x in s if x['B1_correct']=='True' and json.loads(x['B1_removal_predictions'])[e]!=int(x['true_class'])]
   repairs=[x['image_id'] for x in s if x['B1_correct']=='False' and json.loads(x['B1_removal_predictions'])[e]==int(x['true_class'])]
   experts.append(dict(expert=e,removal_harms=harms,removal_repairs=repairs))
  changes=[{k:x[k] for k in ['image_id','class_name','pattern','B0_prediction','B1_prediction','B0_top2_probability_margin','B1_top2_probability_margin']} for x in s if x['B0_prediction']!=x['B1_prediction']]
  result[split]=dict(n=len(s),experts=experts,unique_removal_harmed_ids=sorted(set(i for z in experts for i in z['removal_harms'])),global_full_changed_cases=changes)
 write(r/'expert_removal_overlap.json',result)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from PIL import Image
 names=json.loads((r.parent/'experiment/subset_manifest.json').read_text())['class_names']
 for split in ['source_out','target_in']:
  s=[x for x in rows if x['split']==split]; changed=sorted([x for x in s if x['B0_prediction']!=x['B1_prediction']],key=lambda x:x['image_id']); failed=sorted([x for x in s if x['pattern']=='WWW' and x not in changed],key=lambda x:(-float(x['B0_confidence']),x['image_id'])); chosen=(changed+failed)[:16]
  fig,axes=plt.subplots(4,4,figsize=(16,16))
  for ax in axes.flat:ax.axis('off')
  for ax,x in zip(axes.flat,chosen):
   with Image.open(ROOT/'domainbed/data/terra_incognita'/x['image_id']) as im:ax.imshow(im.convert('RGB'))
   ax.set_title(f"GT {x['class_name']} | {x['pattern']}\nERM {names[int(x['B0_prediction'])]} -> OFF=ON {names[int(x['B1_prediction'])]}\n{x['image_id'].split('/')[-1][:8]}",fontsize=9)
  fig.suptitle(split+' iteration19 selected250: all changed cases then high-confidence shared errors; selected examples, not random');fig.tight_layout();fig.savefig(r/(split+'_cases_4x4.png'),dpi=120);plt.close(fig)
  write(r/(split+'_cases_4x4_manifest.json'),chosen)
 print('Overlap and 4x4 EDA complete',flush=True)
if __name__=='__main__':main()

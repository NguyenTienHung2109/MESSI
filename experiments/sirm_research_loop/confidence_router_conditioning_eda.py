"""Source-only case/RMS/uncertainty analysis of locked factorial diagnostics."""
import json,csv
from pathlib import Path
import numpy as np
import torch
from .confidence_router_conditioning import OUT,BANK
from .shrinkage import write,digest,ROOT

def main():
 results=json.loads((OUT/'results.json').read_text());tables=[];details={};rng=np.random.default_rng(0)
 for state in ['selected','last']:
  details[state]={}
  for key,z in results['states'][state]['None']['fits'].items():
   path=OUT/f'{state}_heldNone_{key}.pt';assert digest(path)==z['model_sha256'];a=torch.load(path,map_location='cpu',weights_only=False)['replays']['source_out'];y=a['y'].numpy();dom=a['domains'].numpy();base=a['global_logits'].argmax(1).numpy()==y;learn=a['policies']['learned'].argmax(1).numpy()==y;constant=a['policies']['uniform_same_RMS'].argmax(1).numpy()==y
   def interval(v):
    draws=[]
    for d in np.unique(dom):
     values=v[dom==d];ix=rng.integers(len(values),size=(10000,len(values)));draws.append(values[ix].mean(1))
    return (np.quantile(np.mean(draws,0),[.025,.975])*100).tolist()
   changes=[]
   for i,id in enumerate(a['ids']):
    if base[i]!=learn[i]:changes.append(dict(id=str(id),y=int(y[i]),global_prediction=int(a['global_logits'][i].argmax()),learned=int(a['policies']['learned'][i].argmax()),kind='repair' if learn[i] else 'harm'))
   v=z['scores']['source_out'];hold=[results['states'][state][str(d)]['fits'][key]['scores']['source_out']['learned'] for d in [0,2,3]];row=dict(state=state,control=key,learned_acc=100*v['learned']['accuracy_equal_domain'],repairs=v['learned']['repairs'],harms=v['learned']['harms'],same_RMS_uniform_acc=100*v['uniform_same_RMS']['accuracy_equal_domain'],learned_native_RMS_acc=100*v['learned_native_RMS']['accuracy_equal_domain'],held_router_domain_acc=100*np.mean([x['accuracy_equal_domain'] for x in hold]),expert_weight_range=v['learned']['relative_expert_range'],CE=v['learned']['CE_equal_domain']);tables.append(row);details[state][key]=dict(summary=row,changed_cases=changes,learned_global_delta_CI_pp=interval(learn.astype(float)-base),learned_uniform_RMS_delta_CI_pp=interval(learn.astype(float)-constant))
 with (OUT/'factorial_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
 write(OUT/'case_and_uncertainty.json',details)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from PIL import Image
 names=json.loads((BANK/'experiment/subset_manifest.json').read_text())['class_names']
 for state in ['selected','last']:
  cases=details[state]['train_standardized_lr0.01']['changed_cases'];cases=sorted(cases,key=lambda x:(x['kind']!='harm',x['id']))[:16];fig,axes=plt.subplots(4,4,figsize=(16,16))
  for ax in axes.flat:ax.axis('off')
  for ax,x in zip(axes.flat,cases):
   with Image.open(ROOT/'domainbed/data/terra_incognita'/x['id']) as im:ax.imshow(im.convert('RGB'))
   ax.set_title(f"{x['kind']} | GT {names[x['y']]}\nERM {names[x['global_prediction']]} -> {names[x['learned']]}\n{x['id'].split('/')[-1][:8]}",fontsize=9)
  fig.suptitle(f'{state} bank: standardized descriptor lr0.01 diagnostic; harms then repairs by ID; extra400 router updates, source only');fig.tight_layout();fig.savefig(OUT/f'{state}_cases_4x4.png',dpi=120);plt.close(fig);write(OUT/f'{state}_grid_manifest.json',cases)
 print(json.dumps(details,indent=2),flush=True)
if __name__=='__main__':main()

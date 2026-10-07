import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
root=Path('/home/ntduc/Desktop/Dat/MESSI');out=root/'results/sirm_research_loop/iteration_19/source_rank3_channel_payoff';r=json.loads((out/'results.json').read_text());names=json.loads((root/'results/sirm_research_loop/iteration_19/experiment/subset_manifest.json').read_text())['class_names'];cases=json.loads((out/'cases.json').read_text());cases.sort(key=lambda c:(not(c['global']==c['label'] and c['full']!=c['label']),c['id']));fig,axs=plt.subplots(4,4,figsize=(16,16))
for ax in axs.flat:ax.axis('off')
for ax,c in zip(axs.flat,cases):
 with Image.open(root/'domainbed/data/terra_incognita'/c['id']) as im:ax.imshow(im.convert('RGB'))
 ax.set_title(f"GT {names[c['label']]} | {Path(c['id']).stem[:8]}\nERM {names[c['global']]} / full {names[c['full']]}\nCLS {names[c['CLS_only']]} / patch-max {names[c['patch_max_only']]}",fontsize=10)
fig.suptitle('Saved rank3 source-out: fixed input-channel contributions; no pixel intervention or new fitting');fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(out/'channel_cases_4x4.png',dpi=120);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5));fitnames=['pooled','held0','held2','held3'];positions=range(4)
for name in ['global','full','CLS','patch_max']:
 vals=[]
 for fit in fitnames:
  x=r['records'][fit]['learned']['source_out'];vals.append(100*(x[name]['accuracy'] if name in ['global','full'] else x['channels'][name]['standalone']['accuracy']))
 ax.plot(positions,vals,'o-',label=name)
ax.set_xticks(list(positions),fitnames);ax.set_ylabel('Source accuracy (%)');ax.legend();ax.set_title('Pooled/auxiliary-held rank3 learned reader; different cohorts across fits');fig.tight_layout();fig.savefig(out/'channel_payoff.png',dpi=150)

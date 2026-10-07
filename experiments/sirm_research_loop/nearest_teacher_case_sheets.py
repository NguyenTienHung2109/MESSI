"""Inspect source-only retrieval pairs on errors/harm; selection is descriptive."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .shrinkage import digest
B=Path('results/sirm_research_loop/iteration_08');out=B/'nearest_teacher_diagnosis/eda';out.mkdir(exist_ok=True);rows=list(csv.DictReader((out.parent/'cases.csv').open()));names=json.loads((B/'experiment/subset_manifest.json').read_text())['class_names'];groups={
'global_wrong_wrong_neighbor': [r for r in rows if r['mode']=='unrestricted' and r['global_pred']!=r['y'] and r['neighbor_y']!=r['y']],
'new_harms': [r for r in rows if r['mode']=='unrestricted' and r['global_pred']==r['y'] and r['neighbor_policy_pred']!=r['y']],
'repairs':[r for r in rows if r['mode']=='unrestricted' and r['global_pred']!=r['y'] and r['neighbor_policy_pred']==r['y']]}
chosen={}
for group,records in groups.items():
 records=sorted(records,key=lambda r:(-float(r['cosine']),r['query_id']))[:6];chosen[group]=records;fig,axes=plt.subplots(len(records),2,figsize=(12,3.4*len(records)),squeeze=False)
 for axrow,r in zip(axes,records):
  for ax,key in zip(axrow,['query_id','neighbor_id']):
   with Image.open(Path('domainbed/data/terra_incognita')/r[key]) as im:ax.imshow(im.convert('RGB'))
   ax.axis('off')
  axrow[0].set_title(f"Query {names[int(r['y'])]} / env{r['domain']}\nERM={names[int(r['global_pred'])]},1NN teacher={names[int(r['neighbor_policy_pred'])]}",fontsize=10);axrow[1].set_title(f"Neighbor {names[int(r['neighbor_y'])]} / env{r['neighbor_domain']}\ncos={float(r['cosine']):.3f},same sequence={r['same_sequence']}",fontsize=10)
 fig.suptitle('Source retrieval / '+group);fig.tight_layout();fig.savefig(out/f'{group}.png',dpi=90);plt.close(fig)
(out/'summary.json').write_text(json.dumps({'scope':'Highest-cosine examples per prespecified error/harm/repair group, not representative rates or training selection. Original source images only.','code_sha256':digest(Path(__file__)),'cases_csv_hash':digest(out.parent/'cases.csv'),'selected_pairs':chosen},indent=2)+'\n')

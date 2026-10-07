from pathlib import Path
import json,math
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
p=Path('results/sirm_research_loop/iteration_13/CLS_control_source_EDA');a=dict(np.load(p/'source_replay.npz'));mp=Path('results/sirm_research_loop/iteration_13/experiment/subset_manifest.json');m=json.loads(mp.read_text());names=m['class_names'];y=a['y'];g=a['g'];full=a['full'];gp=g.argmax(1);fp=full.argmax(1);comp=g.copy();comp[np.arange(len(y)),y]=-np.inf;cp=comp.argmax(1);gap=g[np.arange(len(y)),y]-g[np.arange(len(y)),cp];ii=np.flatnonzero((gp!=y)&(fp!=y));order=sorted(ii,key=lambda i:(gap[i],str(a['ids'][i])));chosen=order[:6]+order[-6:];rows=[]
lookup={r['id']:r for d in [0,2,3] for r in m['sources'][str(d)]['images']}
import hashlib
for i in chosen:
 id=str(a['ids'][i]);file=Path('domainbed/data/terra_incognita')/id;sha=hashlib.sha256(file.read_bytes()).hexdigest();assert sha==lookup[id]['sha256'];rows.append(dict(image_id=id,sha256=sha,environment=int(a['domains'][i]),label=names[int(y[i])],global_prediction=names[int(gp[i])],CLS_prediction=names[int(fp[i])],global_margin=float(gap[i]),single_unscaled_capacity=bool(((g[i][None,:]+a['delta'][i]).argmax(1)==y[i]).any()),selection='deepest_global_margin' if i in order[:6] else 'nearest_global_boundary'))
fig,axes=plt.subplots(4,3,figsize=(15,16));
for ax,r in zip(axes.flat,rows):
 ax.imshow(Image.open(Path('domainbed/data/terra_incognita')/r['image_id']).convert('RGB'));ax.axis('off');ax.set_title(f"env{r['environment']} true={r['label']}\nERM={r['global_prediction']} → CLS={r['CLS_prediction']}\ngap={r['global_margin']:.3f} capacity={r['single_unscaled_capacity']}",fontsize=10)
fig.tight_layout();fig.savefig(p/'persistent_wrong_margin_samples.png',dpi=110);plt.close(fig)
(p/'persistent_wrong_index.json').write_text(json.dumps(dict(source_only=True,target_read=False,scope='12 deterministic margin-extreme samples among persistent source errors; no visual census claim or label changes.',persistent_global_and_CLS_wrong=len(ii),cases=rows,visual_review_complete=False),indent=2)+'\n')
print(json.dumps(rows,indent=2))

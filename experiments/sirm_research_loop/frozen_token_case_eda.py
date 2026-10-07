"""All source correctness changes between fixed CLS and CLS+DIST readouts."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .shrinkage import ROOT,digest,write


def main():
    p=ROOT/'results/sirm_research_loop/iteration_12';out=p/'frozen_token_source_diagnosis';manifest=json.loads((p/'experiment/subset_manifest.json').read_text());lookup={v['id']:v for d in [0,2,3] for v in manifest['sources'][str(d)]['images']};names=manifest['class_names'];a=dict(np.load(out/'source_out_tokens.npz'));g=a['global_logits'];c=np.load(out/'all_source_in_repeat_cls_replay.npz')['source_out_logits'];d=np.load(out/'all_source_in_cls_dist_replay.npz')['source_out_logits'];y=a['y'];changed=np.flatnonzero((c.argmax(1)==y)!=(d.argmax(1)==y));rows=[]
    for i in changed:
        id=str(a['ids'][i]);file=ROOT/'domainbed/data/terra_incognita'/id;assert digest(file)==lookup[id]['sha256'];rows.append(dict(index=int(i),image_id=id,original_sha256=digest(file),label=names[int(y[i])],global_prediction=names[int(g[i].argmax())],CLS_prediction=names[int(c[i].argmax())],CLS_DIST_prediction=names[int(d[i].argmax())],DIST_repairs_CLS=bool(d[i].argmax()==y[i]),domain=int(a['domains'][i])))
    fig,axs=plt.subplots(4,2,figsize=(14,16));axs=axs.ravel()
    for ax,r in zip(axs,rows):
        ax.imshow(Image.open(ROOT/'domainbed/data/terra_incognita'/r['image_id']).convert('RGB'));ax.set_title(f"env{r['domain']} true={r['label']} | ERM={r['global_prediction']}\nCLS={r['CLS_prediction']} → CLS+DIST={r['CLS_DIST_prediction']}\n{Path(r['image_id']).name[:12]}",fontsize=10);ax.axis('off')
    for ax in axs[len(rows):]:ax.axis('off')
    fig.tight_layout();fig.savefig(out/'CLS_DIST_correctness_changes.png',dpi=110);plt.close(fig)
    write(out/'CLS_DIST_case_index.json',dict(scope='All correctness-changing source-out images between the predeclared fixed-L2 all-source-in matched readouts. Labels and selection unchanged; not target EDA or trained SIRM.',cases=rows,repairs=sum(r['DIST_repairs_CLS'] for r in rows),harms=sum(not r['DIST_repairs_CLS'] for r in rows),visual_review_complete=False))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()

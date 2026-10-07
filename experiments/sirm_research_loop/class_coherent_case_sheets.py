"""Every source-out correctness change versus image-hard; original images only."""
import json
import time
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .router_weight import live
from .shrinkage import ROOT, digest, write


def main():
    dest=ROOT/'results/sirm_research_loop/iteration_12';out=dest/'source_comparison'
    pid=json.loads((dest/'source_comparison_worker_pid.json').read_text())['pid']
    while live(pid):time.sleep(10)
    assert json.loads((out/'progress.json').read_text())['status']=='source_comparison_complete'
    path=out/'results.json';r=json.loads(path.read_text())
    manifest=json.loads((dest/'experiment/subset_manifest.json').read_text());names=manifest['class_names']
    lookup={entry['id']:entry for d in [0,2,3] for entry in manifest['sources'][str(d)]['images']}
    index=[]
    for state in ['selected','last']:
        cases=r['splits']['source_out']['correctness_changes'][state]['cases']
        for offset in range(0,len(cases),12):
            page=cases[offset:offset+12];fig,axes=plt.subplots((len(page)+3)//4,4,figsize=(16,4*((len(page)+3)//4)),squeeze=False)
            for ax in axes.flat:ax.axis('off')
            for ax,case in zip(axes.flat,page):
                image=ROOT/'domainbed/data/terra_incognita'/case['image_id']
                assert digest(image)==lookup[case['image_id']]['sha256']
                assert case['label']==lookup[case['image_id']]['class']
                with Image.open(image) as im:ax.imshow(im.convert('RGB'))
                g,old,new=[names[case[k]] for k in ['global_prediction','old_prediction','new_prediction']]
                ax.set_title(f"env{case['domain']} true={names[case['label']]}\nERM / image-hard / class-coherent\n{g} / {old} / {new}\nCE {case['old_CE']:.3f} → {case['new_CE']:.3f}",fontsize=9)
            steps=[r['locks'][key+'_'+state]['step'] for key in ['11','12']]
            fig.suptitle(f'Source-out: {state}, image-hard step{steps[0]} / class-coherent step{steps[1]}; every changed correctness case')
            fig.tight_layout();filename=f'{state}_changed_cases_{offset//12+1}.png';fig.savefig(out/filename,dpi=120);plt.close(fig)
            index.append(dict(filename=filename,state=state,steps=steps,cases=page,original_hashes_verified=True))
    write(out/'sheet_index.json',dict(code_sha256=digest(Path(__file__)),comparison_sha256=digest(path),
                                    target_read=False,sheets=index,
                                    all_source_out_correctness_changes_included=True,
                                    visually_reviewed=False,scope='All correctness changes, not all source images. Official labels unchanged.'))
    print(json.dumps(dict(sheets=len(index),cases=sum(len(x['cases']) for x in index))))

if __name__=='__main__':main()

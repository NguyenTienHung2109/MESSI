"""Matched-image error EDA on locked checkpoints; no fitting or selection."""
import csv,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover_gate.model import GateSIRM
from experiments.terra_10pct_factor_diagnosis.report import data_for
from .shrinkage import ROOT,head,read,digest,write
from .anchor_run import DEST,EXPERIMENT,ANCHOR

def classifier_stats(logits,y):
    prob=logits.softmax(1);top=prob.topk(2,dim=1)
    return dict(prediction=logits.argmax(1),confidence=top.values[:,0],top2_probability_margin=top.values[:,0]-top.values[:,1],true_probability=prob.gather(1,y[:,None])[:,0],CE=F.cross_entropy(logits,y,reduction='none'))

@torch.no_grad()
def run():
    torch.set_num_threads(4);out=DEST/'eda';out.mkdir(exist_ok=True)
    manifest=read(EXPERIMENT/'subset_manifest.json');models={};locks={};reports={}
    for arm in ['OFF','ON']:
        bank=EXPERIMENT/'runs/L38'/arm;report=read(bank/'posthoc/complete.json');sha=digest(bank/'source_best.pkl');assert sha==report['checkpoint_sha256']
        locks[arm]=sha;reports[arm]=report
        payload=torch.load(bank/'source_best.pkl',map_location='cpu',weights_only=False);c=payload['config']
        backbone=DeiTFeaturizer(pretrained=False,model_name=c['backbone']);model=GateSIRM(backbone,backbone.n_outputs,10,3,c)
        model.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()});model.configure_stage();models[arm]=model.eval()
    anchor=torch.load(ANCHOR,map_location='cpu',weights_only=False)['model_dict']
    for m in models.values():
        for name in ['featurizer','global_head']:
            assert all(torch.equal(v,anchor[name+'.'+k]) for k,v in getattr(m,name).state_dict().items())
    write(out/'checkpoint_locks.json',dict(checkpoints=locks,anchor_sha256=digest(ANCHOR),target_selected=False,scope='all cases from fixed source-out and target-in/out; exploratory, not training or selection'))
    all_rows=[];summary={};sheet_specs=[]
    for split,envs,key in [('source_out',[0,2,3],'out'),('target_in',[1],'in'),('target_out',[1],'out')]:
        ids=[];ys=[];domains=[];cols={arm:{k:[] for k in ['g','delta','pi']} for arm in models}
        for env,ds in sorted(data_for(manifest,envs,c,key).items()):
            entries=manifest['sources'][str(env)]['images'];ids.extend(entries[i]['id'] for i in ds.indices)
            for x,y in DataLoader(ds,batch_size=64,num_workers=2):
                z=models['OFF'].featurizer(x);ys.append(y);domains.append(torch.full_like(y,env))
                for arm,m in models.items():
                    for name,value in zip(['g','delta','pi'],head(m,z)):cols[arm][name].append(value)
        y=torch.cat(ys);env=torch.cat(domains);a={arm:{k:torch.cat(v) for k,v in col.items()} for arm,col in cols.items()}
        logits={'B0':a['OFF']['g']};utility={};removal={}
        for arm,label in [('OFF','B1'),('ON','B2')]:
            g=a[arm]['g'];term=a[arm]['delta']*a[arm]['pi'][:,:6,None];full=g+term.sum(1);logits[label]=full
            ce=F.cross_entropy(full,y,reduction='none')
            utility[label]=torch.stack([F.cross_entropy(full-term[:,m],y,reduction='none')-ce for m in range(6)],1)
            removal[label]=torch.stack([(full-term[:,m]).argmax(1) for m in range(6)],1)
        stats={k:classifier_stats(v,y) for k,v in logits.items()};correct={k:v['prediction'].eq(y) for k,v in stats.items()}
        for label,arm in [('B1','OFF'),('B2','ON')]:
            for domain in env.unique():
                mask=env==domain
                expected=reports[arm]['splits'][split]['source_accuracy'][str(int(domain))]
                assert abs(float(correct[label][mask].double().mean())-expected)<1e-12
        np.savez_compressed(out/(split+'_replay.npz'),ids=np.array(ids),y=y.numpy(),domains=env.numpy(),**{k:v.numpy() for k,v in logits.items()},**{arm+'_'+k:v.numpy() for arm,col in a.items() for k,v in col.items()},**{label+'_utility':v.numpy() for label,v in utility.items()})
        records=[]
        for i,image_id in enumerate(ids):
            row=dict(split=split,environment=int(env[i]),image_id=image_id,true_class=int(y[i]),class_name=manifest['class_names'][int(y[i])],pattern=''.join('C' if correct[k][i] else 'W' for k in ['B0','B1','B2']))
            for label,s in stats.items():
                for k,v in s.items():row[label+'_'+k]=int(v[i]) if k=='prediction' else float(v[i])
                row[label+'_correct']=bool(correct[label][i])
            row.update(B1_repairs_global=bool(~correct['B0'][i]&correct['B1'][i]),B1_harms_global=bool(correct['B0'][i]&~correct['B1'][i]),B2_repairs_B1=bool(~correct['B1'][i]&correct['B2'][i]),B2_harms_B1=bool(correct['B1'][i]&~correct['B2'][i]))
            for arm,label in [('OFF','B1'),('ON','B2')]:
                row[label+'_NULL']=float(a[arm]['pi'][i,-1]);row[label+'_routing']=json.dumps(a[arm]['pi'][i,:6].tolist())
                row[label+'_CE_change_vs_global']=row[label+'_CE']-row['B0_CE']
                row[label+'_marginal_utilities']=json.dumps(utility[label][i].tolist())
                row[label+'_removal_predictions']=json.dumps(removal[label][i].tolist())
                row[label+'_correction_norm']=float((a[arm]['delta'][i]*a[arm]['pi'][i,:6,None]).sum(0).norm())
            records.append(row)
        all_rows.extend(records)
        groups={}
        for pattern in sorted(set(r['pattern'] for r in records)):
            group=[r for r in records if r['pattern']==pattern]
            groups[pattern]=dict(n=len(group),fraction=len(group)/len(records),class_counts={name:sum(r['class_name']==name for r in group) for name in manifest['class_names']},mean_B0_confidence=float(np.mean([r['B0_confidence'] for r in group])),mean_B1_NULL=float(np.mean([r['B1_NULL'] for r in group])),mean_B2_NULL=float(np.mean([r['B2_NULL'] for r in group])))
            if pattern!='CCC':
                # High-confidence global failures/repairs are diagnostic examples, not a selected training cohort.
                chosen=sorted(group,key=lambda r:(-r['B0_confidence'],r['image_id']))[:12]
                sheet_specs.append(dict(split=split,pattern=pattern,cases=chosen))
        classes=[]
        for cls,name in enumerate(manifest['class_names']):
            mask=y==cls;n=int(mask.sum())
            classes.append(dict(name=name,n=n,**{k:float(v[mask].double().mean()) if n else None for k,v in correct.items()}))
        confusion={k:np.bincount((y*10+s['prediction']).numpy(),minlength=100).reshape(10,10).tolist() for k,s in stats.items()}
        summary[split]=dict(n=len(y),patterns=groups,per_class=classes,confusion=confusion,accuracy={k:float(v.double().mean()) for k,v in correct.items()},transitions={name:sum(r[name] for r in records) for name in ['B1_repairs_global','B1_harms_global','B2_repairs_B1','B2_harms_B1']})
    with (out/'cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    write(out/'case_summary.json',summary);write(out/'contact_sheet_index.json',sheet_specs)
    make_sheets(out,sheet_specs,manifest)
    for arm,sha in locks.items():assert digest(EXPERIMENT/'runs/L38'/arm/'source_best.pkl')==sha
    write(out/'progress.json',dict(status='replay_and_sheets_complete',remaining='inspect representative original images; distinguish supported observations from speculative mechanisms; derive falsifiable next hypothesis after main metrics'))

def make_sheets(out,specs,manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    folder=ROOT/'domainbed/data/terra_incognita'
    for spec in specs:
        cases=spec['cases'];n=len(cases);fig,axes=plt.subplots((n+3)//4,4,figsize=(16,4*((n+3)//4)),squeeze=False)
        for ax in axes.flat:ax.axis('off')
        for ax,row in zip(axes.flat,cases):
            with Image.open(folder/row['image_id']) as im:ax.imshow(im.convert('RGB'))
            pred='/'.join(manifest['class_names'][row[k+'_prediction']] for k in ['B0','B1','B2'])
            ax.set_title(f"{row['class_name']} | {row['pattern']}\n{pred}\np0={row['B0_confidence']:.3f} NULL1/2={row['B1_NULL']:.2f}/{row['B2_NULL']:.2f}",fontsize=9)
        fig.suptitle(spec['split']+' / '+spec['pattern']+' (B0/B1/B2; C=correct W=wrong)');fig.tight_layout()
        fig.savefig(out/(spec['split']+'_'+spec['pattern']+'.png'),dpi=120);plt.close(fig)

if __name__=='__main__':run()

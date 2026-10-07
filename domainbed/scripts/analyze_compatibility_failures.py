"""Next failure cycle: logit-scale/coverage controls and variance decomposition.

Temperature is a diagnostic fitted only on source calibration data. It changes
the certified predictor class; it is not silently substituted into deployment.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import torch
from torch.nn import functional as F

from domainbed.predictive_compatibility import CandidateMap
from domainbed.compatibility_diagnostics import prediction_metrics
from domainbed.scripts.audit_predictive_compatibility import write_json
from domainbed.scripts.check_compatibility_mechanisms import save_csv


def hierarchical_scatter(z,y,domain):
    """Exact weighted law of total variance; domains receive equal mass."""
    z=np.asarray(z,dtype=float);y=np.asarray(y);domain=np.asarray(domain)
    domains=np.unique(domain)
    weight=np.zeros(len(y))
    for d in domains:weight[domain==d]=1/(len(domains)*np.sum(domain==d))
    mean=(weight[:,None]*z).sum(0)
    total=float((weight*np.square(z-mean).sum(1)).sum())
    between=0.;conditional_domain=0.;residual=0.
    for c in np.unique(y):
        mask=y==c;mass=weight[mask].sum()
        mu=(weight[mask,None]*z[mask]).sum(0)/mass
        between+=mass*np.square(mu-mean).sum()
        for d in domains:
            cell=mask&(domain==d);cell_mass=weight[cell].sum()
            if not cell_mass:continue
            mu_d=(weight[cell,None]*z[cell]).sum(0)/cell_mass
            conditional_domain+=cell_mass*np.square(mu_d-mu).sum()
            residual+=(weight[cell]*np.square(z[cell]-mu_d).sum(1)).sum()
    if not np.isclose(total,between+conditional_domain+residual,atol=1e-10):
        raise AssertionError('Variance decomposition failed')
    return dict(total=total,between_class=float(between),within_class_domain=float(conditional_domain),
                within_class_domain_residual=float(residual),
                between_class_fraction=float(between/max(total,1e-30)),
                domain_given_class_fraction=float(conditional_domain/max(total,1e-30)))


def fit_scale(cells):
    """CE is convex in positive logit multiplier beta; no predictions change."""
    def objective(beta):
        return float(np.mean([F.cross_entropy(beta*x,y).item() for x,y in cells]))
    result=minimize_scalar(objective,bounds=(.125,16.),method='bounded',options={'xatol':1e-7})
    if not result.success:raise RuntimeError('Temperature fit failed')
    return float(result.x)


def load_bank(root):
    manifest=json.loads((root/'env0_seed0/features.json').read_text());bank={}
    for d in range(4):
        data=torch.load(Path('predictive_support_outputs/compatibility_cache')/
            f'env{d}_{manifest[str(d)]["fingerprint"][:16]}.pt',map_location='cpu',weights_only=True)
        assert data['fingerprint']==manifest[str(d)]['fingerprint']
        bank[d]=(data['x'],data['y'])
    return bank


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('predictive_support_outputs/pacs_compatibility_lodo_seed0'))
    parser.add_argument('--revision',type=Path,default=Path('predictive_support_outputs/pacs_transfer_compatibility_seed0'))
    parser.add_argument('--output',type=Path,default=Path('predictive_support_outputs/pacs_failure_cycle2_seed0'))
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use fresh output directory')
    args.output.mkdir(parents=True);torch.set_num_threads(2)
    write_json(args.output/'analysis_protocol.json',dict(seed=0,scale_bounds=[.125,16.],
        scale_fit='calibration only, equal domains',bank_controls=['single_global','all','global_only','proper_pairs_only'],
        variance_identity='total=between_class+domain_given_class+residual',
        note='Diagnostic only; altered scaling is outside the original certificate; no target selection'))
    bank=load_bank(args.root);rows=[];coverage=[];scatters=[];paired=[]
    for variant in ['erm','within_domain','cross_domain']:
        for target in range(4):
            for fold in ['main']+[f'loso_env{d}' for d in range(4) if d!=target]:
                folder=args.revision/variant/f'env{target}_{fold}'
                state=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
                selection=json.loads((folder/'selection.json').read_text());sources=selection['sources'];held=selection['held']
                assert held not in sources and target not in sources
                splits={int(d):{k:np.asarray(v) for k,v in s.items()} for d,s in json.loads((folder/'splits.json').read_text()).items()}
                maps=[]
                for m in state['maps']:
                    model=CandidateMap(384,16,64);model.load_state_dict(m);maps.append(model.eval().requires_grad_(False))
                @torch.no_grad()
                def predict(x):
                    features=torch.stack([m(x) for m in maps],1)
                    logits=torch.stack([features[:,i].double()@w.T for i,w in enumerate(state['heads'])],1)
                    return features,logits
                groups={'single_global':[0],'all':list(range(len(maps))),'global_only':[0,len(maps)-1]}
                if len(maps)>2:groups['proper_pairs_only']=list(range(1,len(maps)-1))
                cal={d:predict(bank[d][0][splits[d]['calibration']])[1] for d in sources}
                scales={name:fit_scale([(cal[d][:,ids].mean(1),bank[d][1][splits[d]['calibration']]) for d in sources])
                        for name,ids in groups.items()}
                print(variant,target,fold,'scales',scales,flush=True)
                combined_z=[];combined_y=[];combined_d=[]
                supports=[c['support'] for c in json.loads((folder/'certificates.json').read_text())]
                eval_sets=[('held',held,*bank[held])]+[('source_selection',d,bank[d][0][splits[d]['selection']],bank[d][1][splits[d]['selection']]) for d in sources]
                for stage,d,x,y in eval_sets:
                    z,logits=predict(x)
                    if stage=='source_selection':
                        combined_z.append(z[:,0].numpy());combined_y.append(y.numpy());combined_d.append(np.full(len(y),d))
                    else:
                        scatters.append(dict(variant=variant,target=target,fold=fold,stage=stage,
                            **hierarchical_scatter(z[:,0].numpy(),y.numpy(),np.full(len(y),d))))
                    centered=logits-logits.mean(2,keepdim=True)
                    for name,ids in groups.items():
                        base=logits[:,ids].mean(1)
                        scaled=base*scales[name]
                        assert torch.equal(base.argmax(1),scaled.argmax(1))
                        for scale_name,prediction in [('raw',base),('calibrated',scaled)]:
                            m=prediction_metrics(prediction,y)
                            ce=F.cross_entropy(prediction,y,reduction='none')
                            correct=prediction.argmax(1)==y
                            base_center=prediction-prediction.mean(1,keepdim=True)
                            margin=prediction[torch.arange(len(y)),y]-prediction.masked_fill(F.one_hot(y,7).bool(),-torch.inf).max(1).values
                            rows.append(dict(variant=variant,target=target,fold=fold,stage=stage,domain=d,
                                control=name,scaling=scale_name,beta=scales[name] if scale_name=='calibrated' else 1.,
                                accuracy=m['accuracy'],ce=m['cross_entropy'],confidence=m['mean_confidence'],ece=m['ece_15'],
                                centered_logit_norm=float(base_center.norm(dim=1).mean()),
                                mean_margin=float(margin.mean()),
                                ce_correct=float(ce[correct].mean()) if correct.any() else None,
                                ce_wrong=float(ce[~correct].mean()) if (~correct).any() else None,
                                logit_cancellation=float((centered[:,ids].square().sum(2).mean(1)-centered[:,ids].mean(1).square().sum(1)).mean())))
                    baseline=logits[:,0].argmax(1);mixture=logits.mean(1).argmax(1)
                    for c in range(7):
                        mask=y==c
                        paired.append(dict(variant=variant,target=target,fold=fold,stage=stage,domain=d,label=c,n=int(mask.sum()),
                            global_correct=int(((baseline==y)&mask).sum()),mixture_correct=int(((mixture==y)&mask).sum()),
                            mixture_rescues=int(((baseline!=y)&(mixture==y)&mask).sum()),
                            mixture_breaks=int(((baseline==y)&(mixture!=y)&mask).sum())))
                    for i,support in enumerate(supports):
                        m=prediction_metrics(logits[:,i],y)
                        coverage.append(dict(variant=variant,target=target,fold=fold,stage=stage,domain=d,expert=i,
                            in_support=d in support,accuracy=m['accuracy'],ce=m['cross_entropy'],
                            macro_class_ce=float(np.mean(list(m['per_class_ce'].values())))))
                scatters.append(dict(variant=variant,target=target,fold=fold,stage='source_selection',
                    **hierarchical_scatter(np.concatenate(combined_z),np.concatenate(combined_y),np.concatenate(combined_d))))
    for name,table in [('scale_and_bank.csv',rows),('support_coverage.csv',coverage),('variance_decomposition.csv',scatters),('class_error_changes.csv',paired)]:
        save_csv(args.output/name,table)
    (args.output/'analysis_source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output/'analysis_status.json',dict(state='completed',folds=48))


if __name__=='__main__':main()

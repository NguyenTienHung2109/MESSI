"""Plot completed mechanism diagnostics without fitting or selecting models."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from domainbed.predictive_compatibility import CandidateMap
from domainbed.scripts.check_compatibility_mechanisms import save_csv


def read_csv(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def spectra(root,output):
    rows=[]
    for t in range(4):
        folder=root/f'env{t}_seed0'
        original=torch.load(folder/'main/model.pt',map_location='cpu',weights_only=True)
        diagnostic=torch.load(output/f'env{t}_main/diagnostic_models.pt',map_location='cpu',weights_only=True)
        models=[]
        for dim,state in [(16,original['maps'][0]),(64,diagnostic['wide_map'])]:
            m=CandidateMap(384,dim,64);m.load_state_dict(state);models.append(m.eval().requires_grad_(False))
        manifest=json.loads((folder/'features.json').read_text())
        splits=json.loads((folder/'main/split_manifest.json').read_text())
        for d in range(4):
            payload=torch.load(Path('predictive_support_outputs/compatibility_cache')/
                f'env{d}_{manifest[str(d)]["fingerprint"][:16]}.pt',map_location='cpu',weights_only=True)
            x=payload['x'] if d==t else payload['x'][splits[str(d)]['selection']]
            for dim,m in zip([16,64],models):
                with torch.no_grad():z=m(x).double()
                z=z-z.mean(0)
                eigen=torch.linalg.eigvalsh(z.T@z/(len(z)-1)).clamp_min(0)
                p=eigen/eigen.sum()
                rows.append(dict(target=t,domain=d,split='held' if d==t else 'source_selection',dimension=dim,
                    variance_trace=float(eigen.sum()),effective_rank=float((-(p*p.clamp_min(1e-30).log()).sum()).exp()),
                    top6_variance_fraction=float(p[-6:].sum())))
    save_csv(output/'representation_spectrum.csv',rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('predictive_support_outputs/pacs_compatibility_lodo_seed0'))
    parser.add_argument('--output',type=Path,default=Path('predictive_support_outputs/pacs_mechanism_checks_seed0'))
    args=parser.parse_args();torch.set_num_threads(2)
    rows=read_csv(args.output/'metrics.csv');matched=read_csv(args.output/'matched_probe_metrics.csv')
    comparison=read_csv(args.output/'comparisons.csv');curve=read_csv(args.output/'bound_sample_complexity.csv')
    spectral=spectra(args.root,args.output)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for dim in [1,16,64,384]:
        r=[x for x in curve if int(x['dimension'])==dim]
        axes[0,0].plot([int(x['n']) for x in r],[2*float(x['epsilon']) for x in r],label=f'd={dim}')
    axes[0,0].axhline(.12,color='black',linestyle='--',label='tau=.12')
    axes[0,0].axvspan(334,786,color='gray',alpha=.2,label='PACS audit size')
    axes[0,0].set(xscale='log',yscale='log',xlabel='Audit samples / domain',ylabel='2 epsilon (nats)',title='1. Statistical power, zero empirical regret')
    axes[0,0].legend(fontsize=8)
    for i,m in enumerate(['router_learned','router_half_uniform','router_constant']):
        values=[float(x['ce_delta']) for x in comparison if x['fold']!='main' and x['split']=='held' and x['first']==m and x['reference']=='router_uniform']
        axes[0,1].scatter(np.full(len(values),i)+np.linspace(-.08,.08,len(values)),values,alpha=.65)
        axes[0,1].plot([i-.2,i+.2],[np.mean(values)]*2,color='black')
    axes[0,1].axhline(0,color='black',linestyle='--')
    axes[0,1].set(xticks=range(3),xticklabels=['Learned','Half-uniform','Constant'],ylabel='CE minus uniform (negative = better)',title='2. Strict excluded-source router checks')
    conditions=[('MLP16 original',rows,'global_original'),('Backbone r1',matched,'backbone384_traincal_mean_r1'),
                ('Backbone r4',matched,'backbone384_traincal_mean_r4'),('MLP16 mean r1',matched,'mlp16_traincal_mean_r1'),
                ('MLP64 minimax',rows,'mlp64_minimax_r1')]
    for i,(_,table,method) in enumerate(conditions):
        target=np.mean([float(x['accuracy']) for x in table if x['method']==method and x['fold']=='main' and x['split']=='held'])*100
        transfer=np.mean([float(x['accuracy']) for x in table if x['method']==method and x['fold']!='main' and x['split']=='held'])*100
        axes[1,0].bar(i-.18,target,width=.35,color='tab:blue',label='4 target mean' if i==0 else None)
        axes[1,0].bar(i+.18,transfer,width=.35,color='tab:orange',label='12 excluded-source mean' if i==0 else None)
    axes[1,0].set(xticks=range(len(conditions)),xticklabels=[x[0] for x in conditions],ylabel='Accuracy (%)',ylim=(0,100),title='3. Probe and head-objective controls')
    axes[1,0].tick_params(axis='x',labelrotation=20);axes[1,0].legend(fontsize=8)
    for dim,color in [(16,'tab:blue'),(64,'tab:orange')]:
        r=[x for x in spectral if x['dimension']==dim and x['split']=='held']
        axes[1,1].plot([x['target'] for x in r],[x['effective_rank'] for x in r],'o-',color=color,label=f'MLP output {dim}')
    axes[1,1].set(xticks=range(4),xticklabels=['Art','Cartoon','Photo','Sketch'],ylabel='Covariance effective rank',title='4. Wider representation: effective rank')
    axes[1,1].legend()
    fig.savefig(args.output/'mechanism_checks.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()

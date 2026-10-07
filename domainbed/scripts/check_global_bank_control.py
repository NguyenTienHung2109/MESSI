"""Same-size global-bank control with matched per-expert sample exposure."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from domainbed.predictive_compatibility import CandidateMap,project,make_cell,solve_common
from domainbed.compatibility_diagnostics import prediction_metrics,log_event
from domainbed.scripts.audit_predictive_compatibility import write_json
from domainbed.scripts.analyze_compatibility_failures import load_bank,fit_scale
from domainbed.scripts.check_compatibility_mechanisms import save_csv


def allocation(total,capacities):
    """Deterministic proportional allocation with exact sum and capacity bound."""
    capacities=np.asarray(capacities,dtype=int)
    if total>capacities.sum() or total<0:raise ValueError('Infeasible allocation')
    ideal=total*capacities/capacities.sum();count=np.floor(ideal).astype(int)
    for j in np.argsort(-(ideal-count),kind='stable')[:total-count.sum()]:count[j]+=1
    assert count.sum()==total and (count<=capacities).all()
    return count


def train_global(bank,splits,sources,args,seed,total_batch,log_path):
    torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed)
    model=CandidateMap(384,16,64);head=nn.Parameter(torch.zeros(7,16))
    optimizer=torch.optim.Adam(list(model.parameters())+[head],lr=args.learning_rate)
    for step in range(args.map_steps):
        losses=[];counts=[]
        for i,d in enumerate(sources):
            # Rotate remainder allocation so no domain systematically receives
            # the extra example. Domain losses still have equal weight.
            size=total_batch//len(sources)+int((i-step)%len(sources)<total_batch%len(sources))
            ix=torch.as_tensor(splits[d]['train'])
            selected=ix[torch.randint(len(ix),(size,),generator=rng)]
            losses.append(F.cross_entropy(model(bank[d][0][selected])@head.T,bank[d][1][selected]));counts.append(size)
        assert sum(counts)==total_batch
        loss=torch.stack(losses).mean();optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():head.copy_(project(head,args.radius))
        if step%100==0 or step==args.map_steps-1:
            log_event(log_path,dict(step=step,expert=seed,ce=float(loss.detach()),domain_batch_sizes=counts,total_batch=total_batch))
    return model.eval().requires_grad_(False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('predictive_support_outputs/pacs_compatibility_lodo_seed0'))
    parser.add_argument('--banks',type=Path,default=Path('predictive_support_outputs/pacs_failure_cycle2_seed0/token_banks'))
    parser.add_argument('--tokens',type=Path,default=Path('predictive_support_outputs/pacs_failure_cycle2_seed0/token_ablation'))
    parser.add_argument('--output',type=Path,default=Path('predictive_support_outputs/pacs_failure_cycle2_seed0/global_bank_control'))
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use fresh control output directory')
    args.output.mkdir(parents=True);torch.set_num_threads(2)
    write_json(args.output/'protocol.json',dict(seed=0,variants=['erm','dist_erm','mean_erm'],
        controls=['original_pair_bank','five_global_bank'],main_runs=12,experts=5,
        shared_experts=[0,4],new_global_experts=[1,2,3],steps=500,
        matched='same map architecture, expert initializations, total train draws/step and calibration sample count per expert',
        different='domain coverage necessarily changes; new global heads fit three domains with fewer calibration examples/domain',
        selection='none, descriptive intervention; two-source folds cannot have proper pair subsets'))
    base=load_bank(args.root);dist={d:torch.load(args.tokens/f'dist_env{d}.pt',map_location='cpu',weights_only=True)['dist'] for d in base}
    rows=[]
    for variant in ['erm','dist_erm','mean_erm']:
        bank={d:((x if variant=='erm' else dist[d] if variant=='dist_erm' else (x+dist[d])/2),y) for d,(x,y) in base.items()}
        for target in range(4):
            folder=args.banks/variant/f'env{target}_main';destination=args.output/variant/f'env{target}';destination.mkdir(parents=True)
            payload=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            config=SimpleNamespace(**payload['config'])
            splits={int(d):{k:np.array(v) for k,v in s.items()} for d,s in json.loads((folder/'splits.json').read_text()).items()}
            sources=sorted(splits);assert target not in sources
            supports=[r['support'] for r in json.loads((folder/'certificates.json').read_text())]
            original=[]
            for state in payload['maps']:
                m=CandidateMap(384,16,64);m.load_state_dict(state);original.append(m.eval().requires_grad_(False))
            global_maps=list(original);global_heads=list(payload['heads']);certificate=[]
            for expert in [1,2,3]:
                print('GLOBAL CONTROL',variant,target,expert,flush=True)
                batch_total=config.batch_size*len(supports[expert])
                model=train_global(bank,splits,sources,config,expert,batch_total,destination/'training.jsonl')
                n_cal=sum(len(splits[d]['calibration']) for d in supports[expert])
                counts=allocation(n_cal,[len(splits[d]['calibration']) for d in sources])
                cells=[];used={}
                for d,n in zip(sources,counts):
                    ix=np.random.default_rng(1009*d+expert).choice(splits[d]['calibration'],int(n),replace=False)
                    used[str(d)]=ix.tolist()
                    with torch.no_grad():z=model(bank[d][0][ix])
                    cells.append(make_cell(z,bank[d][1][ix]))
                w,solver=solve_common(cells,7,1.,1000,1e-5)
                global_maps[expert]=model;global_heads[expert]=w
                certificate.append(dict(expert=expert,original_support=supports[expert],new_support=sources,
                    train_draws_per_step=batch_total,calibration_n=n_cal,calibration_indices=used,solver=solver))
            @torch.no_grad()
            def predict(x,kind):
                maps,heads=(original,payload['heads']) if kind=='original_pair_bank' else (global_maps,global_heads)
                return torch.stack([m(x).double()@w.T for m,w in zip(maps,heads)],1).mean(1)
            scales={kind:fit_scale([(predict(bank[d][0][splits[d]['calibration']],kind),bank[d][1][splits[d]['calibration']]) for d in sources])
                    for kind in ['original_pair_bank','five_global_bank']}
            write_json(destination/'adaptation.json',dict(sources=sources,held=target,reference_heads=certificate,scales=scales))
            torch.save(dict(maps=[m.state_dict() for m in global_maps],heads=global_heads,feature_mode=config.feature_mode),destination/'model.pt')
            for stage,d,x,y in [('held',target,*bank[target])]+[('source_selection',d,bank[d][0][splits[d]['selection']],bank[d][1][splits[d]['selection']]) for d in sources]:
                outputs={}
                for kind in ['original_pair_bank','five_global_bank']:
                    outputs[kind]=predict(x,kind)
                    for calibration,beta in [('raw',1.),('calibrated',scales[kind])]:
                        m=prediction_metrics(outputs[kind]*beta,y)
                        rows.append(dict(variant=variant,target=target,stage=stage,domain=d,bank=kind,
                            scaling=calibration,beta=beta,accuracy=m['accuracy'],ce=m['cross_entropy'],ece=m['ece_15']))
                if stage=='held':np.savez_compressed(destination/'held_predictions.npz',labels=y.numpy(),**{k:v.numpy() for k,v in outputs.items()})
            save_csv(args.output/'metrics.csv',rows)
    (args.output/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output/'status.json',dict(state='completed',seed=0,main_runs=12))


if __name__=='__main__':main()

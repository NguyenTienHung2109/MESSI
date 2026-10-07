"""No-retrain raw-vs-Q-weighted IRMv1 diagnosis at the saved R1 checkpoint."""
import argparse,itertools,json
from pathlib import Path
import torch
import torch.nn.functional as F
from experiments.sirm_h1_h6 import run as base
from experiments.sirm_discover_gate.run import restore
from experiments.sirm_discover.math import component_eligibility
from experiments.sirm_representation_h2.summarize import write_csv
from domainbed.scripts.train_subset_irm_pacs import make_schedule

def norm(grads):return float(sum(g.detach().double().square().sum() for g in grads if g is not None).sqrt())

def inspect(root,out,data_dir):
    out.mkdir(parents=True,exist_ok=True);bank=root/'TerraIncognita/banks/R1';checkpoint=bank/'checkpoints/best.pkl';original_hash=base.digest(checkpoint)
    model,adapter,effective,split=restore(bank,'TerraIncognita','R1',0,data_dir);c=effective['config'];steps=sum(c[k] for k in ['global_steps','discover_steps','invariance_steps'])
    fit=adapter.datasets_for(data_dir,split,c,train=True,cohort='balanced_views');scheduled={e:base.ScheduledImages(ds,0,e) for e,ds in fit.items()};schedules={e:make_schedule(list(range(len(ds))),steps,c['batch_size'],0,e) for e,ds in fit.items()}
    refresh=[json.loads(x) for x in (bank/'q_refresh.jsonl').read_text().splitlines()];refresh=[r for r in refresh if r['stage']=='invariance' and r['eligible_pair_expert_count']>0]
    rows=[];params=list(model.residuals.parameters());model.configure_stage();model.eval()
    for r in refresh:
        batches=[]
        for env in sorted(fit):
            samples=[scheduled[env][(int(i),r['step'],pos)] for pos,i in enumerate(schedules[env][r['step']])]
            batches.append((torch.stack([x for x,y in samples]).cuda(),torch.tensor([y for x,y in samples],device='cuda')))
        x=torch.cat([x for x,y in batches]);y=torch.cat([y for x,y in batches]);ds=torch.arange(3,device='cuda').repeat_interleave(c['batch_size'])
        f=model(x);q=torch.tensor(r['q'],device='cuda');valid,ess,eligible=component_eligibility(f['pi'],ds,q,c['min_ess'])
        term=f['pi'][:,:6,None].detach()*f['delta'];context=f['global_logits'].detach()+term.detach().sum(1)
        for i,j in itertools.combinations(range(3),2):
            for m in range(6):
                a=q[i,m]*q[j,m]
                if a<=0 or not valid[i,m] or not valid[j,m]:continue
                gs=[]
                for d in [i,j]:
                    mask=ds==d;s=term.new_ones((),requires_grad=True)
                    risk=F.cross_entropy(context[mask]-term[mask,m].detach()+s*term[mask,m],y[mask]);gs.append(torch.autograd.grad(risk,s,create_graph=True,retain_graph=True)[0])
                raw=gs[0].square()+gs[1].square();graw=torch.autograd.grad(raw,params,retain_graph=True,allow_unused=True);gweighted=torch.autograd.grad(a*raw,params,retain_graph=True,allow_unused=True)
                rows.append(dict(Q_step=r['step'],batch_step=r['step'],checkpoint_step=int(model.step)-1,source_i=sorted(fit)[i],source_j=sorted(fit)[j],expert=m,pair_weight=float(a),G_i=float(gs[0]),G_j=float(gs[1]),raw_IRM_loss=float(raw),raw_grad_norm=norm(graw),Q_weighted_grad_norm=norm(gweighted),current_lambda_weighted_fixed_divisor_grad_norm=norm(gweighted)*r['effective_lambda_sirm']/3,ESS_i=float(ess[i,m]),ESS_j=float(ess[j,m]),checkpoint_eligible_terms=eligible,recorded_run_eligible_terms=r['eligible_pair_expert_count']))
        print('inspected source batch '+str(r['step']),flush=True)
    assert base.digest(checkpoint)==original_hash and rows
    write_csv(out/'raw_pair_gradients.csv',rows)
    result=dict(checkpoint_sha256=original_hash,checkpoint_step=int(model.step)-1,historical_Q_refreshes=len(refresh),eligible_pairs_inspected=len(rows),raw_grad_min=min(r['raw_grad_norm'] for r in rows),raw_grad_max=max(r['raw_grad_norm'] for r in rows),Q_weight_min=min(r['pair_weight'] for r in rows),Q_weight_max=max(r['pair_weight'] for r in rows),weighted_grad_max=max(r['Q_weighted_grad_norm'] for r in rows),raw_gradient_larger_than_1e_minus10=all(r['raw_grad_norm']>1e-10 for r in rows),method='Saved final source-selected R1 weights, historical eligible Q and exact scheduled source minibatches; original intermediate parameter states were not saved, so these are not historical gradients at those steps.',retraining=False,target_read=False)
    base.write(out/'diagnosis.json',result)
    text=['# Raw IRMv1 / Q attenuation diagnosis','',result['method'],'', 'No training, target reads, parameter updates or checkpoint changes. Eligibility/ESS is recomputed on the saved checkpoint. Pair raw loss = G_i²+G_j²; pair gradient columns exclude lambda and the original fixed domain-pair divisor unless explicitly named.','', '| Q/batch step | Pair/expert | Q weight | Raw loss | Raw gradient norm | Q-weighted gradient norm |','|---|---|---:|---:|---:|---:|']
    for r in rows:text.append(f'| {r["Q_step"]} | {r["source_i"]}–{r["source_j"]}/E{r["expert"]} | {r["pair_weight"]:.7g} | {r["raw_IRM_loss"]:.7g} | {r["raw_grad_norm"]:.7g} | {r["Q_weighted_grad_norm"]:.7g} |')
    text.extend(['',json.dumps(result,indent=2),'', 'The requested binary test is justified when raw gradients are materially larger than the Q-weighted gradients. This is evidence about attenuation at the saved checkpoint, not a claim that binary-gated IRM improves OOD accuracy.'])
    (out/'summary.md').write_text('\n'.join(text)+'\n');print(json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='results/sirm_discover_dynamic/terra_L38_seed0');p.add_argument('--output',default='results/sirm_discover_binary/raw_diagnosis');p.add_argument('--data-dir',default='domainbed/data');a=p.parse_args();torch.set_num_threads(4);inspect(Path(a.root),Path(a.output),a.data_dir)

if __name__=='__main__':main()

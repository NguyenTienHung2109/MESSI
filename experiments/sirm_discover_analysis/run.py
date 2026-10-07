"""Post-hoc equality, assignment support, and residual contribution statistics.

Read-only with respect to training/checkpoints; no selection or retraining.
"""
import argparse,itertools,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from experiments.sirm_h1_h6 import run as base
from experiments.sirm_discover.math import soft_null_q
from experiments.sirm_discover_gate.run import restore
from experiments.sirm_discover_gate.data import SPECS
from experiments.sirm_representation_h2.summarize import write_csv

ARMS=['R0','R1','R2']

def read(p):return json.loads(p.read_text())
def double_center(u):return u-u.mean(1,keepdims=True)-u.mean(0,keepdims=True)+u.mean()
def support(q):
    q=np.asarray(q);s=q>0
    return dict(selected_cells=int(s.sum()),support_per_domain=s.sum(1).tolist(),domains_per_expert=s.sum(0).tolist(),shared_experts=int((s.sum(0)>=2).sum()),pair_expert_count=sum(int((s[i]&s[j]).sum()) for i,j in itertools.combinations(range(len(q)),2)),pair_weight_sum=sum(float((q[i]*q[j]).sum()) for i,j in itertools.combinations(range(len(q)),2)),q_mass_mean=float(q.sum(1).mean()))

def tensors_compare(a,b):
    if a.keys()!=b.keys():raise ValueError('state keys differ')
    keys=[];elements=0;different=0;maximum=0.
    for k,x in a.items():
        y=b[k];elements+=x.numel()
        if not torch.equal(x,y):
            keys.append(k);different+=int((x!=y).sum());maximum=max(maximum,float((x.double()-y.double()).abs().max()))
    return dict(elements=elements,different_elements=different,max_abs_difference=maximum,different_keys=keys)

def checkpoint_stats(bankroot):
    result=[]
    for checkpoint in ['best','last']:
        states={a:base.load(bankroot/a/'checkpoints'/f'{checkpoint}.pkl')['model_dict'] for a in ARMS}
        for a,b in itertools.combinations(ARMS,2):
            modules={}
            for module in ['featurizer','global_head','residuals','router']:
                modules[module]=tensors_compare({k:v for k,v in states[a].items() if k.startswith(module+'.')},{k:v for k,v in states[b].items() if k.startswith(module+'.')})
            allstate=tensors_compare(states[a],states[b]);result.append(dict(checkpoint=checkpoint,pair=a+'/'+b,modules=modules,all_state=allstate))
        del states
    return result

def trajectory_stats(bankroot):
    logs={a:[json.loads(x) for x in (bankroot/a/'diagnostics.jsonl').read_text().splitlines()] for a in ARMS};result=[]
    keys=['loss','loss_cls','loss_discover','loss_router','loss_balance','loss_sirm','residual_rms','router_null_mass','utility_ema','utility_interaction']
    for a,b in itertools.combinations(ARMS,2):
        assert len(logs[a])==len(logs[b])==5001
        for key in keys:
            x=np.array([r[key] for r in logs[a]]);y=np.array([r[key] for r in logs[b]])
            result.append(dict(pair=a+'/'+b,key=key,steps=5001,max_abs_difference=float(np.max(np.abs(x-y))),different_values=int(np.count_nonzero(x!=y))))
    return result,logs

def q_statistics(logs,config):
    rows=[];seen=-1
    for r in logs:
        if r['utility_updates']==seen or r['utility_updates']==0:continue
        seen=r['utility_updates'];u=torch.tensor(r['utility_ema'],dtype=torch.float32)
        valid=torch.ones_like(u,dtype=torch.bool)
        q,null,c,v=soft_null_q(u,valid,config['q_temperature'],config['utility_floor'],config['interaction_floor'])
        row=dict(step=r['step'],stage=r['stage'],U_min=float(u.min()),U_max=float(u.max()),V_min=float(v.min()),V_max=float(v.max()),U_above_floor=int((u>config['utility_floor']).sum()),V_above_floor=int((v>config['interaction_floor']).sum()),joint_above_floors=int(((u>config['utility_floor'])&(v>config['interaction_floor'])).sum()),candidate_NULL_mean=float(null.mean()),candidate_confidence_mean=float(c.mean()),**support(q.numpy()))
        rows.append(row)
    return rows

@torch.no_grad()
def replay(bankroot,dataset,data_dir,out):
    outputs={};details={}
    for arm in ARMS:
        model,adapter,effective,split=restore(bankroot/arm,dataset,arm,0,data_dir);config=effective['config'];env=SPECS[dataset]['target']
        data=ImageFolder(adapter.folder(data_dir,env),base.ViewTransform(config,'original',False,env,'balanced_views'));assert data.classes==split['class_names']
        arrays={k:[] for k in ['logits','global_logits','delta','pi','y']}
        for x,y in DataLoader(data,batch_size=config['eval_batch_size'],num_workers=2):
            f=model(x.cuda())
            for k in arrays:arrays[k].append(y.numpy() if k=='y' else f[k].cpu().numpy())
        arrays={k:np.concatenate(v) for k,v in arrays.items()};outputs[arm]=arrays
        np.savez_compressed(out/f'target_{arm}.npz',**arrays)
        del model;torch.cuda.empty_cache()
    comparisons=[]
    for a,b in itertools.combinations(ARMS,2):
        for k in ['logits','global_logits','delta','pi']:
            x=outputs[a][k];y=outputs[b][k]
            comparisons.append(dict(pair=a+'/'+b,output=k,max_abs_difference=float(np.abs(x-y).max()),different_values=int(np.count_nonzero(x!=y)),argmax_disagreement=int(np.count_nonzero(x.argmax(-1)!=y.argmax(-1)))))
    f=outputs['R0'];full=f['logits'];g=f['global_logits'];y=f['y'];pred=full.argmax(1);gp=g.argmax(1)
    terms=f['pi'][:,:6,None]*f['delta'];corr=terms.sum(1);gc=g-g.mean(1,keepdims=True)
    ratio=lambda a,b:float(np.sqrt(np.mean(np.sum(a*a,axis=-1))/max(np.mean(np.sum(b*b,axis=-1)),1e-16)))
    norm=lambda a:np.linalg.norm(a,axis=-1)
    delta=f['delta'].astype(np.float64);flat=delta.transpose(1,0,2).reshape(6,-1);gram=flat@flat.T;den=np.sqrt(np.outer(np.diag(gram),np.diag(gram)));cos=gram/np.maximum(den,1e-16)
    pi=f['pi'].astype(np.float64);ent=-(pi*np.log(np.maximum(pi,1e-30))).sum(1)
    result=dict(samples=len(y),full_accuracy=float((pred==y).mean()),global_accuracy=float((gp==y).mean()),full_global_disagreement=int((pred!=gp).sum()),helped=int(((pred==y)&(gp!=y)).sum()),harmed=int(((pred!=y)&(gp==y)).sum()),both_wrong_but_different=int(((pred!=y)&(gp!=y)&(pred!=gp)).sum()),correction_global_RMS_ratio=ratio(corr,gc),routed_component_norm_mean=norm(terms).mean(0).tolist(),composed_to_sum_component_norm_ratio=float(norm(corr).mean()/max(norm(terms).sum(1).mean(),1e-16)),expert_delta_flattened_cosine=cos.tolist(),router_mass=pi.mean(0).tolist(),router_entropy_mean=float(ent.mean()),router_NULL_quantiles=np.quantile(pi[:,-1],[0,.1,.5,.9,1]).tolist(),class_accuracy=[dict(label=label,n=int((y==i).sum()),full=float((pred[y==i]==i).mean()),global_accuracy=float((gp[y==i]==i).mean())) for i,label in enumerate(data.classes)])
    return comparisons,result

def matrix_text(matrix,names):
    lines=['| Domain | E0 | E1 | E2 | E3 | E4 | E5 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,row in zip(names,matrix):lines.append('| '+name+' | '+' | '.join(f'{x:.7f}' for x in row)+' |')
    return lines

def analyze(root,data_dir):
    out=root/'stats';out.mkdir(exist_ok=True);md=['# Equality and mechanism stats','', 'Post-hoc only; no training, new checkpoint selection or hyperparameter change. Expert indices are E0–E5. Accuracies/drop values in tables are percentage points; CE utility and derivatives are in loss units. Exact tensor equality is compared by content, not checkpoint file bytes (which include differing metadata).','']
    allresults={}
    for dataset in SPECS:
        folder=out/dataset;folder.mkdir(exist_ok=True);banks=root/dataset/'banks';config=read(banks/'R0/effective_config.json')['config']
        cp=checkpoint_stats(banks);trajectory,logs=trajectory_stats(banks);qrows=q_statistics(logs['R0'],config)
        comparisons,target=replay(banks,dataset,data_dir,folder)
        base.write(folder/'checkpoint_equality.json',cp);write_csv(folder/'trajectory_equality.csv',trajectory);write_csv(folder/'target_equality.csv',comparisons);write_csv(folder/'Q_refresh_counterfactual.csv',qrows);base.write(folder/'target_composition.json',target)
        diagnostics={a:read(banks/a/'source_diagnostics.json') for a in ARMS};d=diagnostics['R0'];names=[SPECS[dataset]['domains'][e] for e in d['source_domains']];u=np.array(d['marginal_utility']);v=np.array(d['utility_interaction']);sel=np.array(d['q'])>0
        stages=[]
        for name in ['before','activation','middle','final']:
            s=read(banks/'R1/milestones'/f'{name}.json');r=logs['R1'][s['step']];ema=np.array(r['utility_ema']);candidate=next(row for row in reversed(qrows) if row['step']<=s['step'])
            stages.append(dict(event=name,step=s['step'],source_accuracy=s['source_mean'],source_global_accuracy=s['global_mean'],source_router_NULL=s['router_mass'][-1],correction_global_RMS_ratio=s['correction_global_norm_ratio_rms'],Uout_min=float(np.min(s['marginal_utility'])),Uout_max=float(np.max(s['marginal_utility'])),G_mean=float(np.mean(s['G_matrix'])),**{'fixed_'+k:val for k,val in support(s['q']).items()},candidate_pair_expert_count=candidate['pair_expert_count'],candidate_Q_mass_mean=candidate['q_mass_mean']))
        write_csv(folder/'milestones.csv',stages)
        selected=read(banks/'R0/summary.json')['selected_step'];metric=read(root/dataset/'evaluation/R0/metrics.json');expert_rows=[]
        for m in range(6):
            expert_rows.append(dict(expert=m,source_U=float(u[:,m].mean()),source_single_gain=d['single_logit_improvement'][m],source_delta_norm=d['residual_norm_per_expert'][m],source_router_mass=d['router_mass'][m],target_U=metric['diagnostics']['marginal_utility'][0][m],target_accuracy_drop_pp=100*metric['diagnostics']['leave_one_out_accuracy_drop'][m],target_router_mass=target['router_mass'][m]))
        write_csv(folder/'experts.csv',expert_rows)
        allresults[dataset]=dict(selected_step=selected,checkpoint_parameters_identical=all(not any(m['different_elements'] for m in c['modules'].values()) for c in cp),full_trajectory_identical=all(r['different_values']==0 for r in trajectory),target_outputs_identical=all(r['different_values']==0 for r in comparisons),fixed_Q_support=support(d['q']),candidate_refresh_shared_updates_after_activation=sum(r['pair_expert_count']>0 for r in qrows if r['step']>=2000),candidate_refresh_updates_after_activation=sum(r['step']>=2000 for r in qrows),source_U_abs_max=float(np.abs(u).max()),source_interaction_abs_max=float(np.abs(v).max()),source_interaction_RMS=float(np.sqrt(np.mean(v*v))),target_composition=target)
        md.extend([f'## {dataset}','',f'Source-selected step: **{selected}**. All three pairs have exactly identical backbone/global/residual/router tensors at both best and last checkpoints. All 5001 steps match exactly for total/classification/discovery/router/balance/SIRM losses, residual RMS, router NULL and U/V EMA. R0/R1/R2 target logits, global logits, residual logits and router probabilities are directly replayed and compared (details in CSV/NPZ). Q buffers/config/metadata may differ; these are not prediction parameters.','',f'**Target replay:** {target["samples"]} images; maximum arm-to-arm logit difference = {max(r["max_abs_difference"] for r in comparisons):.8g}; prediction disagreement = {next(r["argmax_disagreement"] for r in comparisons if r["output"]=="logits")}.','', '| Quantity | Value |','|---|---:|',f'| Full / global-only target accuracy | {100*target["full_accuracy"]:.4f}% / {100*target["global_accuracy"]:.4f}% |',f'| Residual net target delta | {100*(target["full_accuracy"]-target["global_accuracy"]):+.4f} pp |',f'| Global prediction changed / helped / harmed | {target["full_global_disagreement"]} / {target["helped"]} / {target["harmed"]} |',f'| Source / target correction-to-global RMS norm ratio | {d["correction_global_norm_ratio_rms"]:.6f} / {target["correction_global_RMS_ratio"]:.6f} |',f'| Source / target router NULL mass | {d["router_mass"][-1]:.6f} / {target["router_mass"][-1]:.6f} |',f'| Target router entropy (nats; max log(7)=1.946) | {target["router_entropy_mean"]:.6f} |',f'| Fixed-Q selected cells / shared experts / pair-expert terms | {support(d["q"])["selected_cells"]} / {support(d["q"])["shared_experts"]} / {support(d["q"])["pair_expert_count"]} |',f'| Source utility max absolute / interaction max absolute | {np.abs(u).max():.7f} / {np.abs(v).max():.7f} |',f'| Source-utility ↔ target-ablation Spearman (six experts) | {metric["source_utility_target_ablation_spearman"]:.6f} |','', '### Source-out marginal utility at selected checkpoint','',*matrix_text(u,names),'', '### Double-centered source-out utility','',*matrix_text(v,names),'', '### Fixed learned Q','',*matrix_text(d['q'],names),'', '### Residual and router stats per expert','', '| Expert | Source U | CE(g)−CE(g+δ) source | Source ‖δ‖ | Source routing | Target U | Target ΔAcc removal (pp) |','|---|---:|---:|---:|---:|---:|---:|'])
        for row in expert_rows:md.append('| E'+str(row['expert'])+' | '+' | '.join(f'{row[k]:.7f}' for k in ['source_U','source_single_gain','source_delta_norm','source_router_mass','target_U','target_accuracy_drop_pp'])+' |')
        md.extend(['', 'Positive target ΔAcc means removing that expert hurts; negative means removing it improves accuracy. Single-correction gain uses unscaled δ; marginal utility uses routed πδ in context, so their signs need not match. Source U is an equal-domain mean; target statistics average images. Ablation effects overlap and must not be summed as unique causal contributions.','', '### Milestones and the effect of locking Q','', '| Event / step | Source acc | Router NULL | Correction/global | Fixed pair terms | Fresh-Q candidate pair terms | Fresh-Q mean mass |','|---|---:|---:|---:|---:|---:|---:|'])
        for s in stages:md.append(f'| {s["event"]} / {s["step"]} | {100*s["source_accuracy"]:.4f}% | {s["source_router_NULL"]:.6f} | {s["correction_global_RMS_ratio"]:.6f} | {s["fixed_pair_expert_count"]} | {s["candidate_pair_expert_count"]} | {s["candidate_Q_mass_mean"]:.8f} |')
        fresh=allresults[dataset];md.extend(['',f'Replaying the original Q formula with unchanged thresholds on continued **source-in utility EMA** finds structural shared support in {fresh["candidate_refresh_shared_updates_after_activation"]}/{fresh["candidate_refresh_updates_after_activation"]} stage-3 refresh points. This is a post-hoc assignment diagnostic, not a trained dynamic-Q result; potential pair counts exclude ESS checks. It cannot establish accuracy or gradient magnitude under a changed trajectory. Gate floors are U>{config["utility_floor"]}, V>{config["interaction_floor"]}, with temperature {config["q_temperature"]}; detailed pass counts and confidence attenuation are in Q_refresh_counterfactual.csv.',''])
        print(dataset,json.dumps({k:v for k,v in fresh.items() if k!='target_composition'}),flush=True)
    base.write(out/'stats.json',allresults)
    md.extend(['## Interpretation limits','', 'Zero SIRM steps follow from absent shared support, regardless of ESS. Equal arm results test the implementation/no-op path, not the benefit or harm of an active penalty. G under differently permuted masks is a sampling difference, not a treatment effect. Source-out is used for checkpoint selection; target diagnostics are exploratory, one seed, and do not change any training setting. Fresh-Q replay distinguishes no support at activation from potential support emerging later. Old standalone-risk interaction gaps and new marginal-utility interaction gaps are not directly interchangeable.'])
    (out/'summary.md').write_text('\n'.join(md)+'\n')
    main=root/'summary.md';text=main.read_text();marker='\n## Additional equality and mechanism statistics\n'
    text=text.split(marker)[0]+marker+'\nSee [stats/summary.md](stats/summary.md): exact best/last parameter comparisons, all-step loss/EMA differences, direct target output comparisons, global-versus-residual prediction changes, U/V/Q matrices, expert ablations, milestone trajectories and post-hoc Q-refresh support. No retraining or changes to the completed gate.\n';main.write_text(text)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='results/sirm_discover_gate/phase1');p.add_argument('--data-dir',default='domainbed/data');a=p.parse_args();torch.set_num_threads(4);analyze(Path(a.root),a.data_dir)

if __name__=='__main__':main()

"""Read-only post-hoc source diagnostics; never updates the factorial models."""
import argparse,json,math,shutil,itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_h1_h6 import run as base
from experiments.sirm_h1_h6.statistics import fit_linear_probe,probe_logits
from experiments.sirm_representation_h2.protocol import ViewTransform,validate_source
from experiments.sirm_representation_h2.diagnostics import json_write
from experiments.sirm_representation_h2.summarize import write_csv
from domainbed.subset_irm import subset_irm_penalty,top_r_assignment

def cache(run,bank,split,view='original'):
    name=f'source_{split}_{view}.pt';p=run/'cache'/name
    if not p.exists():p=bank/'cache'/name
    c=base.load(p);validate_source(c,json.loads((run/'config.json').read_text())['target_env']);return c

def extract(root,out):
    margins=[];routing=[];curves=[];snapshots=[];inventory=[]
    for p in sorted((root/'runs').glob('*/env*/seed*/metrics.json')):
        run=p.parent;r=json.loads(p.read_text());pr=r['provenance'];bank=Path(pr['bank']);key=dict(condition=pr['condition'],target_env=pr['target_env'],seed=pr['seed'])
        name=f'{key["condition"]}_env{key["target_env"]}_seed{key["seed"]}'
        diag=json.loads((run/'source_diagnostics.json').read_text());comp=diag['competence_matrix'];q=np.array(diag['q']['q'])
        for d,domain in enumerate(comp['source_domains']):
            selected=q[d]>0
            for v,view in enumerate(comp['view_order']):
                risk=np.array(comp['loss_matrix'])[d,:,v];order=np.argsort(risk);top=order[:2]
                margins.append(dict(**key,source_domain=domain,view=view,best_expert=int(order[0]),second_expert=int(order[1]),
                    best_second_margin=float(risk[order[1]]-risk[order[0]]),top2_boundary_margin=float(risk[order[2]]-risk[order[1]]),
                    top2_vs_other_mean_margin=float(np.delete(risk,top).mean()-risk[top].mean()),
                    q_selected_vs_other_margin=float(risk[~selected].mean()-risk[selected].mean()),
                    q_matches_risk_top2_jaccard=len(set(top)&set(np.flatnonzero(selected)))/len(set(top)|set(np.flatnonzero(selected))),
                    expert_risk_sd=float(risk.std(ddof=1)),q_values=json.dumps(q[d].tolist()),risk_values=json.dumps(risk.tolist())))
        for view in ['original','grayscale','blur']:
            c=cache(run,bank,'out',view);dense=c['router_dense'];native=c['router_topk']
            for d in c['domains'].unique().tolist():
                mask=c['domains']==d;pi=dense[mask];n=native[mask]
                for m in range(6):
                    routing.append(dict(**key,view=view,source_domain=comp['source_domains'][d],expert=m,
                        dense_mass=float(pi[:,m].mean()),native_mass=float(n[:,m].mean()),topk_usage=float(n[:,m].gt(0).float().mean()),
                        dense_entropy=float(-(pi*pi.clamp_min(1e-12).log()).sum(1).mean()),
                        native_entropy=float(-(n*n.clamp_min(1e-12).log()).sum(1).mean())))
        for line in (bank/'train_log.jsonl').read_text().splitlines():
            row=json.loads(line);row['source_validation']=json.dumps(row['source_validation']);curves.append(dict(**key,**row))
        for label in ['best','last']:
            checkpoint=base.load(bank/'checkpoints'/f'{label}.pkl');state=checkpoint['model_dict']
            qs=top_r_assignment(state['risk_ema.risk'],state['risk_ema.valid'],2,1.)
            snapshots.append(dict(**key,snapshot=label,step=checkpoint['step'],q=qs.tolist(),risk_ema=state['risk_ema.risk'].tolist(),
                                  source_domains=comp['source_domains']))
        inventory.append(dict(**key,run=str(run),bank=str(bank),checkpoint_sha256=pr['checkpoint_sha256']))
    write_csv(out/'competence_margins.csv',margins);write_csv(out/'router_mass.csv',routing);write_csv(out/'training_curves.csv',curves)
    json_write(out/'q_selected_last.json',snapshots);json_write(out/'inventory.json',inventory)
    for f in ['runs.csv','factorial_effects.csv','view_robustness.csv','router_stability.csv','representation_stability.csv']:
        shutil.copy2(root/f,out/f)
    for folder in ['competence_matrices','q_matrices']:shutil.copytree(root/folder,out/folder,dirs_exist_ok=True)
    for c in ['B','D']:
        row=next(r for r in inventory if r['condition']==c and r['target_env']==0 and r['seed']==0)
        shutil.copy2(Path(row['bank'])/'train_log.jsonl',out/f'{c}_H2_env0_seed0_train_log.jsonl')
    return inventory

def gradients(row,out):
    key={k:row[k] for k in ['condition','target_env','seed']};dest=out/'gradient_runs'/f'{key["condition"]}_env{key["target_env"]}_seed{key["seed"]}.json'
    if dest.exists():return json.loads(dest.read_text())
    bank=Path(row['bank']);model,eff,_=base.selected_model(bank);model.eval()
    for p in model.parameters():p.requires_grad_(True)
    split=json.loads((bank/'split_manifest.json').read_text());base.ViewTransform=ViewTransform
    groups={'backbone':list(model.featurizer.parameters()),'router':list(model.moe_head.router.parameters()),
            'experts':list(model.moe_head.experts.parameters()),'expert_heads':list(model.expert_heads.parameters())}
    params=sum(groups.values(),[]);spans={};start=0
    for g,ps in groups.items():spans[g]=(start,start+len(ps));start+=len(ps)
    spans['all']=(0,len(params));rows=[]
    q=model._assignment(torch.full((3,6),1/6,device='cuda'),torch.arange(3,device='cuda'))
    for view in ['original','grayscale','blur']:
        data=base.source_data('domainbed/data',split,eff['config'],'out',view,eff['cohort'],limit=48)
        assert key['target_env'] not in data
        for repeat in range(3):
            samples=[(d,[data[env][i] for i in range(repeat*16,(repeat+1)*16)]) for d,env in enumerate(sorted(data))]
            x=torch.stack([a for _,ss in samples for a,b in ss]).cuda();y=torch.tensor([b for _,ss in samples for a,b in ss],device='cuda')
            domains=torch.tensor([d for d,ss in samples for _ in ss],device='cuda')
            logits,_,gamma,_,expert_logits,_=model._subset_forward(x);cls=F.cross_entropy(logits,y)
            sirm,info=subset_irm_penalty(expert_logits,y,domains,gamma,q,model.min_effective_samples,model.responsibility_detach)
            gc=torch.autograd.grad(cls,params,retain_graph=True,allow_unused=True)
            gs=torch.autograd.grad(sirm,params,allow_unused=True)
            for group,(lo,hi) in spans.items():
                nc=ns=dot=0.
                for a,b in zip(gc[lo:hi],gs[lo:hi]):
                    if a is not None:nc+=float(a.detach().float().square().sum())
                    if b is not None:ns+=float(b.detach().float().square().sum())
                    if a is not None and b is not None:dot+=float((a.detach().float()*b.detach().float()).sum())
                denom=math.sqrt(nc*ns);raw_ratio=math.sqrt(ns/nc) if nc else None
                rows.append(dict(**key,view=view,repeat=repeat,group=group,loss_cls=float(cls),loss_sirm=float(sirm),
                    norm_cls=math.sqrt(nc),norm_sirm_raw=math.sqrt(ns),raw_ratio=raw_ratio,weighted_ratio_actual=(raw_ratio*eff['hparams']['subset_irm_lambda_sirm']) if raw_ratio is not None else None,
                    weighted_ratio_at_lambda_point1=raw_ratio*.1 if raw_ratio is not None else None,
                    cosine=dot/denom if denom>1e-20 else None,active=bool(info['active']),active_cells=int(info['active_cells'].sum()),
                    measurement='posthoc eval-mode selected checkpoint, paired class-interleaved source-out batches16/domain; not training trajectory'))
            del gc,gs,logits,expert_logits,cls,sirm,x
    json_write(dest,rows);del model;torch.cuda.empty_cache();return rows

def directions(row,out):
    key={k:row[k] for k in ['condition','target_env','seed']};dest=out/'direction_runs'/f'{key["condition"]}_env{key["target_env"]}_seed{key["seed"]}.json'
    if dest.exists():return json.loads(dest.read_text())
    run=Path(row['run']);bank=Path(row['bank']);fit=cache(run,bank,'in');val=cache(run,bank,'out');q=json.loads((run/'source_diagnostics.json').read_text())['q'];rows=[]
    for m in range(6):
        x=fit['expert_features'][:,m].double();mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(1e-6);probes={}
        for d in range(3):
            mask=fit['domains']==d
            assert len(fit['labels'][mask].unique())==7
            probes[d]=fit_linear_probe(x[mask],fit['labels'][mask],7,mean,std,regularization=.01,max_iter=500,tolerance=1e-4)
        for a,b in itertools.combinations(range(3),2):
            wa=probes[a]['w'];wb=probes[b]['w'];wa=wa-wa.mean(1,keepdim=True);wb=wb-wb.mean(1,keepdim=True)
            own=[];cross=[]
            for d,e in [(a,b),(b,a)]:
                mask=val['domains']==d;features=val['expert_features'][mask,m];labels=val['labels'][mask]
                own.append(float(probe_logits(probes[d],features).argmax(1).eq(labels).float().mean()))
                cross.append(float(probe_logits(probes[e],features).argmax(1).eq(labels).float().mean()))
            rows.append(dict(**key,expert=m,domain_a=q['source_domains'][a],domain_b=q['source_domains'][b],
                q_shared=q['q'][a][m]>0 and q['q'][b][m]>0,coefficient_cosine=float(F.cosine_similarity(wa.flatten(),wb.flatten(),dim=0)),
                mean_class_direction_cosine=float(F.cosine_similarity(wa,wb,dim=0).mean()),
                own_domain_val_acc=sum(own)/2,cross_domain_val_acc=sum(cross)/2,probe_gradient_max=max(probes[a]['gradient_max'],probes[b]['gradient_max']),
                preprocessing='shared source-in mean/std; class-centered coefficients; fixed L2 .01; source-out evaluation; no target data'))
    json_write(dest,rows);return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/sirm_representation_h2/full');p.add_argument('--output',default='results/sirm_representation_h2/diagnosis');p.add_argument('--mode',choices=['extract','gradients','directions'],required=True);args=p.parse_args()
    torch.set_num_threads(4);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.mode=='extract':inventory=extract(Path(args.root),out);print(f'exported {len(inventory)} runs',flush=True);return
    rows=[]
    for item in json.loads((out/'inventory.json').read_text()):
        try:rows.extend((gradients if args.mode=='gradients' else directions)(item,out))
        except Exception as error:
            json_write(out/f'failure_{args.mode}.json',dict(run=item,error=repr(error)));raise
        write_csv(out/('gradient_interaction.csv' if args.mode=='gradients' else 'classifier_directions.csv'),rows)
        print(f'{args.mode}: {item["condition"]}/env{item["target_env"]}/seed{item["seed"]}',flush=True)

if __name__=='__main__':main()

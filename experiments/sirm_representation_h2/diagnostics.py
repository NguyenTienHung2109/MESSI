"""One source-selected evaluator and mechanistic diagnostics for all factorial cells."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from experiments.sirm_h1_h6 import run as base
from experiments.sirm_h1_h6.controls import fit_router, metrics, mixture
from experiments.sirm_h1_h6.statistics import expert_losses,fit_linear_probe,probe_logits
from .protocol import CONDITIONS,VIEWS,ViewTransform,validate_source,branch_weights

def json_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    base.write(path,value)

def cosine_margin(features,labels):
    """Macro within-class minus between-class off-diagonal cosine; exact pair sums."""
    z=F.normalize(features.float(),dim=1);classes=labels.unique();within=[];centers=[]
    for c in classes:
        part=z[labels==c];n=len(part);centers.append(part.mean(0))
        if n>1:within.append(((part.sum(0).square().sum()-part.square().sum())/(n*(n-1))).item())
    between=[float(centers[a]@centers[b]) for a in range(len(centers)) for b in range(a+1,len(centers))]
    w=sum(within)/len(within) if within else 0.;b=sum(between)/len(between) if between else 0.
    return dict(within_class_cosine=w,between_class_cosine=b,cosine_margin=w-b,
                feature_variance=features.float().var(0,unbiased=False).mean().item(),mean_feature_norm=features.float().norm(dim=1).mean().item())

def validate_pairs(original,other):
    if original['image_ids']!=other['image_ids'] or not torch.equal(original['labels'],other['labels']):
        raise ValueError('paired views contain different examples or labels')
    if not torch.equal(original['physical_domains'],other['physical_domains']):
        raise ValueError('paired domains changed')

def router_pair(a,b,k=2):
    p,q=a['router_dense'],b['router_dense'];mid=(p+q)/2
    js=.5*((p*(p.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(1)+
            (q*(q.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(1))
    pa=F.one_hot(p.topk(k,dim=1).indices,p.shape[1]).sum(1).bool()
    pb=F.one_hot(q.topk(k,dim=1).indices,q.shape[1]).sum(1).bool()
    return dict(router_cosine=F.cosine_similarity(p,q,dim=1).mean().item(),router_js=js.mean().item(),
                topk_jaccard=((pa&pb).sum(1)/(pa|pb).sum(1)).float().mean().item())

def native_pi(cache):
    return cache.get('router_native',cache['router_topk'])

def competence(caches,sources):
    rows=[];loss_tensors=[];accuracy_tensors=[]
    for view in VIEWS[:3]:
        c=caches[view];loss=expert_losses(c['expert_logits'],c['labels'])
        correct=c['expert_logits'].argmax(2).eq(c['labels'][:,None]).float()
        loss_tensors.append(torch.stack([loss[c['domains']==d].mean(0) for d in range(3)]))
        accuracy_tensors.append(torch.stack([correct[c['domains']==d].mean(0) for d in range(3)]))
    losses=torch.stack(loss_tensors,2);accuracy=torch.stack(accuracy_tensors,2)
    for d,physical in enumerate(sources):
        for m in range(losses.shape[1]):
            for v,view in enumerate(VIEWS[:3]):
                rows.append(dict(source_domain=base.DOMAINS[physical],source_env=physical,expert=m,view=view,
                    loss=float(losses[d,m,v]),accuracy=float(accuracy[d,m,v]),
                    competence_std_over_views=float(losses[d,m].std(unbiased=True)),
                    accuracy_std_over_views=float(accuracy[d,m].std(unbiased=True))))
    return rows,dict(source_domains=[base.DOMAINS[d] for d in sources],view_order=VIEWS[:3],
        loss_matrix=losses.tolist(),accuracy_matrix=accuracy.tolist(),competence_std_over_views=losses.std(2,unbiased=True).tolist(),
        same_validation_examples_for_all_experts=True)

def q_diagnostics(q,caches,sources):
    if not torch.isfinite(q).all() or bool((q<0).any()):raise ValueError('invalid Q')
    torch.testing.assert_close(q.sum(1),torch.ones(3),atol=1e-5,rtol=1e-5)
    support=q.gt(0);pairs=[]
    for d in range(3):
        for e in range(d+1,3):
            pairs.append(dict(first=base.DOMAINS[sources[d]],second=base.DOMAINS[sources[e]],
                jaccard=float((support[d]&support[e]).sum()/(support[d]|support[e]).sum())))
    rows=[]
    for view,c in caches.items():
        for d,env in enumerate(sources):
            mask=c['domains']==d;logits=mixture(c['expert_logits'][mask],q[d].expand(int(mask.sum()),-1));y=c['labels'][mask]
            rows.append(dict(view=view,source_domain=base.DOMAINS[env],q_subset_accuracy=float(logits.argmax(1).eq(y).float().mean()),
                            q_subset_nll=float(F.cross_entropy(logits,y))))
    return dict(q=q.tolist(),source_domains=[base.DOMAINS[d] for d in sources],
        selected_experts=[row.nonzero().flatten().tolist() for row in support],
        pairwise_jaccard=pairs,unique_experts_used=int(support.any(0).sum()),functional_metrics=rows,
        recomputed_on_target=False)

def stability(caches,k):
    original=caches['original'];router=[];representations=[]
    for view,c in caches.items():
        validate_pairs(original,c)
        native_acc=metrics(c,native_pi(c))['accuracy']
        for domain in c['domains'].unique().tolist():
            mask=c['domains']==domain
            a={key:value[mask] for key,value in original.items() if key=='router_dense'}
            b={key:value[mask] for key,value in c.items() if key=='router_dense'}
            routing=router_pair(a,b,k)
            router.append(dict(view=view,source_env=int(c['physical_domains'][mask][0]),**routing,
                               accuracy=float(mixture(c['expert_logits'][mask],native_pi(c)[mask]).argmax(1).eq(c['labels'][mask]).float().mean())))
        for feature_name in ['backbone_feature','expert_features']:
            xa,xb=original[feature_name],c[feature_name]
            entries=[(-1,xa,xb)] if xa.ndim==2 else [(m,xa[:,m],xb[:,m]) for m in range(xa.shape[1])]
            for m,a,b in entries:
                representations.append(dict(view=view,representation='backbone' if m<0 else 'expert',expert=m,
                    paired_cosine=F.cosine_similarity(a,b,dim=1).mean().item(),native_accuracy=native_acc,
                    **cosine_margin(b,c['labels'])))
    return router,representations

def collect_view(model,effective,split,kind,split_name,view,limit=None):
    config=effective['config'];base.ViewTransform=ViewTransform
    if kind=='source':
        data=base.source_data('domainbed/data',split,config,split_name,view,effective['cohort'],limit)
    else:
        env=effective['target_env'];dataset=ImageFolder(base.root_domain('domainbed/data',env),ViewTransform(config,view,False,env,effective['cohort']))
        if dataset.classes!=split['class_names']:raise ValueError('target class map differs')
        ids=list(range(len(dataset)))[:limit] if limit else list(range(len(dataset)))
        data={env:Subset(dataset,ids)}
    meta=dict(kind=kind,split=split_name,view=view,target_env=effective['target_env'],seed=effective['seed'],
              checkpoint_sha256=base.digest(Path(effective['_bank'])/'checkpoints/best.pkl'),split_hash=effective['split_hash'])
    cache=base.collect(model,data,config,meta)
    if not isinstance(model.moe_head.input_proj,torch.nn.Identity):
        raise ValueError('backbone representation requires a hook when input_proj is non-identity')
    # This architecture projects with Identity, so this is exactly the backbone CLS output.
    cache['backbone_feature']=cache['projected_feature']
    cache['router_native']=model._routing_responsibilities(cache['router_dense'].cuda()).cpu()
    if kind=='source':validate_source(cache,effective['target_env'])
    return cache

def source_probe(fit,val,smoke=False):
    if set(fit['image_ids'])&set(val['image_ids']):raise ValueError('source fit/validation overlap')
    x=fit['backbone_feature'].double();mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(1e-6)
    weights=torch.zeros(len(x),dtype=torch.double)
    for d in fit['domains'].unique():
        mask=fit['domains']==d;weights[mask]=1/(len(fit['domains'].unique())*int(mask.sum()))
    candidates=[]
    for reg in [.001,.01,.1]:
        probe=fit_linear_probe(x,fit['labels'],7,mean,std,regularization=reg,max_iter=500,tolerance=1e-4,sample_weights=weights)
        logits=probe_logits(probe,val['backbone_feature']);per=F.cross_entropy(logits,val['labels'],reduction='none')
        nll=torch.stack([per[val['domains']==d].mean() for d in val['domains'].unique()]).mean().item()
        candidates.append((nll,reg,probe))
    nll,reg,probe=min(candidates,key=lambda x:x[0])
    return probe,dict(regularization=reg,validation_nll=nll,gradient_max=probe['gradient_max'],
                      regularization_grid=[.001,.01,.1],target_used=False,selection='equal-domain source-out NLL, earliest tie')

def evaluate_run(bank,output,condition,target,seed,reused=False,smoke=False):
    bank,output=Path(bank),Path(output);output.mkdir(parents=True,exist_ok=True)
    complete=output/'metrics.json'
    if complete.exists():raise FileExistsError('refuse to overwrite evaluated run')
    model,effective,checkpoint_hash=base.selected_model(bank)
    effective=dict(effective,_bank=str(bank));config=effective['config'];split=json.loads((bank/'split_manifest.json').read_text())
    if base.canonical_hash(split)!=effective['split_hash'] or str(target) in split['sources']:
        raise ValueError('target leakage or changed split')
    if not isinstance(model.moe_head.input_proj,torch.nn.Identity):raise ValueError('backbone cache requires true CLS features')
    for entry in split['sources'].values():
        for sample in entry['images']:
            if base.digest(Path('domainbed/data/PACS')/sample['id'])!=sample['sha256']:raise ValueError('source data changed')
    sources=sorted(int(d) for d in split['sources']);limit=8 if smoke else None
    checkpoint_step=int(model.subset_step.item());cache_dir=output/'cache';cache_dir.mkdir(exist_ok=True)
    def get_cache(kind,split_name,view):
        name=f'{kind}_{split_name}_{view}.pt';old=bank/'cache'/name
        saved=cache_dir/name
        if saved.exists():
            c=base.load(saved)
            if c['metadata']['checkpoint_sha256']!=checkpoint_hash or c['metadata']['split_hash']!=effective['split_hash']:
                raise ValueError('partial-run cache provenance changed')
            if kind=='source':validate_source(c,target)
            return c
        if reused and old.exists():
            c=base.load(old)
            if c['metadata']['checkpoint_sha256']!=checkpoint_hash or c['metadata']['split_hash']!=effective['split_hash']:
                raise ValueError('reuse cache provenance mismatch')
            c['backbone_feature']=c['projected_feature']
            c['router_native']=c['router_topk'] if checkpoint_step>=model.topk_warmup_steps else c['router_dense']
            if kind=='source':validate_source(c,target)
            return c
        c=collect_view(model,effective,split,kind,split_name,view,limit)
        torch.save(c,cache_dir/name);return c
    fit=get_cache('source','in','original');validation={v:get_cache('source','out',v) for v in VIEWS}
    validate_source(fit,target)
    for c in validation.values():validate_source(c,target);validate_pairs(validation['original'],c)
    global_fit_config=dict(config,router_epochs=5 if smoke else config['router_epochs'])
    if (output/'source_selected.pt').exists():
        locked=base.load(output/'source_selected.pt')
        if locked['checkpoint_sha256']!=checkpoint_hash:raise ValueError('partial-run source selection changed checkpoint')
        global_router=locked['global_router'];probe=locked['probe'];probe_info=locked['probe_info']
    elif reused:
        prior=base.load(bank/'routers.pt')
        if prior['metadata']['target_used'] or prior['metadata']['checkpoint_sha256']!=checkpoint_hash:
            raise ValueError('global weights not source-selected for this checkpoint')
        for split_name in ['in','out']:
            if prior['metadata']['source_cache_sha256'][split_name]!=base.digest(bank/'cache'/f'source_{split_name}_original.pt'):
                raise ValueError('global fit cache changed')
        global_router=prior['global']
        probe,probe_info=source_probe(fit,validation['original'],smoke)
    else:
        global_router=fit_router(fit,validation['original'],global_fit_config,global_only=True,seed=seed)
        probe,probe_info=source_probe(fit,validation['original'],smoke)
    weights=branch_weights(global_router)
    if not (output/'source_selected.pt').exists():
        torch.save(dict(global_router=global_router,probe=probe,probe_info=probe_info,checkpoint_sha256=checkpoint_hash),output/'source_selected.pt')
    q=model._assignment(torch.full((3,6),1/6,device='cuda'),torch.arange(3,device='cuda')).cpu()
    comp_rows,comp_matrix=competence(validation,sources);q_info=q_diagnostics(q,validation,sources)
    router_rows,representation_rows=stability(validation,model.router_topk)
    aug,enabled=CONDITIONS[condition]
    provenance=dict(dataset='PACS',target_env=target,target_domain=base.DOMAINS[target],seed=seed,condition=condition,
        augmentation=aug,sirm_enabled=enabled,hparams=effective['hparams'],effective_config_hash=base.canonical_hash({k:v for k,v in effective.items() if k!='_bank'}),
        checkpoint_sha256=checkpoint_hash,split_hash=effective['split_hash'],selected_step=json.loads((bank/'summary.json').read_text())['selected_step'],
        selected_source_accuracy=json.loads((bank/'summary.json').read_text())['source_mean'],
        all_lambda_values={k:v for k,v in effective['hparams'].items() if 'lambda' in k},
        optimizer='Adam',source_split_seed=seed,total_steps=effective['steps'],pretrained_sha256=json.loads((bank/'manifest.json').read_text())['pretrained_sha256'],
        pretrained_checkpoint='domainbed/pretrained/deit_small_distilled_patch16_224-649709d9.pth (or identical Torch hub cache)',
        router_config=dict(type='Linear',in_features=model.moe_head.router.in_features,out_features=model.moe_head.router.out_features,
                           bias=model.moe_head.router.bias is not None,topk=model.router_topk,topk_warmup_steps=model.topk_warmup_steps),
        parameters_total=sum(p.numel() for p in model.parameters()),
        git_sha=json.loads((bank/'manifest.json').read_text())['git_sha'],bank=str(bank.resolve()),reused=reused,smoke=smoke,
        global_weights=weights.tolist(),global_source_selected=True,global_validation_nll=global_router['validation_nll'],
        backbone_representation='featurizer CLS; input_proj verified Identity',probe=probe_info,
        target_used_for_model_or_hyperparameter_selection=False)
    json_write(output/'source_diagnostics.json',dict(provenance=provenance,competence_rows=comp_rows,competence_matrix=comp_matrix,
        q=q_info,router_stability=router_rows,representation_stability=representation_rows))
    json_write(output/'config.json',provenance)
    # All source-selected objects and source diagnostics are locked before new target access.
    lock_hash=base.digest(output/'source_selected.pt');target_results={};probe_results={}
    for view in VIEWS:
        c=get_cache('target','all',view)
        pi=native_pi(c);target_results[view]=dict(native=metrics(c,pi),global_weights=metrics(c,weights.expand(len(c['labels']),-1)))
        pred=probe_logits(probe,c['backbone_feature']);probe_results[view]=dict(accuracy=float(pred.argmax(1).eq(c['labels']).float().mean()),nll=float(F.cross_entropy(pred,c['labels'])))
    if base.digest(output/'source_selected.pt')!=lock_hash:raise ValueError('source selection changed after target evaluation')
    summary={}
    summary['source_competence_loss_std']=float(torch.tensor(comp_matrix['competence_std_over_views']).mean())
    summary['source_q_unique_experts']=q_info['unique_experts_used']
    summary['source_q_jaccard_mean']=sum(r['jaccard'] for r in q_info['pairwise_jaccard'])/3
    for view in VIEWS[:3]:
        rows=[r for r in comp_rows if r['view']==view]
        summary[f'source_expert_accuracy_{view}']=sum(r['accuracy'] for r in rows)/len(rows)
    for view in VIEWS:
        backbone=next(r for r in representation_rows if r['view']==view and r['representation']=='backbone')
        experts=[r for r in representation_rows if r['view']==view and r['representation']=='expert']
        summary[f'source_backbone_cos_{view}']=backbone['paired_cosine']
        summary[f'source_backbone_margin_{view}']=backbone['cosine_margin']
        summary[f'source_expert_cos_{view}']=sum(r['paired_cosine'] for r in experts)/len(experts)
        routers=[r for r in router_rows if r['view']==view]
        summary[f'source_router_JS_{view}']=sum(r['router_js'] for r in routers)/len(routers)
        summary[f'source_q_subset_accuracy_{view}']=sum(r['q_subset_accuracy'] for r in q_info['functional_metrics'] if r['view']==view)/3
    for mode in ['native','global_weights']:
        suffix='native' if mode=='native' else 'global'
        for view in VIEWS:
            name={'original':'original','grayscale':'gray','blur':'blur','gray_blur':'gray_blur'}[view]
            summary[f'acc_{name}_{suffix}']=target_results[view][mode]['accuracy']
        values=[target_results[v][mode]['accuracy'] for v in VIEWS[:3]]
        summary[f'worst_view_{suffix}']=min(values);summary[f'mean_view_{suffix}']=sum(values)/3
    for view,result in probe_results.items():summary[f'probe_acc_{view}']=result['accuracy']
    result=dict(provenance=provenance,metrics=summary,views=target_results,linear_probe=probe_results,source_selected_sha256=lock_hash)
    json_write(complete,result);print(f'evaluated {condition}/env{target}/seed{seed}: original native={summary["acc_original_native"]:.5f}',flush=True)
    del model
    torch.cuda.empty_cache()
    return result

"""Matched source-only linear correction readouts on frozen token summaries."""
import json,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .shrinkage import ROOT,digest,write

MODES={'repeat_cls':('cls','cls'),'cls_dist':('cls','dist'),'cls_patch_mean':('cls','patch_mean'),'cls_patch_max':('cls','patch_max'),'repeat_dist':('dist','dist')}


def prepare(train,arrays,mask,mode):
    blocks=[];locks=[]
    for key in MODES[mode]:
        v=torch.tensor(train[key][mask],dtype=torch.double);mean=v.mean(0);std=v.std(0,unbiased=False).clamp_min(1e-8);blocks.append({s:(torch.tensor(a[key],dtype=torch.double)-mean)/std/math.sqrt(2) for s,a in arrays.items()});locks.append(dict(key=key,mean=mean,std=std))
    return {s:torch.cat([v[s] for v in blocks],1) for s in arrays},locks


def score(logits,a,mask):
    y=torch.tensor(a['y']);d=torch.tensor(a['domains']);g=torch.tensor(a['global_logits']);mask=torch.tensor(mask);p=logits.argmax(1);gp=g.argmax(1);ce=F.cross_entropy(logits,y,reduction='none');basece=F.cross_entropy(g,y,reduction='none');domains=d[mask].unique();n=int(mask.sum())
    if not n:return None
    rows={str(int(v)):dict(n=int((mask&d.eq(v)).sum()),accuracy=float(p[mask&d.eq(v)].eq(y[mask&d.eq(v)]).double().mean()),CE=float(ce[mask&d.eq(v)].mean())) for v in domains}
    return dict(n=n,equal_domain_accuracy=float(np.mean([r['accuracy'] for r in rows.values()])),equal_domain_CE=float(np.mean([r['CE'] for r in rows.values()])),image_accuracy=float(p[mask].eq(y[mask]).double().mean()),mean_CE_gain=float((basece-ce)[mask].mean()),
                repairs=int((mask&gp.ne(y)&p.eq(y)).sum()),harms=int((mask&gp.eq(y)&p.ne(y)).sum()),per_domain=rows)


def fit(z,a,mask,l2):
    z=z[mask];y=torch.tensor(a['y'][mask]);g=torch.tensor(a['global_logits'][mask],dtype=torch.double);d=torch.tensor(a['domains'][mask]);weights=torch.zeros(len(y),dtype=torch.double)
    for v in d.unique():weights[d.eq(v)]=1/(len(d.unique())*int(d.eq(v).sum()))
    w=torch.zeros(10,768,dtype=torch.double,requires_grad=True);b=torch.zeros(10,dtype=torch.double,requires_grad=True);opt=torch.optim.LBFGS([w,b],max_iter=1000,tolerance_grad=1e-4,tolerance_change=1e-10,line_search_fn='strong_wolfe');history=[]
    def closure():
        opt.zero_grad();loss=(F.cross_entropy(g+z@w.T+b,y,reduction='none')*weights).sum()+.5*l2*w.square().sum();assert torch.isfinite(loss);loss.backward();history.append(float(loss.detach()));return loss
    closure();initial=history[-1];opt.step(closure);closure();gradient=max(float(w.grad.abs().max()),float(b.grad.abs().max()));assert history[-1]<=initial+1e-8
    return w.detach(),b.detach(),dict(iterations=opt.state[w].get('n_iter'),closures=len(history),initial_objective=initial,final_objective=history[-1],max_objective_gradient=gradient,gradient_tolerance=1e-4,converged_gradient=gradient<=1e-4,objective_history=history)


def main():
    torch.set_num_threads(4);root=ROOT/'results/sirm_research_loop/iteration_12';out=root/'frozen_token_source_diagnosis';audit=json.loads((out/'extraction_audit.json').read_text());arrays={s:dict(np.load(out/(s+'_tokens.npz'))) for s in ['source_in','source_out']};train=arrays['source_in'];val=arrays['source_out'];assert not set(train['ids'])&set(val['ids'])
    for row in audit['rows']:assert digest(out/(row['split']+'_tokens.npz'))==row['cache_sha256']
    metapath=ROOT/'results/sirm_research_loop/iteration_08/source_annotation_audit/selected_source_results.json';meta=json.loads(metapath.read_text());lookup={r['local_id']:r['official_sequence'] for r in meta['records']};sequences={s:np.array([lookup[str(i)] for i in a['ids']]) for s,a in arrays.items()};assert all(not any(v is None for v in a) for a in sequences.values());held=set(sequences['source_out']);seen=np.array([v in set(sequences['source_in']) for v in sequences['source_out']]);scopes={'all_source_in':np.ones(len(train['y']),dtype=bool),'readout_event_disjoint':np.array([v not in held for v in sequences['source_in']])}
    plan=dict(code_sha256=digest(Path(__file__)),extraction_audit_sha256=digest(out/'extraction_audit.json'),metadata_sha256=digest(metapath),modes=MODES,training_scope_sizes={k:int(m.sum()) for k,m in scopes.items()},input_dimension=768,parameters=7690,optimizer='LBFGS convex softmax correction, zero initialization',l2=.001,bias_regularized=False,max_iter=1000,gradient_tolerance=1e-4,objective='equal-source-domain original-view CE(frozen_global + linear_correction) + .5*l2*||W||^2',normalization='Feature-block mean/std fitted on the actual fitting subset only; concatenate two384 blocks divided by sqrt2. repeat_cls has identical dimension/parameters and minimum-norm function capacity to a single standardized CLS linear correction.',selection='No mode, l2 or checkpoint sweep/selection; all fixed endpoint fits reported. Source-out evaluates only.',target_read=False,primary_bank_modified=False,scope='Additional frozen readout diagnostic fitting only; not primary SIRM training, not an H2-trained400 trajectory. Event exclusion applies only to readout fitting; the common anchor previously saw the complete original source-in split. Source-out is previously inspected/used for bank selection, so findings require independent confirmation.')
    if (out/'readout_plan.json').exists():assert json.loads((out/'readout_plan.json').read_text())==plan
    else:write(out/'readout_plan.json',plan)
    results={};locks={}
    for scope,mask in scopes.items():
        results[scope]={};assert set(train['domains'][mask])=={0,2,3}
        for mode in MODES:
            z,normalization=prepare(train,arrays,mask,mode);w,b,fitinfo=fit(z['source_in'],train,mask,.001);pred={s:torch.tensor(a['global_logits'],dtype=torch.double)+z[s]@w.T+b for s,a in arrays.items()}
            metrics={}
            for split,a in arrays.items():
                groups={'all':np.ones(len(a['y']),dtype=bool)}
                if split=='source_out':groups.update(event_seen_in_original_source_in=seen,event_unseen_in_original_source_in=~seen)
                else:groups['fitting_subset']=mask
                metrics[split]={k:score(pred[split],a,m) for k,m in groups.items()}
            results[scope][mode]=dict(fit=fitinfo,splits=metrics)
            saved=out/(scope+'_'+mode+'.pt');torch.save(dict(weight=w,bias=b,normalization=normalization,plan=plan),saved);locks[scope+'/'+mode]=digest(saved)
            np.savez_compressed(out/(scope+'_'+mode+'_replay.npz'),**{s+'_logits':v.numpy() for s,v in pred.items()})
            write(out/'readout_progress.json',dict(status='fitting',completed={k:list(v) for k,v in results.items()}));print(scope,mode,metrics['source_out']['all'],flush=True)
    assert digest(out/'extraction_audit.json')==plan['extraction_audit_sha256'] and digest(metapath)==plan['metadata_sha256'];assert digest(Path(__file__))==plan['code_sha256']
    baseline={s:score(torch.tensor(a['global_logits'],dtype=torch.double),a,np.ones(len(a['y']),dtype=bool)) for s,a in arrays.items()}
    write(out/'readout_results.json',dict(plan=plan,baseline=baseline,results=results,readout_locks=locks,target_read=False,scientific_goal_achieved=False));write(out/'readout_progress.json',dict(status='readouts_complete',completed={k:list(v) for k,v in results.items()}))

if __name__=='__main__':main()

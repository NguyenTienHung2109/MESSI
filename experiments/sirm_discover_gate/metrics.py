"""Exact component-scale derivatives and expert ablations from cached outputs."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from experiments.sirm_discover.math import residual_utilities,center_logits


def avg(values):return float(sum(values)/len(values)) if values else None

@torch.no_grad()
def evaluate(model,data,config,diagnostics=False):
    model.eval();scores={};global_scores={};nlls={};columns={k:[] for k in ['utility','G_signed','norms','single','pi','ratio','ratio_rms_num','ratio_rms_den','loo_accuracy_drop']};domains=[]
    for local,(env,subset) in enumerate(sorted(data.items())):
        correct=global_correct=count=0;nll=0.
        for x,y in DataLoader(subset,batch_size=config['eval_batch_size'],num_workers=config['num_workers']):
            x=x.cuda();y=y.cuda();f=model(x);full=f['logits'];g=f['global_logits'];delta=f['delta'];pi=f['pi']
            correct+=int(full.argmax(1).eq(y).sum());global_correct+=int(g.argmax(1).eq(y).sum());count+=len(y);nll+=float(F.cross_entropy(full,y,reduction='sum'))
            if not diagnostics:continue
            _,term,u,single=residual_utilities(g.double(),delta.double(),pi.double(),y)
            probabilities=full.double().softmax(1);error=probabilities-F.one_hot(y,full.shape[1])
            derivative=(error[:,None,:]*term).sum(-1)
            correction=term.sum(1);global_center=center_logits(g.double());global_norm=global_center.norm(dim=1)
            loo=(full.double()[:,None,:]-term).argmax(-1).eq(y[:,None]);native_correct=full.argmax(1).eq(y)
            vals=dict(utility=u,G_signed=derivative,norms=delta.double().norm(dim=2),single=single,pi=pi.double(),ratio=correction.norm(dim=1)/global_norm.clamp_min(1e-8),ratio_rms_num=correction.square().sum(1),ratio_rms_den=global_center.square().sum(1),loo_accuracy_drop=native_correct[:,None].double()-loo.double())
            for k,v in vals.items():columns[k].append(v.cpu())
            domains.append(torch.full((len(y),),local))
        scores[str(env)]=correct/count;global_scores[str(env)]=global_correct/count;nlls[str(env)]=nll/count
    result=dict(source_accuracy=scores,source_mean=avg(list(scores.values())),global_accuracy=global_scores,global_mean=avg(list(global_scores.values())),domain_nll=nlls)
    if not diagnostics:return result
    c={k:torch.cat(v) for k,v in columns.items()};ds=torch.cat(domains)
    utility=torch.stack([c['utility'][ds==d].mean(0) for d in range(len(data))]);interaction=utility-utility.mean(1,keepdim=True)-utility.mean(0,keepdim=True)+utility.mean()
    G=torch.stack([c['G_signed'][ds==d].mean(0).abs() for d in range(len(data))])
    q=model.q.cpu();sel=q>0
    # Target evaluation has one domain; Q is a source matrix and is never refit.
    src=len(data)==model.num_domains
    selected=[];other=[];isels=[];iothers=[];gsel=[];gnon=[];jaccard=[]
    if src:
        global_order=utility.mean(0).argsort(descending=True)
        for d in range(len(data)):
            a=sel[d];selected.extend(utility[d,a].tolist());other.extend(utility[d,~a].tolist());isels.extend(interaction[d,a].tolist());iothers.extend(interaction[d,~a].tolist());gsel.extend(G[d,a].tolist());gnon.extend(G[d,~a].tolist())
            k=int(a.sum())
            if k:
                global_set=set(global_order[:k].tolist());local_set=set(a.nonzero().flatten().tolist());jaccard.append(len(global_set&local_set)/len(global_set|local_set))
    result.update(source_domains=sorted(data),marginal_utility=utility.tolist(),utility_interaction=interaction.tolist(),G_matrix=G.tolist(),q=q.tolist(),q_learned=model.q_learned.cpu().tolist(),q_null=model.q_null.cpu().tolist(),router_mass=c['pi'].mean(0).tolist(),residual_norm_per_expert=c['norms'].mean(0).tolist(),single_logit_improvement=c['single'].mean(0).tolist(),correction_global_norm_ratio_mean=float(c['ratio'].mean()),correction_global_norm_ratio_rms=float((c['ratio_rms_num'].mean()/c['ratio_rms_den'].mean().clamp_min(1e-16)).sqrt()),leave_one_out_accuracy_drop=c['loo_accuracy_drop'].mean(0).tolist(),selected_utility=avg(selected),nonselected_utility=avg(other),interaction_advantage=avg(isels)-avg(iothers) if isels and iothers else None,G_selected=avg(gsel),G_nonselected=avg(gnon),q_global_topk_jaccard=avg(jaccard),q_fit_split='source-in augmented training minibatches',diagnostic_split='source-out' if src else 'target-posthoc',null_rows=int(q.sum(1).eq(0).sum()))
    return result

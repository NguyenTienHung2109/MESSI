"""Source-only mean/instance decomposition of saved residual expert banks."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover_gate.model import GateSIRM
from experiments.sirm_discover.math import residual_utilities,router_teacher,soft_null_q
from experiments.terra_10pct_factor_diagnosis.report import data_for
from .shrinkage import ROOT,head,read,digest,write
from .bias_control import scores,stability

DEST=ROOT/'results/sirm_research_loop/iteration_06_source_bias_diagnosis'

def evaluate(g,contribution,y,domains,ids,c):
    full=g+contribution.sum(1);ce=F.cross_entropy(full,y,reduction='none')
    U=torch.stack([F.cross_entropy(full-contribution[:,m],y,reduction='none')-ce for m in range(6)],1)
    ds=sorted(set(domains.tolist()));matrix=torch.stack([U[torch.tensor(domains==d)].mean(0) for d in ds])
    q,null,confidence,v=soft_null_q(matrix,torch.ones_like(matrix,dtype=torch.bool),c['q_temperature'],c['utility_floor'],c['interaction_floor'])
    support=q>0;pred=full.argmax(1);global_correct=g.argmax(1).eq(y)
    sc,_=scores(full.numpy(),y.numpy(),domains)
    return dict(scores=sc,U=matrix.tolist(),V=v.tolist(),Q=q.tolist(),Q_NULL=null.tolist(),selected_cells=int(support.sum()),shared_experts=int((support.sum(0)>=2).sum()),structural_pairs=sum(int((support[i]&support[j]).sum()) for i,j in itertools.combinations(range(3),2)),repairs_global=int((~global_correct&pred.eq(y)).sum()),harms_global=int((global_correct&~pred.eq(y)).sum()),per_expert_mean_norm=contribution.norm(dim=2).mean(0).tolist(),utility_halves=stability(dict(y=y.numpy(),domains=domains,ids=ids,B1_utility=U.numpy())))

@torch.no_grad()
def run():
    torch.set_num_threads(4);DEST.mkdir(exist_ok=True);code=Path(__file__);code_sha=digest(code)
    plan=dict(code_sha256=code_sha,iterations=[4,5],scope='Saved source-selected OFF banks; deterministic original source-in calibration and source-out evaluation only. No target cache, training, threshold change or checkpoint selection.',variants=['original','constant_routed','centered_routed','constant_heads','centered_heads'],threshold_rule='Existing U/V floors and Q formula, unchanged; source-out fresh-Q is an assignment diagnostic, not training EMA or execution.')
    if (DEST/'plan.json').exists():assert read(DEST/'plan.json')==plan
    else:write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(code.read_bytes())
    results={}
    for iteration in [4,5]:
        write(DEST/'progress.json',dict(status='source_in_calibration',iteration=iteration))
        source=ROOT/f'results/sirm_research_loop/iteration_{iteration:02d}';bank=source/'experiment/runs/L38/OFF';path=bank/'source_best.pkl';sha=digest(path)
        assert sha==read(bank/'posthoc/complete.json')['checkpoint_sha256']
        payload=torch.load(path,map_location='cpu',weights_only=False);c=payload['config'];manifest=read(source/'experiment/subset_manifest.json')
        backbone=DeiTFeaturizer(pretrained=False,model_name=c['backbone']);model=GateSIRM(backbone,backbone.n_outputs,10,3,c)
        model.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()});model.requires_grad_(False);model.eval()
        means={};raw_means={};calibration_ids=[]
        for d,ds in sorted(data_for(manifest,[0,2,3],c,'in').items()):
            raw=[];routed=[]
            for x,y in DataLoader(ds,batch_size=64,num_workers=0):
                g,delta,pi=head(model,model.featurizer(x));raw.append(delta.double());routed.append((delta*pi[:,:6,None]).double())
            means[d]=torch.cat(routed).mean(0);raw_means[d]=torch.cat(raw).mean(0)
            calibration_ids.extend(manifest['sources'][str(d)]['images'][i]['id'] for i in ds.indices)
        mu=torch.stack(list(means.values())).mean(0);b=torch.stack(list(raw_means.values())).mean(0)
        old_bias=torch.tensor(read(source/'bias_control/source_bias_lock.json')['vector'])
        assert torch.allclose(mu.sum(0),old_bias.double(),atol=1e-7,rtol=1e-6)
        out=DEST/f'iteration_{iteration:02d}';out.mkdir(exist_ok=True)
        write(out/'source_mean_lock.json',dict(checkpoint_sha256=sha,selected_step=payload['selected_step'],manifest_sha256=digest(source/'experiment/subset_manifest.json'),source_in_ids=calibration_ids,raw_expert_means=b.tolist(),routed_component_means=mu.tolist(),per_domain_raw_means={str(k):v.tolist() for k,v in raw_means.items()},per_domain_routed_means={str(k):v.tolist() for k,v in means.items()},target_read=False,optimization=False))
        write(DEST/'progress.json',dict(status='source_out_component_replay',iteration=iteration))
        with np.load(source/'eda/source_out_replay.npz') as cache:
            g=torch.tensor(cache['OFF_g']).double();delta=torch.tensor(cache['OFF_delta']).double();pi=torch.tensor(cache['OFF_pi']).double();y=torch.tensor(cache['y']);domains=cache['domains'];ids=cache['ids']
        assert not set(calibration_ids)&set(ids.tolist())
        actual=pi[:,:6,None]*delta
        variants=dict(original=actual,constant_routed=mu[None].expand_as(actual),centered_routed=actual-mu[None],constant_heads=pi[:,:6,None]*b[None],centered_heads=pi[:,:6,None]*(delta-b[None]))
        global_scores,_=scores(g.numpy(),y.numpy(),domains)
        values={name:evaluate(g,term,y,domains,ids,c) for name,term in variants.items()}
        # Teachers are defined only for the unscaled-head interventions.
        for name,heads in [('original',delta),('constant_heads',b[None].expand_as(delta)),('centered_heads',delta-b[None])]:
            _,_,_,single=residual_utilities(g,heads,pi,y);teacher=router_teacher(single,c['router_temperature'],c['utility_floor']);logits=g+(teacher[:,:6,None]*heads).sum(1)
            values[name]['label_teacher_diagnostic'],_=scores(logits.numpy(),y.numpy(),domains)
            values[name]['label_teacher_NULL']=float(teacher[:,-1].mean())
        # Remove calibration-domain exposure only, not model training exposure.
        held={}
        for d in sorted(means):
            mask=torch.tensor(domains==d);other_mu=torch.stack([v for k,v in means.items() if k!=d]).mean(0)
            other_b=torch.stack([v for k,v in raw_means.items() if k!=d]).mean(0)
            held[str(d)]={}
            for name,logits in [('global',g[mask]),('centered_routed',g[mask]+(actual[mask]-other_mu).sum(1)),('centered_heads',g[mask]+(pi[mask,:6,None]*(delta[mask]-other_b)).sum(1))]:held[str(d)][name],_=scores(logits.numpy(),y[mask].numpy(),domains[mask.numpy()])
        assert digest(path)==sha and digest(code)==code_sha
        result=dict(checkpoint_sha256=sha,selected_step=payload['selected_step'],calibration_images=len(calibration_ids),evaluation_images=len(y),global_scores=global_scores,variants=values,calibration_domain_holdouts=held,scope=plan['scope'],limitations='Source validation previously selected checkpoint; deterministic utility halves are not independent event-level replicates. Fresh-Q excludes training-EMA timing, ESS and actual gradients. Label teacher is not deployable. Holdouts remove only calibration-domain data; bank trained on all sources.')
        write(out/'results.json',result);results[str(iteration)]=result
    write(DEST/'results.json',results);write(DEST/'progress.json',dict(status='completed',remaining='Interpret source-only mean/instance evidence before next training change'))

if __name__=='__main__':
    try:run()
    except Exception as exc:write(DEST/'progress.json',dict(status='failed',error=str(exc)));raise

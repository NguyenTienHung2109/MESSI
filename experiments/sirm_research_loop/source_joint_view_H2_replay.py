"""Fixed utility-probe replay on scheduled H2, with centering controls."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import center_logits,router_teacher,residual_utilities,soft_null_q,component_eligibility
from .routed_conditional_utility import load, STEPS
from .routed_discovery_run import ROOT, DEST, PRIOR
from .source_conditional_accessibility import predict
from .shrinkage import digest,write

OUT=DEST/'source_joint_view_H2_replay'

def inputs(kind,raw,g,delta,oldpi):
    if kind=='CLS_plus_patch_max':return raw.double().numpy()
    return torch.cat([g-g.mean(1,keepdim=True),delta.flatten(1),oldpi],1).double().numpy()

def evaluate(g,delta,y,dom,pi,predicted_u):
    full,_,u,single=residual_utilities(g,delta,pi,y)
    gc=g.argmax(1).eq(y);fc=full.argmax(1).eq(y)
    return dict(n=len(y),global_correct=int(gc.sum()),full_correct=int(fc.sum()),
                global_CE=float(F.cross_entropy(g,y)),full_CE=float(F.cross_entropy(full,y)),
                repairs=int((~gc&fc).sum()),harms=int((gc&~fc).sum()),
                utility_MSE=float((predicted_u-single).square().mean()),
                NULL=float(pi[:,-1].mean())),u

@torch.no_grad()
def run():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    fitroot=DEST/'source_joint_view_utility_probe';fits=json.loads((fitroot/'results.json').read_text())
    proof=json.loads((DEST/'source_conditional_H2/extraction_audit.json').read_text())
    token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis'
    tokenproof=json.loads((token/'extraction_audit.json').read_text());p=token/'source_in_tokens.npz'
    assert digest(p)==next(r['cache_sha256'] for r in tokenproof['rows'] if r['split']=='source_in')
    original=np.load(p);raworiginal=torch.tensor(np.concatenate([original['cls'],original['patch_max']],1))
    originaldom=torch.tensor(original['domains']);lookup={str(id):i for i,id in enumerate(original['ids'])}
    plan=dict(source_only=True,target_read=False,probe_refitting=False,primary_training=False,
              states=list(fits['states']),steps=STEPS,centering=['training_batch_mean','fixed_source_mean'],
              inputs=fits['plan']['inputs'],protocol='Four saved bank states and already locked joint original/H2 all-source probes on40 exact scheduled96-image H2 batches. Original counterpart uses the same image IDs and cohort centering. No validation or target access. H2 batches were included in probe fitting; this replay measures fitted-cohort behavior, not independent augmentation generalization.',
              EMA='Separate hypothetical saved-state EMA initialized with first batch then beta .9, minimum32/domain. Original Q floors .002 and temperature .05, ESS2 unchanged. Never a historical or trained trajectory.',
              code_sha256=digest(Path(__file__)),fit_results_sha256=digest(fitroot/'results.json'),
              H2_audit_sha256=digest(DEST/'source_conditional_H2/extraction_audit.json'),source_in_cache_sha256=digest(p))
    write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    rows=[];locks={};summary=[]
    for state,records in fits['states'].items():
        root=PRIOR if state.startswith('unscaled') else DEST;which='selected' if state.endswith('selected') else 'last'
        m,p,path=load(root,which);locks[state]=digest(path);before={k:v.clone() for k,v in m.state_dict().items()}
        m.calibrate_features(raworiginal,originaldom)
        models={}
        for record in records[:2]:
            kind=record['input'];file=fitroot/f'{state}_{kind}_heldNone.npz';assert digest(file)==record['fit_sha256'];models[kind]=dict(np.load(file))
        for centering in plan['centering']:
            emas={kind:None for kind in models}
            for rec in proof['rows']:
                cache=DEST/'source_conditional_H2'/f"H2_step{rec['step']}.pt";assert digest(cache)==rec['sha256']
                batch=torch.load(cache,map_location='cpu',weights_only=False);raw=batch['raw'];y=batch['y'];dom=batch['domains']
                assert all(int(dom.eq(d).sum())==32 for d in [0,2,3])
                indices=[lookup[id] for id in batch['ids']];assert torch.equal(y,torch.tensor(original['y'][indices]))
                outputs={}
                for view,features in [('H2',raw),('paired_original',raworiginal[indices])]:
                    z=m.bank_features(features);g=m.global_head(features[:,:384]);oldpi=m.router(z).softmax(1)
                    delta=center_logits(torch.stack([e(z) for e in m.residuals],1))
                    delta=delta-(delta.mean(0,keepdim=True) if centering=='training_batch_mean' else m.expert_source_mean[None])
                    outputs[view]=(features,g.double(),delta.double(),oldpi.double())
                for kind,model in models.items():
                    measurements={};probs={};h2_u=None
                    for view,(features,g,delta,oldpi) in outputs.items():
                        pu=torch.tensor(predict(model,inputs(kind,features,g,delta,oldpi)))
                        pi=router_teacher(pu,.1,.002);metric,u=evaluate(g,delta,y,dom,pi,pu)
                        measurements[view]=metric;probs[view]=pi
                        if view=='H2':h2_u=u
                    U=torch.stack([h2_u[dom.eq(d)].mean(0) for d in [0,2,3]])
                    prior=emas[kind];ema=U.clone() if prior is None else .9*prior+.1*U;emas[kind]=ema
                    gate={}
                    mapped=torch.tensor([0 if int(d)==0 else 1 if int(d)==2 else 2 for d in dom])
                    for name,u in [('fresh',U),('hypothetical_EMA',ema)]:
                        q,_,_,v=soft_null_q(u,torch.ones_like(u,dtype=torch.bool),.05,.002,.002)
                        valid,ess,eligible=component_eligibility(probs['H2'],mapped,q,2.)
                        gate[name]=dict(U=u.tolist(),V=v.tolist(),Q=q.tolist(),shared_experts=torch.nonzero((q>0).sum(0)>=2).flatten().tolist(),eligible_pair_count=eligible,ESS=ess.tolist())
                    rows.append(dict(state=state,centering=centering,input=kind,batch_step=rec['step'],metrics=measurements,
                                     pi_L1_original_H2=float((probs['H2']-probs['paired_original']).abs().sum(1).mean()),gate=gate))
            for kind in models:
                rs=[r for r in rows if r['state']==state and r['centering']==centering and r['input']==kind]
                aggregate={}
                for view in ['H2','paired_original']:
                    ms=[r['metrics'][view] for r in rs]
                    aggregate[view]={k:float(np.mean([v[k] for v in ms])) for k in ['global_CE','full_CE','utility_MSE','NULL']}
                    aggregate[view].update({k:sum(v[k] for v in ms) for k in ['n','global_correct','full_correct','repairs','harms']})
                summary.append(dict(state=state,centering=centering,input=kind,aggregate=aggregate,
                                    fresh_shared_refreshes=sum(bool(r['gate']['fresh']['shared_experts']) for r in rs),
                                    EMA_shared_refreshes=sum(bool(r['gate']['hypothetical_EMA']['shared_experts']) for r in rs),
                                    EMA_eligible_refreshes=sum(r['gate']['hypothetical_EMA']['eligible_pair_count']>0 for r in rs)))
        m.load_state_dict(before);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and all(q.grad is None for q in m.parameters())
        assert digest(path)==locks[state]
        print(state,'complete',flush=True)
    write(OUT/'results.json',dict(plan=plan,rows=rows,summary=summary,checkpoint_sha256=locks,
                                 bank_tensors_restored=True,grad_fields_unassigned=True,source_only=True,target_read=False,probe_refitting=False))

if __name__=='__main__':run()

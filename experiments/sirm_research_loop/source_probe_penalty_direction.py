"""Binary IRMv1 finite-direction controls at locked probe-supported source windows."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import center_logits,expert_ce,router_teacher,component_irm
from .routed_conditional_utility import load
from .routed_discovery_run import ROOT,DEST,PRIOR
from .source_conditional_accessibility import predict
from .class_balanced_model import equal_class_cost,gradient_norm
from .class_coherent_discovery import class_coherent_cost
from .class_balanced_margin_diagnosis import margin,classmean
from .discovery_error_gradient_diagnosis import vector
from .shrinkage import digest,write

OUT=DEST/'source_probe_penalty_direction';RADIUS=.01

def outputs(m,raw,pi,training=False):
    with torch.no_grad():z=m.bank_features(raw);g=m.global_head(raw[:,:384])
    delta=center_logits(torch.stack([e(z) for e in m.residuals],1))
    delta=delta-(delta.mean(0,keepdim=True) if training else m.expert_source_mean[None])
    return g,delta,g+(pi[:,:6,None]*delta).sum(1)

def assess(m,raw,y,dom,pi,q):
    g,d,f=outputs(m,raw,pi);correct=f.argmax(1).eq(y);gc=g.argmax(1).eq(y);mar=margin(f,y)
    loss,info=component_irm(g,d,pi,y,dom,q,2.,'binary')
    metrics=dict(CE=float(torch.stack([F.cross_entropy(f[dom==i],y[dom==i]) for i in range(3)]).mean().detach()),
                 accuracy=float(torch.stack([correct[dom==i].double().mean() for i in range(3)]).mean()),
                 repairs=int((~gc&correct).sum()),harms=int((gc&~correct).sum()),bird_correct=int((correct&(y==0)).sum()),
                 residual_RMS=float(d.detach().square().mean().sqrt()),
                 correction_RMS=float((d.detach()*pi[:,:6,None]).sum(1).square().mean().sqrt()),
                 wrong_margin=float(classmean(mar.detach(),y,~gc)),correct_margin=float(classmean(mar.detach(),y,gc)),
                 fixed_mask_loss=float(loss.detach()) if info['active'] else None,validation_pair_count=info['pair_count'])
    return metrics,correct

def run():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    probe=DEST/'source_H2_utility_probe';proof=json.loads((probe/'results.json').read_text());h2proof=json.loads((DEST/'source_conditional_H2/extraction_audit.json').read_text());hashes={r['step']:r['sha256'] for r in h2proof['rows']}
    plan=dict(source_only=True,target_read=False,primary_training=False,probe_refitting=False,optimizer_updates=False,
              states=list(proof['states']),probe='Locked H2 all-source CLS+patch utility ridge; original teacher conversion .1/.002.',
              windows='All hypothetical EMA rows with shared support, not selected by source-out outcomes. ESS2 recomputed before any gradient.',
              directions=['binary_SIRM','diagnostic_base','residual_output_contraction'],radius=RADIUS,
              lambda_SIRM=1.,ratio_scope='Residual parameters; SIRM at full lambda1 / diagnostic base CE+matched discovery+soft balance. Existing router parameters remain fixed; their detached auxiliary targets have no residual gradient. This is not a historical optimizer ratio.',
              evaluation='Original source-validation under fixed probe probabilities, recalibrate original source-in residual means after every equal-radius move. Q fixed per window. Pair eligibility differences reported, no missing G imputed.',
              limits='Saved-state EMA, fixed-bank probe and disposable directions do not establish a trained ON effect, independent generalization or Adam trajectory equivalence.',
              code_sha256=digest(Path(__file__)),probe_results_sha256=digest(probe/'results.json'))
    write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';ta=json.loads((token/'extraction_audit.json').read_text());data={}
    for split in ['source_in','source_out']:
        p=token/f'{split}_tokens.npz';assert digest(p)==next(r['cache_sha256'] for r in ta['rows'] if r['split']==split);a=np.load(p)
        data[split]=(torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains']),a['ids'])
    train,ty,td,tids=data['source_in'];val,vy,vd,vids=data['source_out'];assert not set(tids)&set(vids)
    vdom=torch.tensor([0 if int(d)==0 else 1 if int(d)==2 else 2 for d in vd]);rows=[];locks={}
    for state,records in proof['states'].items():
        rec=records[0];assert rec['input']=='CLS_plus_patch_max' and rec['held_calibration_domain'] is None
        file=probe/f'{state}_CLS_plus_patch_max_heldNone.npz';assert digest(file)==rec['fit_sha256'];model=dict(np.load(file))
        m,p,path=load(PRIOR if state.startswith('unscaled') else DEST,'selected' if state.endswith('selected') else 'last');locks[state]=digest(path);assert locks[state]==proof['checkpoint_sha256'][state]
        original={k:v.clone() for k,v in m.state_dict().items()};m.calibrate_features(train,td);mean=m.expert_source_mean.clone();params=list(m.residuals.parameters());copies=[p.detach().clone() for p in params]
        vpi=router_teacher(torch.tensor(predict(model,val.double().numpy()),dtype=torch.float32),.1,.002)
        for gate in rec['H2_gate']['hypothetical_EMA_rows']:
            if not gate['shared_experts']:continue
            step=gate['batch_step'];cache=DEST/'source_conditional_H2'/f'H2_step{step}.pt';assert digest(cache)==hashes[step];a=torch.load(cache,map_location='cpu',weights_only=False);raw,y,dom=a['raw'],a['y'],a['domains']
            dom=torch.tensor([0 if int(d)==0 else 1 if int(d)==2 else 2 for d in dom]);pi=router_teacher(torch.tensor(predict(model,raw.double().numpy()),dtype=torch.float32),.1,.002);q=torch.tensor(gate['Q'])
            g,d,f=outputs(m,raw,pi,True);sirm,info=component_irm(g,d,pi,y,dom,q,2.,'binary')
            if not info['active']:rows.append(dict(state=state,step=step,shared_but_ESS_ineligible=True));continue
            raw_per=expert_ce(g[:,None]+d,y);reference=class_coherent_cost(raw_per,y)
            per=expert_ce(g[:,None]+pi[:,:6,None]*d,y) if state.startswith('routed') else raw_per
            balanced=equal_class_cost(per,y);oldnorm=gradient_norm(reference,params);newnorm=gradient_norm(balanced,params)
            discovery=balanced*(oldnorm/newnorm).detach().to(balanced.dtype) if newnorm>0 else reference
            teacher=router_teacher(F.cross_entropy(g,y,reduction='none')[:,None]-raw_per,.1,.002)
            loads=(-raw_per/m.config['discovery_temperature']).softmax(1).mean(0)
            balance=(6*loads.square().sum()-1)*(1-teacher[:,-1]).mean()
            base=F.cross_entropy(f,y)+m.config['lambda_discover']*discovery+m.config['lambda_balance']*balance
            vecs={'binary_SIRM':vector(sirm,params),'diagnostic_base':vector(base,params),'residual_output_contraction':vector(d.square().mean(),params)}
            baseline,bc=assess(m,val,vy,vdom,vpi,q);policies={}
            for name,v in vecs.items():
                assert torch.isfinite(v).all() and v.norm()>0;move=(-RADIUS*v/v.norm()).float();offset=0
                with torch.no_grad():
                    for param,copy in zip(params,copies):n=param.numel();param.copy_(copy+move[offset:offset+n].reshape_as(param));offset+=n
                    actual=float(torch.cat([(p-copy).flatten().double() for p,copy in zip(params,copies)]).norm());assert abs(actual-RADIUS)<2e-6
                m.calibrate_features(train,td);metrics,correct=assess(m,val,vy,vdom,vpi,q)
                changes={k:metrics[k]-baseline[k] for k in baseline if k!='validation_pair_count' and metrics[k] is not None and baseline[k] is not None}
                policies[name]=dict(gradient_norm=float(v.norm()),actual_radius=actual,metrics=metrics,delta=changes,
                                    changed_correctness_ids=[str(vids[i]) for i in torch.nonzero(correct!=bc).flatten().tolist()])
                with torch.no_grad():
                    for param,copy in zip(params,copies):param.copy_(copy)
                    m.expert_source_mean.copy_(mean)
            rows.append(dict(state=state,step=step,pair_count=info['pair_count'],G=info['G'],ESS=info['ess'].tolist(),Q=q.tolist(),
                             raw_penalty=float(sirm.detach()),gradient_ratio_to_diagnostic_base=float(vecs['binary_SIRM'].norm()/vecs['diagnostic_base'].norm()),baseline=baseline,policies=policies))
        m.load_state_dict(original);assert all(torch.equal(v,m.state_dict()[k]) for k,v in original.items()) and all(p.grad is None for p in m.parameters()) and digest(path)==locks[state]
        print(state,'complete',flush=True)
    write(OUT/'results.json',dict(plan=plan,rows=rows,checkpoint_sha256=locks,source_only=True,target_read=False,
                                 bank_tensors_restored=True,grad_fields_unassigned=True))

if __name__=='__main__':run()

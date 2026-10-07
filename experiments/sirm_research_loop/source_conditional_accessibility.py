"""Source-in ridge utility prediction with inference-only inputs and fixed bank."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import solve
from scipy.special import logsumexp
from experiments.sirm_discover.math import router_teacher
from .routed_discovery_run import ROOT, DEST
from .source_composition_calibration import STATES
from .shrinkage import digest, write

OUT = DEST/'source_conditional_accessibility'
L2 = .01

def utility(g, delta, y):
    ce = logsumexp(g, axis=1)-g[np.arange(len(y)),y]
    single = g[:,None,:]+delta
    loss = logsumexp(single,axis=2)-single[np.arange(len(y)),:,y]
    return ce[:,None]-loss

def fit(x,u,dom):
    weight=np.array([1/(len(set(dom))*sum(dom==d)) for d in dom])
    mean=x.mean(0);std=x.std(0).clip(1e-8);z=(x-mean)/std
    zm=weight@z;um=weight@u;xc=z-zm;uc=u-um
    normal=xc.T@(weight[:,None]*xc)+L2*np.eye(x.shape[1])
    rhs=xc.T@(weight[:,None]*uc)
    coef=solve(normal,rhs,assume_a='pos')
    residual=float(np.abs(normal@coef-rhs).max());assert residual<1e-8
    return dict(mean=mean,std=std,z_mean=zm,u_mean=um,coef=coef,normal_residual=residual)

def predict(model,x):
    return ((x-model['mean'])/model['std']-model['z_mean'])@model['coef']+model['u_mean']

def evaluate(g,delta,y,dom,ids,predicted_u,true_u):
    pi=router_teacher(torch.tensor(predicted_u),temperature=.1,utility_floor=.002).numpy()
    assert np.allclose(pi.sum(1),1) and (pi>=0).all()
    full=g+np.einsum('nm,nmj->nj',pi[:,:6],delta);correct=full.argmax(1)==y;gw=g.argmax(1)!=y
    ce=logsumexp(full,axis=1)-full[np.arange(len(y)),y]
    return dict(n=len(y),equal_domain_accuracy=float(np.mean([correct[dom==d].mean() for d in sorted(set(dom))])),
                equal_domain_CE=float(np.mean([ce[dom==d].mean() for d in sorted(set(dom))])),
                repairs=int((gw&correct).sum()),harms=int((~gw&~correct).sum()),
                bird_correct=int((correct&(y==0)).sum()),NULL=float(pi[:,-1].mean()),
                utility_MSE=float(np.mean((predicted_u-true_u)**2)),
                utility_error_case_MSE=float(np.mean((predicted_u[gw]-true_u[gw])**2)) if gw.any() else None,
                harmful_standalone_mass=float((pi[:,:6]*(true_u<0)).sum(1).mean()),
                repair_ids=ids[gw&correct].tolist(),harm_ids=ids[~gw&~correct].tolist())

def run():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    assert not (OUT/'results.json').exists(),'Do not silently refit a completed diagnostic'
    plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,probe_fitting=True,
              states=STATES,inputs=['CLS_plus_patch_max','global_and_expert_logits_and_pi'],
              targets='Six standalone raw CE utility values, labels used only to form source-in regression targets.',
              L2=L2,selection='None; fixed existing diagnostic L2 .01, no grid or validation early stopping.',
              weighting='Equal-domain source-in squared utility error; unregularized intercept, standardized inputs, L2 on coefficients.',
              inference='Predicted utilities converted with unchanged router_teacher temperature .1/floor .002; no true label or domain is an input.',
              scopes=['all_source_in','other_two_source_in_for_each_domain'],
              limitations='Diagnostic readout on original views, not primary router training or SIRM. Calibration holds exclude probe fitting only, not bank training. Source-out previously selected checkpoints.',
              code_sha256=digest(Path(__file__)))
    write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    locks=json.loads((DEST/'source_composition_calibration/results.json').read_text())['cache_sha256']
    token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis'
    proof=json.loads((token/'extraction_audit.json').read_text());results={}
    for state in STATES:
        path=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';assert digest(path)==locks[state]
        bank=np.load(path);data={}
        for split in ['source_in','source_out']:
            p=token/f'{split}_tokens.npz';assert digest(p)==next(r['cache_sha256'] for r in proof['rows'] if r['split']==split)
            a=np.load(p);assert np.array_equal(bank[split+'_ids'],a['ids']) and np.array_equal(bank[split+'_y'],a['y'])
            g=a['global_logits'].astype(float);delta=bank[split+'_bank_delta'].astype(float);pi=bank[split+'_bank_pi'].astype(float)
            inputs={'CLS_plus_patch_max':np.concatenate([a['cls'],a['patch_max']],1).astype(float),
                    'global_and_expert_logits_and_pi':np.concatenate([g-g.mean(1,keepdims=True),delta.reshape(len(g),-1),pi],1)}
            data[split]=dict(g=g,delta=delta,y=a['y'],dom=a['domains'],ids=a['ids'],inputs=inputs,u=utility(g,delta,a['y']))
        train=data['source_in'];val=data['source_out'];assert not set(train['ids'])&set(val['ids']);rows=[]
        # Source-out outcomes are computed only after each source-in solve is locked.
        for held in [None,0,2,3]:
            tr=np.ones(len(train['y']),bool) if held is None else train['dom']!=held
            va=np.ones(len(val['y']),bool) if held is None else val['dom']==held
            for kind in plan['inputs']:
                model=fit(train['inputs'][kind][tr],train['u'][tr],train['dom'][tr])
                name=f'{state}_{kind}_held{held}'
                np.savez_compressed(OUT/f'{name}.npz',**model)
                record=dict(held_calibration_domain=held,input=kind,fit_sha256=digest(OUT/f'{name}.npz'),normal_residual=model['normal_residual'])
                for split,mask in [('source_in',tr),('source_out',va)]:
                    a=data[split];pu=predict(model,a['inputs'][kind][mask])
                    record[split]=evaluate(a['g'][mask],a['delta'][mask],a['y'][mask],a['dom'][mask],a['ids'][mask],pu,a['u'][mask])
                rows.append(record)
        results[state]=rows
        print(state,[(r['input'],r['source_out']['equal_domain_accuracy'],r['source_out']['repairs'],r['source_out']['harms']) for r in rows[:2]],flush=True)
        assert digest(path)==locks[state]
    write(OUT/'results.json',dict(plan=plan,states=results,cache_sha256=locks,fits=32,
                                  source_only=True,target_read=False,bank_updates=False,normal_equations_verified=True))

if __name__=='__main__':run()

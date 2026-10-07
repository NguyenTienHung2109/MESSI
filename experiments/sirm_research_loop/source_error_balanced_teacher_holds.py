"""Joint original/H2 source utility probe; fixed bank, no target access."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.sirm_discover.math import center_logits,router_teacher,residual_utilities,soft_null_q
from .routed_conditional_utility import load
from .routed_discovery_run import ROOT,DEST,PRIOR
from .source_accessibility_H2 import inputs
from .source_conditional_accessibility import fit,predict,evaluate,utility,L2
from .shrinkage import digest,write
from .teacher_distribution_helpers import teacher,fit_student,predict_pi,evaluate_pi,gates_pi

OUT=DEST/'source_error_balanced_teacher_holds'

def gates(g,delta,y,dom,pu,batches=False):
    pi=router_teacher(torch.tensor(pu),.1,.002)
    _,_,u,_=residual_utilities(torch.tensor(g),torch.tensor(delta),pi,torch.tensor(y))
    U=torch.stack([u[dom==d].mean(0) for d in [0,2,3]])
    q,_,_,v=soft_null_q(U,torch.ones_like(U,dtype=torch.bool),.05,.002,.002)
    result=dict(U=U.tolist(),V=v.tolist(),Q=q.tolist(),shared_experts=torch.nonzero((q>0).sum(0)>=2).flatten().tolist())
    if batches:
        assert len(u)==40*96;ema=None;rows=[]
        for step in range(40):
            sl=slice(step*96,(step+1)*96);batch=torch.stack([u[sl][dom[sl]==d].mean(0) for d in [0,2,3]])
            ema=batch.clone() if ema is None else .9*ema+.1*batch
            q,_,_,v=soft_null_q(ema,torch.ones_like(ema,dtype=torch.bool),.05,.002,.002)
            rows.append(dict(batch_step=200+5*step,U=ema.tolist(),V=v.tolist(),Q=q.tolist(),shared_experts=torch.nonzero((q>0).sum(0)>=2).flatten().tolist()))
        result.update(hypothetical_EMA_rows=rows,EMA_shared_refreshes=sum(bool(r['shared_experts']) for r in rows))
    return result

def run():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists()
    originalfits=json.loads((DEST/'source_conditional_accessibility/results.json').read_text())
    proof=json.loads((DEST/'source_conditional_H2/extraction_audit.json').read_text())
    token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';tokenproof=json.loads((token/'extraction_audit.json').read_text())
    source={}
    for split in ['source_in','source_out']:
        p=token/f'{split}_tokens.npz';assert digest(p)==next(r['cache_sha256'] for r in tokenproof['rows'] if r['split']==split)
        source[split]=dict(np.load(p))
    plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,probe_fitting=True,
              states=[st for st in originalfits['states'] if st.endswith('selected')],inputs=originalfits['plan']['inputs'],L2=L2,
              held_domains=[0,2,3],intervention='Held calibration domain excluded from selector fitting; bank learning and centering still use all sources. Equal loss mass for frozen-global-correct/errors within every source domain/view. Same locked .1/.002 teacher, linear student, inputs, cohorts and fixed L2; compare completed frequency-weighted fits. No validation-defined groups.',
              fitting='Equal domain/view/correctness-stratum soft-target CE plus .01/2 times coefficient squared norm, standardized pooled fitting inputs, unregularized intercept. LBFGS max500, stationarity1e-5, no validation stopping. L2 numeric value fixed; loss units differ from utility MSE. Held calibration-domain fits on source-selected banks only; no refit of last banks chosen for favorable scores.',
              centering='Actual96-image training batch mean for H2 targets; calibrated original source-in mean for original evaluation.',
              limits='3840 augmented occurrences repeat original images. Held calibration domains exclude fitting only, not bank learning or batch centering. Saved-bank probes are not Stage2 SIRM trajectories.',
              code_sha256=digest(Path(__file__)),helper_sha256=digest(Path(__file__).with_name('teacher_distribution_helpers.py')),cohorts=['original','H2','joint'],H2_cache_audit_sha256=digest(DEST/'source_conditional_H2/extraction_audit.json'))
    write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());results={};locks={}
    for state in plan['states']:
        root=PRIOR if state.startswith('unscaled') else DEST;which='selected' if state.endswith('selected') else 'last'
        m,p,path=load(root,which);m.requires_grad_(False);locks[state]=digest(path);before={k:v.clone() for k,v in m.state_dict().items()}
        si=source['source_in'];raworiginal=torch.tensor(np.concatenate([si['cls'],si['patch_max']],1));m.calibrate_features(raworiginal,torch.tensor(si['domains']))
        ys=[];ds=[];gs=[];deltas=[];ids=[];Xs={k:[] for k in plan['inputs']}
        for rec in proof['rows']:
            file=DEST/'source_conditional_H2'/f"H2_step{rec['step']}.pt";assert digest(file)==rec['sha256']
            b=torch.load(file,map_location='cpu',weights_only=False);raw=b['raw'];z=m.bank_features(raw);g=m.global_head(raw[:,:384]).double();oldpi=m.router(z).softmax(1).double()
            delta=center_logits(torch.stack([e(z) for e in m.residuals],1));delta=(delta-delta.mean(0,keepdim=True)).double()
            gs.append(g.numpy());deltas.append(delta.numpy());ys.append(b['y'].numpy());ds.append(b['domains'].numpy());ids+=b['ids']
            for kind in Xs:Xs[kind].append(inputs(kind,raw,g,delta,oldpi))
        g,delta,y,dom=map(np.concatenate,[gs,deltas,ys,ds]);ids=np.array(ids);u=utility(g,delta,y);Xs={k:np.concatenate(v) for k,v in Xs.items()}
        bankpath=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';assert digest(bankpath)==originalfits['cache_sha256'][state];bank=np.load(bankpath)
        assert not set(ids)&set(bank['source_out_ids']);out={}
        for split in ['source_in','source_out']:
            b=source[split];assert np.array_equal(b['ids'],bank[split+'_ids']);gg=b['global_logits'].astype(float);dd=bank[split+'_bank_delta'].astype(float);pp=bank[split+'_bank_pi'].astype(float)
            out[split]=dict(g=gg,delta=dd,y=b['y'],dom=b['domains'],ids=b['ids'],u=utility(gg,dd,b['y']),
                            X={'CLS_plus_patch_max':np.concatenate([b['cls'],b['patch_max']],1).astype(float),
                               'global_and_expert_logits_and_pi':np.concatenate([gg-gg.mean(1,keepdims=True),dd.reshape(len(gg),-1),pp],1)})
        rows=[]
        for held in [0,2,3]:
            for cohort in ['original','H2','joint']:
                for kind in plan['inputs']:
                    b=out['source_in']
                    if cohort=='original':xx,uu,groups=b['X'][kind],b['u'],b['dom']
                    elif cohort=='H2':xx,uu,groups=Xs[kind],u,dom
                    else:xx,uu,groups=np.concatenate([b['X'][kind],Xs[kind]]),np.concatenate([b['u'],u]),np.concatenate([b['dom'],dom+10])
                    original_wrong=b['g'].argmax(1)!=b['y'];H2_wrong=g.argmax(1)!=y
                    wrong=original_wrong if cohort=='original' else H2_wrong if cohort=='H2' else np.concatenate([original_wrong,H2_wrong])
                    groups=groups+20*wrong
                    keep=(groups%10)!=held;xx,uu,groups=xx[keep],uu[keep],groups[keep]
                    assert len(set(groups))==(8 if cohort=='joint' else 4)
                    counts={int(group):int((groups==group).sum()) for group in sorted(set(groups))};assert min(counts.values())>=2
                    model=fit_student(xx,teacher(uu),groups)
                    model['fitting_prior']=np.mean([teacher(uu)[groups==group].mean(0) for group in sorted(set(groups))],0)
                    file=OUT/f'{state}_{kind}_{cohort}_held{held}.npz';np.savez_compressed(file,**model)
                    record=dict(input=kind,cohort=cohort,held_calibration_domain=held,fit_sha256=digest(file),gradient_max=model['gradient_max'],objective=model['objective'],closure_calls=model['closure_calls'],fit_rows=len(xx),fit_weight_groups=len(set(groups)),weight_group_counts=counts,correctness_loss_weights=[.5,.5])
                    for split,b in out.items():
                        mask=b['dom']!=held if split=='source_in' else b['dom']==held
                        record[split+'_original']=evaluate_pi(b['g'][mask],b['delta'][mask],b['y'][mask],b['dom'][mask],b['ids'][mask],predict_pi(model,b['X'][kind][mask]),b['u'][mask])
                    mask=dom!=held;hp=predict_pi(model,Xs[kind][mask]);record['H2_source_in']=evaluate_pi(g[mask],delta[mask],y[mask],dom[mask],ids[mask],hp,u[mask])
                    rows.append(record)
                    print(state,'held',held,cohort,kind,record['source_out_original']['equal_domain_accuracy'],record['source_out_original']['repairs'],record['source_out_original']['harms'],flush=True)
        results[state]=rows;m.load_state_dict(before)
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and all(p.grad is None for p in m.parameters()) and digest(path)==locks[state]
        print(state,'complete',flush=True)
    write(OUT/'results.json',dict(plan=plan,states=results,checkpoint_sha256=locks,fits=36,
                                  source_only=True,target_read=False,bank_tensors_restored=True,grad_fields_unassigned=True))

if __name__=='__main__':run()

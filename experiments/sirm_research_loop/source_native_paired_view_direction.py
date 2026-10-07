"""Paired original/H2 classification directions on all scheduled source batches."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import router_teacher
from .source_probe_clean_preservation import outputs,assess
from .routed_conditional_utility import load
from .routed_discovery_run import ROOT,DEST,PRIOR
from .source_conditional_accessibility import predict
from .discovery_error_gradient_diagnosis import vector
from .shrinkage import digest,write
OUT=DEST/'source_native_paired_view_direction';RADIUS=.01

def run():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    probe=DEST/'source_H2_utility_probe';proof=json.loads((probe/'results.json').read_text())
    h2proof=json.loads((DEST/'source_conditional_H2/extraction_audit.json').read_text());hashes={r['step']:r['sha256'] for r in h2proof['rows']}
    plan=dict(source_only=True,target_read=False,primary_training=False,probe_refitting=False,optimizer_updates=False,
      states=list(proof['states']),windows='All40 scheduled H2 refresh batches for every state; not selected for support or validation outcomes.',
      directions=['H2_classification','original_classification_own_pi','original_classification_H2_pi','H2_residual_contraction'],
      radius=RADIUS,centering='Each original/H2 gradient uses differentiable mean of exactly the paired96-image cohort. Original source-in equal-domain recalibration for all finite evaluations.',
      routing='Saved native softmax router evaluated per view and detached. Original own-pi and original fixed-H2-pi distinguish view changes from routing changes. Q/shared flag retained only as common probe-derived diagnostic masks; not claimed native Q.',
      evaluation='Fixed readout on original source-in and disjoint source-out, after equal-radius disposable residual moves. No router/global updates.',
      limits='Saved-bank local directions, not a training trajectory or test of generalization to new augmentation draws. Correlated windows, images and bank states.',code_sha256=digest(Path(__file__)),probe_results_sha256=digest(probe/'results.json'))
    write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';ta=json.loads((token/'extraction_audit.json').read_text());data={}
    for split in ['source_in','source_out']:
        p=token/f'{split}_tokens.npz';assert digest(p)==next(r['cache_sha256'] for r in ta['rows'] if r['split']==split);a=np.load(p)
        data[split]=(torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains']),a['ids'])
    train,ty,td,tids=data['source_in'];val,vy,vd,vids=data['source_out'];assert not set(tids)&set(vids)
    lookup={str(id):i for i,id in enumerate(tids)}
    mapped=lambda dom:torch.tensor([0 if int(d)==0 else 1 if int(d)==2 else 2 for d in dom])
    tdom,vdom=mapped(td),mapped(vd);rows=[];locks={}
    for state,records in proof['states'].items():
        rec=records[0];assert rec['input']=='CLS_plus_patch_max' and rec['held_calibration_domain'] is None
        file=probe/f'{state}_CLS_plus_patch_max_heldNone.npz';assert digest(file)==rec['fit_sha256'];model=dict(np.load(file))
        m,p,path=load(PRIOR if state.startswith('unscaled') else DEST,'selected' if state.endswith('selected') else 'last')
        locks[state]=digest(path);assert locks[state]==proof['checkpoint_sha256'][state]
        initial={k:v.clone() for k,v in m.state_dict().items()};m.calibrate_features(train,td);mean=m.expert_source_mean.clone()
        params=list(m.residuals.parameters());copies=[p.detach().clone() for p in params]
        fixedpi=lambda raw:m.router(m.bank_features(raw)).softmax(1).detach()
        vpi,tpi=fixedpi(val),fixedpi(train)
        for gate in rec['H2_gate']['hypothetical_EMA_rows']:
            step=gate['batch_step'];cache=DEST/'source_conditional_H2'/f'H2_step{step}.pt';assert digest(cache)==hashes[step]
            a=torch.load(cache,map_location='cpu',weights_only=False);raw,y,dom=a['raw'],a['y'],a['domains']
            indices=[lookup[str(id)] for id in a['ids']];assert torch.equal(y,ty[indices]) and torch.equal(dom,td[indices])
            original=train[indices];hp,op=fixedpi(raw),fixedpi(original);q=torch.tensor(gate['Q'])
            gh,dh,fh=outputs(m,raw,hp,True);go,do,fo=outputs(m,original,op,True);_,_,fohp=outputs(m,original,hp,True)
            vecs={'H2_classification':vector(F.cross_entropy(fh,y),params),'original_classification_own_pi':vector(F.cross_entropy(fo,y),params),'original_classification_H2_pi':vector(F.cross_entropy(fohp,y),params),'H2_residual_contraction':vector(dh.square().mean(),params)}
            cos={n:float(F.cosine_similarity(v,vecs['H2_classification'],dim=0)) for n,v in vecs.items()}
            view_metrics=dict(H2_global_correct=int(gh.argmax(1).eq(y).sum()),original_global_correct=int(go.argmax(1).eq(y).sum()),pi_L1=float((hp-op).abs().sum(1).mean()))
            baseline,bc=assess(m,val,vy,vdom,vpi,q)
            with torch.no_grad():vg,_,vf=outputs(m,val,vpi);before_predictions=vf.argmax(1);global_predictions=vg.argmax(1)
            tb,tbc=assess(m,train,ty,tdom,tpi,q);policies={}
            for name,v in vecs.items():
                assert torch.isfinite(v).all() and v.norm()>0;move=(-RADIUS*v/v.norm()).float();offset=0
                with torch.no_grad():
                    for param,copy in zip(params,copies):n=param.numel();param.copy_(copy+move[offset:offset+n].reshape_as(param));offset+=n
                    actual=float(torch.cat([(p-copy).flatten().double() for p,copy in zip(params,copies)]).norm());assert abs(actual-RADIUS)<2e-6
                m.calibrate_features(train,td);metrics,correct=assess(m,val,vy,vdom,vpi,q);tm,tc=assess(m,train,ty,tdom,tpi,q)
                with torch.no_grad():_,_,after_f=outputs(m,val,vpi);after_predictions=after_f.argmax(1)
                delta=lambda new,old:{k:new[k]-old[k] for k in old if k!='validation_pair_count' and new[k] is not None and old[k] is not None}
                policies[name]=dict(gradient_norm=float(v.norm()),actual_radius=actual,metrics=metrics,delta=delta(metrics,baseline),train_metrics=tm,train_delta=delta(tm,tb),changed_cases=[dict(id=str(vids[i]),label=int(vy[i]),global_prediction=int(global_predictions[i]),before_prediction=int(before_predictions[i]),after_prediction=int(after_predictions[i]),before_correct=bool(bc[i]),after_correct=bool(correct[i])) for i in torch.nonzero(correct!=bc).flatten().tolist()],changed_correctness_ids=[str(vids[i]) for i in torch.nonzero(correct!=bc).flatten().tolist()])
                with torch.no_grad():
                    for param,copy in zip(params,copies):param.copy_(copy)
                    m.expert_source_mean.copy_(mean)
            rows.append(dict(state=state,step=step,shared_support=bool(gate['shared_experts']),paired_ids=a['ids'],gradient_cosine_to_H2=cos,view_metrics=view_metrics,baseline=baseline,train_baseline=tb,policies=policies))
        m.load_state_dict(initial);assert all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) and all(p.grad is None for p in m.parameters()) and digest(path)==locks[state]
        print(state,'complete',flush=True)
    write(OUT/'results.json',dict(plan=plan,rows=rows,checkpoint_sha256=locks,source_only=True,target_read=False,bank_tensors_restored=True,grad_fields_unassigned=True))
if __name__=='__main__':run()

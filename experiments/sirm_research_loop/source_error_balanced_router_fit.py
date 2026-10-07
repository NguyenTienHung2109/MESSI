"""Prespecified source-in error-balanced teacher fit at immutable iteration16 banks."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import router_teacher
from .source_router_teacher_fit import score, L2
from .source_specialist_routing_audit import ROOT, DEST, replay
from .routed_conditional_utility import load
from .source_probe_penalty_direction import outputs
from .shrinkage import digest, write

OUT = DEST / 'source_error_balanced_router_fit'

def run():
    torch.set_num_threads(4);torch.manual_seed(0)
    OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists()
    audit=json.loads((DEST/'final_analysis_audit.json').read_text())
    for rel,h in audit['sha256'].items():assert digest(DEST/rel)==h,rel
    plan=dict(source_only=True,target_read=False,primary_state='selected200',secondary_state='last399 mechanism only; no bank selection',
        hypothesis='Equal-domain mean teacher loss underweights rare frozen-global errors. Test one fixed 50/50 correct/error cohort control.',
        weighting='Half objective mass for frozen-global correct and half for errors. Within each cohort equal source domains; each domain/cohort needs at least2 distinct images.',
        L2=L2,L2_selection='Fixed .01 matching previous diagnostic; no grid or validation tuning.',
        data='Original source-in teacher targets only; source-out only final reporting. Labels define training weights/teacher, never inference inputs.',
        optimizer='Same native router initialization and linear softmax architecture, float64 LBFGS max_iter1000/max_eval1500/strong_wolfe, tolerance_grad1e-7, tolerance_change1e-12.',
        frozen='All original model parameters frozen; fit new router copy only. Bank/global/backbone immutable.',
        scale_controls='Native correction to fitted source-in RMS; fitted correction to native source-in RMS. Alphas fixed before source-out scoring.',
        comparison='Previous equal-domain teacher fit, no new baseline refit.',
        limits='Diagnostic fitting; not matched400-step discovery/SIRM trajectory. No claim of active SIRM or target gain.',code_sha256=digest(Path(__file__)))
    write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    previous_path=DEST/'source_router_teacher_fit/results.json';previous=json.loads(previous_path.read_text());locks={str(previous_path):digest(previous_path)}
    cache=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';data={}
    for split in ['source_in','source_out']:
        path=cache/f'{split}_tokens.npz';assert digest(path)==previous['input_sha256'][str(path)];locks[str(path)]=digest(path)
        a=dict(np.load(path));data[split]=(torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains']),a['ids'])
    assert not set(data['source_in'][3])&set(data['source_out'][3]);states={}
    for state in ['selected','last']:
        m,p,path=load(DEST,state);m.requires_grad_(False);assert not any(q.requires_grad for q in m.parameters());before=copy.deepcopy(m.state_dict());locks[str(path)]=digest(path)
        m.calibrate_features(data['source_in'][0],data['source_in'][2]);fixed={}
        with torch.no_grad():
            for split,(raw,y,dom,ids) in data.items():
                r=replay(m,raw,y);g,delta,full=outputs(m,raw,r['pi']);teacher=router_teacher(r['raw_gain'],m.config['router_temperature'],m.config['utility_floor'])
                fixed[split]=dict(z=m.bank_features(raw).double(),g=g,delta=delta,full=full,pi=r['pi'],teacher=teacher,y=y,dom=dom,ids=ids)
        router=copy.deepcopy(m.router).double().requires_grad_(True);x=fixed['source_in'];correct=x['g'].argmax(1).eq(x['y']);w=torch.zeros(len(correct),dtype=torch.float64);counts={}
        for flag in [False,True]:
            for d in x['dom'].unique():
                mask=correct.eq(flag)&x['dom'].eq(d);n=int(mask.sum());assert n>=2
                assert len(set(x['ids'][mask.numpy()]))==n
                w[mask]=.5/(len(x['dom'].unique())*n);counts[f'domain{int(d)}_correct{flag}']=n
        assert torch.allclose(w.sum(),torch.tensor(1.,dtype=torch.float64))
        assert abs(float(w[correct].sum())-.5)<1e-12 and abs(float(w[~correct].sum())-.5)<1e-12
        opt=torch.optim.LBFGS(router.parameters(),max_iter=1000,max_eval=1500,tolerance_grad=1e-7,tolerance_change=1e-12,line_search_fn='strong_wolfe');evaluations=0
        def objective():return (w*(-x['teacher'].double()*router(x['z']).log_softmax(1)).sum(1)).sum()+.5*L2*router.weight.square().sum()
        def closure():
            nonlocal evaluations
            opt.zero_grad();loss=objective();loss.backward();evaluations+=1;return loss
        initial=float(objective().detach());opt.step(closure);final=float(closure().detach());grad=max(float(q.grad.abs().max()) for q in router.parameters())
        with torch.no_grad():
            train_pi=router(x['z']).softmax(1).float();native_rms=(x['full']-x['g']).square().mean().sqrt();fit_rms=(train_pi[:,:6,None]*x['delta']).sum(1).square().mean().sqrt();assert native_rms>0 and fit_rms>0
            fit_scale=float(native_rms/fit_rms);native_scale=float(fit_rms/native_rms);splits={};eda={}
            for split,a in fixed.items():
                pi=router(a['z']).softmax(1).float();corr=(pi[:,:6,None]*a['delta']).sum(1);f=a['g']+corr;native_corr=a['full']-a['g']
                policies={'native':(a['full'],a['pi']),'error_balanced_fit':(f,pi),
                    'native_RMS_matched_to_fit':(a['g']+native_scale*native_corr,a['pi']),
                    'fit_RMS_matched_to_native':(a['g']+fit_scale*corr,pi)}
                splits[split]={k:score(a['g'],v,a['y'],a['dom'],p0,a['delta'],a['teacher']) for k,(v,p0) in policies.items()}
                assert splits[split]['native']['accuracy_equal_domain']==previous['states'][state]['splits'][split]['native']['accuracy_equal_domain']
                _,_,oracle=outputs(m,data[split][0],a['teacher']);gc=a['g'].argmax(1).eq(a['y']);fc=f.argmax(1).eq(a['y']);oc=oracle.argmax(1).eq(a['y']);l1=(pi-a['teacher']).abs().sum(1)
                cohorts={}
                for name,mask in [('global_correct',gc),('global_wrong',~gc),('oracle_repairs',~gc&oc)]:
                    cohorts[name]=dict(n=int(mask.sum()),fitted_correct=int(fc[mask].sum()),teacher_fit_L1=float(l1[mask].mean()),teacher_NULL=float(a['teacher'][mask,-1].mean()),fitted_NULL=float(pi[mask,-1].mean()))
                cases=[]
                for i in torch.nonzero((~gc&oc)|(gc&~fc)|(~gc&fc)).flatten().tolist():
                    cases.append(dict(id=str(a['ids'][i]),label=int(a['y'][i]),global_prediction=int(a['g'][i].argmax()),native_prediction=int(a['full'][i].argmax()),fitted_prediction=int(f[i].argmax()),oracle_prediction=int(oracle[i].argmax()),teacher=a['teacher'][i].tolist(),router=pi[i].tolist()))
                eda[split]=dict(cohorts=cohorts,cases=cases)
                np.savez_compressed(OUT/f'{state}_{split}_replay.npz',ids=a['ids'],y=a['y'].numpy(),domains=a['dom'].numpy(),global_logits=a['g'].numpy(),native_logits=a['full'].numpy(),fitted_logits=f.numpy(),fitted_router=pi.numpy())
        fit=OUT/f'{state}_router.pt';torch.save(dict(weight=router.weight.detach(),bias=router.bias.detach(),L2=L2,checkpoint_sha256=digest(path)),fit)
        states[state]=dict(checkpoint_step=p.get('selected_step',p.get('step')),cohort_counts=counts,objective_initial=initial,objective_final=final,max_abs_gradient=grad,evaluations=evaluations,
            iterations=opt.state[next(iter(router.parameters()))]['n_iter'],source_in_fit_to_native_scale=fit_scale,source_in_native_to_fit_scale=native_scale,splits=splits,case_analysis=eda,fit_sha256=digest(fit))
        m.load_state_dict(before);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());assert all(q.grad is None for q in m.parameters())
        write(OUT/'progress.json',dict(state=state,status='state_complete',target_read=False))
    for path,h in locks.items():assert digest(Path(path))==h
    write(OUT/'results.json',dict(source_only=True,target_read=False,diagnostic_fitting=True,primary_training=False,checkpoint_changes=False,states=states,input_sha256=locks,plan_sha256=digest(OUT/'plan.json'),class_names=previous['class_names']))
    for path in OUT.iterdir():audit['sha256'][str(path.relative_to(DEST))]=digest(path)
    write(DEST/'final_analysis_audit.json',audit)
    print(json.dumps({state:dict(gradient=v['max_abs_gradient'],source_out={k:{field:r[field] for field in ['accuracy_equal_domain','CE_equal_domain','repairs','harms','router_NULL']} for k,r in v['splits']['source_out'].items()},cohorts=v['case_analysis']['source_out']['cohorts']) for state,v in states.items()},indent=2))

if __name__=='__main__':run()

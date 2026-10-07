"""Fixed-bank source-only teacher realizability diagnostic; not Stage-2 ON/OFF."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import router_teacher
from .source_specialist_routing_audit import ROOT, DEST, replay
from .routed_conditional_utility import load
from .source_probe_penalty_direction import outputs
from .shrinkage import digest, write

OUT = DEST / 'source_router_teacher_fit'
L2 = .01

def score(g, f, y, dom, pi, delta, teacher):
    gc, fc = g.argmax(1).eq(y), f.argmax(1).eq(y)
    result = dict(accuracy_equal_domain=float(torch.stack([fc[dom.eq(d)].double().mean() for d in dom.unique()]).mean()),
        CE_equal_domain=float(torch.stack([F.cross_entropy(f[dom.eq(d)],y[dom.eq(d)]) for d in dom.unique()]).mean()),
        repairs=int((~gc & fc).sum()), harms=int((gc & ~fc).sum()),
        router_NULL=float(pi[:,-1].mean()), correction_RMS=float((f-g).square().mean().sqrt()),
        teacher_router_L1=float((pi-teacher).abs().sum(1).mean()),
        class_stats={})
    for c in range(10):
        mask=y.eq(c)
        if not mask.any():continue
        result['class_stats'][str(c)]=dict(n=int(mask.sum()),global_correct=int(gc[mask].sum()),full_correct=int(fc[mask].sum()),
            repairs=int((~gc[mask]&fc[mask]).sum()),harms=int((gc[mask]&~fc[mask]).sum()),router_mean=pi[mask].mean(0).tolist())
    return result

def run():
    torch.set_num_threads(4);torch.manual_seed(0)
    OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists()
    audit=json.loads((DEST/'final_analysis_audit.json').read_text())
    for rel,h in audit['sha256'].items():assert digest(DEST/rel)==h,rel
    plan=dict(source_only=True,target_read=False,primary_state='selected200',secondary_state='last399, mechanism diagnostic only; do not choose bank using this fit',
        hypothesis='Fixed-bank native linear router may fail to realize the label-conditioned teacher; fit the same input-only linear softmax architecture using source-in teacher supervision.',
        L2=L2,L2_selection='Fixed .01 before fitting; no hyperparameter grid, validation selection or target access.',
        data='Original source-in only, equal-domain weighted teacher cross entropy; source-out only for final diagnostic.',
        optimizer='LBFGS max_iter1000, max_eval1500, strong_wolfe, tolerance_grad1e-7, tolerance_change1e-12; float64.',
        parameters='New diagnostic copy of router only; every original model tensor immutable. Expert/global/backbone fixed.',
        normalization='Existing frozen bank_features source-in normalization; no new label-conditioned features.',
        controls='Native, fitted router, native uniform contraction to fitted source-in correction RMS, fitted correction rescaled to native source-in RMS. Alphas locked without source-out/target scores.',
        limits='Not a matched400-step Stage-2 experiment, no SIRM execution, no new checkpoint choice, no factor/invariance or target benefit claim.',code_sha256=digest(Path(__file__)))
    write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    data={};locks={};cache=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis'
    previous=json.loads((DEST/'source_specialist_routing/results.json').read_text())
    for split in ['source_in','source_out']:
        path=cache/f'{split}_tokens.npz';assert digest(path)==previous['input_sha256'][str(path)];locks[str(path)]=digest(path)
        a=dict(np.load(path));data[split]=(torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains']),a['ids'])
    assert not set(data['source_in'][3])&set(data['source_out'][3]);results={}
    for state in ['selected','last']:
        m,p,path=load(DEST,state);locks[str(path)]=digest(path);before=copy.deepcopy(m.state_dict());m.calibrate_features(data['source_in'][0],data['source_in'][2])
        fixed={}
        with torch.no_grad():
            for split,(raw,y,dom,ids) in data.items():
                r=replay(m,raw,y);z=m.bank_features(raw).double();g,delta,full=outputs(m,raw,r['pi'])
                teacher=router_teacher(r['raw_gain'],m.config['router_temperature'],m.config['utility_floor'])
                fixed[split]=dict(z=z,g=g,delta=delta,full=full,pi=r['pi'],teacher=teacher,y=y,dom=dom,ids=ids)
        router=copy.deepcopy(m.router).double();x=fixed['source_in'];w=torch.zeros(len(x['y']),dtype=torch.float64)
        for d in x['dom'].unique():w[x['dom'].eq(d)]=1./(len(x['dom'].unique())*int(x['dom'].eq(d).sum()))
        optimizer=torch.optim.LBFGS(router.parameters(),max_iter=1000,max_eval=1500,tolerance_grad=1e-7,tolerance_change=1e-12,line_search_fn='strong_wolfe')
        evaluations=0
        def objective():
            logp=router(x['z']).log_softmax(1)
            return (w*(-x['teacher'].double()*logp).sum(1)).sum()+.5*L2*router.weight.square().sum()
        def closure():
            nonlocal evaluations
            optimizer.zero_grad();loss=objective();loss.backward();evaluations+=1
            return loss
        initial=float(objective().detach());optimizer.step(closure);final=float(closure().detach());grad=max(float(q.grad.abs().max()) for q in router.parameters())
        with torch.no_grad():
            train_pi=router(x['z']).softmax(1).float();native_rms=(x['full']-x['g']).square().mean().sqrt();fit_rms=(train_pi[:,:6,None]*x['delta']).sum(1).square().mean().sqrt()
            fit_scale=float(native_rms/fit_rms) if fit_rms>0 else None;shrink=float(fit_rms/native_rms) if native_rms>0 else None
            states={}
            for split,a in fixed.items():
                pi=router(a['z']).softmax(1).float();corr=(pi[:,:6,None]*a['delta']).sum(1);native_corr=a['full']-a['g']
                policies={'native':(a['full'],a['pi']),'fitted':(a['g']+corr,pi),
                          'native_RMS_matched_to_fit':(a['g']+shrink*native_corr,a['pi']),
                          'fit_RMS_matched_to_native':(a['g']+fit_scale*corr,pi)}
                states[split]={k:score(a['g'],f,a['y'],a['dom'],p0,a['delta'],a['teacher']) for k,(f,p0) in policies.items()}
                np.savez_compressed(OUT/f'{state}_{split}_replay.npz',ids=a['ids'],y=a['y'].numpy(),domains=a['dom'].numpy(),global_logits=a['g'].numpy(),native_logits=a['full'].numpy(),fitted_logits=(a['g']+corr).numpy(),fitted_router=pi.numpy())
        fit_path=OUT/f'{state}_router.pt';torch.save(dict(weight=router.weight.detach(),bias=router.bias.detach(),L2=L2,checkpoint_sha256=digest(path)),fit_path)
        results[state]=dict(checkpoint_step=p.get('selected_step',p.get('step')),objective_initial=initial,objective_final=final,
            evaluations=evaluations,max_abs_gradient=grad,iterations=optimizer.state[next(iter(router.parameters()))]['n_iter'],
            source_in_fit_to_native_scale=fit_scale,source_in_native_to_fit_scale=shrink,splits=states,fit_sha256=digest(fit_path))
        write(OUT/'progress.json',dict(state=state,status='state_complete',target_read=False))
        m.load_state_dict(before);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());assert all(q.grad is None for q in m.parameters())
    for path,h in locks.items():assert digest(Path(path))==h
    write(OUT/'results.json',dict(source_only=True,target_read=False,diagnostic_fitting=True,primary_training=False,checkpoint_changes=False,
        states=results,input_sha256=locks,plan_sha256=digest(OUT/'plan.json'),class_names=previous['class_names']))
    for path in OUT.iterdir():audit['sha256'][str(path.relative_to(DEST))]=digest(path)
    write(DEST/'final_analysis_audit.json',audit)
    print(json.dumps({s:dict(convergence=r['max_abs_gradient'],source_out=r['splits']['source_out']) for s,r in results.items()},indent=2))

if __name__=='__main__':run()

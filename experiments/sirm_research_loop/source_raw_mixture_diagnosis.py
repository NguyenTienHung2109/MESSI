"""Source-fitted raw residual simplex controls and label-informed margin capacity."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar, linprog
from scipy.special import logsumexp
from .routed_discovery_run import ROOT, DEST
from .source_composition_calibration import STATES, scores
from .shrinkage import digest, write

OUT = DEST / 'source_raw_mixture'

def fit(g, delta, y, dom, mode):
    weight = np.array([1 / (len(set(dom)) * sum(dom == d)) for d in dom])
    def calc(w):
        f = g + np.einsum('nmj,m->nj', delta, w)
        loss = logsumexp(f, axis=1) - f[np.arange(len(y)), y]
        prob = np.exp(f - logsumexp(f, axis=1)[:,None])
        prob[np.arange(len(y)),y] -= 1
        grad = np.einsum('n,nj,nmj->m', weight, prob, delta)
        return float(weight @ loss), grad
    if mode == 'uniform':
        r = minimize_scalar(lambda x: calc(np.ones(6)*x/6)[0], bounds=(0,1), method='bounded', options={'xatol':1e-12})
        possibilities = [0.,float(r.x),1.]
        alpha = min(possibilities,key=lambda x:calc(np.ones(6)*x/6)[0])
        w = np.ones(6)*alpha/6
        value,grad=calc(w)
        derivative=grad.mean()
        gap=alpha*derivative-min(0.,derivative)
    else:
        r=minimize(calc,np.ones(6)/6,jac=True,method='SLSQP',bounds=[(0,1)]*6,
                   constraints=[{'type':'ineq','fun':lambda w:1-w.sum(),'jac':lambda w:-np.ones(6)}],
                   options={'maxiter':2000,'ftol':1e-12})
        w=r.x;value,grad=calc(w)
        gap=float(w@grad-min(0.,grad.min()))
    assert r.success and gap < 1e-5 and w.min()>=-1e-9 and w.sum()<=1+1e-9,(r.message,gap,w)
    return dict(weights=w.tolist(),source_in_CE=value,converged=True,FW_gap=float(gap))

def capacity(g, delta, y, dom, ids, native):
    cases=[]
    for i in np.flatnonzero((g.argmax(1)!=y) | (native.argmax(1)!=y)):
        competitors=np.array([j for j in range(g.shape[1]) if j != y[i]])
        base=g[i,y[i]]-g[i,competitors]
        direction=delta[i,:,y[i]][:,None]-delta[i][:,competitors]
        A=np.concatenate([-direction.T,np.ones((len(competitors),1))],1)
        A=np.concatenate([A,np.array([[1.,1.,1.,1.,1.,1.,0.]])],0)
        b=np.concatenate([base,[1.]])
        lp=linprog([0.,0.,0.,0.,0.,0.,-1.],A_ub=A,b_ub=b,bounds=[(0,None)]*6+[(None,None)],method='highs')
        assert lp.success
        w=lp.x[:6];f=g[i]+w@delta[i]
        margin=float(f[y[i]]-f[competitors].max())
        assert abs(margin-lp.x[6])<1e-7 and w.sum()<=1+1e-7
        cases.append(dict(id=str(ids[i]),domain=int(dom[i]),label=int(y[i]),global_correct=bool(g[i].argmax()==y[i]),
                          native_correct=bool(native[i].argmax()==y[i]),simplex_margin=margin,
                          correct=bool(f.argmax()==y[i]),weights=w.tolist()))
    return dict(cases=cases,repairable_global_errors=sum(not c['global_correct'] and c['correct'] for c in cases),
                bird_errors_repairable=sum(c['label']==0 and not c['global_correct'] and c['correct'] for c in cases))

def run():
    OUT.mkdir(exist_ok=True)
    plan=dict(source_only=True,target_read=False,primary_training=False,bank_updates=False,
              states=STATES,controls=['uniform_raw_mix','selective_raw_simplex'],
              weights='Nonnegative raw expert coefficients, total mass <=1. Remaining mass is NULL. No expert correction is amplified beyond its unscaled amplitude.',
              objective='Equal-domain original source-in CE, no regularization grid; all-source and other-two-source fits. Source-out evaluates locked fits only.',
              capacity='Separate true-label-informed per-image linear program maximizes classification margin over raw simplex. Source-out LP is retrospective and never trains/selects a policy.',
              hypothesis='Fixed routed subsets lack capacity, but stronger legal mixtures of existing raw experts may recover useful directions. Compare fitted source transfer against label-informed capacity and uniform composition.',
              limitations='Frozen banks previously saw all source domains; calibration holds are not independently trained domain holds. Source-out previously selected checkpoints. Fixed coefficients do not test a learned instance-dependent router.',
              code_sha256=digest(Path(__file__)))
    write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    locks=json.loads((DEST/'source_composition_calibration/results.json').read_text())['cache_sha256']
    token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis'
    audit=json.loads((token/'extraction_audit.json').read_text());results={}
    for state in STATES:
        path=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';assert digest(path)==locks[state]
        a=np.load(path);data={}
        for split in ['source_in','source_out']:
            p=token/f'{split}_tokens.npz';assert digest(p)==next(r['cache_sha256'] for r in audit['rows'] if r['split']==split)
            b=np.load(p);assert np.array_equal(a[split+'_ids'],b['ids']) and np.array_equal(a[split+'_y'],b['y'])
            data[split]=(b['global_logits'].astype(float),a[split+'_bank_delta'].astype(float),b['y'],b['domains'],b['ids'])
        gi,ti,yi,di,ii=data['source_in'];g,t,y,d,ids=data['source_out'];assert not set(ii)&set(ids)
        fits=[]
        for held in [None,0,2,3]:
            train=np.ones(len(yi),bool) if held is None else di!=held
            val=np.ones(len(y),bool) if held is None else d==held
            for mode in ['uniform','selective']:
                f=fit(gi[train],ti[train],yi[train],di[train],mode)
                fits.append(dict(held_calibration_domain=held,mode=mode,fit=f,source_out=scores(g[val],t[val],y[val],d[val],f['weights'],ids[val])))
        caps={s:capacity(*data[s],a[s+'_bank_full']) for s in data}
        results[state]=dict(fits=fits,capacity=caps)
        print(state,[(f['mode'],f['source_out']['equal_domain_accuracy'],f['source_out']['repairs'],f['source_out']['harms']) for f in fits[:2]],'capacity',[(s,c['repairable_global_errors'],c['bird_errors_repairable']) for s,c in caps.items()],flush=True)
        assert digest(path)==locks[state]
    write(OUT/'results.json',dict(plan=plan,states=results,cache_sha256=locks,source_only=True,target_read=False,
                                  bank_updates=False,converged_fits=32,margin_LP_replay_verified=True))

if __name__=='__main__':run()

"""Source-in calibrated contraction and classifier ensembles for expected utility."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from .routed_discovery_run import DEST,ROOT
from .shrinkage import digest,write
from .label_expected_utility_helpers import predict_label,expected_utility,label_metrics
from .teacher_distribution_helpers import teacher
from .source_composition_calibration import fit,scores
OUT=DEST/'source_expected_utility_controls'

def prob_fit(g,p,y,d):
    q=softmax(g,axis=1);a=q[np.arange(len(y)),y];b=p[np.arange(len(y)),y]-a
    w=np.array([1/(len(set(d))*sum(d==x)) for x in d])
    def calc(x):
        den=(a+x[0]*b).clip(1e-300)
        return float(-w@np.log(den)),np.array([-w@(b/den)])
    result=minimize(calc,[.5],jac=True,method='L-BFGS-B',bounds=[(0,1)],options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
    loss,grad=calc(result.x);pg=np.abs(result.x-np.clip(result.x-grad,0,1)).max()
    assert result.success and pg<1e-6,(result.message,pg)
    return dict(alpha=float(result.x[0]),training_CE=loss,projected_gradient_max=float(pg))

def scalar_fit(g,t,y,d,mode='uniform'):
    f=fit(g,t,y,d,mode);return f

def run():
    OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists()
    token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis'
    tokenproof=json.loads((token/'extraction_audit.json').read_text());source={};locks={}
    for split in ['source_in','source_out']:
        p=token/f'{split}_tokens.npz';locks[str(p)]=digest(p)
        assert digest(p)==next(x['cache_sha256'] for x in tokenproof['rows'] if x['split']==split)
        source[split]=dict(np.load(p))
    plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,checkpoint_selection_changed=False,
              cohorts=['original','H2','joint'],inputs=['CLS_plus_patch_max','global_and_expert_logits_and_pi'],
              controls=['native_uniform_contraction','native_six_expert_suppression','global_posterior_probability_ensemble','global_posterior_geometric_ensemble','expected_utility_contraction','source_in_RMS_matched_native_contraction'],
              fitting='Every free scalar/per-expert scale bounded [0,1]; minimize equal-domain original source-in CE, no validation tuning or grid. Held calibration-domain controls fit scalars on the other two domains using already excluded posterior fits. Probability ensemble is (1-alpha)softmax(global)+alpha*posterior; geometric ensemble interpolates global log probabilities and posterior log probabilities.',
              limits='Native/scalar controls repeat across posterior cohorts for matched rows. RMS matching uses source-in correction RMS only, clipped at1 to forbid native amplification. Banks and centering saw all source domains. This is not a trained SIRM comparison.',code_sha256=digest(Path(__file__)))
    write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());rows=[]
    for folder in ['source_label_expected_utility','source_label_expected_utility_holds']:
        parent=DEST/folder;pr=parent/'results.json';r=json.loads(pr.read_text());locks[str(pr)]=digest(pr)
        for state,readouts in r['states'].items():
            cache=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';bank=dict(np.load(cache));locks[str(cache)]=digest(cache)
            data={}
            for split,b in source.items():
                assert np.array_equal(b['ids'],bank[split+'_ids'])
                g=b['global_logits'].astype(float);delta=bank[split+'_bank_delta'].astype(float);pi=bank[split+'_bank_pi'].astype(float)
                data[split]=dict(g=g,delta=delta,pi=pi,y=b['y'],dom=b['domains'],ids=b['ids'],
                                X={'CLS_plus_patch_max':np.concatenate([b['cls'],b['patch_max']],1).astype(float),
                                   'global_and_expert_logits_and_pi':np.concatenate([g-g.mean(1,keepdims=True),delta.reshape(len(g),-1),pi],1)})
            for readout in readouts:
                held=readout.get('held_calibration_domain');file=parent/readout['fit_file'];assert digest(file)==readout['fit_sha256'];locks[str(file)]=digest(file)
                model=dict(np.load(file));parts={}
                for split,b in data.items():
                    keep=np.ones(len(b['y']),bool) if held is None else (b['dom']!=held if split=='source_in' else b['dom']==held)
                    post=predict_label(model,b['X'][readout['input']][keep]);g=b['g'][keep];delta=b['delta'][keep]
                    epi=teacher(expected_utility(g,delta,post));native=delta*b['pi'][keep,:6,None];expected=delta*epi[:,:6,None]
                    parts[split]=dict(g=g,t=native,et=expected,posterior=post,y=b['y'][keep],d=b['dom'][keep],ids=b['ids'][keep])
                a=parts['source_in'];v=parts['source_out'];gi,yi,di=a['g'],a['y'],a['d'];g,y,d,ids=v['g'],v['y'],v['d'],v['ids']
                record=dict(state=state,input=readout['input'],cohort=readout['cohort'],held_calibration_domain=held,posterior_sha256=digest(file),posterior_file=str(file.relative_to(DEST)),controls={})
                def score(t,w):return scores(g,t,y,d,w,ids)
                record['global_only']=score(v['t'],np.zeros(6));record['native']=score(v['t'],np.ones(6));record['expected_utility']=score(v['et'],np.ones(6))
                assert np.isclose(record['expected_utility']['equal_domain_accuracy'],readout['source_out_original']['equal_domain_accuracy'])
                for name,t,vt,mode in [('native_uniform_contraction',a['t'],v['t'],'uniform'),('native_six_expert_suppression',a['t'],v['t'],'selective'),('expected_utility_contraction',a['et'],v['et'],'uniform')]:
                    fitted=scalar_fit(gi,t,yi,di,mode);record['controls'][name]=dict(fit=fitted,source_out=score(vt,fitted['weights']))
                pf=prob_fit(gi,a['posterior'],yi,di);prob=(1-pf['alpha'])*softmax(g,axis=1)+pf['alpha']*v['posterior'];record['controls']['global_posterior_probability_ensemble']=dict(fit=pf,source_out=label_metrics(prob,y,d,ids,g))
                # g + alpha*(log posterior - g) differs from geometric interpolation only by row-constant offsets.
                trainmix=np.log(a['posterior'].clip(1e-300))-gi;valmix=np.log(v['posterior'].clip(1e-300))-g
                tm=np.repeat(trainmix[:,None,:]/6,6,axis=1);vm=np.repeat(valmix[:,None,:]/6,6,axis=1)
                gf=scalar_fit(gi,tm,yi,di);record['controls']['global_posterior_geometric_ensemble']=dict(fit=gf,source_out=score(vm,gf['weights']))
                native_rms=float(np.sqrt(np.mean(a['t'].sum(1)**2)));expected_rms=float(np.sqrt(np.mean(a['et'].sum(1)**2)))
                alpha=min(1.,expected_rms/max(native_rms,1e-30));record['controls']['source_in_RMS_matched_native_contraction']=dict(alpha=alpha,source_in_native_RMS=native_rms,source_in_expected_RMS=expected_rms,clipped=expected_rms>native_rms,source_out=score(v['t'],np.repeat(alpha,6)))
                rows.append(record)
                if held is None:print(state,readout['cohort'],readout['input'],'expected',record['expected_utility']['repairs'],record['expected_utility']['harms'],'ensembles',[(k,c['source_out']['repairs'],c['source_out']['harms']) for k,c in record['controls'].items() if 'ensemble' in k],flush=True)
    assert len(rows)==60
    for path,sha in locks.items():assert digest(Path(path))==sha
    write(OUT/'results.json',dict(plan=plan,rows=rows,readouts=len(rows),scalar_fits=5*len(rows),source_only=True,target_read=False,bank_updates=False,inputs_sha256=locks))
if __name__=='__main__':run()

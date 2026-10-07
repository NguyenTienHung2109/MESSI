"""Case-level soft posterior controls on source-selected frozen banks only."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
from .label_expected_utility_helpers import predict_label,expected_utility
from .teacher_distribution_helpers import teacher
from .source_composition_calibration import scores
OUT=DEST/'source_expected_utility_controls'

def margins(logits,y):
    competitors=logits.copy();competitors[np.arange(len(y)),y]=-np.inf
    return logits[np.arange(len(y)),y]-competitors.max(1)

def run():
    p=OUT/'results.json';controls=json.loads(p.read_text());parent=DEST/'source_label_expected_utility';r=json.loads((parent/'results.json').read_text())
    token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/source_out_tokens.npz';s=dict(np.load(token));records=[];aggregate=[]
    for state,readouts in r['states'].items():
        if not state.endswith('selected'):continue
        cache=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';b=dict(np.load(cache));assert np.array_equal(s['ids'],b['source_out_ids'])
        g=s['global_logits'].astype(float);delta=b['source_out_bank_delta'].astype(float);oldpi=b['source_out_bank_pi'].astype(float);y=s['y'];d=s['domains'];ids=s['ids'];native=g+np.einsum('nm,nmj->nj',oldpi[:,:6],delta)
        x={'CLS_plus_patch_max':np.concatenate([s['cls'],s['patch_max']],1).astype(float),'global_and_expert_logits_and_pi':np.concatenate([g-g.mean(1,keepdims=True),delta.reshape(len(g),-1),oldpi],1)}
        for fit in readouts:
            file=parent/fit['fit_file'];assert digest(file)==fit['fit_sha256'];posterior=predict_label(dict(np.load(file)),x[fit['input']]);eu=expected_utility(g,delta,posterior);pi=teacher(eu);soft=g+np.einsum('nm,nmj->nj',pi[:,:6],delta)
            hardpi=teacher(expected_utility(g,delta,np.eye(10)[posterior.argmax(1)]));hard=g+np.einsum('nm,nmj->nj',hardpi[:,:6],delta)
            exactpi=teacher(expected_utility(g,delta,np.eye(10)[y]));exact=g+np.einsum('nm,nmj->nj',exactpi[:,:6],delta)
            metrics={name:scores(g,delta*route[:,:6,None],y,d,np.ones(6),ids) for name,route in [('soft',pi),('hard_argmax',hardpi),('true_label_nondeployable',exactpi)]}
            assert metrics['soft']['repair_ids']==fit['source_out_original']['repair_ids'];assert metrics['soft']['harm_ids']==fit['source_out_original']['harm_ids']
            c=next(z for z in controls['rows'] if z['state']==state and z['input']==fit['input'] and z['cohort']==fit['cohort'] and z['held_calibration_domain'] is None)
            alpha=c['controls']['global_posterior_probability_ensemble']['fit']['alpha'];ens=(1-alpha)*softmax(g,axis=1)+alpha*posterior
            gm,nm,sm,hm,em=map(lambda a:margins(a,y),[g,native,soft,hard,exact])
            aggregate.append(dict(state=state,input=fit['input'],cohort=fit['cohort'],metrics=metrics,soft_repaired_but_posterior_wrong=ids[(soft.argmax(1)==y)&(g.argmax(1)!=y)&(posterior.argmax(1)!=y)].tolist(),soft_repaired_but_ensemble_wrong=ids[(soft.argmax(1)==y)&(g.argmax(1)!=y)&(ens.argmax(1)!=y)].tolist()))
            for i,image in enumerate(ids):
                comp=int(np.argmax(np.where(np.arange(10)==y[i],-np.inf,g[i])));gap=delta[i,:,y[i]]-delta[i,:,comp]
                records.append(dict(state=state,input=fit['input'],cohort=fit['cohort'],image_id=str(image),domain=int(d[i]),label=int(y[i]),global_prediction=int(g[i].argmax()),native_prediction=int(native[i].argmax()),soft_prediction=int(soft[i].argmax()),hard_prediction=int(hard[i].argmax()),posterior_prediction=int(posterior[i].argmax()),ensemble_prediction=int(ens[i].argmax()),global_margin=float(gm[i]),native_margin=float(nm[i]),soft_margin=float(sm[i]),hard_margin=float(hm[i]),exact_margin=float(em[i]),posterior_true_probability=float(posterior[i,y[i]]),posterior_probabilities=posterior[i].tolist(),expected_utilities=eu[i].tolist(),native_pi=oldpi[i].tolist(),soft_pi=pi[i].tolist(),hard_pi=hardpi[i].tolist(),global_competitor=comp,native_fixed_competitor_contributions=(oldpi[i,:6]*gap).tolist(),soft_fixed_competitor_contributions=(pi[i,:6]*gap).tolist(),source_global_wrong=bool(gm[i]<0),soft_repair=bool(gm[i]<0 and sm[i]>0),soft_harm=bool(gm[i]>0 and sm[i]<0)))
    write(OUT/'case_audit.json',dict(source_only=True,target_read=False,fitting=False,rows=records,aggregates=aggregate,limits='True-label routing is diagnostic only; hard posterior argmax is a fixed ablation, no fit selection. Fixed-global-competitor contributions need not equal actual final margin after competitor changes.',input_results_sha256=digest(p)))
if __name__=='__main__':run()

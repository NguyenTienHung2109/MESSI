"""Locked posterior agreement/uncertainty and conservative utility readouts."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
from .label_expected_utility_helpers import predict_label,expected_utility,label_metrics
from .teacher_distribution_helpers import teacher,gates_pi
from .source_residual_rank_diagnosis import summarize
from .source_expected_utility_controls import prob_fit,scalar_fit
from .source_composition_calibration import scores
OUT=DEST/'source_posterior_agreement_diagnosis'

def calibration(p,y):
 confidence=p.max(1);correct=p.argmax(1)==y;bins=[];ece=0
 for b in range(10):
  keep=(confidence>=b/10)&((confidence<(b+1)/10) if b<9 else confidence<=1)
  n=int(keep.sum())
  if n:
   accuracy=float(correct[keep].mean());conf=float(confidence[keep].mean());ece+=n/len(y)*abs(accuracy-conf);bins.append(dict(bin=b,n=n,accuracy=accuracy,confidence=conf))
 return dict(bins=bins,image_weighted_ECE=ece,image_weighted_Brier=float(np.mean(np.sum((p-np.eye(10)[y])**2,1))))

def suppress(pi,mask):
 out=pi.copy();out[:,:6]*=mask;out[:,-1]=1-out[:,:6].sum(1);assert np.allclose(out.sum(1),1) and out.min()>-1e-12;return out

def run():
 OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();locks={}
 source={};proof=json.loads((ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/extraction_audit.json').read_text())
 for split in ['source_in','source_out']:
  path=ROOT/f'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/{split}_tokens.npz';assert digest(path)==next(r['cache_sha256'] for r in proof['rows'] if r['split']==split);source[split]=dict(np.load(path));locks[str(path)]=digest(path)
 plan=dict(source_only=True,target_read=False,primary_training=False,bank_updates=False,posterior_refitting=False,checkpoint_selection_changed=False,policy_selection=False,policies=['original_teacher','all_argmax_agreement_suppression','minimum_utility_teacher','minimum_utility_original_suppression','mean_posterior_teacher','geometric_posterior_teacher','original_entropy_attenuation'],rules='Three locked original/H2/joint posterior fits at same original image features. Agreement requires all3 argmax equal. Min utility uses worst expected utility per expert across3 posteriors, unchanged .1/.002 teacher. Expert suppression requires all3 utility>.002; removed mass goes NULL. Entropy attenuation is 1-H(original)/log10, without fitted cutoff.',controls='Plain posterior classifiers; arithmetic/geometric posterior classifier; bounded[0,1] equal-domain source-in CE probability ensemble and uniform residual contraction. Excluded calibration-domain readouts fit scalars only on remaining source-in domains.',limits='Posteriors trained on source-in; training confidence optimistic. Holds exclude calibration fitting only, bank/centering still saw allsource domains. Source-out already inspected. Readouts are not new trained methods or proof of SIRM.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());rows=[];cases=[]
 for state in ['unscaled_OFF_selected','routed_OFF_selected']:
  cache=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';bank=dict(np.load(cache));rank=json.loads((DEST/'source_residual_rank_diagnosis/results.json').read_text());assert digest(cache)==rank['input_sha256'][str(cache)];locks[str(cache)]=digest(cache)
  data={}
  for split,b in source.items():
   assert np.array_equal(b['ids'],bank[split+'_ids']) and np.array_equal(b['y'],bank[split+'_y'])
   g=b['global_logits'].astype(float);d=bank[split+'_bank_delta'].astype(float);pi=bank[split+'_bank_pi'].astype(float)
   data[split]=dict(g=g,delta=d,native=pi,y=b['y'],dom=b['domains'],ids=b['ids'],X={'CLS_plus_patch_max':np.concatenate([b['cls'],b['patch_max']],1).astype(float),'global_and_expert_logits_and_pi':np.concatenate([g-g.mean(1,keepdims=True),d.reshape(len(g),-1),pi],1)})
  for held in [None,0,2,3]:
   parent=DEST/('source_label_expected_utility' if held is None else 'source_label_expected_utility_holds');result=parent/'results.json';fits=json.loads(result.read_text());locks[str(result)]=digest(result)
   for kind in ['CLS_plus_patch_max','global_and_expert_logits_and_pi']:
    models=[];modelsha=[]
    for cohort in ['original','H2','joint']:
     rec=next(r for r in fits['states'][state] if r['input']==kind and r['cohort']==cohort and r.get('held_calibration_domain')==held);file=parent/rec['fit_file'];assert digest(file)==rec['fit_sha256'];models.append(dict(np.load(file)));modelsha.append(digest(file));locks[str(file)]=digest(file)
    parts={}
    for split,whole in data.items():
     mask=np.ones(len(whole['y']),bool) if held is None else (whole['dom']!=held if split=='source_in' else whole['dom']==held)
     b={k:v[mask] for k,v in whole.items() if k!='X'};post=np.stack([predict_label(m,whole['X'][kind][mask]) for m in models]);utilities=np.stack([expected_utility(b['g'],b['delta'],p) for p in post]);original=teacher(utilities[0]);agree=np.all(post.argmax(2)==post.argmax(2)[0],axis=0);mean=post.mean(0);geo=softmax(np.log(post.clip(1e-300)).mean(0),axis=1);entropy=-np.sum(post[0]*np.log(post[0].clip(1e-300)),1);js=np.mean(np.sum(post*(np.log(post.clip(1e-300))-np.log(mean.clip(1e-300))[None]),2),0)
     routes={'original_teacher':original,'all_argmax_agreement_suppression':suppress(original,agree[:,None]),'minimum_utility_teacher':teacher(utilities.min(0)),'minimum_utility_original_suppression':suppress(original,(utilities>.002).all(0)),'mean_posterior_teacher':teacher(expected_utility(b['g'],b['delta'],mean)),'geometric_posterior_teacher':teacher(expected_utility(b['g'],b['delta'],geo)),'original_entropy_attenuation':suppress(original,(1-entropy/np.log(10))[:,None])}
     b.update(routes=routes,post=post,mean=mean,geo=geo,agreement=agree,entropy=entropy,JS=js);parts[split]=b
     for policy,pi in routes.items():
      full=b['g']+np.einsum('nm,nmc->nc',pi[:,:6],b['delta']);correct=full.argmax(1)==b['y'];globalcorrect=b['g'].argmax(1)==b['y'];metrics=summarize(b['g'],b['delta'],pi,b['y'],b['dom'],b['ids'])
      record=dict(state=state,held_calibration_domain=held,input=kind,split=split,policy=policy,metrics=metrics,all_argmax_agreement_fraction=float(agree.mean()),Q=gates_pi(b['g'],b['delta'],b['y'],b['dom'],pi) if held is None else None)
      rows.append(record)
      for i in range(len(full)):
       cases.append(dict(state=state,held_calibration_domain=held,input=kind,split=split,policy=policy,image_id=str(b['ids'][i]),domain=int(b['dom'][i]),label=int(b['y'][i]),global_correct=bool(globalcorrect[i]),full_correct=bool(correct[i]),full_prediction=int(full[i].argmax()),all_argmax_agree=bool(agree[i]),posterior_argmax=post[:,i].argmax(1).tolist(),original_confidence=float(post[0,i].max()),entropy=float(entropy[i]),JS=float(js[i]),NULL=float(pi[i,-1]),expert_mass=pi[i,:6].tolist()))
    train=parts['source_in'];val=parts['source_out'];controls={}
    for name,p,pv in [('original',train['post'][0],val['post'][0]),('H2',train['post'][1],val['post'][1]),('joint',train['post'][2],val['post'][2]),('mean',train['mean'],val['mean']),('geometric',train['geo'],val['geo'])]:
     fit=prob_fit(train['g'],p,train['y'],train['dom']);prob=(1-fit['alpha'])*softmax(val['g'],1)+fit['alpha']*pv
     controls[name]=dict(source_in_calibration=calibration(p,train['y']),source_out_calibration=calibration(pv,val['y']),plain_source_out=label_metrics(pv,val['y'],val['dom'],val['ids'],val['g']),ensemble_fit=fit,ensemble_source_out=label_metrics(prob,val['y'],val['dom'],val['ids'],val['g']))
    native=scalar_fit(train['g'],train['delta']*train['native'][:,:6,None],train['y'],train['dom']);controls['native_contraction']=dict(fit=native,source_out=scores(val['g'],val['delta']*val['native'][:,:6,None],val['y'],val['dom'],native['weights'],val['ids']))
    rows.append(dict(state=state,held_calibration_domain=held,input=kind,controls=controls))
    print(state,held,kind,'complete',flush=True)
 for path,h in locks.items():assert digest(Path(path))==h
 assert len(rows)==16*15 and len(cases)==4*(3*1172+2*304)*7
 write(OUT/'results.json',dict(plan=plan,rows=rows,case_file='cases.json',case_rows=len(cases),source_only=True,target_read=False,bank_updates=False,input_sha256=locks));write(OUT/'cases.json',cases)
if __name__=='__main__':run()

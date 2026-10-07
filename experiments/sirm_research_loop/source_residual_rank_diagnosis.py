"""Source-fitted output rank controls for frozen residual experts."""
import json
from pathlib import Path
import numpy as np
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
from .label_expected_utility_helpers import predict_label,expected_utility
from .teacher_distribution_helpers import teacher,gates_pi
from .source_composition_calibration import scores
OUT=DEST/'source_residual_rank_diagnosis'

def summarize(g,delta,pi,y,dom,ids):
 result=scores(g,delta*pi[:,:6,None],y,dom,np.ones(6),ids)
 weights=np.array([1/(len(set(dom))*sum(dom==d)) for d in dom])
 result.update(equal_domain_residual_RMS=float(np.sqrt(np.einsum('n,nmc->',weights,delta**2)/(6*10))),equal_domain_correction_RMS=float(np.sqrt(np.einsum('n,nc->',weights,np.einsum('nm,nmc->nc',pi[:,:6],delta)**2)/10)),NULL=float(np.mean(pi[:,-1])))
 return result

def run():
 OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists()
 parent=DEST/'source_label_expected_utility';pr=json.loads((parent/'results.json').read_text());originalfits=json.loads((DEST/'source_conditional_accessibility/results.json').read_text());token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';proof=json.loads((token/'extraction_audit.json').read_text());source={};locks={}
 for split in ['source_in','source_out']:
  p=token/f'{split}_tokens.npz';assert digest(p)==next(x['cache_sha256'] for x in proof['rows'] if x['split']==split);locks[str(p)]=digest(p);source[split]=dict(np.load(p))
 plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,basis_fitting=True,states=[s for s in pr['states'] if s.endswith('selected')],variants=['full','rank1','rank1_source_in_RMS_matched'],basis='Per-expert SVD of raw class-centered residual outputs with equal-source-domain source-in weights; non-centered second moment, no label information in fit. Sign by largest absolute class loading. No rank chosen by validation.',routing=['native_fixed','original/H2/joint feature posterior fixed full-bank routing','same posterior recomputed utility routing','true_label_nondeployable'],class_domain='Descriptive response groups minimum2 images, no fitted class probe.',limits='Source-out already selected and inspected. Rank1 truncation of an already trained bank is not rank1 training or proof of a semantic/causal factor. RMS matching is a source-in amplitude control and can amplify retained mode; gates unchanged.',code_sha256=digest(Path(__file__)),posterior_results_sha256=digest(parent/'results.json'))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());states={}
 for state in plan['states']:
  path=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';assert digest(path)==originalfits['cache_sha256'][state];locks[str(path)]=digest(path);bank=dict(np.load(path));data={}
  for split,b in source.items():
   assert np.array_equal(b['ids'],bank[split+'_ids']) and np.array_equal(b['y'],bank[split+'_y'])
   data[split]=dict(g=b['global_logits'].astype(float),delta=bank[split+'_bank_delta'].astype(float),native=bank[split+'_bank_pi'].astype(float),y=b['y'],dom=b['domains'],ids=b['ids'],x=np.concatenate([b['cls'],b['patch_max']],1).astype(float))
  a=data['source_in'];weights=np.array([1/(3*sum(a['dom']==d)) for d in a['dom']]);axes=[];spectrum=[];responses=[]
  for expert in range(6):
   _,singular,vh=np.linalg.svd(a['delta'][:,expert]*np.sqrt(weights[:,None]),full_matrices=False);v=vh[0];v*=1 if v[np.abs(v).argmax()]>0 else -1;axes.append(v);energy=singular**2;assert np.allclose(vh@vh.T,np.eye(10),atol=1e-10);assert np.allclose(a['delta'][:,expert]@vh.T@vh,a['delta'][:,expert],atol=1e-10)
   spectrum.append(dict(expert=expert,weighted_singular_values=singular.tolist(),energy_fraction=(energy/energy.sum()).tolist(),rank1_energy_fraction=float(energy[0]/energy.sum()),participation_rank=float(energy.sum()**2/(energy**2).sum()),rank_for_90pct=int(np.searchsorted(np.cumsum(energy)/energy.sum(),.9)+1),leading_class_loadings=v.tolist(),source_in_RMS_rescale=float(np.sqrt(energy.sum()/energy[0]))))
   for split,b in data.items():
    for domain in [0,2,3]:
     mask=b['dom']==domain;_,sv,dvh=np.linalg.svd(b['delta'][mask,expert],full_matrices=False);cos=float(abs(dvh[0]@v)) if sv[0]>1e-12 else None
     for label in sorted(set(b['y'][mask])):
      cm=mask&(b['y']==label)
      if cm.sum()<2:continue
      response=b['delta'][cm,expert].mean(0);responses.append(dict(split=split,domain=domain,label=int(label),count=int(cm.sum()),expert=expert,mean_logit_response=response.tolist(),mean_rank1_score=float(response@v),rank1_reconstruction_error_RMS=float(np.sqrt(np.mean((b['delta'][cm,expert]-np.outer(b['delta'][cm,expert]@v,v))**2))),domain_leading_axis_abs_cosine=cos))
  axes=np.stack(axes);basisfile=OUT/f'{state}_source_basis.npz';np.savez_compressed(basisfile,axes=axes,rescales=np.array([s['source_in_RMS_rescale'] for s in spectrum]));basislock=digest(basisfile)
  posterior={}
  for rec in pr['states'][state]:
   if rec['input']!='CLS_plus_patch_max':continue
   file=parent/rec['fit_file'];assert digest(file)==rec['fit_sha256'];locks[str(file)]=digest(file);model=dict(np.load(file));posterior[rec['cohort']]={split:predict_label(model,b['x']) for split,b in data.items()}
  rows=[]
  for variant in plan['variants']:
   for split,b in data.items():
    original=b['delta'];score=np.einsum('nmc,mc->nm',original,axes);projected=score[:,:,None]*axes[None]
    delta=original if variant=='full' else projected if variant=='rank1' else projected*np.array([s['source_in_RMS_rescale'] for s in spectrum])[None,:,None]
    if split=='source_in' and variant=='rank1_source_in_RMS_matched':assert np.allclose(np.einsum('n,nmc->m',weights,delta**2),np.einsum('n,nmc->m',weights,original**2),rtol=1e-9)
    routes={'native_fixed':b['native'],'true_label_nondeployable':teacher(expected_utility(b['g'],delta,np.eye(10)[b['y']]))}
    for cohort,posts in posterior.items():
     p=posts[split];routes[cohort+'_posterior_fixed']=teacher(expected_utility(b['g'],original,p));routes[cohort+'_posterior_recomputed']=teacher(expected_utility(b['g'],delta,p))
    rawpred=(b['g'][:,None]+delta).argmax(2);capacity=int(((b['g'].argmax(1)!=b['y'])&(rawpred==b['y'][:,None]).any(1)).sum())
    for name,pi in routes.items():
     record=dict(variant=variant,split=split,routing=name,metrics=summarize(b['g'],delta,pi,b['y'],b['dom'],b['ids']),raw_standalone_error_repair_capacity=capacity,Q=gates_pi(b['g'],delta,b['y'],b['dom'],pi))
     if variant=='full' and name.endswith('_posterior_recomputed'):
      cohort=name.split('_posterior')[0];ref=next(x for x in pr['states'][state] if x['cohort']==cohort and x['input']=='CLS_plus_patch_max')[split+'_original'];assert record['metrics']['repair_ids']==ref['repair_ids'] and record['metrics']['harm_ids']==ref['harm_ids']
     rows.append(record)
  assert len(rows)==48 and digest(basisfile)==basislock
  states[state]=dict(spectrum=spectrum,class_domain_responses=responses,source_in_basis_file=basisfile.name,source_in_basis_sha256=basislock,rows=rows)
  print(state,'rank1 energy',[round(x['rank1_energy_fraction'],4) for x in spectrum],flush=True)
 for path,h in locks.items():assert digest(Path(path))==h
 write(OUT/'results.json',dict(plan=plan,states=states,input_sha256=locks,source_only=True,target_read=False,bank_updates=False,rank_selected_by_validation=False))
if __name__=='__main__':run()

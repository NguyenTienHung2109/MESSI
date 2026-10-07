"""Source-only correction-risk guard from leave-location-out posterior cases."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write
from .label_expected_utility_helpers import predict_label,expected_utility,fit_label
from .teacher_distribution_helpers import teacher,gates_pi
from .source_residual_rank_diagnosis import summarize
OUT=DEST/'source_crossfit_correction_guard'

def features(g,d,post):
 pi=teacher(expected_utility(g,d,post));corr=np.einsum('nm,nmc->nc',pi[:,:6],d);f=g+corr;old=g.argmax(1);new=f.argmax(1);ix=np.arange(len(g));gs=np.sort(g,1);fs=np.sort(f,1);prob=softmax(g,1)
 # Fixed eight scalar features; same-old/new cases receive no guard intervention.
 x=np.stack([gs[:,-1]-gs[:,-2],fs[:,-1]-fs[:,-2],g[ix,new]-g[ix,old],corr[ix,new]-corr[ix,old],np.log(post[ix,new].clip(1e-12))-np.log(post[ix,old].clip(1e-12)),-(post*np.log(post.clip(1e-12))).sum(1),-(prob*np.log(prob.clip(1e-12))).sum(1),np.sqrt((corr**2).mean(1))],1)
 return x,pi,old,new

def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists();rank=json.loads((DEST/'source_residual_rank_diagnosis/results.json').read_text());source={};locks={}
 for split in ['source_in','source_out']:
  path=ROOT/f'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis/{split}_tokens.npz';assert digest(path)==rank['input_sha256'][str(path)];source[split]=dict(np.load(path));locks[str(path)]=digest(path)
 plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,hypothesis='Auxiliary in-sample composition has no harmful changes; leave-location-out posterior predictions expose risks that a fixed low-dimensional class-contrast guard might distinguish.',fit='Binary softmax L2=.01 on source-in cross-fitted changed predictions where exactly one of global/composition is correct. Equal present-domain CE weights. Minimum2 cases per present domain/class cell and both classes; otherwise skip, never synthesize labels. Features fixed eight global/composition margins, contrasts, posterior contrast, two entropies and correction RMS. LBFGS500 gradient<1e-5.',policies=['original','hard_guard_0.5','soft_guard'],calibration_holds=[None,0,2,3],limits='Posterior cross-fitting excludes probe fitting only, not prior bank/backbone learning. Full guard evaluation uses full posterior; held evaluation excludes the evaluated domain from guard training and posterior fitting. Source-out already inspected; diagnostic guard fitting is not new400-step SIRM or target improvement.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());rows=[];case_rows=[];coverage=[]
 for state in rank['states']:
  path=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';assert digest(path)==rank['input_sha256'][str(path)];bank=dict(np.load(path));locks[str(path)]=digest(path)
  for kind in ['CLS_plus_patch_max','global_and_expert_logits_and_pi']:
   def data(split):
    b=source[split];g=b['global_logits'].astype(float);d=bank[split+'_bank_delta'].astype(float);assert np.array_equal(b['ids'],bank[split+'_ids']);x=np.concatenate([b['cls'],b['patch_max']],1).astype(float) if kind=='CLS_plus_patch_max' else np.concatenate([g-g.mean(1,keepdims=True),d.reshape(len(g),-1),bank[split+'_bank_pi']],1);return b,g,d,x
   b,g,d,x=data('source_in');oof=np.zeros_like(g)
   def posterior(held,x):
    parent=DEST/('source_label_expected_utility' if held is None else 'source_label_expected_utility_holds');path=parent/'results.json';result=json.loads(path.read_text());locks[str(path)]=digest(path);rec=next(r for r in result['states'][state] if r['input']==kind and r['cohort']=='original' and r.get('held_calibration_domain')==held);f=parent/rec['fit_file'];assert digest(f)==rec['fit_sha256'];locks[str(f)]=digest(f);return predict_label(dict(np.load(f)),x)
   for domain in [0,2,3]:
    k=b['domains']==domain;oof[k]=posterior(domain,x[k])
   z,pi,old,new=features(g,d,oof);y=b['y'];changed=old!=new;repair=changed&(new==y);harm=changed&(old==y);valid=repair|harm
   for domain in [0,2,3]:
    k=b['domains']==domain;coverage.append(dict(state=state,input=kind,domain=domain,changed=int((changed&k).sum()),repairs=int((repair&k).sum()),harms=int((harm&k).sum()),both_wrong=int((changed&~valid&k).sum())))
   for held in [None,0,2,3]:
    k=valid if held is None else valid&(b['domains']!=held);dom=b['domains'][k];labels=repair[k].astype(int);counts={f'{domain},{label}':int(((dom==domain)&(labels==label)).sum()) for domain in sorted(set(dom)) for label in [0,1] if ((dom==domain)&(labels==label)).any()};rec=dict(state=state,input=kind,held_calibration_domain=held,fit_count=int(k.sum()),fit_cell_counts=counts)
    if len(set(labels))<2 or min(counts.values(),default=0)<2:
     rec.update(skipped=True,reason='Insufficient minimum2 risk-class coverage or missing one risk class.');rows.append(rec);continue
    model=fit_label(z[k],np.eye(2)[labels],dom);f=OUT/f'{state}_{kind}_held{held}.npz';np.savez_compressed(f,**model);rec.update(skipped=False,fit_file=f.name,fit_sha256=digest(f),gradient_max=model['gradient_max']);vp,vg,vd,vx=data('source_out');mask=np.ones(len(vg),bool) if held is None else vp['domains']==held;vg=vg[mask];vd=vd[mask];post=posterior(held,vx[mask]);vf,vpi,vo,vn=features(vg,vd,post);prob=predict_label(model,vf)[:,1];keep=vo!=vn;rec['policies']={}
    for name in plan['policies']:
     scale=np.ones(len(vg)) if name=='original' else np.where(keep,(prob>=.5).astype(float) if name=='hard_guard_0.5' else prob,1.);guardpi=vpi.copy();guardpi[:,:6]*=scale[:,None];guardpi[:,-1]=1-guardpi[:,:6].sum(1);assert guardpi.min()>-1e-12 and np.allclose(guardpi.sum(1),1);yy=vp['y'][mask];dd=vp['domains'][mask];ids=vp['ids'][mask];rec['policies'][name]=summarize(vg,vd,guardpi,yy,dd,ids)
     if held is None:rec['policies'][name]['Q']=gates_pi(vg,vd,yy,dd,guardpi)
     full=vg+np.einsum('nm,nmc->nc',guardpi[:,:6],vd);pred=full.argmax(1)
     for i,id in enumerate(ids):
      if keep[i]:case_rows.append(dict(state=state,input=kind,held_calibration_domain=held,policy=name,image_id=str(id),label=int(yy[i]),global_prediction=int(vo[i]),original_prediction=int(vn[i]),guard_prediction=int(pred[i]),guard_repair_probability=float(prob[i]),scale=float(scale[i])))
    rows.append(rec);print(state,kind,held,'fit',int(k.sum()),'source',[round(v['equal_domain_accuracy'],5) for v in rec['policies'].values()],flush=True)
 for path,h in locks.items():assert digest(Path(path))==h
 assert len(rows)==16
 write(OUT/'results.json',dict(plan=plan,rows=rows,crossfit_source_in_coverage=coverage,input_sha256=locks,source_only=True,target_read=False,bank_updates=False));write(OUT/'cases.json',case_rows)
if __name__=='__main__':run()

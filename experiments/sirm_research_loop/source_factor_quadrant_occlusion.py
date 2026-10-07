"""Fixed quadrant mean-fill occlusion intervention on locked source-out images; frozen banks."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from experiments.sirm_h1_h6.run import MEAN,STD
from experiments.terra_10pct_factor_diagnosis.report import data_for
from experiments.sirm_discover.math import center_logits
from .spatial_bank_model import pooled_raw_features
from .routed_conditional_utility import load
from .routed_discovery_run import ROOT,DEST
from .source_residual_rank_diagnosis import summarize
from .label_expected_utility_helpers import predict_label,expected_utility
from .teacher_distribution_helpers import teacher
from .shrinkage import digest,write
OUT=DEST/'source_factor_quadrant_occlusion'

@torch.no_grad()
def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists()
 rankpath=DEST/'source_residual_rank_diagnosis/results.json';rank=json.loads(rankpath.read_text());postroot=DEST/'source_label_expected_utility';posts=json.loads((postroot/'results.json').read_text());token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';source={}
 for split in ['source_in','source_out']:
  p=token/f'{split}_tokens.npz';assert digest(p)==rank['input_sha256'][str(p)];source[split]=dict(np.load(p))
 m,p,path=load(DEST,'selected');manifestpath=DEST/'experiment/subset_manifest.json';manifest=json.loads(manifestpath.read_text());before={k:v.cpu().clone() for k,v in m.featurizer.state_dict().items()};phi=m.featurizer.cuda().eval();assert all(not v.requires_grad for v in phi.parameters());c=p['config'];records=[];raws={str(a):[] for a in ['original','top_left','top_right','bottom_left','bottom_right']};ids=[];ys=[];doms=[];clips={str(a):[] for a in ['original','top_left','top_right','bottom_left','bottom_right']}
 plan=dict(source_only=True,target_read=False,bank_updates=False,primary_training=False,posterior_refitting=False,occlusion_views=['original','top_left','top_right','bottom_left','bottom_right'],pixels='After exact Resize224 replace one predetermined112x112 quadrant with each image original per-channel mean RGB. Original plus4 quadrants;25pct pixels occluded. No selected ROI or fitted mask. Normalize existing MEAN/STD.',extraction='Frozen original final normalized CLS plus patch-max, excludes both prefix tokens from max. Original identity/hash/raw-feature replay required.',readouts='Native full view, global-only view, residual-only fixed-original-global/fixed-original-routing, frozen original feature posterior expected composition. Source-in calibration mean stays at original view.',limits='Occlusion removes both object/background information, shifts image distribution and creates boundaries. It localizes response only, not semantic causality; no quadrant is selected for training. Global and residual counterfactuals separated. Source-out already inspected. No target or training.',code_sha256=digest(Path(__file__)),rank_results_sha256=digest(rankpath),manifest_sha256=digest(manifestpath))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
 mean=torch.tensor(MEAN)[None,:,None,None];std=torch.tensor(STD)[None,:,None,None]
 for domain,ds in sorted(data_for(manifest,[0,2,3],c,'out').items()):
  current=[manifest['sources'][str(domain)]['images'][i] for i in ds.indices]
  for entry in current:assert digest(ROOT/'domainbed/data/terra_incognita'/entry['id'])==entry['sha256']
  offset=0
  for x,y in DataLoader(ds,batch_size=32,shuffle=False,num_workers=0):
   rgb=(x*std+mean).clamp(0,1)
   for factor in ['original','top_left','top_right','bottom_left','bottom_right']:
    scaled=rgb.clone()
    if factor!='original':
     row=0 if factor.startswith('top') else 112;col=0 if factor.endswith('left') else 112;scaled[:,:,row:row+112,col:col+112]=rgb.mean((2,3),keepdim=True)
    clips[str(factor)].extend([0.]*len(x));xx=(scaled.clamp(0,1)-mean)/std;_,raw=pooled_raw_features(phi,xx.cuda());raws[str(factor)].append(raw.cpu().numpy())
   ids.extend([s['id'] for s in current[offset:offset+len(y)]]);ys.extend(y.tolist());doms.extend([domain]*len(y));offset+=len(y)
 assert np.array_equal(ids,source['source_out']['ids']) and np.array_equal(ys,source['source_out']['y']);raws={k:np.concatenate(v) for k,v in raws.items()};reference=np.concatenate([source['source_out']['cls'],source['source_out']['patch_max']],1);error=float(np.max(np.abs(raws['original']-reference)));assert error<1e-4,error
 phi.cpu();assert all(torch.equal(v,m.featurizer.state_dict()[k].cpu()) for k,v in before.items()) and all(v.grad is None for v in m.featurizer.parameters());del m
 cache=OUT/'paired_source_out_raw.npz';np.savez_compressed(cache,**{'raw_'+k:v for k,v in raws.items()},ids=np.array(ids),y=np.array(ys),domains=np.array(doms));cachelock=digest(cache);locks={};y=np.array(ys);dom=np.array(doms);ids=np.array(ids)
 for state,info in rank['states'].items():
  from .routed_discovery_run import PRIOR
  m,p,path=load(PRIOR if state.startswith('unscaled') else DEST,'selected');locks[state]=digest(path);before={k:v.clone() for k,v in m.state_dict().items()};si=source['source_in'];m.calibrate_features(torch.tensor(np.concatenate([si['cls'],si['patch_max']],1)),torch.tensor(si['domains']))
  bankpath=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';assert digest(bankpath)==rank['input_sha256'][str(bankpath)];bank=dict(np.load(bankpath));g0=source['source_out']['global_logits'].astype(float);pi0=bank['source_out_bank_pi'].astype(float);d0=bank['source_out_bank_delta'].astype(float);rec=next(x for x in posts['states'][state] if x['input']=='CLS_plus_patch_max' and x['cohort']=='original');file=postroot/rec['fit_file'];assert digest(file)==rec['fit_sha256'];posterior=dict(np.load(file));post0=predict_label(posterior,reference.astype(float));pp0=teacher(expected_utility(g0,d0,post0));axispath=DEST/'source_residual_rank_diagnosis'/info['source_in_basis_file'];assert digest(axispath)==info['source_in_basis_sha256'];axes=np.load(axispath)['axes'];original_scores=np.einsum('nmc,mc->nm',d0,axes)
  for factor,arr in raws.items():
   raw=torch.tensor(arr);z=m.bank_features(raw);g=m.global_head(raw[:,:384]).numpy().astype(float);delta=(center_logits(torch.stack([e(z) for e in m.residuals],1))-m.expert_source_mean[None]).numpy().astype(float);native=m.router(z).softmax(1).numpy().astype(float);post=predict_label(posterior,arr.astype(float));pi=teacher(expected_utility(g,delta,post));score=np.einsum('nmc,mc->nm',delta,axes)
   natural=summarize(g,delta,native,y,dom,ids);expected=summarize(g,delta,pi,y,dom,ids);held=summarize(g0,delta,pi0,y,dom,ids);heldpost=summarize(g0,delta,pp0,y,dom,ids);globalonly=summarize(g,delta,np.tile([0,0,0,0,0,0,1],(len(y),1)),y,dom,ids)
   if factor=='original':assert np.array_equal(g.argmax(1),g0.argmax(1)) and np.isclose(natural['equal_domain_accuracy'],next(r['metrics']['equal_domain_accuracy'] for r in info['rows'] if r['variant']=='full' and r['split']=='source_out' and r['routing']=='native_fixed'))
   record=dict(state=state,occlusion=factor,mean_clipped_fraction=float(np.mean(clips[factor])),global_only_view=globalonly,native_full_view=natural,original_posterior_full_view=expected,residual_only_global_and_native_pi_held=held,residual_only_global_and_original_posterior_pi_held=heldpost,factor_scores=score.tolist(),factor_score_changes=(score-original_scores).tolist(),ids=ids.tolist(),group_score_changes=[])
   for domain in [0,2,3]:
    for label in sorted(set(y[dom==domain])):
     mask=(dom==domain)&(y==label)
     if mask.sum()<2:continue
     record['group_score_changes'].append(dict(domain=domain,label=int(label),n=int(mask.sum()),mean_score_changes=(score[mask]-original_scores[mask]).mean(0).tolist()))
   records.append(record)
  m.load_state_dict(before);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and all(v.grad is None for v in m.parameters()) and digest(path)==locks[state]
 assert digest(cache)==cachelock
 write(OUT/'results.json',dict(plan=plan,source_only=True,target_read=False,bank_tensors_restored=True,grad_fields_unassigned=True,original_raw_max_error=error,all_image_hashes_verified=True,paired_cache_sha256=cachelock,checkpoint_sha256=locks,rows=records))
if __name__=='__main__':run()

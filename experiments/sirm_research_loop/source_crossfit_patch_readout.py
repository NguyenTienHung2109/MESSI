"""Domain-crossfit conditional patch readers, source only, no primary SIRM change."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from .source_conditional_patch_readout import Head,score
from .source_low_rank_readout import CACHE
from .shrinkage import ROOT,write,digest
DEST=ROOT/'results/sirm_research_loop/iteration_19/source_crossfit_patch_readout'
def ridge_fit(c,p,d,mask):
 mean_c=c[mask].mean(0);std_c=c[mask].std(0,unbiased=False).clamp_min(1e-6);mean_p=p[mask].mean(0);std_p=p[mask].std(0,unbiased=False).clamp_min(1e-6);cc=((c[mask]-mean_c)/std_c).double();pp=((p[mask]-mean_p)/std_p).double();dd=d[mask];domains=dd.unique();weights=torch.zeros(len(dd),dtype=torch.float64)
 for domain in domains:weights[dd.eq(domain)]=1/(len(domains)*int(dd.eq(domain).sum()))
 r=torch.linalg.solve(cc.T@(weights[:,None]*cc)+torch.eye(384,dtype=torch.float64),cc.T@(weights[:,None]*pp)).float();return dict(cmean=mean_c,cstd=std_c,pmean=mean_p,pstd=std_p,ridge=r,fit_domains=domains.tolist())
def predict(c,model):return model['pmean']+((c-model['cmean'])/model['cstd'])@model['ridge']*model['pstd']
def main():
 torch.set_num_threads(4);DEST.mkdir(exist_ok=True);assert not (DEST/'results.json').exists();data={s:dict(np.load(CACHE/f'{s}_tokens.npz')) for s in ['source_in','source_out']};c={s:torch.tensor(v['cls']) for s,v in data.items()};p={s:torch.tensor(v['patch_max']) for s,v in data.items()};g={s:torch.tensor(v['global_logits']) for s,v in data.items()};y={s:torch.tensor(v['y']) for s,v in data.items()};d={s:torch.tensor(v['domains']) for s,v in data.items()}
 plan=dict(source_only=True,target_read=False,primary_SIRM_training=False,previous_goal_turn='progress:32CLS/raw/conditional/sham readout fits and EDA; pooled benefit but held transfer mixed',hypothesis='Fit-domain CLS-to-patch prediction does not generalize; crossfit residual training may reduce resulting harms. Test crossfit conditional versus crossfit sham, using existing self-fit conditional/raw/CLS controls.',fits='2 families xlearned/fixed xpooled/held0/2/3=16 new fits,400updates',transform='For each outer-fit source-in domain, fit ridge using only other outer-fit domains, including inner mean/std. Reconstruct raw predicted patch. Train row uses its exclusion predictor; source-out uses arithmetic mean of inner predictors. Outer held domain excluded from every ridge/head/calibration fit.',sham='Normalize raw predicted patch in outer patch coordinates, rotate384dimensions using same seed0 Q as previous controls; subtraction norm identical before family normalization.',normalization='Outer source-in patch mean/std, then family mean/std on crossfit train rows, floor1e-6/sqrt2. No source-out normalization.',ridge_lambda=1,training='Same zero-input-coefficient source-risk rank3/Adam.001/CE+KLcoef1+.01effectiveW squared, seed0 exact schedules, final399',selection='Report all16 fits and existing comparators, no source-out family/map/scale/checkpoint selection; no target reads',amplitude='Match crossfit conditional/sham and existing self-fit conditional/raw/CLS to common mean source-in correction RMS per outerfit/policy, report all',limits='Auxiliary exclusion is not unseen-domain backbone test. Inner ridge training domain count differs from prior self-fit. Averaging inner regressors differs from pooled-regression deployment. Patch-max not object-localization proof.',input_sha256={str(CACHE/f'{s}_tokens.npz'):digest(CACHE/f'{s}_tokens.npz') for s in data},code_sha256=digest(Path(__file__)))
 write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());rotation=torch.linalg.qr(torch.randn(384,384,generator=torch.Generator().manual_seed(0)))[0];result={}
 for held in [None,0,2,3]:
  fit='pooled' if held is None else f'held{held}';keep=torch.ones(len(y['source_in']),dtype=torch.bool) if held is None else d['source_in'].ne(held);take=torch.ones(len(y['source_out']),dtype=torch.bool) if held is None else d['source_out'].eq(held);domains=d['source_in'][keep].unique();outermean=p['source_in'][keep].mean(0);outerstd=p['source_in'][keep].std(0,unbiased=False).clamp_min(1e-6);regressors={};trainpred=torch.zeros_like(p['source_in']);valpred=[]
  for dd in domains:
   inner=keep&d['source_in'].ne(dd);assert int(inner.sum())>0;model=ridge_fit(c['source_in'],p['source_in'],d['source_in'],inner);assert int(dd) not in model['fit_domains'];assert held is None or held not in model['fit_domains'];regressors[int(dd)]=model;rows=keep&d['source_in'].eq(dd);trainpred[rows]=predict(c['source_in'][rows],model);valpred.append(predict(c['source_out'],model))
  predictions={'source_in':trainpred,'source_out':torch.stack(valpred).mean(0)};pre={name:{} for name in ['crossfit_conditional','crossfit_sham']};z={name:{} for name in pre};normstats={};regression={}
  for split in data:
   patch=(p[split]-outermean)/outerstd;prediction=(predictions[split]-outermean)/outerstd;pre['crossfit_conditional'][split]=patch-prediction;pre['crossfit_sham'][split]=patch-prediction@rotation;mask=keep if split=='source_in' else take;regression[split]=dict(raw_patch_MSE=float(patch[mask].square().mean()),conditional_patch_MSE=float(pre['crossfit_conditional'][split][mask].square().mean()),sham_patch_MSE=float(pre['crossfit_sham'][split][mask].square().mean()),subtraction_norm_max_error=float((prediction[mask].norm(dim=1)-(prediction[mask]@rotation).norm(dim=1)).abs().max()))
  for name in pre:
   mean=pre[name]['source_in'][keep].mean(0);std=pre[name]['source_in'][keep].std(0,unbiased=False).clamp_min(1e-6);normstats[name]=(mean,std);z[name]={s:(v-mean)/std/2**.5 for s,v in pre[name].items()}
  idx=torch.nonzero(keep).flatten();gen=torch.Generator().manual_seed(0);schedule=[torch.cat([ii[torch.randint(len(ii),(32,),generator=gen)] for dd in domains for ii in [idx[d['source_in'][idx].eq(dd)]]]) for _ in range(400)];schedule_hash=hashlib.sha256(torch.stack(schedule).numpy().tobytes()).hexdigest();prior=json.loads((DEST.parent/'source_conditional_patch_readout/results.json').read_text())['fits'][fit];assert schedule_hash==prior['schedule_sha256'];out={'regression':regression,'families':{},'global':score(g['source_out'][take],g['source_out'][take],y['source_out'][take],d['source_out'][take]),'schedule_sha256':schedule_hash};dvsave={}
  for name in pre:
   desired=F.one_hot(y['source_in'],10).float()-g['source_in'].softmax(1);moment=torch.stack([desired[keep&d['source_in'].eq(dd)].T@z[name]['source_in'][keep&d['source_in'].eq(dd)]/int((keep&d['source_in'].eq(dd)).sum()) for dd in domains]).mean(0);u,sv,vh=torch.linalg.svd(moment,full_matrices=False);b=torch.linalg.qr(u[:,:3]-u[:,:3].mean(0,keepdim=True))[0]
   for j in range(3):
    if b[b[:,j].abs().argmax(),j]<0:b[:,j]*=-1
   out['families'][name]={}
   for policy in ['learned','fixed']:
    h=Head(b,policy=='learned');assert h(z[name]['source_in'][:2]).count_nonzero()==0;opt=torch.optim.Adam([p for p in h.parameters() if p.requires_grad],lr=.001)
    for ix in schedule:
     f=g['source_in'][ix]+h(z[name]['source_in'][ix]);loss=F.cross_entropy(f,y['source_in'][ix])+F.kl_div(F.log_softmax(f,1),F.softmax(g['source_in'][ix],1),reduction='batchmean')+.01*h.weight().square().sum();assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
     dt=h(z[name]['source_in'][keep]);dv=h(z[name]['source_out'][take]);record=dict(source_in=score(g['source_in'][keep]+dt,g['source_in'][keep],y['source_in'][keep],d['source_in'][keep]),source_out=score(g['source_out'][take]+dv,g['source_out'][take],y['source_out'][take],d['source_out'][take]),source_in_RMS=float(dt.square().mean().sqrt()),risk_gradient_top3_energy=float(sv[:3].square().sum()/sv.square().sum()));out['families'][name][policy]=record;dvsave[name,policy]=dv
    torch.save(dict(a=h.a.detach(),b=h.b.detach(),regressors=regressors,rotation=rotation,outer_patch_mean=outermean,outer_patch_std=outerstd,family_stats=normstats[name],family=name,held=held,steps=400,policy=policy,schedule_sha256=schedule_hash),DEST/f'{fit}_{name}_{policy}.pt');np.savez_compressed(DEST/f'{fit}_{name}_{policy}_source_out.npz',ids=data['source_out']['ids'][take.numpy()],y=y['source_out'][take].numpy(),domains=d['source_out'][take].numpy(),global_logits=g['source_out'][take].numpy(),full_logits=(g['source_out'][take]+dv).numpy());print(fit,name,policy,record['source_out'],flush=True);write(DEST/'progress.json',dict(status='fit_complete',fit=fit,family=name,policy=policy))
  for policy in ['learned','fixed']:
   common=np.mean([out['families'][name][policy]['source_in_RMS'] for name in pre]+[prior['families'][name][policy]['source_in_RMS'] for name in ['CLS','patch_raw','patch_conditional']]);out.setdefault('common_RMS_controls',{})[policy]={}
   for name in pre:
    scale=common/out['families'][name][policy]['source_in_RMS'];out['common_RMS_controls'][policy][name]=dict(scale=float(scale),source_out=score(g['source_out'][take]+dvsave[name,policy]*scale,g['source_out'][take],y['source_out'][take],d['source_out'][take]))
   for name in ['CLS','patch_raw','patch_conditional']:
    stored=np.load(DEST.parent/'source_conditional_patch_readout'/f'{fit}_{name}_{policy}_source_out.npz');base=torch.tensor(stored['global_logits']);delta=torch.tensor(stored['full_logits'])-base;scale=common/prior['families'][name][policy]['source_in_RMS'];out['common_RMS_controls'][policy][name]=dict(scale=float(scale),source_out=score(base+delta*scale,base,y['source_out'][take],d['source_out'][take]))
  result[fit]=out;write(DEST/'partial_results.json',result)
 for path,sha in plan['input_sha256'].items():assert digest(Path(path))==sha
 write(DEST/'results.json',dict(source_only=True,target_read=False,primary_SIRM_training=False,fits=result,input_sha256=plan['input_sha256'],scientific_goal_achieved=False));write(DEST/'progress.json',dict(status='complete'))
if __name__=='__main__':main()

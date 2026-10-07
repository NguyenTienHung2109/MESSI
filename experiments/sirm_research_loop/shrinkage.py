"""Source-calibrated residual-scaling diagnostic; no model training or selection."""
import argparse,csv,json,hashlib,copy,time,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover_gate.model import GateSIRM
from experiments.sirm_discover.math import center_logits
from experiments.terra_10pct_factor_diagnosis.report import data_for

ROOT=Path(__file__).resolve().parents[2]
SUITE=ROOT/'results/terra_10pct_factor_diagnosis'
DEST=ROOT/'results/sirm_research_loop/iteration_01'
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)

def head(model,z):
    delta=center_logits(torch.stack([expert(z) for expert in model.residuals],1));pi=model.router(z).softmax(1)
    return model.global_head(z),delta,pi

def quantities(g,delta,pi,y,domains,alpha):
    g=g.double();term=alpha*pi[:,:delta.shape[1],None].double()*delta.double();logits=g+term.sum(1)
    error=logits.softmax(1)-F.one_hot(y,g.shape[1]);deriv=(error[:,None,:]*term).sum(-1)
    correct=logits.argmax(1).eq(y);globalcorrect=g.argmax(1).eq(y);ce=F.cross_entropy(logits,y,reduction='none')
    n=len(set(domains.tolist()));accuracy=[];globalacc=[];G=[];utility=[]
    loo=logits[:,None,:]-term
    looces=torch.stack([F.cross_entropy(loo[:,m,:],y,reduction='none') for m in range(delta.shape[1])],1)
    drop=correct[:,None].double()-loo.argmax(-1).eq(y[:,None]).double()
    for d in range(n):
        mask=domains==d;accuracy.append(float(correct[mask].double().mean()));globalacc.append(float(globalcorrect[mask].double().mean()));G.append(deriv[mask].mean(0).abs().tolist());utility.append((looces[mask]-ce[mask,None]).mean(0).tolist())
    residual=alpha*delta.double();flat=center_logits(residual).permute(1,0,2).reshape(delta.shape[1],-1);gram=flat@flat.T;diag=gram.diag()
    cosine=gram/(diag[:,None]*diag[None,:]).sqrt().clamp_min(1e-16) if alpha!=0 else None
    U=np.array(utility);interaction=U-U.mean(1,keepdims=True)-U.mean(0,keepdims=True)+U.mean()
    return dict(residual_norm_per_expert=residual.norm(dim=2).mean(0).tolist(),residual_cosine_matrix=cosine.tolist() if cosine is not None else None,residual_cosine_mean=float(cosine.numpy()[np.triu_indices(delta.shape[1],1)].mean()) if cosine is not None else None,router_NULL=float(pi[:,-1].mean()),utility_interaction=interaction.tolist(),alpha=float(alpha),accuracy=float(np.mean(accuracy)),per_domain_accuracy=accuracy,global_accuracy=float(np.mean(globalacc)),residual_delta=float(np.mean(accuracy)-np.mean(globalacc)),CE=float(ce.mean()),G_matrix=G,utility_matrix=utility,expert_accuracy_drop=drop.mean(0).tolist(),correction_RMS=float(term.sum(1).square().mean().sqrt()),correction_global_RMS=float(term.sum(1).square().mean().sqrt()/center_logits(g).square().mean().sqrt()),predictions=logits.argmax(1))

@torch.no_grad()
def run(suite=SUITE,dest=DEST):
    torch.set_num_threads(4);dest.mkdir(parents=True,exist_ok=True)
    manifest=read(suite/'subset_manifest.json');baseline={m:read(suite/'runs/L38'/m/'posthoc/complete.json') for m in ['ERM_H2','OFF','ON']}
    paths={m:suite/'runs/L38'/m/'source_best.pkl' for m in ['ERM_H2','OFF','ON']}
    locks={m:digest(p) for m,p in paths.items()}
    for m in paths:assert locks[m]==baseline[m]['checkpoint_sha256']
    write(dest/'checkpoint_locks.json',dict(checkpoints=locks,manifest_sha256=digest(suite/'subset_manifest.json'),selection='source-out only; locked before scaling evaluation',scale_selection='source-out correction RMS only; never target'))
    models={}
    for m,p in paths.items():
        if m=='ERM_H2':continue
        payload=torch.load(p,map_location='cpu',weights_only=False);c=payload['config'];backbone=DeiTFeaturizer(pretrained=False,model_name=c['backbone']);model=GateSIRM(backbone,backbone.n_outputs,10,3,c)
        model.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()});model.configure_stage();models[m]=model.eval()
    for name in ['featurizer','global_head']:
        a=getattr(models['OFF'],name).state_dict();b=getattr(models['ON'],name).state_dict();assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a),'global anchors differ'
    envs=[0,2,3];out={};calibration=None;mask=np.array(baseline['ON']['mechanism']['union_mask'],bool)
    for split,indices,key in [('source_out',envs,'out'),('target_in',[1],'in'),('target_out',[1],'out')]:
        ds=data_for(manifest,indices,c,key);columns={m:{k:[] for k in ['g','delta','pi']} for m in models};ys=[];dd=[]
        for d,(env,data) in enumerate(sorted(ds.items())):
            for x,y in DataLoader(data,batch_size=64,num_workers=2):
                z=models['OFF'].featurizer(x)
                for m,model in models.items():
                    g,delta,pi=head(model,z)
                    for k,v in zip(['g','delta','pi'],[g,delta,pi]):columns[m][k].append(v)
                ys.append(y);dd.append(torch.full_like(y,d))
        y=torch.cat(ys);domain=torch.cat(dd);arrays={m:{k:torch.cat(v) for k,v in vals.items()} for m,vals in columns.items()}
        evals={m:quantities(**arrays[m],y=y,domains=domain,alpha=1.) for m in models}
        # Validate original composition before interpreting interventions.
        for m in models:assert abs(evals[m]['accuracy']-baseline[m]['splits'][split]['source_mean'])<1e-12,(m,split,'CPU replay accuracy mismatch')
        if split=='source_out':
            denom=evals['OFF']['correction_RMS'];assert denom>0,'zero OFF correction prevents RMS calibration'
            calibration=evals['ON']['correction_RMS']/denom
            # Freeze alpha before target tensors are read.
            write(dest/'source_scale_lock.json',dict(alpha=calibration,OFF_source_correction_RMS=denom,ON_source_correction_RMS=evals['ON']['correction_RMS'],target_used=False))
        evals['OFF_source_matched']=quantities(**arrays['OFF'],y=y,domains=domain,alpha=calibration)
        evals['OFF_global_only']=quantities(**arrays['OFF'],y=y,domains=domain,alpha=0.)
        orig=evals['OFF']['predictions'];on=evals['ON']['predictions'];scaled=evals['OFF_source_matched']['predictions'];change=on!=orig
        out[split]=dict(samples=len(y),methods={},prediction_control=dict(ON_changed=int(change.sum()),scaled_changed=int((scaled!=orig).sum()),ON_scaled_disagreement=int((on!=scaled).sum()),ON_change_reproduced=int((change&(on==scaled)).sum()),ON_net_correct=int((on==y).sum()-(orig==y).sum()),scaled_net_correct=int((scaled==y).sum()-(orig==y).sum())))
        for m,vals in evals.items():
            if split=='source_out':
                selected=np.array(baseline['OFF' if m.startswith('OFF') else 'ON']['splits']['source_out']['q'])>0;inter=np.array(vals['utility_interaction'])
                vals['interaction_advantage']=float(inter[selected].mean()-inter[~selected].mean()) if selected.any() and (~selected).any() else None
                vals['Q']=baseline['OFF' if m.startswith('OFF') else 'ON']['splits']['source_out']['q']
            else:
                sourceU=np.array(out['source_out']['methods'][m]['utility_matrix']).mean(0);targetU=np.array(vals['utility_matrix'])[0]
                for name,value in [('source_target_CE_spearman',targetU),('source_target_accuracy_spearman',vals['expert_accuracy_drop'])]:
                    corr=float(spearmanr(sourceU,value).statistic);vals[name]=corr if np.isfinite(corr) else None
                vals['useful_experts']=sum(x>0 for x in vals['expert_accuracy_drop']);vals['harmful_experts']=sum(x<0 for x in vals['expert_accuracy_drop'])
            vals.pop('predictions');out[split]['methods'][m]=vals
            if split=='source_out':vals['G_common_executed']=float(np.array(vals['G_matrix'])[mask].mean()) if mask.any() else None
    for m,p in paths.items():assert digest(p)==locks[m]
    on=out['target_in']['methods']['ON'];off=out['target_in']['methods']['OFF'];scaled=out['target_in']['methods']['OFF_source_matched'];gain=on['accuracy']-off['accuracy'];sgain=scaled['accuracy']-off['accuracy']
    change=out['target_in']['prediction_control'];active=baseline['ON']['mechanism']['active_steps']
    training_rows=[json.loads(line) for line in (suite/'runs/L38/ON/diagnostics.jsonl').read_text().splitlines()]
    selected_active=sum(row.get('sirm_active',False) for row in training_rows if row['step']<=baseline['ON']['selected_step'])
    # Discrete one-image tolerance is explicit, not statistical evidence.
    tol=1/out['target_in']['samples']
    if not selected_active or gain<=0:status='INCONCLUSIVE';reason='No positive ON gain (or no SIRM exposure before the selected checkpoint) to explain by shrinkage.'
    elif calibration>=1:status='FALSIFIED';reason='Source-calibrated residual magnitude does not shrink; uniform residual shrinkage cannot explain the positive ON gain.'
    elif abs(sgain-gain)<=tol and change['ON_scaled_disagreement']<=1:status='SUPPORTED';reason='Source-matched scaling reproduces the ON gain and predictions within one image.'
    elif sgain>=gain-tol:status='PARTIALLY SUPPORTED';reason='Source-matched scaling reproduces aggregate gain but not all prediction changes.'
    else:status='FALSIFIED';reason='Source-matched scaling fails to reproduce the positive ON gain; this rejects this uniform-shrinkage control, not all shrinkage mechanisms.'
    result=dict(status=status,reason=reason,selected_checkpoint_active_steps=selected_active,alpha_source_matched=calibration,baseline=baseline,splits=out,checkpoint_hashes=locks,common_mask=mask.tolist(),accuracy_tolerance=tol,training=False,target_selected=False)
    write(dest/'shrinkage_results.json',result)
    report=['ITERATION 1','','Goal:','Test whether the ON benefit can be reproduced by source-calibrated uniform residual shrinkage.','','Current bottleneck:','Historical full-data source factors transfer poorly; fast-protocol global-anchor fitting and factor transfer remain separate possibilities.','','Hypothesis:','If uniform shrinkage explains ON, source-RMS-matched OFF scaling should reproduce its target benefit.','','Intervention:',f'Inference-only OFF scaling; alpha={calibration:.8g}, fitted from source-out only. Fixed controls alpha=0 and alpha=1; no target-selected alpha.','','Expected evidence:','Predeclared in preregistration.md; compare changed predictions, CE utilities, residual deltas and fixed-mask G.','','Experiment:','Fixed 10% Terra L38 seed 0, locked source-selected ERM/OFF/ON checkpoints. CPU replay, no fitting or checkpoint changes.','','Results:',f'{status}: {reason}','', '| Split | Model | Accuracy % | Global % | Residual delta pp | CE | Correction/global RMS | G common mask |','|---|---|---:|---:|---:|---:|---:|---:|']
    for split,d in out.items():
        for m,v in d['methods'].items():report.append(f'| {split} | {m} | {100*v["accuracy"]:.6f} | {100*v["global_accuracy"]:.6f} | {100*v["residual_delta"]:+.6f} | {v["CE"]:.7g} | {v["correction_global_RMS"]:.7g} | {v.get("G_common_executed")} |')
    report+=['','Mechanism analysis:',f'ON active training steps: {active}; exposures before selected checkpoint: {selected_active}. The scale is locked before target evaluation. Global anchors are exactly equal. Uniform scaling preserves residual directions and router probabilities; reducing G this way is attenuation, not evidence of new invariant factors. Full utility matrices and changed-prediction counts are in shrinkage_results.json.','','Accuracy analysis:',f'ON−OFF: {100*gain:+.6f} pp; source-matched scaled OFF−OFF: {100*sgain:+.6f} pp. Main pure ERM-H2: {100*baseline["ERM_H2"]["splits"]["target_in"]["source_mean"]:.6f}%. These exploratory one-seed small-subset scores cannot establish reproducibility.','','Hypothesis status:',status,'','Keep/Revert:','Keep this diagnostic; no training modification to retain or revert. Do not promote scalar shrinkage as invariant-factor learning.','','Next hypothesis:','Audit source validation/global-only gaps to determine whether early global freezing is the primary confound before redesigning factor discovery. A candidate intervention requires a separate preregistration and paired OFF/ON comparison.']
    (dest/'report.md').write_text('\n'.join(report)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');args=p.parse_args()
    if args.wait:
        while not all((SUITE/'runs/L38'/m/'posthoc/complete.json').exists() for m in ['ERM_H2','OFF','ON']):
            prog=read(SUITE/'progress.json');pid=read(SUITE/'controller_pid.json')['pid']
            try:state=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]
            except FileNotFoundError:raise RuntimeError('Baseline controller missing before L38 completion')
            if state in ['Z','X'] or prog['status'] in ['failed','stopped_by_user']:raise RuntimeError('Baseline suite stopped before L38 completion')
            write(DEST/'progress.json',dict(status='waiting_for_locked_L38_baselines',verified_live_controller_pid=pid,observed_unix=time.time(),baseline_progress=prog));time.sleep(15)
    write(DEST/'progress.json',dict(status='analyzing',observed_unix=time.time()));result=run();write(DEST/'progress.json',dict(status='completed',hypothesis_status=result['status'],observed_unix=time.time()))

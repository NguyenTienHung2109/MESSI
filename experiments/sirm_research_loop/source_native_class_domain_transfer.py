"""Source-in class/domain classification directions and native validation margin transfer."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import center_logits
from .source_probe_clean_preservation import outputs,assess
from .routed_conditional_utility import load
from .routed_discovery_run import ROOT,DEST,PRIOR
from .discovery_error_gradient_diagnosis import vector
from .class_balanced_margin_diagnosis import margin
from .shrinkage import digest,write
OUT=DEST/'source_native_class_domain_transfer';RADIUS=.01

def run():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    plan=dict(source_only=True,target_read=False,primary_training=False,optimizer_updates=False,probe_used=False,
      states=['unscaled_OFF_selected','unscaled_OFF_last','routed_OFF_selected','routed_OFF_last'],
      cohort='All original source-in class/domain cells and their frozen-global-error subcohorts with at least2 images. No validation-derived groups.',
      loss='Mean classification CE in each group; native pi detached, differentiable equal-domain full-source-in residual centering.',
      radius=RADIUS,evaluation='Original disjoint source-out and source-in under their own fixed native pi, recalibrating equal-domain source-in centering after every finite move.',
      controls=['all_source_equal_domain_CE','residual_contraction','source_in_mean_logit_response'],response_control='Equal-domain mean finite logit change on source-in, applied as a constant vector to baseline source-out logits; fitted neither on validation labels nor target.',
      limits='Equal-radius saved-state diagnostics are not SGD/Adam trajectories or estimates of independent experiments. Class/domain conditioning uses known source labels; not deployable gates.',code_sha256=digest(Path(__file__)))
    write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';proof=json.loads((token/'extraction_audit.json').read_text());data={}
    for split in ['source_in','source_out']:
        p=token/f'{split}_tokens.npz';assert digest(p)==next(r['cache_sha256'] for r in proof['rows'] if r['split']==split);a=np.load(p)
        data[split]=(torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains']),list(map(str,a['ids'])))
    train,ty,td,tids=data['source_in'];val,vy,vd,vids=data['source_out'];assert not set(tids)&set(vids)
    mapped=lambda dom:torch.tensor([0 if int(d)==0 else 1 if int(d)==2 else 2 for d in dom]);tdom,vdom=mapped(td),mapped(vd)
    cases=json.loads((DEST/'source_native_paired_view_direction/case_margins.json').read_text());caseids=sorted(set(r['id'] for r in cases['rows']));caseindices=[vids.index(id) for id in caseids]
    def grouped(f,g,y,dom):
        rows=[];ce=F.cross_entropy(f,y,reduction='none').detach();mar=margin(f,y).detach();gc=g.argmax(1).eq(y);fc=f.argmax(1).eq(y)
        for cls in range(10):
            for domain in range(3):
                mask=y.eq(cls)&dom.eq(domain);n=int(mask.sum())
                if not n:continue
                rows.append(dict(label=cls,domain=domain,n=n,CE=float(ce[mask].mean()),margin=float(mar[mask].mean()),correct=int(fc[mask].sum()),repairs=int((~gc&fc&mask).sum()),harms=int((gc&~fc&mask).sum())))
        return rows
    rows=[];pairs=[];coverage=[];locks={}
    for state in plan['states']:
        m,p,path=load(PRIOR if state.startswith('unscaled') else DEST,'selected' if state.endswith('selected') else 'last');locks[state]=digest(path);initial={k:v.clone() for k,v in m.state_dict().items()}
        m.calibrate_features(train,td);mean=m.expert_source_mean.clone();params=list(m.residuals.parameters());copies=[p.detach().clone() for p in params]
        with torch.no_grad():tz=m.bank_features(train);tg=m.global_head(train[:,:384]);tpi=m.router(tz).softmax(1);vpi=m.router(m.bank_features(val)).softmax(1)
        delta=center_logits(torch.stack([e(tz) for e in m.residuals],1));delta=delta-torch.stack([delta[tdom==d].mean(0) for d in range(3)]).mean(0)[None];tf=tg+(tpi[:,:6,None]*delta).sum(1)
        loss=F.cross_entropy(tf,ty,reduction='none');gc=tg.argmax(1).eq(ty)
        dirs={'all_source_equal_domain_CE':vector(torch.stack([loss[tdom==d].mean() for d in range(3)]).mean(),params),'residual_contraction':vector(delta.square().mean(),params)};groups={}
        for cls in range(10):
            for domain in range(3):
                for cohort in ['all','global_errors']:
                    mask=ty.eq(cls)&tdom.eq(domain)
                    if cohort=='global_errors':mask=mask&~gc
                    n=int(mask.sum());coverage.append(dict(state=state,label=cls,domain=domain,cohort=cohort,n=n,eligible=n>=2))
                    if n<2:continue
                    name=f'{cohort}_class{cls}_domain{domain}';dirs[name]=vector(loss[mask].mean(),params);groups[name]=dict(label=cls,domain=domain,cohort=cohort,n=n)
        for cls in range(10):
            for cohort in ['all','global_errors']:
                names=[n for n,g in groups.items() if g['label']==cls and g['cohort']==cohort]
                for i,n1 in enumerate(names):
                    for n2 in names[i+1:]:pairs.append(dict(state=state,label=cls,cohort=cohort,domains=[groups[n1]['domain'],groups[n2]['domain']],cosine=float(F.cosine_similarity(dirs[n1],dirs[n2],dim=0))))
        del delta,tf,loss
        q=torch.zeros(3,6);baseline,bc=assess(m,val,vy,vdom,vpi,q);tb,tbc=assess(m,train,ty,tdom,tpi,q)
        with torch.no_grad():vg,_,vf=outputs(m,val,vpi);vgroup=grouped(vf,vg,vy,vdom);beforemargin=margin(vf,vy);beforepred=vf.argmax(1);globalpred=vg.argmax(1);_,_,trainf=outputs(m,train,tpi);tgroup=grouped(trainf,tg,ty,tdom)
        for name,v in dirs.items():
            assert torch.isfinite(v).all() and v.norm()>0;move=(-RADIUS*v/v.norm()).float();offset=0
            with torch.no_grad():
                for param,copy in zip(params,copies):n=param.numel();param.copy_(copy+move[offset:offset+n].reshape_as(param));offset+=n
                actual=float(torch.cat([(p-copy).flatten().double() for p,copy in zip(params,copies)]).norm());assert abs(actual-RADIUS)<2e-6
            m.calibrate_features(train,td);metrics,correct=assess(m,val,vy,vdom,vpi,q);tm,tc=assess(m,train,ty,tdom,tpi,q)
            with torch.no_grad():_,_,afterf=outputs(m,val,vpi);afmargin=margin(afterf,vy);afpred=afterf.argmax(1);avgroup=grouped(afterf,vg,vy,vdom);_,_,aftertrainf=outputs(m,train,tpi);atgroup=grouped(aftertrainf,tg,ty,tdom)
            differences=lambda new,old:{k:new[k]-old[k] for k in old if k!='validation_pair_count' and new[k] is not None and old[k] is not None}
            groupdiff=lambda new,old:[dict(label=r['label'],domain=r['domain'],n=r['n'],delta={k:r[k]-s[k] for k in ['CE','margin','correct','repairs','harms']}) for r,s in zip(new,old)]
            with torch.no_grad():
                train_change=aftertrainf-trainf
                constant=torch.stack([train_change[tdom==d].mean(0) for d in range(3)]).mean(0)
                energy=torch.stack([train_change[tdom==d].square().sum(1).mean() for d in range(3)]).mean()
                cf=vf+constant[None];cmargin=margin(cf,vy);cpred=cf.argmax(1);cc=cpred.eq(vy);vgc=globalpred.eq(vy)
                constant_control=dict(vector=constant.tolist(),source_response_energy_fraction=float(constant.square().sum()/energy),
                  CE_change=float(torch.stack([F.cross_entropy(cf[vdom==d],vy[vdom==d])-F.cross_entropy(vf[vdom==d],vy[vdom==d]) for d in range(3)]).mean()),
                  correct_change=int(cc.sum()-bc.sum()),repair_change=int((~vgc&cc).sum()-(~vgc&bc).sum()),harm_change=int((vgc&~cc).sum()-(vgc&~bc).sum()),
                  cases=[dict(id=vids[i],margin_change=float(cmargin[i]-beforemargin[i]),after_prediction=int(cpred[i])) for i in caseindices])
                assert -.00001<=constant_control['source_response_energy_fraction']<=1.00001
            rows.append(dict(source_constant_response_control=constant_control,state=state,direction=name,group=groups.get(name),gradient_norm=float(v.norm()),actual_radius=actual,baseline=baseline,delta=differences(metrics,baseline),train_delta=differences(tm,tb),validation_groups=groupdiff(avgroup,vgroup),train_groups=groupdiff(atgroup,tgroup),cases=[dict(id=vids[i],label=int(vy[i]),global_prediction=int(globalpred[i]),before_prediction=int(beforepred[i]),after_prediction=int(afpred[i]),before_margin=float(beforemargin[i]),margin_change=float(afmargin[i]-beforemargin[i])) for i in caseindices],changed_cases=[dict(id=vids[i],label=int(vy[i]),before_correct=bool(bc[i]),after_correct=bool(correct[i])) for i in torch.nonzero(correct!=bc).flatten().tolist()]))
            with torch.no_grad():
                for param,copy in zip(params,copies):param.copy_(copy)
                m.expert_source_mean.copy_(mean)
        m.load_state_dict(initial);assert all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) and all(p.grad is None for p in m.parameters()) and digest(path)==locks[state]
        print(state,len(dirs),'directions complete',flush=True)
    write(OUT/'results.json',dict(plan=plan,rows=rows,cross_domain_gradient_pairs=pairs,coverage=coverage,checkpoint_sha256=locks,source_only=True,target_read=False,bank_tensors_restored=True,grad_fields_unassigned=True))
if __name__=='__main__':run()

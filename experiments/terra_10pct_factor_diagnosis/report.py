"""Locked-checkpoint diagnostics; no target-selected results or tuning."""
import copy,csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from .subsets import ROOT,RESULT
from .run import NAMES,METHODS,write,digest
from experiments.sirm_discover_gate.data import Adapter
from experiments.sirm_discover_gate.metrics import evaluate
from experiments.sirm_discover_binary.run import restore
from experiments.sirm_discover_binary_analysis.run import pooled_cosine,rows,describe,ratio_records,evolution_stats

def csvwrite(path,data,fields=None):
    fields=fields or list(data[0]) if data else fields or ['status']
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)

def audit_training(bank,target):
    logs=rows(bank/'results.jsonl');assert logs and (bank/'source_best.pkl').exists()
    assert all(not any(k.startswith(f'env{target}_') for k in r) for r in logs),'target evaluated during training'
    selection=json.loads((bank/'selection.json').read_text());assert selection['target_used'] is False
    hp=json.loads((bank/'args.json').read_text()) if (bank/'args.json').exists() else None
    write(bank/'training_audit.json',dict(passed=True,target_evaluation_rows=0,checkpoint_sha256=digest(bank/'source_best.pkl'),selection=selection))

def data_for(manifest,envs,c,split):
    a=Adapter('TerraIncognita');a.classes=manifest['class_names'];m=copy.deepcopy(manifest);m['sources']={str(e):m['sources'][str(e)] for e in envs}
    return a.source_data(ROOT/'domainbed/data',m,c,split)

@torch.no_grad()
def plain_scores(model,data,c):
    scores={}
    for e,ds in data.items():
        n=correct=0
        for x,y in DataLoader(ds,batch_size=c['eval_batch_size'],num_workers=c['num_workers']):
            pred=model.predict(x.cuda()).argmax(1).cpu();correct+=int(pred.eq(y).sum());n+=len(y)
        scores[str(e)]=correct/n
    return dict(source_accuracy=scores,source_mean=float(np.mean(list(scores.values()))))

def correlation(a,b):
    v=float(spearmanr(a,b).statistic);return v if np.isfinite(v) else None

def analyze(bank,target,method,manifest):
    dest=bank/'posthoc';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():
        assert json.loads((dest/'complete.json').read_text())['checkpoint_sha256']==digest(bank/'source_best.pkl');return
    audit=json.loads((bank/'training_audit.json').read_text());sha=digest(bank/'source_best.pkl');assert sha==audit['checkpoint_sha256']
    # Save this lock BEFORE constructing target loaders or evaluating target images.
    write(dest/'selection_lock.json',dict(checkpoint_sha256=sha,target_used_for_selection=False))
    payload=torch.load(bank/'source_best.pkl',map_location='cpu',weights_only=False);c=payload['config'];sources=[e for e in range(4) if e!=target]
    if method=='ERM_H2':
        from .erm import ERMH2Diagnostic
        model=ERMH2Diagnostic((3,224,224),len(manifest['class_names']),3,dict(sirm_discover_config=c,sirm_output_dir=str(dest),sirm_source_envs=sources));model.load_state_dict(payload['model_dict']);model=model.cuda().eval()
    else:model,c,payload=restore(bank)
    record=dict(target=NAMES[target],target_env=target,method=method,selected_step=payload['selected_step'],checkpoint_sha256=sha,splits={})
    for label,envs,split in [('source_out',sources,'out'),('target_in',[target],'in'),('target_out',[target],'out')]:
        data=data_for(manifest,envs,c,split)
        if method=='ERM_H2':d=plain_scores(model,data,c)
        else:
            d=evaluate(model,data,c,True);d['cosine']=pooled_cosine(model,data,c)
            if label!='source_out':
                u=np.array(record['splits']['source_out']['marginal_utility']).mean(0);tu=np.array(d['marginal_utility'])[0];drop=d['leave_one_out_accuracy_drop']
                d.update(source_target_CE_spearman=correlation(u,tu),source_target_accuracy_spearman=correlation(u,drop),useful_experts=int((np.array(drop)>0).sum()),harmful_experts=int((np.array(drop)<0).sum()))
        record['splits'][label]=d
    if method!='ERM_H2':
        logs=rows(bank/'diagnostics.jsonl');stage=[r for r in logs if r['stage']=='invariance'];refs=[r for r in rows(bank/'q_refresh.jsonl') if r['stage']=='invariance'];ratios=ratio_records(refs);active=[r['gradient_ratio'] for r in ratios if r['active'] and r['gradient_ratio'] is not None]
        mask=np.zeros((3,c['num_experts']),bool)
        for r in stage:
            if r['step']>payload['selected_step']:continue
            for key in r.get('G',{}):
                d,m=map(int,key.split(','));mask[d,m]=True
        record['mechanism']=dict(active_steps=sum(r['sirm_active'] for r in stage),stage3_steps=len(stage),active_fraction=sum(r['sirm_active'] for r in stage)/len(stage),first_nonzero_step=next((r['step'] for r in stage if r['loss_sirm']>0),None),Q=evolution_stats(refs),gradient_ratio=describe(active),union_mask=mask.tolist(),no_shared_refreshes=sum(r['shared_experts']==0 for r in refs),ESS_blocked_refreshes=sum(r['shared_experts']>0 and r['eligible_pair_expert_count']==0 for r in refs))
        csvwrite(dest/'gradient_ratio_history.csv',ratios)
        write(dest/'Q_refresh_history.json',refs)
    assert digest(bank/'source_best.pkl')==sha
    write(dest/'complete.json',record);del model;torch.cuda.empty_cache()

def aggregate(root,manifest):
    records=[]
    for target in NAMES:
        for method in METHODS:
            p=root/'runs'/target/method/'posthoc/complete.json'
            if p.exists():records.append(json.loads(p.read_text()))
    accuracy=[];decomp=[];marginal=[];interaction=[];transfer=[];globalrows=[];mechanism=[];refresh_history=[]
    lookup={(r['target'],r['method']):r for r in records}
    for r in records:
        base=dict(target=r['target'],method=r['method'],selected_step=r['selected_step']);splits=r['splits']
        accuracy.append(dict(**base,target_test_accuracy=splits['target_in']['source_mean'],target_validation_accuracy=splits['target_out']['source_mean'],source_validation_accuracy=splits['source_out']['source_mean'],checkpoint_sha256=r['checkpoint_sha256']))
        if r['method']=='ERM_H2':continue
        for label,d in splits.items():
            globalrows.append(dict(**base,split=label,full_accuracy=d['source_mean'],global_only_accuracy=d['global_mean'],residual_net_delta=d['source_mean']-d['global_mean']))
            for m,norm in enumerate(d['residual_norm_per_expert']):
                decomp.append(dict(**base,split=label,expert=m,residual_norm=norm,correction_global_RMS=d['correction_global_norm_ratio_rms'],router_NULL=d['router_mass'][-1],cosine_mean=d['cosine']['pair_mean'],cosine_matrix=json.dumps(d['cosine']['matrix'])))
            if label=='source_out':
                for local,e in enumerate(d['source_domains']):
                    for m,u in enumerate(d['marginal_utility'][local]):
                        marginal.append(dict(**base,domain=manifest['sources'][str(e)]['images'][0]['id'].split('/')[0],expert=m,utility=u))
                        interaction.append(dict(**base,domain=e,expert=m,interaction=d['utility_interaction'][local][m],selected=d['q'][local][m]>0,interaction_advantage=d['interaction_advantage']))
            else:
                source=np.array(splits['source_out']['marginal_utility']).mean(0)
                for m,u in enumerate(d['marginal_utility'][0]):
                    transfer.append(dict(**base,split=label,expert=m,source_utility=source[m],target_CE_utility=u,target_accuracy_drop=d['leave_one_out_accuracy_drop'][m],CE_spearman=d['source_target_CE_spearman'],accuracy_spearman=d['source_target_accuracy_spearman'],useful_experts=d['useful_experts'],harmful_experts=d['harmful_experts']))
        if r['method']=='ON':
            mech=r['mechanism'];off=lookup.get((r['target'],'OFF'));onG=np.array(splits['source_out']['G_matrix']);offG=np.array(off['splits']['source_out']['G_matrix']) if off else None
            refs=rows(root/'runs'/r['target']/'ON/q_refresh.jsonl')
            for qrow in refs:
                if qrow['stage']!='invariance':continue
                ratio=next(iter(ratio_records([qrow])))['gradient_ratio']
                refresh_history.append(dict(**base,step=qrow['step'],selected_cells=qrow['selected_cells'],shared_experts=qrow['shared_experts'],eligible_terms=qrow['eligible_pair_expert_count'],executed_terms=qrow['sirm_pair_count'],SIRM_loss=qrow['loss_sirm'],gradient_ratio=ratio,source_validation_accuracy=qrow['source_validation_accuracy'],Q=json.dumps(qrow['q']),utility_EMA=json.dumps(qrow['utility_ema']),interaction_EMA=json.dumps(qrow['utility_interaction'])))
            masks={'ON_selected_Q':np.array(splits['source_out']['q'])>0,'ON_executed_union':np.array(mech['union_mask'],bool)}
            for name,mask in masks.items():
                mechanism.append(dict(**base,mask=name,mask_matrix=json.dumps(mask.tolist()),G_ON=float(onG[mask].mean()) if mask.any() else None,G_OFF=float(offG[mask].mean()) if offG is not None and mask.any() else None,first_shared_support_step=mech['Q']['first_shared_support_step'],first_nonzero_step=mech['first_nonzero_step'],active_steps=mech['active_steps'],stage3_steps=mech['stage3_steps'],active_fraction=mech['active_fraction'],Q_support_changes=mech['Q']['Q_support_changes'],ratio_median=mech['gradient_ratio']['median'],ratio_mean=mech['gradient_ratio']['mean'],ratio_p90=mech['gradient_ratio']['p90'],ratio_max=mech['gradient_ratio']['max'],no_shared_refreshes=mech['no_shared_refreshes'],ESS_blocked_refreshes=mech['ESS_blocked_refreshes']))
    for row in accuracy:
        t=row['target']; trio=[lookup.get((t,m)) for m in METHODS]
        vals=[x['splits']['target_in']['source_mean'] if x else None for x in trio]
        row.update(architecture_gain=vals[1]-vals[0] if vals[1] is not None and vals[0] is not None else None,SIRM_gain=vals[2]-vals[1] if vals[2] is not None and vals[1] is not None else None,total_gain=vals[2]-vals[0] if vals[2] is not None and vals[0] is not None else None)
    if len(records)==12:
        for method in METHODS:
            same=[r for r in accuracy if r['method']==method];mean=dict(same[0],target='MEAN_4',selected_step=None,checkpoint_sha256=None)
            for field in ['target_test_accuracy','target_validation_accuracy','source_validation_accuracy','architecture_gain','SIRM_gain','total_gain']:mean[field]=float(np.mean([r[field] for r in same]))
            accuracy.append(mean)
    for name,data in [('accuracy',accuracy),('residual_decomposition',decomp),('marginal_utility',marginal),('interaction_utility',interaction),('target_factor_transfer',transfer),('global_vs_residual',globalrows),('sirm_mechanism',mechanism),('sirm_refresh_history',refresh_history)]:csvwrite(root/(name+'.csv'),data)
    text=['# Terra 10% factor diagnosis','',f'Completed post-hoc analyses: {len(records)}/12. Seed 0. Target-in is the primary test score; target-out is post-hoc only. Checkpoints selected solely by equal-source-domain out accuracy, steps 200–500, every 25 steps (earliest best tie). Pure ERM trains end-to-end all 501 steps. OFF/ON use the shortened 100/100/301 stages; ON lambda=1, 50-step ramp, dynamic Q every 5 steps and binary eligible-pair gate.','', 'CSV accuracies/deltas are fractions; tables are percentages/percentage points. Missing results remain pending.','', '| Target | ERM-H2 | OFF | ON | OFF−ERM | ON−OFF | ON−ERM |','|---|---:|---:|---:|---:|---:|---:|']
    complete=[]
    for t in NAMES:
        vals=[lookup.get((t,m),{}).get('splits',{}).get('target_in',{}).get('source_mean') for m in METHODS]
        fmt=lambda v:f'{100*v:.4f}' if v is not None else 'pending'
        gains=[vals[1]-vals[0],vals[2]-vals[1],vals[2]-vals[0]] if all(v is not None for v in vals) else [None]*3
        if all(v is not None for v in vals):complete.append(vals)
        text.append('| '+t+' | '+' | '.join(fmt(v) for v in vals+gains)+' |')
    if len(complete)==4:
        a,b,c=np.mean(complete,axis=0);text.append('| Mean four targets | '+' | '.join(fmt(v) for v in [a,b,c,b-a,c-b,c-a])+' |')
    text+=['','## Eight diagnostic questions','']
    if len(records)<12:text+=['All eight answers remain provisional until all 12 source-selected results exist. The following descriptive results cover completed runs only.','']
    if globalrows:
        src=[r['residual_net_delta'] for r in globalrows if r['split']=='source_out'];tgt=[r['residual_net_delta'] for r in globalrows if r['split']=='target_in']
        text += [f'1. Non-redundant factors: inspect full residual cosine matrices together with marginal utilities and removal effects; low cosine alone does not establish unique useful information.', '2. Stronger interaction than old SIRM: not directly identifiable from a comparison of old standalone-risk interactions with new routed marginal-utility interactions. interaction_utility.csv reports the current complete matrices/selection advantage; different datasets/sample sizes further confound historical comparisons.',f'3. Residual source benefit: mean full−global across completed OFF/ON runs = {100*np.mean(src):+.4f} pp.',f'4. Residual target benefit: mean full−global = {100*np.mean(tgt):+.4f} pp. Per-target/arm signs are in global_vs_residual.csv.', '5. Source utility predicts transfer: CE and accuracy-removal Spearman values across six experts are in target_factor_transfer.csv; undefined constant-vector correlations remain blank. Six experts make estimates coarse.']
        gvalid=[r for r in mechanism if r['mask']=='ON_executed_union' and r['G_ON'] is not None and r['G_OFF'] is not None]
        text += [f'6. IRMv1 violation: G_ON < G_OFF in {sum(r["G_ON"]<r["G_OFF"] for r in gvalid)}/{len(gvalid)} completed target comparisons using ON executed-union common masks.', '7. G reduction versus target improvement: compare the target-paired ON−OFF score and G in sirm_mechanism.csv. Lower G alone is not an OOD success criterion.', '8. Bottleneck: diagnose per target using source/target residual deltas, source-to-target utility ranks, Q shared-support/ESS blocks and gradient strength. This one-seed 10% experiment cannot establish a unique causal bottleneck; mixed signs should be reported rather than forced into one explanation.']
    else:text+=['1–8. Pending training and selected-checkpoint replay.']
    if records and globalrows:
        text += ['', 'Completed-run factor evidence:', '', '| Target / method | Source residual Δ pp | Target residual Δ pp | Source interaction max abs | Selected interaction advantage | Target CE rank correlation | Target accuracy rank correlation | Useful / harmful experts |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in records:
            if r['method']=='ERM_H2':continue
            ds=r['splits']['source_out'];dt=r['splits']['target_in']
            val=lambda x:f'{x:.6g}' if x is not None else 'undefined'
            text.append(f"| {r['target']} / {r['method']} | {100*(ds['source_mean']-ds['global_mean']):+.4f} | {100*(dt['source_mean']-dt['global_mean']):+.4f} | {np.abs(ds['utility_interaction']).max():.6g} | {val(ds['interaction_advantage'])} | {val(dt['source_target_CE_spearman'])} | {val(dt['source_target_accuracy_spearman'])} | {dt['useful_experts']} / {dt['harmful_experts']} |")
        adverse=sum(r['splits']['source_out']['source_mean']>r['splits']['source_out']['global_mean'] and r['splits']['target_in']['source_mean']<r['splits']['target_in']['global_mean'] for r in records if r['method']!='ERM_H2')
        positive=sum(r['splits']['target_in']['source_mean']>r['splits']['target_in']['global_mean'] for r in records if r['method']!='ERM_H2')
        text += ['', f'Source-helpful / target-harmful residual pattern: {adverse} completed OFF/ON runs; target-helpful residual composition: {positive}. This describes factor transfer in the sampled environments, not a causal identification of factors.']
        if mechanism:
            text += ['', '| Target | G OFF (ON executed union) | G ON | ON−OFF target pp | Active fraction | First shared support | Ratio median / p90 / max |', '|---|---:|---:|---:|---:|---:|---:|']
            for row in mechanism:
                if row['mask']!='ON_executed_union':continue
                off=lookup.get((row['target'],'OFF'));on=lookup[(row['target'],'ON')]
                delta=100*(on['splits']['target_in']['source_mean']-off['splits']['target_in']['source_mean']) if off else None
                val=lambda x:f'{x:.6g}' if x is not None else 'undefined'
                text.append('| '+row['target']+' | '+' | '.join(val(v) for v in [row['G_OFF'],row['G_ON'],delta,row['active_fraction'],row['first_shared_support_step']])+' | '+ ' / '.join(val(row[k]) for k in ['ratio_median','ratio_p90','ratio_max'])+' |')
            both=sum(row['G_ON']<row['G_OFF'] and lookup[(row['target'],'ON')]['splits']['target_in']['source_mean']>lookup[(row['target'],'OFF')]['splits']['target_in']['source_mean'] for row in gvalid)
            text += ['', f'Both G reduction and positive target ON−OFF gain: {both}/{len(gvalid)} targets with a defined common-union comparison. Mechanism-versus-transfer conclusions remain exploratory; one seed, small test subsets, and different source-selected times limit causal attribution.']
    text+=['','Full per-run metrics, cosine matrices, source/target scores and locked hashes: runs/<target>/<method>/posthoc/complete.json. ON refresh histories include Q/EMA, shared experts, eligible terms, ESS and per-domain source validation. Gradient norms cover residual parameters at Q refreshes, not all optimizer steps. G is the absolute domain-mean residual-scale derivative (can cancel), not the squared minibatch penalty. Fixed common masks use ON assignments; executed union includes exposure up to the ON selected step; selected steps may differ. Target diagnostics never alter checkpoints/configuration. Ablation effects overlap and must not be summed. Rare classes use min-two rounding capped by original class count and thus can exceed 10%; counts/IDs are explicit. No target history or target Oracle selection is generated.']
    (root/'summary.md').write_text('\n'.join(text)+'\n')


def audit_controls(target_root,smoke=False):
    logs={m:rows(target_root/m/'diagnostics.jsonl') for m in METHODS}
    schedules=[json.loads((target_root/m/'source_schedules.json').read_text()) for m in METHODS];assert schedules[0]==schedules[1]==schedules[2]
    assert json.loads((target_root/'OFF/initialization.json').read_text())==json.loads((target_root/'ON/initialization.json').read_text())
    stop=2 if smoke else 201
    for a,b in zip(logs['OFF'][:stop],logs['ON'][:stop]):
        for key in ['step','loss_cls','loss_discover','loss_router','loss_balance','utility_ema','q']:assert a[key]==b[key],(a['step'],key)
    assert logs['ERM_H2'][0]['loss_cls']==logs['OFF'][0]['loss_cls']
    for method in ['OFF','ON']:
        assert len(logs[method])==(3 if smoke else 501)
        assert all(not r['sirm_active'] and r['loss_sirm']==0 for r in logs[method] if r['stage']!='invariance')
    assert all(not r['sirm_active'] and r['loss_sirm']==0 for r in logs['OFF'])
    write(target_root/'control_audit.json',dict(passed=True,source_schedules_equal=True,OFF_ON_initialization_equal=True,OFF_ON_preactivation_exact=True,ERM_SIRM_initial_batch_loss_equal=True,OFF_penalty_never_executed=True,target_training_scores_absent=True))

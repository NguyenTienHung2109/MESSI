"""Source competence interactions and paired B->D expert coverage; no training."""
import argparse, csv, json, statistics
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.sirm_q_followup.diagnose import digest, load_cache
from experiments.sirm_representation_h2.summarize import write_csv, describe, paired_bootstrap

DOMAINS=['art_painting','cartoon','photo','sketch']


def decomposition(risk):
    r=np.asarray(risk,dtype=np.float64)
    if r.ndim!=2 or not np.isfinite(r).all():raise ValueError('complete finite risk matrix required')
    mu=r.mean();alpha=r.mean(1)-mu;beta=r.mean(0)-mu
    interaction=r-mu-alpha[:,None]-beta[None,:]
    ss_domain=float(r.shape[1]*np.square(alpha).sum())
    ss_expert=float(r.shape[0]*np.square(beta).sum())
    ss_interaction=float(np.square(interaction).sum());total=float(np.square(r-mu).sum())
    assert np.allclose(interaction.mean(0),0,atol=1e-12)
    assert np.allclose(interaction.mean(1),0,atol=1e-12)
    assert np.isclose(ss_domain+ss_expert+ss_interaction,total)
    return dict(mu=mu,alpha=alpha,beta=beta,interaction=interaction,ss_total=total,
                ss_domain=ss_domain,ss_expert=ss_expert,ss_interaction=ss_interaction,
                expert_share=ss_expert/total if total else 0.,interaction_share=ss_interaction/total if total else 0.)


def gap(matrix,selected):
    # Equal weight for source rows; supports can differ by reference.
    r=np.broadcast_to(matrix,selected.shape)
    return float(np.mean([r[d,~selected[d]].mean()-r[d,selected[d]].mean() for d in range(len(r))]))


def target_means(risk,selected):
    return (float(np.mean([risk[s].mean() for s in selected])),
            float(np.mean([risk[~s].mean() for s in selected])))


def top2(risk):
    r=np.asarray(risk);m=np.zeros(r.shape,dtype=bool)
    for d in range(len(r)):m[d,np.argsort(r[d],kind='stable')[:2]]=True
    return m


def ce(logits,labels):
    n,m,c=logits.shape
    return F.cross_entropy(logits.double().reshape(-1,c),labels[:,None].expand(n,m).reshape(-1),reduction='none').reshape(n,m).numpy()


def summarize_rows(rows,group_keys,metrics):
    result=[]
    for group in sorted({tuple(r[k] for k in group_keys) for r in rows},key=str):
        rs=[r for r in rows if tuple(r[k] for k in group_keys)==group]
        for metric in metrics:
            vs=[r[metric] for r in rs if r.get(metric) is not None]
            if vs:result.append(dict(zip(group_keys,group),metric=metric,**describe(vs)))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/sirm_representation_h2/full');p.add_argument('--output',default='results/sirm_q_followup/interaction');args=p.parse_args()
    root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    cellrows=[];domainrows=[];runrows=[];targetrows=[];inventory=[];objects={}
    for path in sorted((root/'runs').glob('*/env*/seed*/config.json')):
        cfg=json.loads(path.read_text());key={k:cfg[k] for k in ['condition','target_env','seed']};run=path.parent;bank=Path(cfg['bank']);checkpoint=bank/'checkpoints/best.pkl'
        assert digest(checkpoint)==cfg['checkpoint_sha256']
        eff=json.loads((bank/'effective_config.json').read_text());split=json.loads((bank/'split_manifest.json').read_text());manifest=json.loads((bank/'manifest.json').read_text())
        assert str(cfg['target_env']) not in split['sources']
        qinfo=json.loads((run/'source_diagnostics.json').read_text())['q'];q=np.asarray(qinfo['q']);selected=q>0
        assert selected.shape==(3,6) and (selected.sum(1)==2).all()
        s=load_cache(run,bank,'source_out_original.pt',cfg);t=load_cache(run,bank,'target_all_original.pt',cfg)
        for i,name in enumerate(qinfo['source_domains']):
            physical=DOMAINS.index(name) if isinstance(name,str) else name
            assert s['physical_domains'][s['domains']==i].eq(physical).all()
        source_loss=ce(s['expert_logits'],s['labels']);target_loss=ce(t['expert_logits'],t['labels'])
        risk=np.stack([source_loss[s['domains'].numpy()==d].mean(0) for d in range(3)])
        target_risk=target_loss.mean(0)
        # Full-domain diagnostic and saved routed EMA are distinct estimators.
        checkpoint_payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        ema=checkpoint_payload['model_dict']['risk_ema.risk'].double().numpy();valid=checkpoint_payload['model_dict']['risk_ema.valid'].numpy();assert valid.all()
        del checkpoint_payload
        for estimator,matrix in [('source_out',risk),('risk_ema',ema)]:
            dec=decomposition(matrix);global_pair=np.argsort(matrix.mean(0),kind='stable')[:2];global_mask=np.zeros(6,dtype=bool);global_mask[global_pair]=True
            residual=dec['interaction'];raw_gap=gap(matrix,selected);expert_gap=gap(dec['beta'][None,:],selected);interaction_gap=gap(residual,selected)
            assert np.isclose(raw_gap,expert_gap+interaction_gap)
            jaccards=[]
            for d,physical in enumerate(qinfo['source_domains']):
                j=float(np.logical_and(selected[d],global_mask).sum()/np.logical_or(selected[d],global_mask).sum());jaccards.append(j)
                rsel=residual[d,selected[d]].mean();rnon=residual[d,~selected[d]].mean()
                domainrows.append(dict(**key,estimator=estimator,source_domain=physical,q_global_jaccard=j,global_pair=json.dumps(global_pair.tolist()),q_pair=json.dumps(np.flatnonzero(selected[d]).tolist()),interaction_selected=float(rsel),interaction_nonselected=float(rnon),interaction_gap=float(rnon-rsel)))
                for m in range(6):cellrows.append(dict(**key,estimator=estimator,source_domain=physical,expert=m,q=float(q[d,m]),selected=bool(selected[d,m]),risk=float(matrix[d,m]),domain_effect=float(dec['alpha'][d]),expert_effect=float(dec['beta'][m]),interaction=float(residual[d,m]),selection_frequency=int(selected[:,m].sum())))
            runrows.append(dict(**key,estimator=estimator,raw_gap=raw_gap,expert_effect_gap=expert_gap,interaction_gap=interaction_gap,interaction_fraction_of_gap=interaction_gap/raw_gap if abs(raw_gap)>1e-12 else None,
                interaction_selected=float(residual[selected].mean()),interaction_nonselected=float(residual[~selected].mean()),global_jaccard=float(np.mean(jaccards)),exact_global_rows=int(np.all(selected==global_mask,axis=1).sum()),expert_share=dec['expert_share'],interaction_share=dec['interaction_share'],ss_total=dec['ss_total'],ss_domain=dec['ss_domain'],ss_expert=dec['ss_expert'],ss_interaction=dec['ss_interaction']))
            trq,trnon=target_means(target_risk,selected);tgq,tgnon=target_means(target_risk,np.tile(global_mask,(3,1)))
            targetrows.append(dict(**key,estimator=estimator,target_ce_q=trq,target_ce_nonq=trnon,target_gap=trnon-trq,target_ce_global_top2=tgq,target_ce_other=tgnon,global_target_gap=tgnon-tgq))
        native=t.get('router_native',t['router_topk']).double().numpy();dense=t['router_dense'].double().numpy()
        assert np.allclose(native.sum(1),1,atol=1e-6) and np.allclose(dense.sum(1),1,atol=1e-6)
        if cfg['condition'] in ['B','D']:
            objects[(cfg['condition'],cfg['target_env'],cfg['seed'])]=dict(q=q,selected=selected,target_risk=target_risk,source_risk=risk,loss=target_loss,native=native,dense=dense,logits=t['expert_logits'].numpy(),labels=t['labels'].numpy(),ids=t['image_ids'],manifest=manifest,checkpoint_sha256=cfg['checkpoint_sha256'],selected_step=cfg['selected_step'],sources=qinfo['source_domains'])
        inventory.append(dict(**key,bank=str(bank),checkpoint_sha256=cfg['checkpoint_sha256'],split_hash=cfg['split_hash'],source_ids_sha256=__import__('hashlib').sha256(json.dumps(s['image_ids']).encode()).hexdigest(),target_ids_sha256=__import__('hashlib').sha256(json.dumps(t['image_ids']).encode()).hexdigest()))
        print('interaction '+str(key),flush=True)
        del s,t
    assert len(inventory)==48
    write_csv(out/'competence_cells.csv',cellrows);write_csv(out/'q_global_by_domain.csv',domainrows);write_csv(out/'competence_runs.csv',runrows);write_csv(out/'target_absolute.csv',targetrows)
    expertrows=[];pairedrows=[];strata=[];counterfactual=[];audits=[];g_pairs=[]
    # Existing exact G measurements let us hold membership fixed and distinguish
    # changes in eligibility from changes in measured scalar violation.
    with Path('results/sirm_q_followup/diagnosis/irm_cells.csv').open() as f:
        g_lookup={(r['condition'],int(r['target_env']),int(r['seed']),r['source_domain'],int(r['expert'])):r for r in csv.DictReader(f)}
    for target in range(4):
        for seed in range(3):
            b=objects['B',target,seed];d=objects['D',target,seed]
            for k in ['initialization_sha256','split_hash','schedule_sha256','parameters_total','pretrained_sha256']:assert b['manifest'][k]==d['manifest'][k],(target,seed,k)
            assert b['ids']==d['ids'] and np.array_equal(b['labels'],d['labels']) and b['sources']==d['sources']
            basekey=dict(target_env=target,seed=seed);sb=b['selected'].sum(0);sd=d['selected'].sum(0)
            for ref,bmask,dmask in [('own_Q',b['selected'],d['selected']),('fixed_B_Q',b['selected'],b['selected']),('fixed_D_Q',d['selected'],d['selected'])]:
                bq,bn=target_means(b['target_risk'],bmask);dq,dn=target_means(d['target_risk'],dmask)
                pairedrows.append(dict(**basekey,reference=ref,B_target_ce_q=bq,B_target_ce_nonq=bn,D_target_ce_q=dq,D_target_ce_nonq=dn,delta_q=dq-bq,delta_nonq=dn-bn,delta_gap=(dn-dq)-(bn-bq),q_jaccard=float(np.mean([np.logical_and(b['selected'][i],d['selected'][i]).sum()/np.logical_or(b['selected'][i],d['selected'][i]).sum() for i in range(3)]))))
            # Exact symmetric decomposition of routing-weighted *expert* CE.
            quality=.5*(b['native']+d['native'])*(d['loss']-b['loss'])
            routing=.5*(b['loss']+d['loss'])*(d['native']-b['native'])
            delta_selection=(d['native']*d['loss']-b['native']*b['loss']).mean(0)
            assert np.allclose(quality.mean(0)+routing.mean(0),delta_selection,atol=1e-10)
            for m in range(6):expertrows.append(dict(**basekey,expert=m,s_B=int(sb[m]),s_D=int(sd[m]),B_target_ce=float(b['target_risk'][m]),D_target_ce=float(d['target_risk'][m]),delta_target_ce=float(d['target_risk'][m]-b['target_risk'][m]),B_source_ce=float(b['source_risk'][:,m].mean()),D_source_ce=float(d['source_risk'][:,m].mean()),B_target_native_mass=float(b['native'][:,m].mean()),D_target_native_mass=float(d['native'][:,m].mean()),B_target_dense_mass=float(b['dense'][:,m].mean()),D_target_dense_mass=float(d['dense'][:,m].mean()),B_target_usage=float((b['native'][:,m]>0).mean()),D_target_usage=float((d['native'][:,m]>0).mean()),quality_contribution=float(quality[:,m].mean()),routing_contribution=float(routing[:,m].mean()),selection_ce_delta_contribution=float(delta_selection[m])))
            for reference,freq in [('B_Q',sb),('D_Q',sd)]:
                for frequency in range(4):
                    mask=freq==frequency
                    if not mask.any():continue
                    strata.append(dict(**basekey,reference=reference,frequency=frequency,expert_count=int(mask.sum()),B_target_ce=float(b['target_risk'][mask].mean()),D_target_ce=float(d['target_risk'][mask].mean()),delta_target_ce=float((d['target_risk'][mask]-b['target_risk'][mask]).mean()),B_native_mass=float(b['native'][:,mask].sum(1).mean()),D_native_mass=float(d['native'][:,mask].sum(1).mean()),B_dense_mass=float(b['dense'][:,mask].sum(1).mean()),D_dense_mass=float(d['dense'][:,mask].sum(1).mean()),quality_contribution=float(quality[:,mask].sum(1).mean()),routing_contribution=float(routing[:,mask].sum(1).mean()),selection_ce_delta_contribution=float(delta_selection[mask].sum())))
                for name,mask in [('low_0_1',freq<=1),('high_2_3',freq>=2)]:
                    if not mask.any():continue
                    strata.append(dict(**basekey,reference=reference,frequency=name,expert_count=int(mask.sum()),B_target_ce=float(b['target_risk'][mask].mean()),D_target_ce=float(d['target_risk'][mask].mean()),delta_target_ce=float((d['target_risk'][mask]-b['target_risk'][mask]).mean()),B_native_mass=float(b['native'][:,mask].sum(1).mean()),D_native_mass=float(d['native'][:,mask].sum(1).mean()),B_dense_mass=float(b['dense'][:,mask].sum(1).mean()),D_dense_mass=float(d['dense'][:,mask].sum(1).mean()),quality_contribution=float(quality[:,mask].sum(1).mean()),routing_contribution=float(routing[:,mask].sum(1).mean()),selection_ce_delta_contribution=float(delta_selection[mask].sum())))
            for logits_name,router_name,logits,pi in [('B','B',b['logits'],b['native']),('D','D',d['logits'],d['native']),('D','B',d['logits'],b['native']),('B','D',b['logits'],d['native'])]:
                mixed=torch.tensor((logits*pi[:,:,None]).sum(1),dtype=torch.float64);y=torch.tensor(b['labels'])
                if logits_name==router_name:
                    expected=json.loads((root/'runs'/logits_name/f'env{target}'/f'seed{seed}'/'metrics.json').read_text())['metrics']['acc_original_native']
                    assert np.isclose(float(mixed.argmax(1).eq(y).double().mean()),expected,atol=1e-7)
                counterfactual.append(dict(**basekey,logits=logits_name,router=router_name,accuracy=float(mixed.argmax(1).eq(y).double().mean()),nll=float(F.cross_entropy(mixed,y)),selection_ce=float((pi*(b['loss'] if logits_name=='B' else d['loss'])).sum(1).mean())))
            for reference,mask in [('B_Q',b['selected']),('D_Q',d['selected'])]:
                for selected in [True,False]:
                    vals=[]
                    for i,domain in enumerate(b['sources']):
                        for m in range(6):
                            if mask[i,m]!=selected:continue
                            physical=DOMAINS.index(domain) if isinstance(domain,str) else domain
                            gb=g_lookup['B',target,seed,str(physical),m];gd=g_lookup['D',target,seed,str(physical),m]
                            if gb['valid_routed']=='True' and gd['valid_routed']=='True':vals.append((float(gb['G_routed']),float(gd['G_routed'])))
                    if vals:g_pairs.append(dict(**basekey,reference=reference,selected=selected,matched_valid_cells=len(vals),B_G=float(np.mean([v[0] for v in vals])),D_G=float(np.mean([v[1] for v in vals])),delta_G=float(np.mean([v[1]-v[0] for v in vals]))))
            audits.append(dict(**basekey,initialization_split_schedule_capacity_matched=True,target_ids_labels_order_matched=True,B_step=b['selected_step'],D_step=d['selected_step'],B_checkpoint=b['checkpoint_sha256'],D_checkpoint=d['checkpoint_sha256']))
    write_csv(out/'paired_target_absolute.csv',pairedrows);write_csv(out/'paired_experts.csv',expertrows);write_csv(out/'frequency_runs.csv',strata);write_csv(out/'router_swap.csv',counterfactual);write_csv(out/'paired_irm_fixed_membership.csv',g_pairs)
    aggregates=summarize_rows(runrows,['condition','estimator'],['raw_gap','expert_effect_gap','interaction_gap','interaction_fraction_of_gap','global_jaccard','expert_share','interaction_share'])
    write_csv(out/'competence_aggregate.csv',aggregates)
    write_csv(out/'target_aggregate.csv',summarize_rows(targetrows,['condition','estimator'],['target_ce_q','target_ce_nonq','target_gap','target_ce_global_top2','global_target_gap']))
    write_csv(out/'frequency_aggregate.csv',summarize_rows(strata,['reference','frequency'],['delta_target_ce','B_target_ce','D_target_ce','B_native_mass','D_native_mass','B_dense_mass','D_dense_mass','quality_contribution','routing_contribution','selection_ce_delta_contribution']))
    write_csv(out/'frequency_by_target.csv',summarize_rows(strata,['target_env','reference','frequency'],['delta_target_ce','B_native_mass','D_native_mass','quality_contribution','routing_contribution']))
    write_csv(out/'selection_transitions.csv',summarize_rows(expertrows,['s_B','s_D'],['delta_target_ce','B_target_native_mass','D_target_native_mass']))
    interaction_effects=[]
    for condition in 'ABCD':
        for estimator in ['source_out','risk_ema']:
            part=[r for r in runrows if r['condition']==condition and r['estimator']==estimator]
            for metric in ['interaction_gap','global_jaccard']:
                values=[r[metric] for r in part]
                interaction_effects.append(dict(condition=condition,estimator=estimator,metric=metric,**describe(values),**paired_bootstrap(values,[(r['target_env'],r['seed']) for r in part])))
    write_csv(out/'interaction_uncertainty.csv',interaction_effects)
    frequency_contrasts=[]
    for ref in ['B_Q','D_Q']:
        lo={(r['target_env'],r['seed']):r for r in strata if r['reference']==ref and r['frequency']=='low_0_1'}
        hi={(r['target_env'],r['seed']):r for r in strata if r['reference']==ref and r['frequency']=='high_2_3'}
        units=sorted(set(lo)&set(hi));values=[lo[u]['delta_target_ce']-hi[u]['delta_target_ce'] for u in units]
        frequency_contrasts.append(dict(reference=ref,metric='delta_low_minus_high',**describe(values),**paired_bootstrap(values,units)))
    write_csv(out/'frequency_contrasts.csv',frequency_contrasts)
    paired_summary=[]
    for reference in ['own_Q','fixed_B_Q','fixed_D_Q']:
        part=[r for r in pairedrows if r['reference']==reference]
        for metric in ['delta_q','delta_nonq','delta_gap']:
            vals=[r[metric] for r in part];units=[(r['target_env'],r['seed']) for r in part]
            paired_summary.append(dict(reference=reference,metric=metric,**describe(vals),**paired_bootstrap(vals,units)))
    write_csv(out/'paired_target_effects.csv',paired_summary)
    write_csv(out/'paired_target_by_target.csv',summarize_rows(pairedrows,['target_env','reference'],['B_target_ce_q','D_target_ce_q','B_target_ce_nonq','D_target_ce_nonq','delta_q','delta_nonq']))
    def mean(rows,key):return statistics.mean(r[key] for r in rows)
    bprimary=[r for r in runrows if r['condition']=='B' and r['estimator']=='source_out'];dprimary=[r for r in runrows if r['condition']=='D' and r['estimator']=='source_out']
    fixed=[r for r in pairedrows if r['reference']=='fixed_B_Q']
    lines=['# Global expert quality / subset interaction / coverage','', 'Completed four post-hoc analyses on 48 source-selected factorial checkpoints; paired B→D has 4 targets × 3 seeds. Original deterministic view, full target domain. CPU only; no fitting, retraining or target-selected settings.', '', '## Findings', '',
        f'Global expert quality dominates the source-out Q advantage: interaction contributes {100*mean(bprimary,"interaction_gap")/mean(bprimary,"raw_gap"):.2f}% of the aggregate B gap and {100*mean(dprimary,"interaction_gap")/mean(dprimary,"raw_gap"):.2f}% of the D gap. The residual advantage is small, not identically zero. Q is not equivalent to global Top-2 (B/D Jaccard {mean(bprimary,"global_jaccard"):.3f}/{mean(dprimary,"global_jaccard"):.3f}).', '',
        f'The predicted selective-sacrifice pattern is not supported overall. Fixed B membership gives Δ Q target CE {mean(fixed,"delta_q"):+.5f}, Δ non-Q {mean(fixed,"delta_nonq"):+.5f}; both worsen. Low-frequency experts do not deteriorate more than high-frequency experts on average. Sketch has a larger low-frequency increase, so consult per-target results rather than treating the pooled result as uniform.', '',
        'Cached routing swaps place most of the observed accuracy difference on the expert-logit bank: D probabilities on B logits preserve B accuracy, while B probabilities on D logits do not recover it. This supports broad prediction degradation, not a complete causal mechanism. Interaction-aware soft Q remains an architectural hypothesis to test, not a validated fix from these diagnostics.', '',
        '## 1. Remove global expert quality','', 'R = grand mean + domain effect + expert effect + double-centered interaction. Positive gap means non-Q minus Q risk, so selected is better. Rows and experts have equal weight; source-out CE is primary. Saved routed risk EMA is a separate estimator. SS shares describe this finite matrix, not a population ANOVA or causal mechanisms.', '', '| Condition | Raw Q gap | Expert effect gap | Interaction gap | Expert SS share | Interaction SS share | Q/global Top-2 Jaccard |','|---|---:|---:|---:|---:|---:|---:|']
    for c in 'ABCD':
        rs=[r for r in runrows if r['condition']==c and r['estimator']=='source_out'];lines.append('| '+c+' | '+' | '.join(f'{mean(rs,k):.5f}' for k in ['raw_gap','expert_effect_gap','interaction_gap','expert_share','interaction_share','global_jaccard'])+' |')
    lines.extend(['', 'Estimator sensitivity: B/D saved risk-EMA interaction gaps are '+ '/'.join(f'{mean([r for r in runrows if r["condition"]==c and r["estimator"]=="risk_ema"],"interaction_gap"):.5f}' for c in 'BD') + ', and EMA-based global Jaccards are '+ '/'.join(f'{mean([r for r in runrows if r["condition"]==c and r["estimator"]=="risk_ema"],"global_jaccard"):.5f}' for c in 'BD') + '. EMA constructed Q, so its residual advantage is not independent validation. Source-out was also used for checkpoint selection; it is not an untouched diagnostic holdout. Observed domain/expert interactions can reflect class mix or calibration rather than shared mechanisms.'])
    lines.extend(['','## 2. Absolute target losses and matched groups','', '| Q reference | B Q CE | D Q CE | B non-Q CE | D non-Q CE | Δ Q | Δ non-Q |','|---|---:|---:|---:|---:|---:|---:|'])
    for ref in ['own_Q','fixed_B_Q','fixed_D_Q']:
        rs=[r for r in pairedrows if r['reference']==ref];lines.append('| '+ref+' | '+' | '.join(f'{mean(rs,k):.5f}' for k in ['B_target_ce_q','D_target_ce_q','B_target_ce_nonq','D_target_ce_nonq','delta_q','delta_nonq'])+' |')
    lines.extend(['','Each Q/non-Q loss is an equal mean over three source-row masks, then over runs; an expert can appear in multiple rows. own_Q uses each model’s own assignment and can change composition. fixed_B_Q/fixed_D_Q hold cell membership constant across models. Expert IDs share initialization; this does not prove their learned roles stay aligned. Source-selected steps may differ.', '', '## 3. Expert frequency and target routing','', '| Frequency reference | Group | Δ target expert CE | B native mass | D native mass |','|---|---|---:|---:|---:|'])
    for ref in ['B_Q','D_Q']:
        for freq in [0,1,2,3,'low_0_1','high_2_3']:
            rs=[r for r in strata if r['reference']==ref and r['frequency']==freq]
            if rs:lines.append(f'| {ref} | {freq} (n={len(rs)} runs) | '+' | '.join(f'{mean(rs,k):.5f}' for k in ['delta_target_ce','B_native_mass','D_native_mass'])+' |')
    lines.extend(['','CE averages experts within each present run/group; native mass sums expert probabilities within that group. Empty groups are omitted and n is explicit. Dense mass is exported separately: it is not actual Top-2 prediction mass. Frequency means selected-domain count at the best checkpoint, not cumulative selection frequency during training.', '', '## 4. Fixed-bank router swap diagnostic','', '| Expert logits | Router probabilities | Target accuracy | Target NLL | Selection CE |','|---|---|---:|---:|---:|'])
    for l,r in [('B','B'),('D','D'),('D','B'),('B','D')]:
        rs=[x for x in counterfactual if x['logits']==l and x['router']==r];lines.append(f'| {l} | {r} | {mean(rs,"accuracy")*100:.3f}% | {mean(rs,"nll"):.5f} | {mean(rs,"selection_ce"):.5f} |')
    lines.extend(['','These swaps reuse cached logits and routing on the same image IDs, without fitting. They test output composition, not retraining or causal mediation. The exact symmetric quality/routing decomposition in frequency CSVs applies to routing-weighted expert CE, not mixture-logit CE or accuracy. Single-expert loss increases do not alone imply the target needs that expert.', '', 'Matched-valid routed-G comparisons under fixed B/D membership are in paired_irm_fixed_membership.csv. This avoids interpreting changing Q or ESS support as a fixed-cell change; still, full-domain scalar stationarity is not the minibatch training penalty. No early checkpoints are available.', '', 'Per-target effects, group counts, raw matrices, estimator sensitivity and paired uncertainty are exported. Three training seeds limit uncertainty precision; do not treat cells/experts as independent training replicates. The pilot remains a separate experiment; no intervention was selected from target results.'])
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(runs=inventory,paired_audits=audits,code_sha256=digest(Path(__file__)),target_used_for_selection=False,model_updates=False),indent=2))
    print('Completed '+str(out),flush=True)

if __name__=='__main__':main()

"""Research metric log from locked reports; missing metrics remain blank."""
import csv,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SUITE=ROOT/'results/terra_10pct_factor_diagnosis';OUT=ROOT/'results/sirm_research_loop'
FIELDS=['iteration','method','hypothesis','source_val','target_acc','global_only_acc','residual_target_delta','interaction_advantage','residual_cosine','SIRM_G','SIRM_active_fraction','grad_ratio_mean','shrinkage_control_result','status','target','protocol','selected_step','active_steps_before_selected','source_report','checkpoint_sha256']
def update():
    records=[];locks={}
    for p in sorted(SUITE.glob('runs/*/*/posthoc/complete.json')):
        d=json.loads(p.read_text());m=d['method'];source=d['splits']['source_out'];target=d['splits']['target_in'];mech=d.get('mechanism',{})
        r={k:None for k in FIELDS};r.update(iteration=0,method={'ERM_H2':'B0 ERM-H2','OFF':'B1 OFF','ON':'B2 ON lambda1'}[m],hypothesis='locked fast-protocol baseline',source_val=source['source_mean'],target_acc=target['source_mean'],global_only_acc=target.get('global_mean'),residual_target_delta=target['source_mean']-target['global_mean'] if 'global_mean' in target else None,interaction_advantage=source.get('interaction_advantage'),residual_cosine=target.get('cosine',{}).get('pair_mean'),SIRM_active_fraction=mech.get('active_fraction'),grad_ratio_mean=mech.get('gradient_ratio',{}).get('mean'),shrinkage_control_result='pending' if m=='ON' else 'not applicable',status='LOCKED BASELINE',target=d['target'],protocol='10pct_seed0_501steps',selected_step=d['selected_step'],source_report=str(p.relative_to(ROOT)),checkpoint_sha256=d['checkpoint_sha256'])
        if m!='ERM_H2':
            logs=[json.loads(l) for l in (p.parents[1]/'diagnostics.jsonl').read_text().splitlines()];r['active_steps_before_selected']=sum(x.get('sirm_active',False) for x in logs if x['step']<=d['selected_step'])
        records.append(r);locks[f'{d["target"]}/{m}']=dict(source_report=r['source_report'],checkpoint_sha256=d['checkpoint_sha256'],selected_step=d['selected_step'])
    shrink=OUT/'iteration_01/shrinkage_results.json'
    if shrink.exists():
        x=json.loads(shrink.read_text());ds=x['splits']['source_out'];dt=x['splits']['target_in'];b=x['baseline']['OFF'];v=dt['methods']['OFF_source_matched'];r={k:None for k in FIELDS};r.update(iteration=1,method='OFF source-RMS scaling',hypothesis='uniform shrinkage reproduces ON benefit',source_val=ds['methods']['OFF_source_matched']['accuracy'],target_acc=v['accuracy'],global_only_acc=v['global_accuracy'],residual_target_delta=v['residual_delta'],residual_cosine=b['splits']['target_in']['cosine']['pair_mean'],SIRM_G=ds['methods']['OFF_source_matched']['G_common_executed'],SIRM_active_fraction=0,shrinkage_control_result=x['reason'],status=x['status'],target='L38',protocol='10pct_seed0_501steps',selected_step=b['selected_step'],source_report=str(shrink.relative_to(ROOT)),checkpoint_sha256=b['checkpoint_sha256']);records.append(r)
        for r in records:
            if r['target']=='L38' and r['method']=='B2 ON lambda1':r['shrinkage_control_result']=x['reason'];r['SIRM_G']=ds['methods']['ON']['G_common_executed']
            if r['target']=='L38' and r['method']=='B1 OFF':r['SIRM_G']=ds['methods']['OFF']['G_common_executed']
    # New protocol uses a different common anchor and must be a separate block.
    candidate=OUT/'iteration_03/experiment'
    control_path=OUT/'iteration_03/shrinkage_control/shrinkage_results.json'
    control=json.loads(control_path.read_text()) if control_path.exists() else None
    verdict_path=OUT/'iteration_03/scientific_verdict.json'
    verdict=json.loads(verdict_path.read_text()) if verdict_path.exists() else None
    for p in sorted(candidate.glob('runs/L38/*/posthoc/complete.json')):
        d=json.loads(p.read_text());m=d['method'];s=d['splits']['source_out'];t=d['splits']['target_in'];mech=d.get('mechanism',{})
        row={k:None for k in FIELDS}
        row.update(iteration=3,method={'ERM_H2':'B0 frozen full-budget ERM','OFF':'B1 frozen-ERM400 OFF','ON':'B2 frozen-ERM400 ON'}[m],hypothesis='complementary factors on a common fully trained frozen ERM',source_val=s['source_mean'],target_acc=t['source_mean'],global_only_acc=t.get('global_mean',t['source_mean'] if m=='ERM_H2' else None),residual_target_delta=t['source_mean']-t['global_mean'] if m!='ERM_H2' else 0.,interaction_advantage=s.get('interaction_advantage'),residual_cosine=t.get('cosine',{}).get('pair_mean'),SIRM_active_fraction=mech.get('active_fraction',0.),grad_ratio_mean=mech.get('gradient_ratio',{}).get('mean'),shrinkage_control_result='pending' if m!='ERM_H2' else 'not applicable',status='PENDING SCIENTIFIC ANALYSIS',target='L38',protocol='frozen_ERM_endpoint500_stage2_400',selected_step=d['selected_step'],source_report=str(p.relative_to(ROOT)),checkpoint_sha256=d['checkpoint_sha256'])
        if m!='ERM_H2':
            logs=[json.loads(l) for l in (p.parents[1]/'diagnostics.jsonl').read_text().splitlines()];row['active_steps_before_selected']=sum(x.get('sirm_active',False) for x in logs if x['step']<=d['selected_step'])
            if control:
                row['SIRM_G']=control['splits']['source_out']['methods'][m]['G_common_executed'];row['shrinkage_control_result']=control['reason']
        if verdict:row['status']=verdict['status'] if m!='ERM_H2' else 'LOCKED FULL-BUDGET ENDPOINT; NOT SOURCE-BEST'
        records.append(row)
    if control:
        s=control['splits']['source_out']['methods']['OFF_source_matched'];t=control['splits']['target_in']['methods']['OFF_source_matched'];b=control['baseline']['OFF'];row={k:None for k in FIELDS}
        row.update(iteration=3,method='B1 source-RMS scaling control',hypothesis='uniform magnitude control for B2−B1',source_val=s['accuracy'],target_acc=t['accuracy'],global_only_acc=t['global_accuracy'],residual_target_delta=t['residual_delta'],residual_cosine=t['residual_cosine_mean'],SIRM_G=s['G_common_executed'],SIRM_active_fraction=0.,shrinkage_control_result=control['reason'],status=control['status'],target='L38',protocol='frozen_ERM_endpoint500_stage2_400',selected_step=b['selected_step'],source_report=str(control_path.relative_to(ROOT)),checkpoint_sha256=b['checkpoint_sha256']);records.append(row)
    for iteration in [4,5,7,8]:
        directory=OUT/f'iteration_{iteration:02d}'
        verdict_path=directory/'scientific_verdict.json'
        verdict=json.loads(verdict_path.read_text()) if verdict_path.exists() else None
        control_path=directory/'shrinkage_control/shrinkage_results.json'
        control=json.loads(control_path.read_text()) if control_path.exists() else None
        for p in sorted((directory/'experiment').glob('runs/L38/*/posthoc/complete.json')):
            d=json.loads(p.read_text());m=d['method'];s=d['splits']['source_out'];t=d['splits']['target_in'];mech=d.get('mechanism',{})
            row={k:None for k in FIELDS}
            row.update(iteration=iteration,method={'ERM_H2':'B0 frozen source-selected ERM','OFF':'B1 frozen-ERM400 OFF','ON':'B2 frozen-ERM400 ON'}[m],hypothesis={4:'correct source-selected common anchor',5:'router distillation weight0.1 to1',7:'differentiable raw-expert centering',8:'hard-min discovery with symmetric ties and original soft balance'}[iteration],source_val=s['source_mean'],target_acc=t['source_mean'],global_only_acc=t.get('global_mean',t['source_mean'] if m=='ERM_H2' else None),residual_target_delta=t['source_mean']-t['global_mean'] if m!='ERM_H2' else 0.,interaction_advantage=s.get('interaction_advantage'),residual_cosine=t.get('cosine',{}).get('pair_mean'),SIRM_active_fraction=mech.get('active_fraction',0.),grad_ratio_mean=mech.get('gradient_ratio',{}).get('mean'),shrinkage_control_result='pending' if m!='ERM_H2' else 'not applicable',status=(verdict['status'] if verdict else 'PENDING SCIENTIFIC ANALYSIS') if m!='ERM_H2' else 'LOCKED SOURCE-BEST FROM COMPLETED ERM RUN',target='L38',protocol=f'frozen_ERM_sourcebest375_stage2_400_router{1.0 if iteration==5 else 0.1}'+('_centered'+('_hard_discovery' if iteration==8 else '') if iteration in [7,8] else ''),selected_step=d['selected_step'],source_report=str(p.relative_to(ROOT)),checkpoint_sha256=d['checkpoint_sha256'])
            if m!='ERM_H2':
                logs=[json.loads(l) for l in (p.parents[1]/'diagnostics.jsonl').read_text().splitlines()];row['active_steps_before_selected']=sum(x.get('sirm_active',False) for x in logs if x['step']<=d['selected_step'])
                if control:
                    row['SIRM_G']=control['splits']['source_out']['methods'][m]['G_common_executed'];row['shrinkage_control_result']=control['reason']
            records.append(row)
    OUT.mkdir(parents=True,exist_ok=True);tmp=OUT/'leaderboard.csv.tmp'
    with tmp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(records)
    tmp.replace(OUT/'leaderboard.csv');(OUT/'baseline_locks.json').write_text(json.dumps(locks,indent=2)+'\n')
    return len(records)
if __name__=='__main__':print('Logged',update(),'locked baseline/control rows')

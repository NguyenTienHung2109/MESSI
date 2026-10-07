"""Locked diagnostic pipeline following the corrected source-selected pair."""
import json,time,os
from pathlib import Path
from .class_balanced_run import DEST,EXPERIMENT,ANCHOR,ROOT,PRIOR
from .router_weight import live,write
from experiments.terra_10pct_factor_diagnosis.run import digest

def main():
    files=[Path(__file__),Path(__file__).with_name('shrinkage.py'),Path(__file__).with_name('case_eda.py'),Path(__file__).with_name('bias_control.py'),Path(__file__).with_name('spatial_bank_replay.py'),Path(__file__).with_name('spatial_bank_model.py'),Path(__file__).with_name('class_balanced_model.py')]
    plan=dict(code_hashes={str(p.relative_to(ROOT)):digest(p) for p in files},destination=str(DEST),experiment=str(EXPERIMENT),anchor=str(ANCHOR),anchor_sha256=digest(ANCHOR),runtime_scope='reuse identical diagnostic functions with this explicit experiment/anchor destination; no training patches')
    if (DEST/'analysis_plan.json').exists():assert json.loads((DEST/'analysis_plan.json').read_text())==plan
    else:
        write(DEST/'analysis_plan.json',plan)
        for p in files:
            dst=DEST/'analysis_code_snapshot'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    def check():
        assert all(digest(ROOT/p)==h for p,h in plan['code_hashes'].items()) and digest(ANCHOR)==plan['anchor_sha256']
    pid=json.loads((DEST/'worker_pid.json').read_text())['pid']
    while live(pid):
        check();write(DEST/'analysis_progress.json',dict(status='waiting_for_paired_training',worker_pid=pid,verified_live=True,observed_unix=time.time()));time.sleep(20)
    assert json.loads((DEST/'progress.json').read_text())['status']=='training_and_locked_posthoc_complete';check()
    write(DEST/'analysis_progress.json',dict(status='source_calibrated_shrinkage'))
    from .spatial_bank_replay import install
    install()
    from .shrinkage import run
    run(suite=EXPERIMENT,dest=DEST/'shrinkage_control');check()
    write(DEST/'analysis_progress.json',dict(status='matched_case_replay'))
    from . import case_eda
    case_eda.DEST=DEST;case_eda.EXPERIMENT=EXPERIMENT;case_eda.ANCHOR=ANCHOR;case_eda.run();check()
    write(DEST/'analysis_progress.json',dict(status='source_bias_and_utility_controls'))
    from . import bias_control
    bias_control.DEST=DEST;bias_control.EXPERIMENT=EXPERIMENT;bias_control.run();check()
    write(DEST/'analysis_progress.json',dict(status='matched_original_weighting_case_replay'))
    control=DEST/'matched_weighting_control';suite=control/'experiment';banks=suite/'runs/L38';banks.mkdir(parents=True,exist_ok=True)
    import shutil
    shutil.copyfile(EXPERIMENT/'subset_manifest.json',suite/'subset_manifest.json')
    for label,arm in [('OFF','OFF'),('ON','OFF'),('ERM_H2','ERM_H2')]:
        link=banks/label;target=(PRIOR if label=='OFF' else DEST)/'experiment/runs/L38'/arm
        if link.is_symlink():assert link.resolve()==target.resolve()
        else:link.symlink_to(target,target_is_directory=True)
    write(control/'arm_mapping.json',dict(B0='frozen ERM',B1='iteration13 original class-image-weighted spatial OFF',B2='iteration14 equal-class norm-matched spatial OFF',SIRM_comparison=False,target_used_for_selection=False))
    case_eda.DEST=control;case_eda.EXPERIMENT=suite;case_eda.run();check()
    write(DEST/'analysis_progress.json',dict(status='diagnostics_complete',remaining='visual EDA review; exact required report; scientific verdict; audit and next hypothesis'))

if __name__=='__main__':
    write(DEST/'analysis_worker_pid.json',dict(pid=os.getpid(),started_unix=time.time(),module='experiments.sirm_research_loop.class_balanced_posthoc'))
    try:main()
    except Exception as exc:write(DEST/'analysis_progress.json',dict(status='failed',error=str(exc)));raise

"""Locked class-weight intervention from completed source EDA and finite controls."""
import copy,json,os,shutil,subprocess,time
from pathlib import Path
from experiments.terra_10pct_factor_diagnosis.run import ROOT,MANIFEST,command,digest,write,hashes
from .frozen_erm_audit import audit_anchor,audit_pair
from .router_weight import live

DEST=ROOT/'results/sirm_research_loop/iteration_14';EXPERIMENT=DEST/'experiment';PRIOR=ROOT/'results/sirm_research_loop/iteration_13';ANCHOR=ROOT/'results/terra_10pct_factor_diagnosis/runs/L38/ERM_H2/source_best.pkl'
ARMS=['OFF','ON']

def read(p):return json.loads(p.read_text())

def configuration(arm,bank):
    old=read(PRIOR/'plan.json');hp=copy.deepcopy(old['hparams']['ON' if arm=='ON' else 'OFF']);hp['sirm_output_dir']=str(bank);c=hp['sirm_discover_config']
    c.update(protocol='sirm_research_iter14_equal_class_norm_match',discovery_class_weighting='equal_class_original_gradient_norm')
    return hp


def invocation(arm,bank):
    cmd=command(1,'ON' if arm=='ON' else 'OFF',bank);cmd[cmd.index('domainbed.scripts.train')]='experiments.sirm_research_loop.class_balanced_train';cmd[cmd.index('--steps')+1]='400';cmd[cmd.index('--hparams')+1]=json.dumps(configuration(arm,bank));return cmd


def main():
    prior=read(PRIOR/'final_analysis_audit.json');assert prior['analysis_complete'] and not prior['scientific_goal_achieved'];assert read(PRIOR/'source_class_allocation/completion_audit.json')['complete'];assert read(DEST/'prelaunch_integration_audit.json')['passed'];validation=read(DEST/'validation.json');assert validation['synthetic_tests_passed'];assert all(digest(ROOT/f)==h for f,h in validation['code_sha256'].items());assert (DEST/'preregistration.md').exists()
    old=read(PRIOR/'plan.json');assert digest(ANCHOR)==old['anchor_sha256'] and digest(MANIFEST)==old['manifest_sha256'];assert all(digest(ROOT/p)==h for p,h in old['base_code_hashes'].items())
    if (PRIOR/'worker_pid.json').exists():assert not live(read(PRIOR/'worker_pid.json')['pid'])
    assert (ANCHOR.parent/'done').exists() and not read(ANCHOR.parent/'selection.json')['target_used'];assert [json.loads(l)['step'] for l in (ANCHOR.parent/'diagnostics.jsonl').read_text().splitlines()]==list(range(501))
    variant=['class_balanced_run.py','class_balanced_model.py','class_balanced_train.py','class_balanced_prelaunch.py','test_class_balanced.py','spatial_bank_run.py','spatial_bank_train.py','spatial_bank_model.py','spatial_bank_replay.py','spatial_bank_prelaunch.py','test_spatial_bank.py','class_coherent_model.py','class_coherent_discovery.py','hard_router_cap_model.py','hard_discovery.py','centered_model.py','centered_train.py','anchor_train.py','frozen_erm_audit.py','protocol_audit.py']
    plan=dict(anchor_sha256=digest(ANCHOR),anchor_completed_updates=501,anchor_selected_step=375,manifest_sha256=digest(MANIFEST),feature_cache_sha256=digest(PRIOR/'source_feature_cache.pt'),base_code_hashes=hashes(),variant_code_hashes={str(Path(__file__).with_name(f).relative_to(ROOT)):digest(Path(__file__).with_name(f)) for f in variant},hparams={a:configuration(a,EXPERIMENT/'runs/L38'/a) for a in ARMS},arm_order=ARMS,source_hypothesis_evidence_sha256=digest(PRIOR/'source_class_allocation/interpretation.md'),hypothesis='Outer class weighting changes conflicting discovery allocation. Equal-class direction with detached original residual gradient norm may improve useful source corrections and replicated shared utility. Source evidence only, no target selection.',changed='Only outer discovery class weighting and its detached raw residual gradient norm match change; all spatial architecture, class winners, original balance, frozen anchor, router cap, centering, Q/ESS, lambda and400 schedule retained.',target=1,seed=0,training_target_reads=False)
    allowed={'protocol','discovery_class_weighting'}
    for arm in ARMS:
        before=old['hparams']['ON' if arm=='ON' else 'OFF']['sirm_discover_config'];after=plan['hparams'][arm]['sirm_discover_config'];assert {k:v for k,v in before.items() if k not in allowed}=={k:v for k,v in after.items() if k not in allowed}
    if (DEST/'plan.json').exists():assert read(DEST/'plan.json')==plan
    else:
        write(DEST/'plan.json',plan)
        for f in {**plan['base_code_hashes'],**plan['variant_code_hashes']}:
            target=DEST/'code_snapshot'/f;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/f).read_bytes())
    def check():
        assert all(digest(ROOT/f)==h for f,h in {**plan['base_code_hashes'],**plan['variant_code_hashes']}.items());assert digest(ANCHOR)==plan['anchor_sha256'] and digest(MANIFEST)==plan['manifest_sha256'];assert digest(PRIOR/'source_feature_cache.pt')==plan['feature_cache_sha256']
    manifest=read(MANIFEST);write(EXPERIMENT/'subset_manifest.json',manifest);trio=EXPERIMENT/'runs/L38';base=trio/'ERM_H2';(base/'posthoc').mkdir(parents=True,exist_ok=True);shutil.copyfile(ANCHOR,base/'source_best.pkl');shutil.copyfile(ANCHOR.parent/'posthoc/complete.json',base/'posthoc/complete.json');write(base/'posthoc/selection_lock.json',dict(checkpoint_sha256=digest(ANCHOR),selected_step=375,completed_training_updates=501,retraining=False,target_used_for_selection=False))
    env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    for arm in ARMS:
        check();bank=trio/arm;bank.mkdir(parents=True,exist_ok=True)
        if not (bank/'done').exists():
            assert not (bank/'diagnostics.jsonl').exists(),'refuse overwriting partial run'
            with (bank/'console.log').open('w') as log:
                proc=subprocess.Popen(invocation(arm,bank),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    rows=bank/'diagnostics.jsonl';step=None
                    if rows.exists():
                        for line in reversed(rows.read_text().splitlines()[-2:]):
                            try:step=json.loads(line)['step'];break
                            except ValueError:pass
                    write(DEST/'progress.json',dict(status='training',arm=arm,pid=proc.pid,step=step,observed_unix=time.time()));time.sleep(10)
                assert proc.returncode==0,f'{arm} failed; inspect console.log'
        from experiments.terra_10pct_factor_diagnosis.report import audit_training
        audit_training(bank,1);audit_anchor(bank,ANCHOR)
        write(DEST/'progress.json',dict(status='arm_training_complete',arm=arm))
    audit_pair(trio)
    # Verify matched-source control schedule. Parameter equality at initialization
    # is separately hashed by the model adapter, rather than including different
    # mode-specific normalization buffers in the control hash.
    initial=[read(trio/a/'spatial_initialization.json') for a in ARMS]
    assert len({v['parameter_sha256'] for v in initial})==1
    assert initial[0]['parameter_sha256']==read(PRIOR/'experiment/runs/L38/OFF/spatial_initialization.json')['parameter_sha256']
    assert all(v['bank_parameters']==1201987 and v['bank_dimension']==768 and v['global_dimension']==384 for v in initial)
    write(DEST/'matched_initialization_audit.json',dict(passed=True,arms=ARMS,parameters_equal=True,rows=initial,scope='Named parameters only; feature-mode normalization buffers intentionally differ. Spatial OFF/ON full initialization is checked by audit_pair.'))
    schedules=[read(trio/a/'source_schedules.json') for a in ARMS];assert all(v==schedules[0] for v in schedules)
    assert schedules[0]==read(PRIOR/'experiment/runs/L38/OFF/source_schedules.json')
    write(DEST/'original_control_reuse_audit.json',dict(passed=True,source_only=True,control_directory=str(PRIOR/'experiment/runs/L38'),initial_parameters_exact=True,schedules_exact=True,unchanged_configuration_except=sorted(allowed),checkpoint_hashes={a:{f:digest(PRIOR/'experiment/runs/L38'/a/f) for f in ['source_best.pkl','last.pkl']} for a in ARMS}))
    from .spatial_bank_replay import install
    install()
    from experiments.terra_10pct_factor_diagnosis.report import analyze
    for arm in ARMS:
        check();write(DEST/'progress.json',dict(status='posthoc',arm=arm));analyze(trio/arm,1,arm,manifest)
        r=read(trio/arm/'posthoc/complete.json');b=read(base/'posthoc/complete.json');assert all(r['splits'][s]['global_mean']==b['splits'][s]['source_mean'] for s in ['source_out','target_in','target_out'])
    check();write(DEST/'progress.json',dict(status='training_and_locked_posthoc_complete',remaining='full source/control comparison, visual EDA, utility/Q reliability, shrinkage/G controls and scientific report'))

if __name__=='__main__':
    write(DEST/'worker_pid.json',dict(pid=os.getpid(),started_unix=time.time(),module='experiments.sirm_research_loop.class_balanced_run'))
    try:main()
    except Exception as exc:write(DEST/'progress.json',dict(status='failed',error=str(exc)));raise

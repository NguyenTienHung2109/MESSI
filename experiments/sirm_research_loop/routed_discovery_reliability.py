"""Source-only utility/support reliability at selected and common endpoint states."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover.math import residual_utilities,soft_null_q
from .spatial_bank_model import SpatialBankGateSIRM
from .spatial_bank_replay import spatial_head
from .routed_discovery_run import DEST,PRIOR,ROOT
TOKEN_ROOT=ROOT/'results/sirm_research_loop/iteration_12'
from .shrinkage import write,digest
from .class_conditional_support_diagnosis import groups


def support(u,domains,subset):
    counts=[int(((domains==d)&subset).sum()) for d in [0,2,3]]
    U=torch.stack([u[torch.tensor((domains==d)&subset)].mean(0) for d in [0,2,3]])
    q,null,confidence,V=soft_null_q(U,torch.tensor([[n>=2]*6 for n in counts]),temperature=.05,utility_floor=.002,interaction_floor=.002)
    return dict(counts=counts,U=U.tolist(),V=V.tolist(),Q=q.tolist(),U_positive=(U>0).tolist(),U_floor_pass=(U>.002).tolist(),joint_floor_pass=((U>.002)&(V>.002)).tolist(),shared_U_floor_experts=torch.nonzero((U>.002).sum(0)>=2).flatten().tolist(),shared_Q_experts=torch.nonzero(q.gt(0).sum(0)>=2).flatten().tolist())

@torch.no_grad()
def main():
    torch.set_num_threads(4);out=DEST/'source_spatial_reliability';out.mkdir(exist_ok=True)
    audit=json.loads((TOKEN_ROOT/'frozen_token_source_diagnosis/extraction_audit.json').read_text());caches={};locks={}
    for split in ['source_in','source_out']:
        path=TOKEN_ROOT/'frozen_token_source_diagnosis'/f'{split}_tokens.npz';assert digest(path)==next(r['cache_sha256'] for r in audit['rows'] if r['split']==split);caches[split]=dict(np.load(path));locks[split]=digest(path)
    assert not set(caches['source_in']['ids'])&set(caches['source_out']['ids'])
    results={}
    for label,root,arm in [('unscaled_OFF',PRIOR,'OFF'),('routed_OFF',DEST,'OFF'),('routed_ON',DEST,'ON')]:
        assert (root/'experiment/runs/L38'/arm/'done').exists()
        for state,file in [('selected','source_best.pkl'),('last','last.pkl')]:
            key=f'{label}_{state}';path=root/'experiment/runs/L38'/arm/file;p=torch.load(path,map_location='cpu',weights_only=False);c=p['config'];assert c['utility_floor']==.002 and c['interaction_floor']==.002 and c['q_temperature']==.05
            m=SpatialBankGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c);m.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()},strict=True);m.configure_stage();m.eval();results[key]=dict(checkpoint_sha256=digest(path),step=int(p.get('selected_step',p.get('step'))),splits={})
            replays={}
            for split,a in caches.items():
                raw=torch.tensor(np.concatenate([a['cls'],a['patch_max']],1));g,delta,pi=spatial_head(m,raw);y=torch.tensor(a['y']);f,contrib,u,single=residual_utilities(g,delta,pi,y)
                assert np.array_equal(g.argmax(1).numpy(),a['global_logits'].argmax(1))
                allrows=np.ones(len(y),bool);v=support(u,a['domains'],allrows)
                replays[split]=dict(ids=a['ids'],y=a['y'],domains=a['domains'],bank_full=f.numpy(),bank_delta=delta.numpy(),bank_pi=pi.numpy())
                v['class_conditioned']=groups(replays[split],'bank',allrows)
                v['global_correct']=int(g.argmax(1).eq(y).sum());v['bank_correct']=int(f.argmax(1).eq(y).sum());results[key]['splits'][split]=v
                if split=='source_out':
                    halves=[np.zeros(len(y),bool) for _ in range(2)]
                    for d in [0,2,3]:
                        for cls in range(10):
                            ix=np.flatnonzero((a['domains']==d)&(a['y']==cls));ix=sorted(ix,key=lambda i:hashlib.sha256(str(a['ids'][i]).encode()).hexdigest())
                            for j,i in enumerate(ix):halves[j%2][i]=True
                    assert np.all(halves[0]^halves[1]);results[key]['source_out_halves']=[support(u,a['domains'],h) for h in halves];results[key]['class_halves']=[groups(replays[split],'bank',h) for h in halves]
            matrices=[np.array(results[key]['splits'][s]['U']) for s in ['source_in','source_out']]+[np.array(v['U']) for v in results[key]['source_out_halves']]
            results[key]['positive_in_all_four']=np.logical_and.reduce([v>0 for v in matrices]).tolist();results[key]['U_floor_in_all_four']=np.logical_and.reduce([v>.002 for v in matrices]).tolist()
            results[key]['stable_class_shared']={str(cls):sorted(set.intersection(*[set(v[str(cls)]['shared_experts']) for v in [results[key]['splits'][s]['class_conditioned'] for s in ['source_in','source_out']]+results[key]['class_halves']])) for cls in range(10)}
            np.savez_compressed(out/f'{key}_source_replay.npz',**{s+'_'+k:v for s,a in replays.items() for k,v in a.items()})
    record=dict(training=False,fitting=False,target_read=False,actual_Q_changed=False,checkpoint_selection_changed=False,minimum_samples=2,cache_sha256=locks,code_sha256=digest(Path(__file__)),scope='Original source-only saved-state U/Q formula and class-conditioned counterfactual; not historical minibatch EMA, independent validation or a new policy. Source-selected states and class/domain halves are descriptive. Invalid complete class matrices cannot establish absence of real support.',states=results)
    write(out/'results.json',record)
    print(json.dumps({k:dict(positive_cells_all_four=int(np.array(v['positive_in_all_four']).sum()),floor_cells_all_four=int(np.array(v['U_floor_in_all_four']).sum()),shared_U={s:x['shared_U_floor_experts'] for s,x in v['splits'].items()},shared_Q={s:x['shared_Q_experts'] for s,x in v['splits'].items()},stable_class_shared=v['stable_class_shared']) for k,v in results.items()},indent=2))

if __name__=='__main__':
    import os,time
    from .router_weight import live
    write(DEST/'reliability_worker_pid.json',dict(pid=os.getpid(),started_unix=time.time(),module='experiments.sirm_research_loop.class_balanced_reliability'))
    pid=json.loads((DEST/'worker_pid.json').read_text())['pid']
    while live(pid):
        write(DEST/'reliability_progress.json',dict(status='waiting_for_locked_pair',worker_pid=pid,verified_live=True,observed_unix=time.time()));time.sleep(20)
    assert json.loads((DEST/'progress.json').read_text())['status']=='training_and_locked_posthoc_complete'
    try:
        main();write(DEST/'reliability_progress.json',dict(status='source_matrices_complete',remaining='Interpret utility replication/coverage with original-image EDA'))
    except Exception as exc:
        write(DEST/'reliability_progress.json',dict(status='failed',error=str(exc)));raise

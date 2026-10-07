"""Source-only paired original/H2 teacher stability and cross-view oracle controls."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from domainbed.deit_transformer import DeiTFeaturizer
from domainbed.scripts.train_subset_irm_pacs import make_schedule
from experiments.sirm_discover.math import router_teacher
from .centered_model import CenteredGateSIRM
from .hard_router_diagnosis import outputs
from .bias_control import scores
from .shrinkage import ROOT,digest,write


def main():
    torch.set_num_threads(4);root=ROOT/'results/sirm_research_loop/iteration_11';out=root/'paired_teacher_stability_diagnosis';out.mkdir(exist_ok=True)
    path=root/'experiment/runs/L38/OFF/source_best.pkl';p=torch.load(path,map_location='cpu',weights_only=False);assert p['selected_step']==399
    c=p['config'];model=CenteredGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()});model.configure_stage();model.eval();model.requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    manifest=json.loads((root/'experiment/subset_manifest.json').read_text());lock=json.loads((root/'experiment/runs/L38/OFF/source_schedules.json').read_text())
    cache=ROOT/'results/sirm_research_loop/iteration_09/source_router_diagnosis/source_in_original.npz'
    verified=json.loads((root/'composed_bias_source_diagnosis/results.json').read_text());assert digest(cache)==verified['plan']['source_cache_hashes']['source_in']
    with np.load(cache) as a:features={i:torch.from_numpy(z) for i,z in zip(a['ids'].tolist(),a['z'])}
    schedules={d:make_schedule(list(range(len(lock[str(d)]['split_keys']))),400,32,0,d) for d in [0,2,3]}
    assert all(hashlib.sha256(schedules[d].tobytes()).hexdigest()==lock[str(d)]['schedule_sha256'] for d in schedules)
    plan=dict(code_sha256=digest(Path(__file__)),checkpoint_sha256=digest(path),source_cache_sha256=digest(cache),
        target_read=False,fitting=False,training=False,selection=False,
        scope='Six scheduled source-in minibatches, same saved399 model; original/H2 views paired by exact image IDs/labels. '
              'Not historical intermediate gradients/states or independent samples. H2 includes crop/flip/jitter/gray/blur; transform-specific effects cannot be isolated.')
    rows=[];case_rows=[]
    for step in [200,215,250,300,350,395]:
        path_h2=ROOT/'results/sirm_research_loop/iteration_08/pair_shared_parameter_audit'/f'H2_step{step}.pt'
        b=torch.load(path_h2,map_location='cpu',weights_only=False)
        assert digest(path_h2)==next(r['cache_sha256'] for r in verified['paired_batches'] if r['step']==step)
        entries=[manifest['sources'][str(d)]['images'][lock[str(d)]['split_keys'][int(i)]] for d in [0,2,3] for i in schedules[d][step]]
        ids=[e['id'] for e in entries];assert b['y'].tolist()==[e['class'] for e in entries]
        original=torch.stack([features[i] for i in ids]);y=b['y'];domains=b['domains']
        with torch.no_grad():
            for mode in ['saved','batch']:
                f={view:outputs(model,z,y,mode) for view,z in [('original',original),('H2',b['z'])]}
                t={view:row['teacher'] for view,row in f.items()}
                robust=router_teacher(torch.minimum(f['original']['single'],f['H2']['single']),c['router_temperature'],c['utility_floor'])
                policies=dict(original_teacher=t['original'],H2_teacher=t['H2'],mean_teacher=(t['original']+t['H2'])/2,minimum_utility_teacher=robust)
                row=dict(step=step,centering=mode,n=len(y),cache_sha256=digest(path_h2),
                    expert_winner_agreement=float((f['original']['winner']==f['H2']['winner']).double().mean()),
                    teacher_argmax_agreement=float((t['original'].argmax(1)==t['H2'].argmax(1)).double().mean()),
                    teacher_L1=float((t['original']-t['H2']).abs().sum(1).mean()),
                    router_L1=float((f['original']['pi']-f['H2']['pi']).abs().sum(1).mean()),evaluations={})
                for view,bank in f.items():
                    globalpred=bank['g'].argmax(1);wrong=globalpred!=y
                    row['evaluations'][view]={}
                    for policy,target in {**policies,'native':bank['pi']}.items():
                        logits=bank['g']+(target[:,:6,None]*bank['delta']).sum(1)
                        score,pred=scores(logits.numpy(),y.numpy(),domains.numpy());pred=torch.from_numpy(pred)
                        score.update(repairs=int((wrong&(pred==y)).sum()),harms=int((~wrong&(pred!=y)).sum()),NULL=float(target[:,-1].mean()))
                        row['evaluations'][view][policy]=score
                if mode=='batch':
                    l1=(t['original']-t['H2']).abs().sum(1)
                    for i,image in enumerate(ids):case_rows.append(dict(id=image,step=step,label=int(y[i]),domain=int(domains[i]),teacher_L1=float(l1[i]),
                        original_teacher_top=int(t['original'][i].argmax()),H2_teacher_top=int(t['H2'][i].argmax()),
                        original_global_prediction=int(f['original']['g'][i].argmax()),H2_global_prediction=int(f['H2']['g'][i].argmax())))
                rows.append(row)
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items()) and digest(path)==plan['checkpoint_sha256']
    write(out/'results.json',dict(plan=plan,records=rows,cases=case_rows,distinct_source_images=len(set(r['id'] for r in case_rows)),state_tensors_unchanged=True))
    for mode in ['saved','batch']:
        rs=[r for r in rows if r['centering']==mode]
        print(mode,{k:float(np.mean([r[k] for r in rs])) for k in ['expert_winner_agreement','teacher_argmax_agreement','teacher_L1','router_L1']})
        for view in ['original','H2']:
            print(view,{p:{k:float(np.mean([r['evaluations'][view][p][k] for r in rs])) for k in ['accuracy','CE','repairs','harms','NULL']} for p in rows[0]['evaluations'][view]})


if __name__=='__main__':main()

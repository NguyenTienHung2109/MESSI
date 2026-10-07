"""Saved-state source H2 replay: specialization versus centering context.

Exact historical image/augmentation schedules, but saved selected/final weights:
these are not gradients or predictions from historical intermediate states.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from domainbed.deit_transformer import DeiTFeaturizer
from domainbed.scripts.train_subset_irm_pacs import make_schedule
from experiments.sirm_discover.math import center_logits, multiple_choice_loss, residual_utilities, router_teacher
from experiments.sirm_discover_gate.data import Adapter
from experiments.sirm_h1_h6.run import ScheduledImages
from .centered_model import CenteredGateSIRM

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'results/sirm_research_loop/iteration_07'
STEPS = [200, 215, 250, 300, 350, 395]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    torch.set_num_threads(4)
    out = DEST/'centering_batch_diagnosis'; out.mkdir(exist_ok=True)
    bank = DEST/'experiment/runs/L38/OFF'
    paths = {name:bank/file for name,file in [('selected','source_best.pkl'),('last','last.pkl')]}
    models = {}; locks = {}
    for name, path in paths.items():
        p = torch.load(path, map_location='cpu', weights_only=False); config = p['config']
        phi = DeiTFeaturizer(pretrained=False, model_name=config['backbone'])
        m = CenteredGateSIRM(phi, phi.n_outputs, 10, 3, config)
        m.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()}); m.configure_stage(); m.eval()
        models[name] = m; locks[name] = dict(sha256=digest(path), step=p['selected_step'] if name=='selected' else p['step'])
    manifest = json.loads((DEST/'experiment/subset_manifest.json').read_text())
    schedules_meta = json.loads((bank/'source_schedules.json').read_text())
    source_manifest = dict(manifest, sources={str(d):manifest['sources'][str(d)] for d in [0,2,3]})
    a = Adapter('TerraIncognita'); a.classes=manifest['class_names']
    original = a.source_data(ROOT/'domainbed/data', source_manifest, config, 'in')
    augmented = a.source_data(ROOT/'domainbed/data', source_manifest, config, 'in', train=True)
    schedules = {d:make_schedule(list(range(len(ds))), 400, 32, 0, d) for d,ds in augmented.items()}
    for d, schedule in schedules.items():
        assert list(augmented[d].indices)==schedules_meta[str(d)]['split_keys']
        assert hashlib.sha256(schedule.tobytes()).hexdigest()==schedules_meta[str(d)]['schedule_sha256']
    plan=dict(checkpoints=locks, scheduled_steps=STEPS, code_sha256=digest(Path(__file__)),
              scope='Exact source-in image and augmentation schedules evaluated at saved selected200/final399 OFF weights; not historical intermediate states. Original paired images and H2 features share image/label/domain IDs. No target, fitting, updates, tuning or selection.',
              modes=['differentiable_training_batch_mean_formula', 'saved_source_original_mean_formula'])
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    phi = models['selected'].featurizer.cuda(); records=[]
    with torch.no_grad():
        for step in STEPS:
            tensors={'H2':[], 'original':[]}; ys=[]; ids=[]
            for d in [0,2,3]:
                scheduled=ScheduledImages(augmented[d],0,d)
                x=[]; xo=[]; y=[]
                for position,index in enumerate(schedules[d][step]):
                    image,label=scheduled[(int(index),step,position)]; clean,clean_label=original[d][int(index)]
                    assert label==clean_label
                    x.append(image);xo.append(clean);y.append(label)
                    ids.append(manifest['sources'][str(d)]['images'][original[d].indices[int(index)]]['id'])
                tensors['H2'].extend(x);tensors['original'].extend(xo);ys.extend(y)
            y=torch.tensor(ys)
            features={view:phi(torch.stack(x).cuda()).cpu() for view,x in tensors.items()}
            for state,model in models.items():
                for view,z in features.items():
                    g=model.global_head(z);raw=center_logits(torch.stack([e(z) for e in model.residuals],1));pi=model.router(z).softmax(1)
                    means={'batch':raw.mean(0),'saved_source':model.expert_source_mean}
                    predictions={}; metrics={}
                    for mode,mean in means.items():
                        delta=raw-mean[None];full,_,utility,single=residual_utilities(g,delta,pi,y)
                        _,responsibility=multiple_choice_loss(g,delta,y,config['discovery_temperature'])
                        entropy=-(responsibility*responsibility.clamp_min(1e-30).log()).sum(1)
                        teacher=router_teacher(single,config['router_temperature'],config['utility_floor'])
                        teacher_logits=g+(teacher[:,:6,None]*delta).sum(1)
                        predictions[mode]=full.argmax(1)
                        metrics[mode]=dict(CE=float(F.cross_entropy(full,y)),accuracy=float(full.argmax(1).eq(y).float().mean()),
                          normalized_responsibility_entropy=float(entropy.mean()/np.log(6)),mean_max_responsibility=float(responsibility.max(1).values.mean()),
                          fraction_responsibility_over_half=float((responsibility.max(1).values>.5).float().mean()),
                          mean_U=utility.mean(0).tolist(),mean_expert_norm=float(delta.norm(dim=2).mean()),
                          teacher_repairs=int((~g.argmax(1).eq(y)&teacher_logits.argmax(1).eq(y)).sum()),
                          teacher_harms=int((g.argmax(1).eq(y)&~teacher_logits.argmax(1).eq(y)).sum()))
                    records.append(dict(state=state,scheduled_batch_step=step,view=view,n=len(y),image_ids=ids,
                      global_CE=float(F.cross_entropy(g,y)),global_accuracy=float(g.argmax(1).eq(y).float().mean()),
                      raw_mean_vs_saved_mean_norm=float((means['batch']-means['saved_source']).norm()),
                      prediction_changes_due_to_mean=int(predictions['batch'].ne(predictions['saved_source']).sum()),metrics=metrics))
    assert all(digest(path)==locks[name]['sha256'] for name,path in paths.items())
    result=dict(plan=plan,records=records,limitation='Six overlapping scheduled batches are descriptive. Original versus H2 input and batch versus source mean are controlled replay contrasts, not independent confirmation or proof that either caused training failure. No optimizer or parameter gradients are computed.')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'records':len(records),'completed':True,'target_read':False}))


if __name__ == '__main__':
    run()

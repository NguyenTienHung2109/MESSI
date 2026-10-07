"""Source-only soft-min versus hard-winner gradient diagnosis; no updates."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from domainbed.deit_transformer import DeiTFeaturizer
from domainbed.scripts.train_subset_irm_pacs import make_schedule
from experiments.sirm_discover.math import center_logits,expert_ce,multiple_choice_loss
from experiments.sirm_discover_gate.data import Adapter
from experiments.sirm_h1_h6.run import ScheduledImages
from .centered_model import CenteredGateSIRM
from .router_gradient_diagnosis import norm,cosine

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'results/sirm_research_loop/iteration_07'
STEPS=[200,215,250,300,350,395]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run():
    torch.set_num_threads(4);bank=DEST/'experiment/runs/L38/OFF';out=DEST/'discovery_assignment_gradient';out.mkdir(exist_ok=True)
    path=bank/'source_best.pkl';sha=digest(path);payload=torch.load(path,map_location='cpu',weights_only=False);c=payload['config']
    phi=DeiTFeaturizer(pretrained=False,model_name=c['backbone']);m=CenteredGateSIRM(phi,phi.n_outputs,10,3,c)
    m.load_state_dict({k.removeprefix('model.'):v for k,v in payload['model_dict'].items()});m.configure_stage();m.eval()
    assert all(not p.requires_grad for name in ['featurizer','global_head'] for p in getattr(m,name).parameters())
    manifest=json.loads((DEST/'experiment/subset_manifest.json').read_text());manifest=dict(manifest,sources={str(d):manifest['sources'][str(d)] for d in [0,2,3]})
    adapter=Adapter('TerraIncognita');adapter.classes=manifest['class_names'];meta=json.loads((bank/'source_schedules.json').read_text())
    aug=adapter.source_data(ROOT/'domainbed/data',manifest,c,'in',train=True);orig=adapter.source_data(ROOT/'domainbed/data',manifest,c,'in')
    schedules={d:make_schedule(list(range(len(ds))),400,32,0,d) for d,ds in aug.items()}
    for d,s in schedules.items():assert list(aug[d].indices)==meta[str(d)]['split_keys'] and hashlib.sha256(s.tobytes()).hexdigest()==meta[str(d)]['schedule_sha256']
    plan=dict(checkpoint_sha256=sha,checkpoint_step=payload['selected_step'],code_sha256=digest(Path(__file__)),scheduled_steps=STEPS,scope='Saved selected200 centered OFF model, exact source-in scheduled H2 and paired original images. No historical intermediate weights, target reads, fitting, parameter updates or selection.',candidate='Replace only discovery loss soft-min with minimum expert CE; keep existing soft responsibility for the unchanged differentiable balance objective. Router teacher and all numerical settings unchanged.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');m.featurizer.cuda();records=[]
    parameters=list(m.residuals.parameters());head_sizes=[len(list(head.parameters())) for head in m.residuals]
    for step in STEPS:
        xs={'H2':[],'original':[]};ys=[];ids=[]
        for d in [0,2,3]:
            scheduled=ScheduledImages(aug[d],0,d)
            for position,index in enumerate(schedules[d][step]):
                x,y=scheduled[(int(index),step,position)];xo,yo=orig[d][int(index)];assert y==yo
                xs['H2'].append(x);xs['original'].append(xo);ys.append(y);ids.append(manifest['sources'][str(d)]['images'][orig[d].indices[int(index)]]['id'])
        y=torch.tensor(ys);winners={}
        for view,x in xs.items():
            with torch.no_grad():z=m.featurizer(torch.stack(x).cuda()).cpu();g=m.global_head(z);pi=m.router(z).softmax(1)
            raw=center_logits(torch.stack([head(z) for head in m.residuals],1));delta=raw-raw.mean(0,keepdim=True)
            full=g+(pi[:,:6,None]*delta).sum(1);cls=F.cross_entropy(full,y)
            soft,choice=multiple_choice_loss(g,delta,y,c['discovery_temperature']);losses=expert_ce(g[:,None]+delta,y);best,winner=losses.min(1);hard=losses.amin(1).mean();winners[view]=winner
            grads={name:torch.autograd.grad(loss,parameters,retain_graph=True) for name,loss in [('classification',cls),('soft',c['lambda_discover']*soft),('hard',c['lambda_discover']*hard)]}
            head_cos={}
            for name in ['soft','hard']:
                vectors=[];offset=0
                for size in head_sizes:
                    vectors.append(torch.cat([v.detach().flatten().double() for v in grads[name][offset:offset+size]]));offset+=size
                norms=torch.tensor([float(v.norm()) for v in vectors]);matrix=torch.stack([F.normalize(v,dim=0) for v in vectors])@torch.stack([F.normalize(v,dim=0) for v in vectors]).T
                valid=norms[:,None].gt(0)&norms[None,:].gt(0)&~torch.eye(6,dtype=torch.bool)
                head_cos[name]=dict(matrix=matrix.tolist(),nonzero_heads=int(norms.gt(0).sum()),mean_valid_off_diagonal=float(matrix[valid].mean()) if valid.any() else None)
            records.append(dict(scheduled_batch_step=step,view=view,n=len(y),image_ids=ids,
              hard_winner_counts=torch.bincount(winner,minlength=6).tolist(),hard_winner_counts_by_domain=[torch.bincount(winner[i*32:(i+1)*32],minlength=6).tolist() for i in range(3)],
              exact_minimum_tie_images=int(((losses.detach()==best.detach()[:,None]).sum(1)>1).sum()),
              hard_winner_counts_by_class={str(k):torch.bincount(winner[y==k],minlength=6).tolist() for k in y.unique().tolist()},
              soft_entropy_normalized=float(-(choice*choice.clamp_min(1e-30).log()).sum(1).mean()/np.log(6)),
              residual_gradient_norm={name:norm(v) for name,v in grads.items()},weighted_discovery_to_classification={name:norm(grads[name])/norm(grads['classification']) for name in ['soft','hard']},
              classification_discovery_cosine={name:cosine(grads['classification'],grads[name]) for name in ['soft','hard']},head_parameter_gradient_cosine=head_cos))
            assert all(p.grad is None for p in m.parameters())
        records[-1]['paired_H2_original_winner_agreement']=float(winners['H2'].eq(winners['original']).float().mean())
    assert digest(path)==sha
    result=dict(plan=plan,records=records,parameter_updates=False,target_read=False,limitation='Gradients are saved-state residual-head parameter gradients on six overlapping batches. Hard loss uses amin, distributing gradient equally across exact ties; argmin counts select the first tied expert for display only. Different winner gradients are expected by construction, not proof of useful specialization, stable factors or OOD transfer. Cross-head parameter cosine compares independently initialized hidden coordinates and is not representation invariant; use within-same-parameter objective alignment for strength. Winner changes across H2/original views may signal transform-sensitive partitioning. No candidate training is performed.')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'completed':True,'records':len(records)}))

if __name__=='__main__':run()

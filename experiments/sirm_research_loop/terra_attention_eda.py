"""Paired object-localization EDA: raw rollout, class Grad-CAM and bbox interventions."""
import csv, json, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover_gate.model import GateSIRM
from experiments.sirm_h1_h6.run import ViewTransform
from .routed_conditional_utility import load
from .routed_discovery_run import ROOT, ANCHOR
from .shrinkage import digest, write

OUT=ROOT/'results/terra_attention_diagnosis/phase1'
CURRENT=ROOT/'results/sirm_research_loop/iteration_16'
HISTORICAL=ROOT/'results/sirm_discover_binary/terra_L38_seed0_lambda1/banks/R1/source_best.pkl'

def norm(a):
    a=np.maximum(a,0).astype(float);s=a.sum()
    return a/s if s>1e-20 else None

def box_mask(boxes):
    mask=np.zeros((224,224),dtype=bool)
    for x,y,w,h in boxes:mask[max(0,int(np.floor(y*224))):min(224,int(np.ceil((y+h)*224))),max(0,int(np.floor(x*224))):min(224,int(np.ceil((x+w)*224)))]=True
    return mask

def localization(heat,mask):
    if heat is None or not mask.any():return dict(valid=False)
    area=mask.mean();patch=mask.reshape(14,16,14,16).mean((1,3));mass=float((heat*patch).sum());row,col=np.unravel_index(heat.argmax(),heat.shape)
    return dict(valid=True,box_area_fraction=float(area),heat_mass_in_box=mass,area_normalized_enrichment=mass/area,peak_patch_center_in_box=bool(mask[row*16+8,col*16+8]))

def load_models():
    current,p,path=load(CURRENT,'selected');current.requires_grad_(False);current.eval().cuda()
    h=torch.load(HISTORICAL,map_location='cpu',weights_only=False);c=h['config'];phi=DeiTFeaturizer(pretrained=False,model_name=c['backbone']);historical=GateSIRM(phi,384,10,3,c)
    historical.load_state_dict({k.removeprefix('model.'):v for k,v in h['model_dict'].items()});historical.configure_stage();historical.requires_grad_(False);historical.eval().cuda()
    anchor=torch.load(ANCHOR,map_location='cpu',weights_only=False)['model_dict']
    for module in ['featurizer','global_head']:
        for k,v in getattr(current,module).state_dict().items():assert torch.equal(v.cpu(),anchor[module+'.'+k]),module+'.'+k
    funcs={'strong_ERM':lambda x:current.global_head(current.featurizer(x)), 'current_SIRM_inactive':lambda x:current(x)['logits'],
           'historical_ERM_global':lambda x:historical.global_head(historical.featurizer(x)), 'historical_SIRM_IRMv1_ON':lambda x:historical(x)['logits']}
    return current,historical,funcs,{str(path):digest(path),str(HISTORICAL):digest(HISTORICAL),str(ANCHOR):digest(ANCHOR)}

def explain(model,forward,x,label):
    attn=[];activ=[];hooks=[]
    for block in model.featurizer.vit.blocks:
        hooks.append(block.attn.attn_drop.register_forward_hook(lambda module,args,out:attn.append(out)))
    hooks.append(model.featurizer.vit.blocks[-1].norm1.register_forward_hook(lambda module,args,out:activ.append(out)))
    try:
        inp=x.detach().clone().requires_grad_(True);logits=forward(inp);prediction=int(logits.argmax(1));maps={}
        assert len(attn)==12 and len(activ)==1 and activ[0].shape[1]==198
        for cls in sorted({label,prediction}):
            gradient=torch.autograd.grad(logits[0,cls],activ[0],retain_graph=True)[0]
            feature=activ[0][0,2:];grad=gradient[0,2:];weights=grad.mean(0)
            maps[cls]=norm((feature*weights).sum(1).relu().detach().cpu().numpy().reshape(14,14))
        rollout=torch.eye(198,device=x.device)
        with torch.no_grad():
            for a in attn:
                flow=a[0].mean(0)+torch.eye(198,device=x.device);flow/=flow.sum(1,keepdim=True);rollout=flow@rollout
        return logits.detach()[0].cpu().numpy(),norm(rollout[0,2:].cpu().numpy().reshape(14,14)),maps[label],maps[prediction]
    finally:
        for hook in hooks:hook.remove()

def draw_grid(name,records,images,maps,names,model,key):
    fig,axes=plt.subplots(4,4,figsize=(14,15))
    for i,(ax,r,img,h) in enumerate(zip(axes.flat,records,images,maps),1):
        ax.imshow(img);heat=h[model][key]
        if heat is not None:
            expanded=F.interpolate(torch.tensor(heat)[None,None],size=(224,224),mode='bilinear',align_corners=False)[0,0].numpy();ax.imshow(expanded,cmap='jet',alpha=.45,vmin=0,vmax=expanded.max())
        for x,y,w,height in r['boxes']:ax.add_patch(Rectangle((x*224,y*224),w*224,height*224,fill=False,edgecolor='lime',linewidth=1.5))
        m=r['models'][model];loc=m['localization'][key];stats=f"enrich={loc['area_normalized_enrichment']:.2f}" if loc['valid'] else 'no matching official box'
        ax.set_title(f"{i:02d} {r['split']} GT:{r['class_name']}\npred:{names[m['prediction']]} p={m['confidence']:.2f} | {stats}",fontsize=8);ax.axis('off')
    fig.suptitle(f'{model} | {key} | green: official class-matched bounding boxes',fontsize=13)
    fig.text(.5,.015,'Selected diagnostic cases, not a random sample. Each heatmap normalized separately. Missing box is not empty ground truth.\nGrad-CAM uses last-block pre-attention norm1 patch tokens; raw rollout is class-agnostic.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.97));fig.savefig(OUT/(name+'.png'),dpi=130);plt.close(fig)

def run():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists()
    casepath=CURRENT/'eda/cases.csv';cases=list(csv.DictReader(casepath.open()));metadata=OUT/'caltech_bboxes_20200316.json';official=json.loads(metadata.read_text());cats={r['id']:r['name'] for r in official['categories']};sizes={r['id']:(r['width'],r['height']) for r in official['images']};boxes={}
    for a in official['annotations']:
        if a.get('bbox'):boxes.setdefault(a['image_id'],[]).append((cats[a['category_id']],a['bbox']))
    selected=[r for r in cases if r['split'] in ['source_out','target_in'] and r['class_name']=='bird']
    for stem in ['59bfdfaa','59f2c51e','599be932']:
        selected += [r for r in cases if Path(r['image_id']).stem.startswith(stem) and r not in selected]
    for split in ['source_out','target_in']:
        pool=sorted([r for r in cases if r['split']==split and r['B0_correct']=='True' and r['class_name']!='empty' and Path(r['image_id']).stem in boxes],key=lambda r:r['image_id'])
        for r in pool[:2]:
            if r not in selected:selected.append(r)
    for r in sorted(cases,key=lambda r:r['image_id']):
        if len(selected)>=16:break
        if r['split']=='source_out' and r['class_name']!='empty' and Path(r['image_id']).stem in boxes and r not in selected:selected.append(r)
    assert len(selected)==16
    names=json.loads((CURRENT/'experiment/subset_manifest.json').read_text())['class_names']
    plan=dict(hypothesis='Small-object loss or background-driven decisions may prevent useful factor transfer; heatmaps alone do not prove reliance.',
        selection='All8 source-out/target-in birds,3 predefined harms,2 boxed correct nonempty examples per split sorted by ID; fill16 with boxed source-out examples sorted by ID.',
        selected_ids=[r['image_id'] for r in selected],case_sha256=digest(casepath),metadata_sha256=digest(metadata),
        metadata_source='https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_bboxes_20200316.json',
        models=['strong_ERM','current_SIRM_inactive','historical_ERM_global','historical_SIRM_IRMv1_ON'],
        comparison='Within each shared backbone raw attention should match; strong-vs-historical comparison confounded by training/data/budget. No matched vanilla IRM checkpoint present in local output paths.',
        maps='Raw CLS attention rollout: head mean + identity, row-normalized each layer. Positive Grad-CAM on last-block norm1: true and predicted class separately; omit two prefix tokens. 14x14 coarse map.',
        tests='Official class-matched bbox mass/enrichment/peak; bbox erasure, bbox-only mean-filled background, enlarged bbox crop, area-matched nonobject erasure when feasible. These are post-hoc label-aware interventions, not deployment accuracy.',
        missing='Missing class bbox excluded from localization/intervention metrics, never interpreted as no object. Boxes are coarse, not segmentation.',
        frozen=True,training=False,target_tuning=False,checkpoint_selection_changes=False,source_only_next_experiment=True,
        references=['https://arxiv.org/abs/2005.00928','https://arxiv.org/abs/1610.02391','https://lila.science/datasets/caltech-camera-traps'],code_sha256=digest(Path(__file__)))
    write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    current,historical,forwards,locks=load_models();versions={name:[q._version for q in m.parameters()] for name,m in [('current',current),('historical',historical)]};records=[];images=[];allmaps=[]
    transform=ViewTransform(current.config,'original',False)
    for i,r in enumerate(selected):
        path=ROOT/'domainbed/data/terra_incognita'/r['image_id'];locks[str(path)]=digest(path);img=Image.open(path).convert('RGB');x=transform(img)[None].cuda();display=img.resize((224,224),Image.Resampling.BILINEAR);id=Path(r['image_id']).stem;bb=[]
        if id in sizes:
            width,height=sizes[id]
            bb=[[xx/width,yy/height,w/width,h/height] for label,(xx,yy,w,h) in boxes.get(id,[]) if label==r['class_name']]
        mask=box_mask(bb);record=dict(image_id=r['image_id'],split=r['split'],class_name=r['class_name'],label=int(r['true_class']),boxes=bb,box_available=bool(mask.any()),models={});maps={}
        perturbations={}
        if mask.any():
            pix=np.array(display).astype(float)/255.;mean=np.array([.485,.456,.406]);erased=pix.copy();erased[mask]=mean;only=pix.copy();only[~mask]=mean
            ys,xs=np.nonzero(mask);crop=display.crop((int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)))
            perturbations={'erase_object':transform(Image.fromarray(np.uint8(np.clip(erased,0,1)*255)))[None].cuda(), 'object_only':transform(Image.fromarray(np.uint8(np.clip(only,0,1)*255)))[None].cuda(),'object_crop':transform(crop)[None].cuda()}
            # Same rectangular erasure translated to nonobject area, first feasible raster position.
            bh,bw=int(ys.max()-ys.min()+1),int(xs.max()-xs.min()+1);control=None
            for yy in range(0,224-bh+1,8):
                for xx in range(0,224-bw+1,8):
                    if not mask[yy:yy+bh,xx:xx+bw].any():control=(yy,xx);break
                if control:break
            if control:
                cp=pix.copy();yy,xx=control;cp[yy:yy+bh,xx:xx+bw]=mean;perturbations['erase_background_area_matched']=transform(Image.fromarray(np.uint8(np.clip(cp,0,1)*255)))[None].cuda()
        for name,forward in forwards.items():
            model=current if name in ['strong_ERM','current_SIRM_inactive'] else historical
            logits,rollout,gt,pred=explain(model,forward,x,record['label']);heat=dict(rollout=rollout,GT_GradCAM=gt,predicted_GradCAM=pred);maps[name]=heat;prediction=int(logits.argmax());prob=torch.tensor(logits).softmax(0).numpy()
            values=dict(prediction=prediction,confidence=float(prob[prediction]),logits=logits.tolist(),localization={k:localization(v,mask) for k,v in heat.items()},interventions={})
            with torch.no_grad():
                for key,px in perturbations.items():
                    f=forward(px)[0].cpu();values['interventions'][key]=dict(prediction=int(f.argmax()),GT_probability=float(f.softmax(0)[record['label']]),GT_logit=float(f[record['label']]),GT_logit_change=float(f[record['label']]-logits[record['label']]))
            record['models'][name]=values
            np.savez_compressed(OUT/f'case{i:02d}_{name}.npz',logits=logits,rollout=rollout,GT_GradCAM=gt,predicted_GradCAM=pred,box_mask=mask)
        assert np.array_equal(maps['strong_ERM']['rollout'],maps['current_SIRM_inactive']['rollout'])
        assert np.array_equal(maps['historical_ERM_global']['rollout'],maps['historical_SIRM_IRMv1_ON']['rollout'])
        records.append(record);images.append(display);allmaps.append(maps);write(OUT/'progress.json',dict(status='heatmaps',completed=len(records),total=16));print('case',i+1,'/16',r['image_id'],flush=True)
    for model in forwards:
        for key in ['rollout','GT_GradCAM','predicted_GradCAM']:draw_grid(model+'_'+key+'_4x4',records,images,allmaps,names,model,key)
    for name,m in [('current',current),('historical',historical)]:assert versions[name]==[q._version for q in m.parameters()] and all(q.grad is None for q in m.parameters()) and not any(q.requires_grad for q in m.parameters())
    for path,h in locks.items():assert digest(Path(path))==h
    write(OUT/'results.json',dict(records=records,class_names=names,input_sha256=locks,raw_attention_exactly_equal_within_shared_backbone=True,parameters_unchanged=True,training=False,checkpoint_changes=False,target_tuning=False,
        coverage=dict(selected=16,official_class_matched_box=sum(r['box_available'] for r in records),bird_box=sum(r['box_available'] for r in records if r['class_name']=='bird'),bird_total=8)))
    write(OUT/'progress.json',dict(status='complete',cases=16,grids=12))

if __name__=='__main__':run()

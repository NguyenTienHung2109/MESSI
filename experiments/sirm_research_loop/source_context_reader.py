"""Source-only paired context supervision: fixed backbone cache and diagnostic readers."""
import json,copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import gaussian_blur
from .conditioned_router_source_analysis import load,DEST as BANK
from .terra_attention_eda import OUT,box_mask
from .spatial_bank_model import pooled_raw_features
from .shrinkage import ROOT,write,digest
from experiments.sirm_h1_h6.run import ViewTransform
DEST=BANK/'source_context_reader';CACHE=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis'
def setup():
 DEST.mkdir(exist_ok=True);manifest=json.loads((BANK/'experiment/subset_manifest.json').read_text());ann=OUT/'caltech_bboxes_20200316.json';a=json.loads(ann.read_text());cats={r['id']:r['name'] for r in a['categories']};sizes={r['id']:(r['width'],r['height']) for r in a['images']};boxes={}
 for b in a['annotations']:
  if b.get('bbox'):boxes.setdefault(b['image_id'],[]).append((cats[b['category_id']],b['bbox']))
 data={s:dict(np.load(CACHE/f'{s}_tokens.npz')) for s in ['source_in','source_out']};donorids={}
 for domain in [0,2,3]:
  z=manifest['sources'][str(domain)];donorids[domain]=sorted(z['images'][i]['id'] for i in z['in'] if z['images'][i]['class']==5)[0]
 return manifest,ann,sizes,boxes,data,donorids

def extract():
 manifest,ann,sizes,boxes,data,donorids=setup();m,p,path=load(BANK,'selected');m.cuda().eval();c=m.config;tr=ViewTransform(c,'original',False);before={k:v.cpu().clone() for k,v in m.featurizer.state_dict().items()};donors={}
 for d,ident in donorids.items():
  with Image.open(ROOT/'domainbed/data/terra_incognita'/ident) as im:donors[d]=tr(im.convert('RGB'))[None].cuda()
 ids=data['source_in']['ids'];raw=torch.tensor(np.concatenate([data['source_in']['cls'],data['source_in']['patch_max']],1)) if 'patch_max' in data['source_in'] else torch.tensor(np.concatenate([data['source_in']['cls'],data['source_in']['patchmax']],1));g=torch.tensor(data['source_in']['global_logits']);y=torch.tensor(data['source_in']['y']);dom=torch.tensor(data['source_in']['domains']);available=[];job=[]
 for i,(ident,label) in enumerate(zip(ids,y.tolist())):
  stem=Path(str(ident)).stem;bb=[]
  if stem in sizes:
   ow,oh=sizes[stem];bb=[[x/ow,z/oh,w/ow,h/oh] for name,(x,z,w,h) in boxes.get(stem,[]) if name==manifest['class_names'][label]]
  if bb:job.append((i,bb));available.append(i)
 plan=dict(hypothesis='Paired object-retained context supervision may improve natural source correction payoff',source_only=True,target_read=False,scope='Diagnostic linear residual reader, not a new primary SIRM family or invariant-factor proof',checkpoint_sha256=digest(path),frozen_anchor_sha256=c['strong_anchor_sha256'],annotation_sha256=digest(ann),source_input_sha256={s:digest(CACHE/f'{s}_tokens.npz') for s in data},donors=donorids,boxed_training_images=len(job),training_images=len(ids),transform='Photometric-matched feather15/sigma3 outside true-class boxes only; objectpixels exact. Dataset-empty donors may include people.',fits='Original-only vs paired-original/context perpooled and3 auxiliary-held domains. Original-only repeats same view for equal two-branch weighting. Heldfit excludes held images and held-domain donors; bank/backbone previously saw allsources.',updates=400,seed=0,architecture='Zero-initialized linear768->10 additive residual, trueCLS+patchmax, train-only mean/std/sqrt2; frozen view-specific global logits',optimizer='Adam lr.001 weight_decay1e-6 fixed, diagnostic only',loss='Equal present-domain mean .5CE(originalglobal+delta)+.5CE(viewglobal+delta), no SIRM/router/discovery',schedule='32 images with replacement per presentdomain, same indices both policies; donor deterministic step%numberallowed, unboxed/empty repeat original',evaluation='Original source-out only, final400 update without selection; natural repair/harm, CE, RMS and zero-correction baseline; target excluded',minimum_samples=2,code_sha256=digest(Path(__file__)))
 if (DEST/'plan.json').exists():assert json.loads((DEST/'plan.json').read_text())==plan
 else:write(DEST/'plan.json',plan);(DEST/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
 fp=DEST/'features.pt'
 if fp.exists():assert json.loads((DEST/'extraction_audit.json').read_text())['cache_sha256']==digest(fp);return
 swapped=raw[:,None,:].repeat(1,3,1);sg=g[:,None,:].repeat(1,3,1);errors=[]
 for start in range(0,len(job),16):
  group=job[start:start+16];images=[];mapping=[]
  for i,bb in group:
   with Image.open(ROOT/'domainbed/data/terra_incognita'/str(ids[i])) as im:x=tr(im.convert('RGB'))[None].cuda()
   mask=torch.tensor(box_mask(bb),device='cuda')[None,None];outside=(~mask).expand_as(x);alpha=torch.where(mask,torch.ones_like(mask).float(),gaussian_blur(mask.float(),[15,15],[3.,3.]));tm=x.masked_select(outside).view(3,-1).mean(1)[None,:,None,None];ts=x.masked_select(outside).view(3,-1).std(1,unbiased=False).clamp_min(1e-4)[None,:,None,None]
   images.append(x);mapping.append((i,None))
   for di,d in enumerate([0,2,3]):
    bg=donors[d];dm=bg.masked_select(outside).view(3,-1).mean(1)[None,:,None,None];ds=bg.masked_select(outside).view(3,-1).std(1,unbiased=False).clamp_min(1e-4)[None,:,None,None];nb=(bg-dm)/ds*ts+tm;view=alpha*x+(1-alpha)*nb;assert torch.equal(view.masked_select(mask.expand_as(x)),x.masked_select(mask.expand_as(x)));images.append(view);mapping.append((i,di))
  with torch.no_grad():_,rr=pooled_raw_features(m.featurizer,torch.cat(images));gg=m.global_head(rr[:,:384])
  for k,(i,di) in enumerate(mapping):
   if di is None:errors.append(float((rr[k].cpu()-raw[i]).abs().max()));assert errors[-1]<5e-5
   else:swapped[i,di]=rr[k].cpu();sg[i,di]=gg[k].cpu()
  write(DEST/'progress.json',dict(status='extracting_source',boxed_completed=min(start+16,len(job)),boxed_total=len(job)));print('sourcecache',min(start+16,len(job)),flush=True)
 assert all(torch.equal(v,m.featurizer.state_dict()[k].cpu()) for k,v in before.items());assert all(q.grad is None for q in m.parameters());assert digest(path)==plan['checkpoint_sha256'];out=data['source_out'];vo=torch.tensor(np.concatenate([out['cls'],out['patch_max'] if 'patch_max' in out else out['patchmax']],1));assert torch.isfinite(swapped).all() and torch.isfinite(sg).all();torch.save(dict(raw=raw,global_logits=g,y=y,domains=dom,ids=ids,context_raw=swapped,context_global=sg,boxed_indices=available,validation_raw=vo,validation_global=torch.tensor(out['global_logits']),validation_y=torch.tensor(out['y']),validation_domains=torch.tensor(out['domains']),validation_ids=out['ids']),fp);write(DEST/'extraction_audit.json',dict(cache_sha256=digest(fp),source_only=True,target_read=False,frozen_backbone_exact=True,original_feature_max_error=max(errors),object_pixels_exact=True));print('Extraction complete',flush=True)

def fit():
 a=torch.load(DEST/'features.pt',weights_only=False,map_location='cpu');result={};torch.set_num_threads(4)
 for held in [None,0,2,3]:
  name='pooled' if held is None else f'held{held}';keep=torch.ones(len(a['y']),dtype=torch.bool) if held is None else a['domains'].ne(held);ii=torch.nonzero(keep).flatten();domains=torch.unique(a['domains'][ii]).tolist();assert all((a['domains'][ii]==d).sum()>=2 for d in domains);mean=a['raw'][ii].mean(0);std=a['raw'][ii].std(0,unbiased=False).clamp_min(1e-6);z=(a['raw']-mean)/std/2**.5;v=(a['validation_raw']-mean)/std/2**.5;context=(a['context_raw']-mean)/std/2**.5;allowed=[j for j,d in enumerate([0,2,3]) if d!=held];gen=torch.Generator().manual_seed(0);schedule=[torch.cat([ix[torch.randint(len(ix),(32,),generator=gen)] for d in domains for ix in [ii[a['domains'][ii]==d]]]) for _ in range(400)];states={}
  for policy in ['original','paired_context']:
   head=torch.nn.Linear(768,10);torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias);opt=torch.optim.Adam(head.parameters(),lr=.001,weight_decay=1e-6);curve=[]
   for step,idx in enumerate(schedule):
    f=a['global_logits'][idx]+head(z[idx]);di=allowed[step%len(allowed)];ff=f if policy=='original' else a['context_global'][idx,di]+head(context[idx,di]);loss=.5*F.cross_entropy(f,a['y'][idx])+.5*F.cross_entropy(ff,a['y'][idx]);opt.zero_grad();loss.backward();opt.step()
    if step%25==0:curve.append(dict(step=step,loss=float(loss)))
   take=torch.ones(len(v),dtype=torch.bool) if held is None else a['validation_domains'].eq(held)
   with torch.no_grad():delta=head(v);f=a['validation_global']+delta;yy=a['validation_y'];gc=a['validation_global'].argmax(1)==yy;fc=f.argmax(1)==yy;ce=F.cross_entropy(f,yy,reduction='none');gce=F.cross_entropy(a['validation_global'],yy,reduction='none');evaldom=torch.unique(a['validation_domains'][take]);acc=float(torch.stack([fc[take&a['validation_domains'].eq(d)].float().mean() for d in evaldom]).mean());base=float(torch.stack([gc[take&a['validation_domains'].eq(d)].float().mean() for d in evaldom]).mean());meanCE=float(torch.stack([ce[take&a['validation_domains'].eq(d)].mean() for d in evaldom]).mean())
   states[policy]=dict(accuracy=acc,global_accuracy=base,CE=meanCE,repairs=int((take&~gc&fc).sum()),harms=int((take&gc&~fc).sum()),correction_RMS=float(delta[take].square().mean().sqrt()),CE_change_correct=float((ce-gce)[take&gc].mean()),CE_change_wrong=float((ce-gce)[take&~gc].mean()),curve=curve)
   np.savez_compressed(DEST/f'{name}_{policy}_validation.npz',ids=a['validation_ids'][take.numpy()],y=yy[take].numpy(),domains=a['validation_domains'][take].numpy(),global_logits=a['validation_global'][take].numpy(),full_logits=f[take].numpy(),delta=delta[take].numpy());torch.save(dict(head=head.state_dict(),mean=mean,std=std,held=held,policy=policy),DEST/f'{name}_{policy}_reader.pt')
  result[name]=states;write(DEST/'progress.json',dict(status='fit_complete',fit=name));print(name,states,flush=True)
 write(DEST/'results.json',dict(source_only=True,target_read=False,primary_SIRM_training=False,states=result,cache_sha256=digest(DEST/'features.pt'),plan_sha256=digest(DEST/'plan.json'),scientific_goal_achieved=False));write(DEST/'progress.json',dict(status='complete'));print('Source reader diagnostic complete',flush=True)
if __name__=='__main__':
 torch.set_num_threads(4);extract();fit()

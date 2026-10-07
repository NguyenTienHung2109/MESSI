"""Frozen source fusion: object versus equal-area background erasure, no fits."""
import json
from pathlib import Path
import torch,numpy as np
from PIL import Image
from torchvision.transforms.functional import gaussian_blur
from .terra_attention_eda import ROOT,OUT,load_models,box_mask
from .terra_object_mask_population import bg_masks
from .terra_local_class_hierarchical_uniform import joint_logprob
from .terra_local_fusion_control import DEST,SOURCE
from experiments.sirm_h1_h6.run import ViewTransform
from .shrinkage import write,digest

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;output=DEST/'object_background_intervention';output.mkdir(exist_ok=True);assert not (output/'plan.json').exists();population=OUT/'source_object_population/results.json';records=json.loads(population.read_text())['records'];probe_path=SOURCE/'pooled_probe.pt';reader_path=DEST/'global_local.pt';probe=torch.load(probe_path,map_location='cpu',weights_only=False);reader=torch.load(reader_path,map_location='cpu',weights_only=False);locks={str(p):digest(p) for p in [population,probe_path,reader_path]};write(output/'plan.json',dict(source_only=True,target_read=False,fitting=False,scope='All206 boxed source-out,196 with up to3 translated nonoverlapping equal-area background controls; normalized tensor zero-fill and Gaussian blur31 sigma7, same controls as original population',code_sha256=digest(Path(__file__)),input_sha256=locks));(output/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());current,historical,fw,model_locks=load_models();del historical;version=[p._version for p in current.parameters()];t=ViewTransform(current.config,'original',False);result=[]
 for n,r in enumerate(records):
  x=t(Image.open(ROOT/'domainbed/data/terra_incognita'/r['image_id']).convert('RGB'))[None].cuda();obj=box_mask(r['boxes']);bg=bg_masks(obj);blur=gaussian_blur(x,[31,31],[7.,7.]);views={'original':x}
  for region,m in [('object',obj)]+[(f'background{k}',m) for k,m in enumerate(bg)]:
   mask=torch.tensor(m,device='cuda')[None,None];views[f'{region}_mean']=torch.where(mask,torch.zeros_like(x),x);views[f'{region}_blur']=torch.where(mask,blur,x)
  captured=[];hook=current.featurizer.vit.norm.register_forward_hook(lambda m,args,out:captured.append(out))
  try:
   with torch.no_grad():g=fw['strong_ERM'](torch.cat(list(views.values()))).cpu()
  finally:hook.remove()
  patches=captured[0][:,2:].cpu();prob=joint_logprob(((patches-probe['mean'])/probe['std'])@probe['weight'].T+probe['bias']).exp();ix=(1-prob[:,:,5]).topk(8,1).indices;q=prob.gather(1,ix[:,:,None].expand(-1,-1,10)).mean(1).clamp_min(1e-8).log();feature=torch.cat([g,q],1);correction=((feature-reader['mean'])/reader['std']/np.sqrt(2))@reader['weight'].T+reader['bias']-reader['center'];f=g+correction;label=r['label'];entries={}
  for i,key in enumerate(views):
   others=f[i].clone();others[label]=-torch.inf;entries[key]=dict(global_prediction=int(g[i].argmax()),fusion_prediction=int(f[i].argmax()),correct=bool(f[i].argmax()==label),GT_margin=float(f[i,label]-others.max()),local_logprob_change=float((q[i]-q[0]).norm()),correction_change=float((correction[i]-correction[0]).norm()))
  assert np.allclose(g[0].numpy(),r['models']['strong_ERM']['views']['original']['logits'],atol=1e-4,rtol=1e-5);result.append(dict(id=r['image_id'],label=label,boxes=r['boxes'],background_controls=len(bg),views=entries))
  if (n+1)%25==0:print(n+1,'/',len(records),flush=True)
 summary={}
 for kind in ['mean','blur']:
  paired=[r for r in result if r['background_controls']];summary[kind]=dict(n=len(paired))
  for metric in ['local_logprob_change','correction_change','GT_margin_drop']:
   arr=[]
   for r in paired:
    a=r['views']['original'];b=r['views'][f'object_{kind}'];controls=[r['views'][f'background{k}_{kind}'] for k in range(r['background_controls'])];od=a['GT_margin']-b['GT_margin'] if metric=='GT_margin_drop' else b[metric];bd=np.mean([a['GT_margin']-z['GT_margin'] if metric=='GT_margin_drop' else z[metric] for z in controls]);arr.append([od,bd])
   ar=np.array(arr);summary[kind][metric]=dict(object_median=float(np.median(ar[:,0])),mean_background_median=float(np.median(ar[:,1])),object_larger_fraction=float((ar[:,0]>ar[:,1]).mean()))
 assert version==[p._version for p in current.parameters()] and all(p.grad is None and not p.requires_grad for p in current.parameters())
 for p,h in {**locks,**model_locks}.items():assert digest(Path(p))==h
 write(output/'results.json',dict(records=result,summary=summary,source_only=True,target_read=False,fitting=False,parameters_unchanged=True,input_sha256={**locks,**model_locks}));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

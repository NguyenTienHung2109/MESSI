"""Source-only native-resolution ROI controls, no fitting or checkpoint selection."""
import json,csv
from pathlib import Path
import torch
from PIL import Image
from .terra_attention_eda import ROOT,CURRENT,OUT,load_models
from experiments.sirm_h1_h6.run import ViewTransform
from .shrinkage import write,digest

def run():
 torch.set_num_threads(4)
 out=OUT/'native_source_roi';out.mkdir(exist_ok=True);assert not (out/'plan.json').exists()
 official=json.loads((OUT/'caltech_bboxes_20200316.json').read_text());cats={a['id']:a['name'] for a in official['categories']};sizes={a['id']:(a['width'],a['height']) for a in official['images']};boxes={}
 for a in official['annotations']:
  if a.get('bbox'):boxes.setdefault(a['image_id'],[]).append((cats[a['category_id']],a['bbox']))
 cases=list(csv.DictReader((CURRENT/'eda/cases.csv').open()));rows=[]
 for r in cases:
  id=Path(r['image_id']).stem
  bb=[b for label,b in boxes.get(id,[]) if label==r['class_name']]
  if r['split']=='source_out' and bb:rows.append((r,bb))
 write(out/'plan.json',dict(scope='All source-out images with official class-matched boxes; target excluded',padding=[0,.25,1],controls='Native ROI vs crop after 224 resize vs centered same-size native crop',cases=len(rows),training=False,selection=False,code_sha256=digest(Path(__file__))))
 (out/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
 current,historical,forwards,locks=load_models();t=ViewTransform(current.config,'original',False);results=[]
 with torch.no_grad():
  for r,bb in rows:
   path=ROOT/'domainbed/data/terra_incognita'/r['image_id'];img=Image.open(path).convert('RGB');w,h=img.size;ow,oh=sizes[Path(r['image_id']).stem];x0=min(b[0] for b in bb)/ow;y0=min(b[1] for b in bb)/oh;x1=max(b[0]+b[2] for b in bb)/ow;y1=max(b[1]+b[3] for b in bb)/oh;label=int(r['true_class']);views={'original':img}
   for pad in [0,.25,1]:
    bw=x1-x0;bh=y1-y0;rect=(max(0,x0-pad*bw),max(0,y0-pad*bh),min(1,x1+pad*bw),min(1,y1+pad*bh))
    native=tuple(round(v*s) for v,s in zip(rect,[w,h,w,h]));small=tuple(round(v*224) for v in rect)
    views[f'native_pad{pad}']=img.crop(native);views[f'resized_pad{pad}']=img.resize((224,224),Image.Resampling.BILINEAR).crop(small)
    cw=native[2]-native[0];ch=native[3]-native[1];cx=(w-cw)//2;cy=(h-ch)//2;views[f'center_control_pad{pad}']=img.crop((cx,cy,cx+cw,cy+ch))
   batch=torch.cat([t(v)[None] for v in views.values()]).cuda();rec=dict(image_id=r['image_id'],class_name=r['class_name'],label=label,area=(x1-x0)*(y1-y0),models={})
   for name,forward in forwards.items():
    logits=forward(batch).cpu();rec['models'][name]={key:dict(prediction=int(z.argmax()),correct=bool(z.argmax()==label),GT_logit=float(z[label])) for key,z in zip(views,logits)}
   results.append(rec)
 summary={}
 for name in forwards:
  summary[name]={}
  for key in results[0]['models'][name]:
   summary[name][key]=dict(n=len(results),correct=sum(r['models'][name][key]['correct'] for r in results),repairs=sum(not r['models'][name]['original']['correct'] and r['models'][name][key]['correct'] for r in results),harms=sum(r['models'][name]['original']['correct'] and not r['models'][name][key]['correct'] for r in results))
 write(out/'results.json',dict(records=results,summary=summary,source_only=True,oracle_box_diagnostic=True,deployment_accuracy=False,training=False));print(json.dumps(summary,indent=2))
if __name__=='__main__':run()

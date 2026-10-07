"""Immutable fixed-bank routing magnitude controls and source case audit."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .source_direct_mixture_reliability_router import OUT
from .source_view_reliability_router import OUT as TEACHER
from .source_router_teacher_fit import score
from .terra_attention_eda import ROOT
from .shrinkage import write,digest

def main():
 torch.set_num_threads(4);manifest=json.loads((OUT.parent/'experiment/subset_manifest.json').read_text());names=manifest['class_names'];source=json.loads((OUT/'results.json').read_text());result={}
 for state in ['selected','last']:
  folds={}
  for held in [None,0,2,3]:
   controls={}
   for control in ['confidence_only','confidence_plus_view']:
    p=torch.load(OUT/f'{state}_held{held}_{control}.pt',map_location='cpu',weights_only=False);train=p['replays']['source_in'];val=p['replays']['source_out'];corr=train['learned_logits']-train['global_logits'];reference=corr.square().mean().sqrt();nativecorr=train['native_logits']-train['global_logits'];native_scale=float(reference/nativecorr.square().mean().sqrt());meanpi=torch.stack([train['learned_pi'][train['domains'].eq(d)].mean(0) for d in train['domains'].unique()]).mean(0);const_corr=(meanpi[:6][None,:,None]*train['delta']).sum(1);constant_scale=float(reference/const_corr.square().mean().sqrt());g=val['global_logits'];y=val['y'];d=val['domains'];dd=val['delta'];teacher=val['teacher'];pi=val['learned_pi'];learned=val['learned_logits'];constant=g+constant_scale*(meanpi[:6][None,:,None]*dd).sum(1);native=g+native_scale*(val['native_logits']-g);scores={}
    for name,f,pp in [('global',g,torch.cat([torch.zeros(len(y),6),torch.ones(len(y),1)],1)),('learned',learned,pi),('native_scaled_to_learned',native,pi),('constant_scaled_to_learned',constant,meanpi[None].expand(len(y),-1))]:scores[name]=score(g,f,y,d,pp,dd,teacher)
    # pi field for the native-scaled score is not its actual native router, so
    # router_NULL/teacher_L1 are omitted rather than silently attributed.
    for key in ['router_NULL','teacher_router_L1']:scores['native_scaled_to_learned'].pop(key)
    gain=F.cross_entropy(g,y,reduction='none')-F.cross_entropy(learned,y,reduction='none');domains={int(dom):dict(n=int(d.eq(dom).sum()),CE_gain=float(gain[d.eq(dom)].mean())) for dom in d.unique()};cases=[]
    for i,id in enumerate(val['ids']):
     gp=int(g[i].argmax());fp=int(learned[i].argmax());cp=int(constant[i].argmax());npred=int(native[i].argmax());label=int(y[i])
     if gp!=fp:cases.append(dict(id=id,label=names[label],global_prediction=names[gp],learned_prediction=names[fp],scaled_native_prediction=names[npred],scaled_constant_prediction=names[cp],event='repair' if fp==label else 'harm' if gp==label else 'wrong_to_wrong',CE_gain=float(gain[i]),pi=pi[i].tolist()))
    controls[control]=dict(scores=scores,native_source_train_RMS_match_scale=native_scale,constant_source_train_RMS_match_scale=constant_scale,source_train_mean_pi=meanpi.tolist(),domain_CE_gain=domains,cases=cases)
   folds[str(held)]=controls
  result[state]=folds
 write(OUT/'magnitude_and_case_eda.json',dict(states=result,source_only=True,target_read=False,primary_state='selected200',secondary_state='last399 mechanism only',fitting=False,code_sha256=digest(Path(__file__))))
 # 16 cases: all8 primary endpoint harms + first8 of13 endpoint repairs.
 c=result['last']['None']['confidence_only']['cases'];harms=sorted([x for x in c if x['event']=='harm'],key=lambda x:x['id']);repairs=sorted([x for x in c if x['event']=='repair'],key=lambda x:x['id']);selected=(harms+repairs[:16-len(harms)]);assert len(selected)==16;fig,axes=plt.subplots(4,4,figsize=(14,15))
 for j,(ax,c) in enumerate(zip(axes.flat,selected),1):
  ax.imshow(Image.open(ROOT/'domainbed/data/terra_incognita'/c['id']).convert('RGB').resize((224,224),Image.Resampling.BILINEAR));ax.set_title(f"{j:02d} {c['event']} | GT:{c['label']}\nERM:{c['global_prediction']} → routed:{c['learned_prediction']}\nScaled native:{c['scaled_native_prediction']} | constant:{c['scaled_constant_prediction']}",fontsize=8);ax.axis('off')
 fig.suptitle('Frozen endpoint399 bank: direct-risk router harms and repairs',fontsize=13);fig.text(.5,.015,'Source-only. New400-update diagnostic router fit; frozen bank/global unchanged.\nEndpoint399 is secondary, not the source-selected checkpoint. No SIRM execution or target result.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.045,1,.97));fig.savefig(OUT/'endpoint_cases_4x4.png',dpi=130);plt.close(fig);write(OUT/'grid_manifest.json',dict(cases=selected,selection='All8 harms then first8 of13 repairs, sorted imageID. Last399 secondary diagnostic only.',source_only=True))
 print({state:{name:{k:v[k] for k in ['accuracy_equal_domain','CE_equal_domain','repairs','harms']} for name,v in result[state]['None']['confidence_only']['scores'].items()} for state in result})
if __name__=='__main__':main()

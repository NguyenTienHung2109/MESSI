import json,torch
from pathlib import Path
from experiments.sirm_research_loop.terra_object_pool_readout import score
root=Path('results/terra_attention_diagnosis/phase1/source_object_probe');A=torch.load(root/'local_class_probe/all_image_replay/pooled_replay.pt',map_location='cpu',weights_only=False);B=torch.load(root/'local_class_hard_negative/all_image_replay/pooled_replay.pt',map_location='cpu',weights_only=False)
scale=float((A['source_in']['additive_logits']-A['source_in']['global_logits']).square().mean().sqrt()/(B['source_in']['additive_logits']-B['source_in']['global_logits']).square().mean().sqrt());out=B['source_out'];result={'scale_fixed_from_source_in_RMS':scale,'train_reference':'mean-empty-loss additive residual RMS','scaled_hard_source_out':score(out['global_logits'],out['global_logits']+scale*(out['additive_logits']-out['global_logits']),out['labels'],out['domains']),'source_only':True,'fitting':False,'target_read':False}
boxes={r['image_id']:r['boxes'] for r in json.loads((root.parent/'source_object_population/results.json').read_text())['records']}
for name,data in [('mean_empty_loss',A),('hard_negative_loss',B)]:
 p=data['source_out'];counts={'boxed_n':0,'top1_inside':0,'any_top8_inside':0,'correct_class_given_top1_inside':0,'correct_additive_given_top1_inside':0}
 for i,id in enumerate(p['ids']):
  if id not in boxes:continue
  counts['boxed_n']+=1;selected=p['top8_indices'][i].tolist();inside=[]
  for ix in selected:
   r,c=divmod(ix,14);x=(c+.5)/14;y=(r+.5)/14;inside.append(any(bx<=x<=bx+w and by<=y<=by+h for bx,by,w,h in boxes[id]))
  counts['top1_inside']+=inside[0];counts['any_top8_inside']+=any(inside)
  if inside[0]:
   counts['correct_class_given_top1_inside']+=int(p['standalone_logits'][i].argmax()==p['labels'][i]);counts['correct_additive_given_top1_inside']+=int(p['additive_logits'][i].argmax()==p['labels'][i])
 result[name]=counts
(root/'local_class_hard_negative/amplitude_and_localization_controls.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

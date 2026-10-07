"""Source-only learned score decomposition; no fitting or parameter changes."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from .conditioned_router_source_analysis import DEST,load
from .source_direct_mixture_reliability_router import FEATURES
from .shrinkage import ROOT,digest,write

def main():
 torch.set_num_threads(4);locks={};records={};cache=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';data={}
 for split in ['source_in','source_out']:
  p=cache/f'{split}_tokens.npz';locks[str(p)]=digest(p);a=np.load(p);data[split]=torch.tensor(np.concatenate([a['cls'],a['patch_max']],1))
 for state in ['selected','last']:
  m,p,path=load(DEST,state);locks[str(path)]=digest(path);m.requires_grad_(False);before=copy.deepcopy(m.state_dict());splits={}
  for split,raw in data.items():
   captured=[];hook=m.router.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
   with torch.no_grad():pi=m.router(m.bank_features(raw)).softmax(1)
   hook.remove();assert len(captured)==1;x=captured[0].double();w=m.router.head.weight.detach().double()[0];bias=m.router.head.bias.detach().double()[0];contribution=x*w;score=contribution.sum(2)+bias;replayed=torch.cat([score,score.new_zeros(len(score),1)],1).softmax(1);err=float((replayed-pi.double()).abs().max());assert err<1e-6
   centered=contribution-contribution.mean(1,keepdim=True);centered_score=score-score.mean(1,keepdim=True);common_score=score.mean(1);features=[]
   for j,name in enumerate(FEATURES[:11]):features.append(dict(feature=name,weight=float(w[j]),mean_feature_value=float(x[:,:,j].mean()),feature_sd_across_images_and_experts=float(x[:,:,j].std(unbiased=False)),within_image_expert_feature_RMS=float((x[:,:,j]-x[:,:,j].mean(1,keepdim=True)).square().mean().sqrt()),within_image_expert_score_contribution_RMS=float(centered[:,:,j].square().mean().sqrt())))
   splits[split]=dict(n=len(raw),feature_decomposition=features,mean_common_expert_score=float(common_score.mean()),common_score_sd=float(common_score.std(unbiased=False)),expert_contrast_score_RMS=float(centered_score.square().mean().sqrt()),score_probability_replay_error=err)
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and all(q.grad is None for q in m.parameters());records[state]=dict(global_logit_unit=float(m.router.global_logit_unit),head_bias=float(m.router.head.bias),head_weight_norm=float(m.router.head.weight.norm()),splits=splits)
 for p,h in locks.items():assert digest(Path(p))==h
 write(DEST/'OFF_descriptor_audit.json',dict(source_only=True,target_read=False,fitting=False,parameter_updates=False,states=records,input_sha256=locks,code_sha256=digest(Path(__file__)),scope='OFF current source-selected/final399 normalized score decomposition. Contributions correlated and cannot be summed as unique causal effects. Global descriptor columns are identical across experts by construction. No optimization setting changed.'))
 print(json.dumps({s:{'weight_norm':v['head_weight_norm'],'bias':v['head_bias'],'source_out_expert_contrast_RMS':v['splits']['source_out']['expert_contrast_score_RMS'],'source_out_common_score_mean':v['splits']['source_out']['mean_common_expert_score']} for s,v in records.items()},indent=2))
if __name__=='__main__':main()

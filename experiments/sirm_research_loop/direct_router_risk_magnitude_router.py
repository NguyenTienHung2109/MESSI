"""Fixed-bank, source-only shared correction reliability routing diagnostic."""
import json,copy,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from experiments.sirm_discover.math import router_teacher
BANK=Path('results/sirm_research_loop/iteration_17')
from .routed_conditional_utility import load
from .source_probe_penalty_direction import outputs
from .source_router_teacher_fit import score
from .terra_attention_eda import OUT as ATTENTION
from .shrinkage import write,digest
OUT=BANK/'source_magnitude_aware_reliability_router'
FEATURES=['global_confidence','global_entropy','global_margin','candidate_confidence','candidate_entropy','candidate_margin','global_class_probability_change','margin_change','prediction_changed','delta_RMS','delta_at_global_prediction','blur_relative_delta_change','blur_delta_cosine','candidate_view_probability_L2','candidate_view_prediction_changed','candidate_view_entropy_change']

def entropy(p):return -(p*p.clamp_min(1e-12).log()).sum(-1)

def descriptor(g,delta,blur_delta):
 gp=g.softmax(-1);candidate=g[:,None]+delta;cp=candidate.softmax(-1);bp=(g[:,None]+blur_delta).softmax(-1);gt=g.topk(2,1).values;ct=candidate.topk(2,2).values;idx=g.argmax(1)[:,None,None].expand(-1,6,1);repeat=lambda x:x[:,None].expand(-1,6)
 feats=[repeat(gp.max(1).values),repeat(entropy(gp)),repeat(gt[:,0]-gt[:,1]),cp.max(2).values,entropy(cp),ct[:,:,0]-ct[:,:,1],cp.gather(2,idx).squeeze(2)-repeat(gp.max(1).values),(ct[:,:,0]-ct[:,:,1])-repeat(gt[:,0]-gt[:,1]),candidate.argmax(2).ne(g.argmax(1)[:,None]).float(),delta.square().mean(2).sqrt(),delta.gather(2,idx).squeeze(2),(delta-blur_delta).norm(dim=2)/delta.norm(dim=2).clamp_min(1e-8),F.cosine_similarity(delta,blur_delta,dim=2), (cp-bp).norm(dim=2),cp.argmax(2).ne(bp.argmax(2)).float(),entropy(bp)-entropy(cp)]
 x=torch.stack(feats,2).double();assert x.shape[1:]==(6,len(FEATURES)) and torch.isfinite(x).all();return x

def run():
 torch.set_num_threads(4);torch.manual_seed(0);OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists();audit=json.loads((BANK/'final_analysis_audit.json').read_text())
 for rel,h in audit['sha256'].items():assert digest(BANK/rel)==h,rel
 paths={s:ATTENTION/'source_object_probe/blur_residual_control'/f'{s}_features.pt' for s in ['source_in','source_out']};locks={str(p):digest(p) for p in paths.values()};data={s:torch.load(p,map_location='cpu',weights_only=False) for s,p in paths.items()};write(OUT/'plan.json',dict(hypothesis='Single-expert teacher imitation may differ from true mixture risk. Sign-balanced confidence probes distinguish many small utility signs yet selected groups have negative mean utility in2/3 source domains. With expert tensors fixed, actual mixture CE may account for correction magnitude better; compare constant/native routing at matched train RMS before integration.',source_only=True,target_read=False,primary_state='selected350',secondary_state='last399 mechanism only; no checkpoint selection',features=FEATURES,architecture='One shared linear16->1 scalar for every expert; NULL score fixed0; no expert ID, labels, domain ID or raw backbone inputs at inference; last5 descriptor weights remain unused',controls=['confidence_only'],confidence_only='Last5 view descriptors zeroed; identical parameter count, optimizer, teacher targets and sampling',training='400 fixed Adam.01 updates,L2.01, actual mixture CE with fixed residual experts; raw-gain teacher only initializes prior and supports reporting; confidence-only sign-probe scores are not deployed. Each source domain16 correct+16 error images/update, minimum2/cohort. Initialize shared bias from training teacher NULL prior. Train-only pooled expert-feature normalization; no source-out selection.',auxiliary_holdouts='All-source plus3 fits excluding a source domain from router fitting only; frozen bank/backbone saw all3 sources',limits='Fixed-bank diagnostic fit adds400 router updates after old400 bank updates; not a matched400 Stage2/SIRM run. No new factors/SIRM/target gain claim.',code_sha256=digest(Path(__file__)),input_sha256=locks));(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes());states={}
 for state in ['selected','last']:
  m,p,path=load(BANK,state);m.requires_grad_(False);before=copy.deepcopy(m.state_dict());locks[str(path)]=digest(path);m.calibrate_features(data['source_in']['original'],data['source_in']['domains']);fixed={}
  with torch.no_grad():
   for split,a in data.items():
    raw=a['original'];pi=m.router(m.bank_features(raw)).softmax(1);g,delta,full=outputs(m,raw,pi);br=a['blur'];bpi=m.router(m.bank_features(br)).softmax(1);_,bd,_=outputs(m,br,bpi);raw_ce=torch.stack([F.cross_entropy(g+delta[:,e],a['labels'],reduction='none') for e in range(6)],1);gain=F.cross_entropy(g,a['labels'],reduction='none')[:,None]-raw_ce;teacher=router_teacher(gain,m.config['router_temperature'],m.config['utility_floor']);fixed[split]=dict(x=descriptor(g,delta,bd),g=g,delta=delta,full=full,pi=pi,teacher=teacher,y=a['labels'],dom=a['domains'],ids=a['ids'])
  fits={}
  for held in [None,0,2,3]:
   train=fixed['source_in'];keep=torch.ones(len(train['y']),dtype=torch.bool) if held is None else train['dom'].ne(held);dom=train['dom'][keep];y=train['y'][keep];g=train['g'][keep];correct=g.argmax(1).eq(y);teacher=train['teacher'][keep].double();domains=sorted(dom.unique().tolist());cohorts={};gen=torch.Generator().manual_seed(0)
   for d in domains:
    for flag in [False,True]:ix=torch.nonzero(dom.eq(d)&correct.eq(flag)).flatten();assert len(ix)>=2;cohorts[(d,flag)]=ix
   schedule=[torch.cat([ix[torch.randint(len(ix),(16,),generator=gen)] for ix in cohorts.values()]) for _ in range(400)];controls={}
   for control in ['confidence_only']:
    raw=train['x'][keep].clone()
    if control=='confidence_only':raw[:,:,11:]=0
    mean=raw.mean((0,1));std=raw.std((0,1),unbiased=False).clamp_min(1e-8);x=(raw-mean)/std;head=torch.nn.Linear(len(FEATURES),1).double();head.weight.data.zero_();prior=teacher[:,-1].mean().clamp(.001,.999);head.bias.data.fill_(float(((1-prior)/(6*prior)).log()));opt=torch.optim.Adam(head.parameters(),lr=.01)
    for ix in schedule:
     logits=torch.cat([head(x[ix]).squeeze(-1),torch.zeros(len(ix),1,dtype=torch.float64)],1);pi=logits.softmax(1);mixture=g[ix].double()+(pi[:,:6,None]*train['delta'][keep][ix].double()).sum(1);loss=F.cross_entropy(mixture,y[ix])+.01*head.weight.square().sum();opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
     splits={};replays={};trainpi=torch.cat([head(x).squeeze(-1),torch.zeros(len(x),1,dtype=torch.float64)],1).softmax(1).float();td=train['delta'][keep];nativecorr=(train['pi'][keep,:6,None]*td).sum(1);fitcorr=(trainpi[:,:6,None]*td).sum(1);scale=float(nativecorr.square().mean().sqrt()/fitcorr.square().mean().sqrt())
     for split,a in fixed.items():
      take=keep if split=='source_in' else (torch.ones(len(a['y']),dtype=torch.bool) if held is None else a['dom'].eq(held));raw=a['x'][take].clone()
      if control=='confidence_only':raw[:,:,11:]=0
      xx=(raw-mean)/std;pi=torch.cat([head(xx).squeeze(-1),torch.zeros(len(xx),1,dtype=torch.float64)],1).softmax(1).float();gg=a['g'][take];dd=a['delta'][take];yy=a['y'][take];doms=a['dom'][take];teacher0=a['teacher'][take];corr=(pi[:,:6,None]*dd).sum(1);f=gg+corr;oracle=gg+(teacher0[:,:6,None]*dd).sum(1);splits[split]={policy:score(gg,ff,yy,doms,pp,dd,teacher0) for policy,ff,pp in [('native',a['full'][take],a['pi'][take]),('learned',f,pi),('learned_train_RMS_matched',gg+scale*corr,pi),('label_aware_teacher',oracle,teacher0)]};repairs=gg.argmax(1).ne(yy)&oracle.argmax(1).eq(yy);splits[split]['oracle_repair_recovery']=dict(n=int(repairs.sum()),learned_correct=int(f.argmax(1)[repairs].eq(yy[repairs]).sum()));replays[split]=dict(ids=[id for id,flag in zip(a['ids'],take.tolist()) if flag],y=yy,domains=doms,global_logits=gg,delta=dd,native_logits=a['full'][take],learned_logits=f,learned_pi=pi,teacher=teacher0)
     file=OUT/f'{state}_held{held}_{control}.pt';torch.save(dict(weight=head.weight,bias=head.bias,mean=mean,std=std,replays=replays),file);controls[control]=dict(splits=splits,source_train_RMS_match_scale=scale,model_sha256=digest(file));print(state,held,control,{s:{k:v['learned'][k] for k in ['accuracy_equal_domain','CE_equal_domain','repairs','harms','router_NULL']} for s,v in splits.items()},flush=True)
   fits[str(held)]=dict(controls=controls,schedule_sha256=hashlib.sha256(torch.stack(schedule).numpy().tobytes()).hexdigest())
  states[state]=dict(checkpoint_step=p.get('selected_step',p.get('step')),fits=fits);m.load_state_dict(before);assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and all(q.grad is None for q in m.parameters())
 for path,h in locks.items():assert digest(Path(path))==h
 write(OUT/'results.json',dict(states=states,source_only=True,target_read=False,diagnostic_fitting=True,primary_training=False,checkpoint_changes=False,input_sha256=locks));print('complete',flush=True)
if __name__=='__main__':run()

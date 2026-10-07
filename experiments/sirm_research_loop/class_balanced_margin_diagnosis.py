"""Source-only gradient and disposable finite margin controls after error EDA."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover.math import center_logits,expert_ce
from .class_balanced_model import equal_class_cost
from .class_coherent_discovery import class_coherent_cost
from .class_balanced_run import DEST,PRIOR,ROOT
from .spatial_bank_model import SpatialBankGateSIRM
from .spatial_bank_replay import spatial_head
from .discovery_error_gradient_diagnosis import vector
from .shrinkage import write,digest
RADIUS=.01

def graph(m,raw,y,centering):
    z=m.bank_features(raw)
    with torch.no_grad():g=m.global_head(raw[:,:384]);pi=m.router(z).softmax(1)
    d=center_logits(torch.stack([e(z) for e in m.residuals],1));d=d-(d.mean(0,keepdim=True) if centering=='batch' else m.expert_source_mean[None]);f=g+(d*pi[:,:6,None]).sum(1)
    return g,d,f

def margin(f,y):
    competing=f.clone();competing[torch.arange(len(y)),y]=-torch.inf
    return f[torch.arange(len(y)),y]-competing.max(1).values

def classmean(v,y,mask):
    labels=y[mask].unique();return torch.stack([v[mask&y.eq(c)].mean() for c in labels]).mean() if len(labels) else v.sum()*0

def gradients(m,raw,y,dom,centering):
    g,d,f=graph(m,raw,y,centering);losses=expert_ce(g[:,None]+d,y);wrong=g.argmax(1).ne(y);mar=margin(f,y);params=list(m.residuals.parameters())
    with torch.no_grad():pi=m.router(m.bank_features(raw)).softmax(1)
    routed_losses=expert_ce(g[:,None]+pi[:,:6,None]*d,y)
    objectives={'original_discovery':class_coherent_cost(losses,y),'equal_discovery':equal_class_cost(losses,y),'routed_discovery':equal_class_cost(routed_losses,y),'routed_CE':torch.stack([F.cross_entropy(f[dom==x],y[dom==x]) for x in dom.unique()]).mean(),'wrong_margin':-classmean(mar,y,wrong)}
    return {k:vector(v,params) for k,v in objectives.items()}

@torch.no_grad()
def metrics(m,raw,y,dom):
    g,d,pi=spatial_head(m,raw);f=g+(d*pi[:,:6,None]).sum(1);wrong=g.argmax(1).ne(y);correct=f.argmax(1).eq(y);mar=margin(f,y)
    ce=torch.stack([F.cross_entropy(f[dom==x],y[dom==x]) for x in dom.unique()]).mean()
    return {'CE':float(ce),'accuracy':float(torch.stack([correct[dom==x].double().mean() for x in dom.unique()]).mean()),'repairs':int((wrong&correct).sum()),'harms':int((~wrong&~correct).sum()),'wrong_margin_class_mean':float(classmean(mar,y,wrong)),'correct_margin_class_mean':float(classmean(mar,y,~wrong)),'bird_margin_mean':float(mar[y==0].mean()),'bird_correct':int((correct&(y==0)).sum())},mar,correct,wrong

def run():
    torch.set_num_threads(4);out=DEST/'source_routed_discovery_diagnosis';out.mkdir(exist_ok=True)
    plan={'source_only':True,'target_read':False,'training':False,'optimizer_updates':False,'checkpoint_changes':False,'method_changes':False,'radius':RADIUS,'radius_chosen_before_outcomes':True,'policies':['original_discovery','equal_discovery','routed_discovery','routed_CE','wrong_margin'],'scope':'Follow-up of completed margin diagnostic: add class-coherent individual routed-expert discovery to test unscaled-versus-deployed mismatch. Equal14 OFF saved375 and399, original source-in and six exact source H2 batches. Residual-only gradient directions with detached router; excludes router auxiliary and SIRM. Wrong-margin and routed-discovery directions are diagnostics, not trained candidates. Equal-L2 disposable perturbations recalibrate source-in expert means, preserve backbone/global/router, then restore all tensors. Source-out is previously used for selection; not independent replication.','rejection':'An error-margin improvement that harms correct cases or fails source-out CE/accuracy does not justify a new discovery loss.'}
    write(out/'plan.json',plan);(out/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';audit=json.loads((token/'extraction_audit.json').read_text());cache={};locks={}
    for split in ['source_in','source_out']:
        path=token/f'{split}_tokens.npz';assert digest(path)==next(r['cache_sha256'] for r in audit['rows'] if r['split']==split);locks[split]=digest(path)
        a=np.load(path);cache[split]=(torch.tensor(np.concatenate([a['cls'],a['patch_max']],1)),torch.tensor(a['y']),torch.tensor(a['domains']),a['ids'])
    assert not set(cache['source_in'][3])&set(cache['source_out'][3]);train,ty,td,_=cache['source_in'];val,vy,vd,vids=cache['source_out'];h2audit=json.loads((PRIOR/'source_class_allocation/extraction_audit.json').read_text());batches=[('original',None,train,ty,td,'saved')]
    for r in h2audit['rows']:
        path=PRIOR/'source_class_allocation'/f"H2_step{r['step']}.pt";assert digest(path)==r['sha256'];a=torch.load(path,map_location='cpu',weights_only=False);batches.append(('H2',r['step'],a['raw'],a['y'],a['domains'],'batch'));locks[str(r['step'])]=r['sha256']
    results=[];checkpoints={}
    for state,file in [('selected','source_best.pkl'),('last','last.pkl')]:
        path=DEST/'experiment/runs/L38/OFF'/file;p=torch.load(path,map_location='cpu',weights_only=False);c=p['config'];m=SpatialBankGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c);m.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()});m.configure_stage();m.eval();checkpoints[state]=digest(path);original={k:v.clone() for k,v in m.state_dict().items()};assert all(not q.requires_grad for mod in [m.featurizer,m.global_head] for q in mod.parameters())
        m.calibrate_features(train,td);mean=m.expert_source_mean.clone();params=list(m.residuals.parameters());copies=[q.detach().clone() for q in params];base,bmar,bc,bwrong=metrics(m,val,vy,vd)
        g,d,f=graph(m,val,vy,'saved');wm=vector(-classmean(margin(f,vy),vy,bwrong),params);cm=vector(-classmean(margin(f,vy),vy,~bwrong),params)
        for view,step,raw,y,dom,centering in batches:
            vecs=gradients(m,raw,y,dom,centering);policies={}
            for key,v in vecs.items():
                if not v.norm():policies[key]={'zero_gradient':True};continue
                move=(-v/v.norm()*RADIUS).float();pos=0
                with torch.no_grad():
                    for q,copy in zip(params,copies):n=q.numel();q.copy_(copy+move[pos:pos+n].reshape_as(q));pos+=n
                    actual=float(torch.cat([(q-copy).flatten().double() for q,copy in zip(params,copies)]).norm());assert abs(actual-RADIUS)<2e-6
                m.calibrate_features(train,td);s,mar,correct,wrong=metrics(m,val,vy,vd);changes=mar-bmar
                policies[key]={'gradient_norm':float(v.norm()),'actual_perturbation_norm':actual,'predicted_wrong_margin_class_mean_change':float(RADIUS*wm.dot(v)/v.norm()),'predicted_correct_margin_class_mean_change':float(RADIUS*cm.dot(v)/v.norm()),'metrics':s,'delta':{k:s[k]-base[k] for k in base},'wrong_cases_margin_improved':int(((changes>0)&bwrong).sum()),'wrong_cases_margin_worsened':int(((changes<0)&bwrong).sum()),'correct_cases_margin_worsened':int(((changes<0)&~bwrong).sum()),'new_correct_to_wrong':int((bc&~correct).sum()),'new_wrong_to_correct':int((~bc&correct).sum()),'changed_correctness_ids':[str(vids[i]) for i in torch.nonzero(bc!=correct).flatten().tolist()]}
                with torch.no_grad():
                    for q,copy in zip(params,copies):q.copy_(copy)
                    m.expert_source_mean.copy_(mean)
            results.append({'state':state,'checkpoint_step':p.get('selected_step',p.get('step')),'view':view,'batch_step':step,'baseline':base,'policies':policies});write(out/'progress.json',{'status':'finite_source_controls','state':state,'view':view,'batch_step':step,'records':len(results)})
        m.load_state_dict(original);assert all(torch.equal(v,m.state_dict()[k]) for k,v in original.items());assert all(q.grad is None for q in m.parameters());assert digest(path)==checkpoints[state]
    summary=[]
    for state in ['selected','last']:
        for view in ['original','H2']:
            rr=[r for r in results if r['state']==state and r['view']==view];summary.append({'state':state,'view':view,'n':len(rr),'policies':{k:{'mean_delta':{metric:float(np.mean([r['policies'][k]['delta'][metric] for r in rr])) for metric in rr[0]['baseline']},'CE_improved':sum(r['policies'][k]['delta']['CE']<0 for r in rr),'new_harms':sum(r['policies'][k]['new_correct_to_wrong'] for r in rr),'new_repairs':sum(r['policies'][k]['new_wrong_to_correct'] for r in rr)} for k in plan['policies']}})
    write(out/'results.json',{'plan':plan,'source_cache_sha256':locks,'checkpoint_sha256':checkpoints,'code_sha256':digest(Path(__file__)),'all_original_tensors_restored':True,'grad_fields_unassigned':True,'records':results,'summary':summary});write(out/'progress.json',{'status':'complete','scientific_goal_achieved':False,'remaining':'Interpret source-only results before proposing a new method.'});print(json.dumps(summary,indent=2))
if __name__=='__main__':run()

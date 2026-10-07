"""Saved-state exact-scheduled H2 utility diagnostic, never historical gradients."""
import json
from pathlib import Path
import numpy as np
import torch
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.sirm_discover.math import center_logits,residual_utilities
from .class_balanced_run import DEST,PRIOR,ROOT
from .spatial_bank_model import SpatialBankGateSIRM
from .shrinkage import write,digest
from .class_balanced_reliability import support

@torch.no_grad()
def run():
    torch.set_num_threads(4);out=DEST/'source_component_diagnosis';audit=json.loads((PRIOR/'source_class_allocation/extraction_audit.json').read_text());reuse=json.loads((DEST/'original_control_reuse_audit.json').read_text());assert reuse['passed'];rows=[];locks={}
    for root,label in [(PRIOR,'original'),(DEST,'equal')]:
        for state,file in [('selected','source_best.pkl'),('last','last.pkl')]:
            path=root/'experiment/runs/L38/OFF'/file;p=torch.load(path,map_location='cpu',weights_only=False);c=p['config'];m=SpatialBankGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c);m.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()});m.configure_stage();m.eval();before={k:v.clone() for k,v in m.state_dict().items()};locks[label+'/'+state]=digest(path)
            for row in audit['rows']:
                cache=PRIOR/'source_class_allocation'/f"H2_step{row['step']}.pt";assert digest(cache)==row['sha256'];a=torch.load(cache,map_location='cpu',weights_only=False);raw=a['raw'];y=a['y'];dom=a['domains'];z=m.bank_features(raw);g=m.global_head(raw[:,:384]);pi=m.router(z).softmax(1);delta=center_logits(torch.stack([e(z) for e in m.residuals],1))
                for center in ['batch','saved']:
                    d=delta-(delta.mean(0,keepdim=True) if center=='batch' else m.expert_source_mean[None]);f,_,u,_=residual_utilities(g,d,pi,y);v=support(u,dom.numpy(),np.ones(len(y),bool));birds=[]
                    for domain in [0,2,3]:
                        mask=(dom==domain)&(y==0);birds.append({'domain':domain,'n':int(mask.sum()),'valid_minimum_two':bool(mask.sum()>=2),'E2_U':float(u[mask,2].mean()) if mask.any() else None,'bird_correct':int((f.argmax(1).eq(y)&mask).sum())})
                    rows.append({'bank':label,'state':state,'checkpoint_step':p.get('selected_step',p.get('step')),'scheduled_batch_step':row['step'],'centering':center,'pooled':v,'bird':birds,'H2_cache_sha256':row['sha256']})
            assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items());assert all(q.grad is None for q in m.parameters());assert digest(path)==locks[label+'/'+state]
    result={'source_only':True,'target_read':False,'training':False,'parameter_updates':False,'checkpoint_changes':False,'threshold_changes':False,'checkpoint_sha256':locks,'code_sha256':digest(Path(__file__)),'scope':'Four saved states on six exact source H2 minibatches. Batch centering matches training form; saved centering separates evaluation convention. These are saved-state replay utilities, not historical utility EMA or original intermediate states. Bird groups below2 are invalid for shared-support claims. Actual Q remains unchanged.','rows':rows};write(out/'H2_utility_results.json',result)
    print(json.dumps([{'bank':r['bank'],'state':r['state'],'step':r['scheduled_batch_step'],'sharedU':r['pooled']['shared_U_floor_experts'],'sharedQ':r['pooled']['shared_Q_experts'],'bird':r['bird']} for r in rows if r['centering']=='batch'],indent=2))
if __name__=='__main__':run()

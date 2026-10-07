"""Source-only selected/common399 replay from previously locked original tokens."""
import json
from pathlib import Path
import numpy as np
import torch
from domainbed.deit_transformer import DeiTFeaturizer
from .spatial_bank_model import SpatialBankGateSIRM
from .spatial_bank_replay import spatial_head
from .spatial_bank_run import DEST,PRIOR,ARMS
from .shrinkage import write,digest
from .bias_control import scores

@torch.no_grad()
def run():
    torch.set_num_threads(4);source=PRIOR/'frozen_token_source_diagnosis/source_out_tokens.npz'
    audit=json.loads((PRIOR/'frozen_token_source_diagnosis/extraction_audit.json').read_text());assert digest(source)==next(v['cache_sha256'] for v in audit['rows'] if v['split']=='source_out')
    with np.load(source) as a:
        raw=torch.tensor(np.concatenate([a['cls'],a['patch_max']],1));y=torch.tensor(a['y']);domains=a['domains'].copy();ids=a['ids'].copy();g_cache=torch.tensor(a['global_logits'])
    out=DEST/'source_common_state';out.mkdir(exist_ok=True);result={};arrays={};locks={}
    for arm in ARMS:
        for state,file in [('selected','source_best.pkl'),('last','last.pkl')]:
            bank=DEST/'experiment/runs/L38'/arm;path=bank/file
            # Saved final filename is verified from disk, never inferred from selected step.
            if state=='last' and not path.exists():path=bank/'model_last.pkl'
            locks[f'{arm}/{state}']=digest(path);p=torch.load(path,map_location='cpu',weights_only=False);c=p['config'];assert c['strong_anchor_sha256']==audit['anchor_sha256']
            m=SpatialBankGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c)
            m.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()},strict=True);m.configure_stage();m.eval()
            g,d,pi=spatial_head(m,raw);full=g+(d*pi[:,:6,None]).sum(1);assert torch.equal(g.argmax(1),g_cache.argmax(1))
            v,pred=scores(full,y,domains);globalpred=g.argmax(1);correct=full.argmax(1).eq(y);globalcorrect=globalpred.eq(y)
            if state=='selected':
                report=json.loads((bank/'posthoc/complete.json').read_text());assert abs(v['accuracy']-report['splits']['source_out']['source_mean'])<1e-12
                replay=DEST/('eda' if arm!='CLS_CONTROL' else 'matched_CLS_control/eda')/'source_out_replay.npz'
                with np.load(replay) as a:
                    assert a['ids'].tolist()==ids.tolist()
                    ref=a['B1'] if arm in ['OFF','CLS_CONTROL'] else a['B2'];assert np.array_equal(ref.argmax(1),pred)
                    v['max_abs_cached_vs_image_logits']=float(np.max(np.abs(full.numpy()-ref)))
            v.update(step=int(p.get('selected_step',p.get('step'))),repairs=int((~globalcorrect&correct).sum()),harms=int((globalcorrect&~correct).sum()),single_unscaled_repair_capacity=int((~globalcorrect&(g[:,None,:]+d).argmax(-1).eq(y[:,None]).any(1)).sum()),checkpoint_sha256=locks[f'{arm}/{state}'])
            result[f'{arm}/{state}']=v;arrays[f'{arm}_{state}']=full.numpy()
    np.savez_compressed(out/'source_outputs.npz',ids=ids,y=y.numpy(),domains=domains,**arrays)
    write(out/'results.json',dict(source_only=True,target_read=False,token_cache_sha256=digest(source),scope='Source-only saved selected/common399 states. Cached norm-token extraction is independently verified; selected image-replay predictions are checked exactly. Not historical gradients or target ranking.',states=result))
    return result

if __name__=='__main__':print(json.dumps(run()))

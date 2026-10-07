"""Read-only normalized DeiT token extraction on the locked source images."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.terra_10pct_factor_diagnosis.report import data_for
from .shrinkage import ROOT,digest,write


@torch.no_grad()
def main():
    torch.set_num_threads(4);root=ROOT/'results/sirm_research_loop/iteration_12';out=root/'frozen_token_source_diagnosis';out.mkdir(exist_ok=True)
    anchor=ROOT/'results/terra_10pct_factor_diagnosis/runs/L38/ERM_H2/source_best.pkl';manifestpath=root/'experiment/subset_manifest.json';plan=json.loads((root/'plan.json').read_text())
    assert digest(anchor)==plan['anchor_sha256'];assert digest(manifestpath)==plan['manifest_sha256']
    for key in ['base_code_hashes','variant_code_hashes']:assert all(digest(ROOT/p)==h for p,h in plan[key].items())
    payload=torch.load(anchor,map_location='cpu',weights_only=False);c=payload['config'];assert payload['selected_step']==375
    backbone=DeiTFeaturizer(pretrained=False,model_name=c['backbone']);backbone.load_state_dict({k.removeprefix('featurizer.'):v for k,v in payload['model_dict'].items() if k.startswith('featurizer.')},strict=True)
    backbone.requires_grad_(False);backbone.cuda().eval();before={k:v.cpu().clone() for k,v in backbone.state_dict().items()}
    global_head=torch.nn.Linear(384,10);global_head.load_state_dict({k.removeprefix('global_head.'):v for k,v in payload['model_dict'].items() if k.startswith('global_head.')});global_head.requires_grad_(False);global_head.cuda().eval()
    manifest=json.loads(manifestpath.read_text());sourceids={s:{r['id']:r for d in [0,2,3] for i in manifest['sources'][str(d)][s] for r in [manifest['sources'][str(d)]['images'][i]]} for s in ['in','out']}
    assert not set(sourceids['in'])&set(sourceids['out']);summaries=[]
    captured=[]
    hook=backbone.vit.norm.register_forward_hook(lambda module,args,result:captured.append(result))
    try:
        for split in ['in','out']:
            name='source_'+split;refpath=ROOT/'results/sirm_research_loop/iteration_09/source_router_diagnosis'/(name+'_original.npz');ref=dict(np.load(refpath));lookup={str(i):j for j,i in enumerate(ref['ids'])};features={k:[] for k in ['cls','dist','patch_mean','patch_max','global_logits']};ids=[];labels=[];domains=[];reference_error=0.
            for d,ds in sorted(data_for(manifest,[0,2,3],c,split).items()):
                current=[manifest['sources'][str(d)]['images'][i]['id'] for i in ds.indices]
                for image in current:assert digest(ROOT/'domainbed/data/terra_incognita'/image)==sourceids[split][image]['sha256']
                offset=0
                for x,y in DataLoader(ds,batch_size=32,num_workers=0,shuffle=False):
                    captured.clear();cls=backbone(x.cuda());assert len(captured)==1;t=captured[0];assert t.shape==(len(x),198,384);assert torch.equal(cls,t[:,0]);assert torch.isfinite(t).all()
                    currentids=current[offset:offset+len(y)];ix=[lookup[i] for i in currentids];assert np.array_equal(y.numpy(),ref['y'][ix]);assert np.all(ref['domains'][ix]==d)
                    error=float((cls.cpu()-torch.tensor(ref['z'][ix])).abs().max());reference_error=max(reference_error,error);assert error<1e-4
                    for key,value in [('cls',t[:,0]),('dist',t[:,1]),('patch_mean',t[:,2:].mean(1)),('patch_max',t[:,2:].amax(1)),('global_logits',global_head(cls))]:features[key].append(value.cpu().numpy())
                    ids.extend(currentids);labels.extend(y.tolist());domains.extend([d]*len(y));offset+=len(y)
            assert len(ids)==len(sourceids[split]) and set(ids)==set(sourceids[split]);arrays={k:np.concatenate(v) for k,v in features.items()};arrays.update(ids=np.array(ids),y=np.array(labels),domains=np.array(domains));file=out/(name+'_tokens.npz');np.savez_compressed(file,**arrays)
            ix=[lookup[i] for i in ids];assert np.array_equal(arrays['global_logits'].argmax(1),ref['g'][ix].argmax(1));summaries.append(dict(split=name,n=len(ids),normalized_tokens=198,prefix_tokens=2,patch_tokens=196,CLS_max_abs_reference_error=reference_error,CLS_exact_same_forward_output=True,global_predictions_match=True,all_original_image_hashes_match=True,cache_sha256=digest(file),reference_cache_sha256=digest(refpath)))
            write(out/'extraction_progress.json',dict(status='extracting',completed_splits=[r['split'] for r in summaries]))
    finally:hook.remove()
    assert all(torch.equal(v,backbone.state_dict()[k].cpu()) for k,v in before.items());assert all(not p.requires_grad and p.grad is None for p in backbone.parameters());assert digest(anchor)==plan['anchor_sha256'] and digest(manifestpath)==plan['manifest_sha256']
    write(out/'extraction_audit.json',dict(anchor_sha256=digest(anchor),anchor_selected_step=375,anchor_training_updates=501,manifest_sha256=digest(manifestpath),code_sha256=digest(Path(__file__)),source_envs=[0,2,3],training=False,fitting=False,target_read=False,backbone_tensors_unchanged=True,global_head_tensors_loaded_exact=True,original_view_only=True,rows=summaries,scope='Final normalized 2 prefix +196 patch tokens from a hook on the original vit.norm, no modified forward implementation. Spatial pooling is an availability diagnostic, not foreground localization or causal factor evidence.'))
    write(out/'extraction_progress.json',dict(status='extraction_complete',completed_splits=['source_in','source_out']))
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()

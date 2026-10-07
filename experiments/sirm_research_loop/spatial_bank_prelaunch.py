"""Real-image one-update paired integration audit and raw source token cache."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.terra_10pct_factor_diagnosis.report import data_for
from .spatial_bank_model import SpatialBankGateSIRM,pooled_raw_features
from .spatial_bank_replay import RawTokenFeaturizer,spatial_head
from .anchor_train import load_anchor
from .shrinkage import ROOT,digest,write


def main():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    dest=ROOT/'results/sirm_research_loop/iteration_13';dest.mkdir(exist_ok=True);prior=ROOT/'results/sirm_research_loop/iteration_12';assert json.loads((prior/'completion_audit.json').read_text())['primary_EDA_complete'];token=prior/'frozen_token_source_diagnosis';proof=json.loads((token/'extraction_audit.json').read_text());cachefile=token/'source_in_tokens.npz';assert digest(cachefile)==next(r['cache_sha256'] for r in proof['rows'] if r['split']=='source_in')
    a=dict(np.load(cachefile));raw=torch.tensor(np.concatenate([a['cls'],a['patch_max']],1));cache=dict(features=raw,domains=torch.tensor(a['domains']),ids=a['ids'].tolist(),anchor_sha256=proof['anchor_sha256'],manifest_sha256=proof['manifest_sha256'],view='original',label_conditioned=False,target_read=False,feature_blocks=['original_CLS','final_normalized_patch_coordinatewise_max'],extraction_audit_sha256=digest(token/'extraction_audit.json'))
    path=dest/'source_feature_cache.pt'
    if path.exists():
        existing=torch.load(path,weights_only=False);assert all(torch.equal(existing[k],v) if isinstance(v,torch.Tensor) else existing[k]==v for k,v in cache.items())
    else:torch.save(cache,path)
    anchor=ROOT/'results/terra_10pct_factor_diagnosis/runs/L38/ERM_H2/source_best.pkl';assert digest(anchor)==proof['anchor_sha256'];payload=torch.load(anchor,map_location='cpu',weights_only=False);old=json.loads((prior/'plan.json').read_text());manifest=json.loads((prior/'experiment/subset_manifest.json').read_text());models=[];configs=[]
    for arm in ['OFF','ON']:
        c=copy.deepcopy(old['hparams'][arm]['sirm_discover_config']);c.update(protocol='sirm_research_iter13_spatial',center_feature_cache=str(path),center_feature_cache_sha256=digest(path),bank_input_mode='cls_patch_max')
        torch.manual_seed(0);m=SpatialBankGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c);load_anchor(m,payload);m.configure_stage();m=m.cuda();m._source_features=raw;m._source_domains=cache['domains'];m.runtime_calibration=True;models.append(m);configs.append(c)
    assert all(torch.equal(v,models[1].state_dict()[k]) for k,v in models[0].state_dict().items());before={k:v.cpu().clone() for k,v in models[0].state_dict().items() if k.startswith(('featurizer.','global_head.'))}
    batches=[]
    for d,ds in sorted(data_for(manifest,[0,2,3],configs[0],'in').items()):
        x,y=next(iter(DataLoader(ds,batch_size=32,num_workers=0)));batches.append((x.cuda(),y.cuda()))
    rows=[]
    for m in models:
        rows.append(m.update(batches));m.eval();assert int(m.centering_calibration_step)==1
        assert all(torch.equal(v,m.state_dict()[k].cpu()) for k,v in before.items());assert all(not p.requires_grad and p.grad is None for name in ['featurizer','global_head'] for p in getattr(m,name).parameters())
    assert rows[0]==rows[1] and all(torch.equal(v,models[1].state_dict()[k]) for k,v in models[0].state_dict().items())
    x=torch.cat([b[0] for b in batches]);m=models[0]
    with torch.no_grad():
        f=m(x);expected=m.global_head(m.featurizer(x));assert torch.equal(f['global_logits'],expected);cls,raw_batch=pooled_raw_features(m.featurizer,x);gh,dh,pih=spatial_head(m,raw_batch);assert torch.equal(gh,f['global_logits']) and torch.equal(dh,f['delta']) and torch.equal(pih,f['pi']);diagnostic=RawTokenFeaturizer(pretrained=False,model_name=configs[0]['backbone']).cuda().eval();diagnostic.load_state_dict(m.featurizer.state_dict(),strict=True);assert torch.equal(diagnostic(x),raw_batch);z=m.bank_features(raw.cuda());delta=m.residual_logits(raw.cuda());means=torch.stack([delta[cache['domains'].to(delta.device).eq(d)].double().mean(0) for d in [0,2,3]]).mean(0);assert float(means.abs().max())<1e-6
    params=sum(p.numel() for mod in [m.residuals,m.router] for p in mod.parameters())
    write(dest/'prelaunch_integration_audit.json',dict(passed=True,anchor_sha256=digest(anchor),feature_cache_sha256=digest(path),source_original_images=1172,source_only=True,target_read=False,diagnostic_updates=1,primary_training_started=False,bank_trainable_parameters=params,bank_feature_dimension=768,global_dimension=384,
        real_paired_OFF_ON_update_exact=True,global_CLS_prediction_exact=True,raw_token_diagnostic_and_head_replay_exact=True,all_frozen_tensors_unchanged=True,runtime_source_calibration_verified=True,equal_domain_centering_error=float(means.abs().max()),optimizer='rebuilt after replacing modules; no discarded module parameters',observed_update=rows[0],code_sha256=digest(Path(__file__)),model_code_sha256=digest(ROOT/'experiments/sirm_research_loop/spatial_bank_model.py'),scope='Fresh disposable models and one real96-image original-source update. No final experiment checkpoint or target read; synthetic supported-SIRM tests provide independent penalty-path coverage.'))
    print('Real paired update/global/source calibration passed; bank trainable parameters',params)

if __name__=='__main__':main()

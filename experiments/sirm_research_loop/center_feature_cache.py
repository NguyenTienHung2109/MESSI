"""Fixed original source-in frozen-ERM features for label-free mean calibration."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from domainbed.deit_transformer import DeiTFeaturizer
from experiments.terra_10pct_factor_diagnosis.report import data_for
from experiments.terra_10pct_factor_diagnosis.run import ROOT,MANIFEST,digest,write

@torch.no_grad()
def prepare(path,anchor):
    manifest=__import__('json').loads(MANIFEST.read_text());anchor_sha=digest(anchor);manifest_sha=digest(MANIFEST)
    if path.exists():
        p=torch.load(path,map_location='cpu',weights_only=False)
        assert p['anchor_sha256']==anchor_sha and p['manifest_sha256']==manifest_sha
        assert set(p['domains'].tolist())=={0,2,3} and p['view']=='original'
        return p
    payload=torch.load(anchor,map_location='cpu',weights_only=False);c=payload['config']
    backbone=DeiTFeaturizer(pretrained=False,model_name=c['backbone'])
    state={k[len('featurizer.'):]:v for k,v in payload['model_dict'].items() if k.startswith('featurizer.')}
    backbone.load_state_dict(state,strict=True);backbone.requires_grad_(False);backbone=backbone.cuda().eval()
    features=[];domains=[];ids=[]
    for d,ds in sorted(data_for(manifest,[0,2,3],c,'in').items()):
        ids.extend(manifest['sources'][str(d)]['images'][i]['id'] for i in ds.indices)
        for x,_ in DataLoader(ds,batch_size=64,num_workers=0):
            z=backbone(x.cuda()).cpu();features.append(z);domains.append(torch.full((len(z),),d,dtype=torch.long))
    p=dict(features=torch.cat(features),domains=torch.cat(domains),ids=ids,anchor_sha256=anchor_sha,manifest_sha256=manifest_sha,view='original',label_conditioned=False,target_read=False)
    assert len(ids)==1172 and torch.isfinite(p['features']).all() and digest(anchor)==anchor_sha
    torch.save(p,path);write(path.with_suffix('.json'),dict(anchor_sha256=anchor_sha,manifest_sha256=manifest_sha,cache_sha256=digest(path),source_in_images=len(ids),source_domains=[0,2,3],view='original',target_read=False,label_conditioned=False))
    del backbone;torch.cuda.empty_cache();return p

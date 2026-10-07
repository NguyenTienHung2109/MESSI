"""PACS/Terra source-only adapters preserving existing DomainBed split keys."""
from pathlib import Path
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from experiments.sirm_h1_h6 import run as base

SPECS={
 'PACS':dict(directory='PACS',domains=['art_painting','cartoon','photo','sketch'],target=3),
 'TerraIncognita':dict(directory='terra_incognita',domains=['location_100','location_38','location_43','location_46'],target=1),
}

class Adapter:
    def __init__(self,dataset):self.dataset=dataset;self.spec=SPECS[dataset];self.classes=None
    def folder(self,root,env):return Path(root)/self.spec['directory']/self.spec['domains'][env]
    def inventory(self,root,envs,seed,config):
        result=dict(protocol='DomainBed80_20',seed=seed,holdout_fraction=.2,sources={},class_names=None)
        for env in envs:
            data=ImageFolder(self.folder(root,env))
            if result['class_names'] is None:result['class_names']=data.classes
            if data.classes!=result['class_names']:raise ValueError('source class maps differ')
            fit,out=base.split_keys(len(data),.2,seed,env)
            images=[dict(id=str(Path(p).relative_to(Path(root)/self.spec['directory'])),**{'class':y},sha256=base.digest(p)) for p,y in data.samples]
            assert not set(fit)&set(out) and len(fit)+len(out)==len(data)
            result['sources'][str(env)]=dict(**{'in':fit,'out':out},images=images)
        self.classes=result['class_names'];return result
    def datasets_for(self,root,split,config,view='original',train=False,cohort='balanced_views',limit=None):
        return self.source_data(root,split,config,'in' if train else 'out',view,cohort,limit,train)
    def source_data(self,root,split,config,split_name,view='original',cohort='balanced_views',limit=None,train=False):
        result={}
        for key,entry in split['sources'].items():
            env=int(key);data=ImageFolder(self.folder(root,env),base.ViewTransform(config,view,train,env,cohort))
            actual=[(str(Path(p).relative_to(Path(root)/self.spec['directory'])),y) for p,y in data.samples]
            assert actual==[(s['id'],s['class']) for s in entry['images']]
            ids=entry[split_name]
            if limit:
                if train:ids=ids[:limit]
                else:
                    groups={c:[i for i in ids if data.targets[i]==c] for c in range(len(data.classes))}
                    ids=[i for pos in range(max(map(len,groups.values()))) for c in groups for i in groups[c][pos:pos+1]][:limit]
            result[env]=Subset(data,ids)
        return result

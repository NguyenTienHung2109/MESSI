"""Process-local spatial-bank training; inherited frozen-anchor/selection hooks."""
import hashlib,json,runpy
from domainbed import sirm_discover
from .centered_train import StrongCenteredAnchor
from .spatial_bank_model import SpatialBankGateSIRM

class StrongSpatialAnchor(StrongCenteredAnchor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        h=hashlib.sha256()
        for name,p in self.model.named_parameters():
            h.update(name.encode());h.update(str(tuple(p.shape)).encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        (self.output/'spatial_initialization.json').write_text(json.dumps(dict(
            parameter_sha256=h.hexdigest(),bank_input_mode=self.config['bank_input_mode'],
            bank_dimension=768,global_dimension=384,
            bank_parameters=sum(p.numel() for mod in [self.model.residuals,self.model.router] for p in mod.parameters()),
            anchor_sha256=self.config['strong_anchor_sha256']),indent=2)+'\n')

if __name__=='__main__':
    sirm_discover.GateSIRM=SpatialBankGateSIRM
    sirm_discover.SIRMDiscoverV2=StrongSpatialAnchor
    runpy.run_module('domainbed.scripts.train',run_name='__main__')

"""Process-local replay adapter: expose raw CLS+patch summaries to cached heads."""
import torch
from domainbed.deit_transformer import DeiTFeaturizer
from .spatial_bank_model import SpatialBankGateSIRM

class RawTokenFeaturizer(DeiTFeaturizer):
    def forward(self,x):
        captured=[];hook=self.vit.norm.register_forward_hook(lambda module,args,result:captured.append(result))
        try:cls=super().forward(x)
        finally:hook.remove()
        assert len(captured)==1 and torch.equal(cls,captured[0][:,0])
        return torch.cat([cls,captured[0][:,2:].amax(1)],1)


def spatial_head(model,raw):
    assert isinstance(model,SpatialBankGateSIRM) and raw.shape[1]==2*model.original_feature_dim
    return model.global_head(raw[:,:model.original_feature_dim]),model.residual_logits(raw),model.router(model.bank_features(raw)).softmax(1)


def install():
    from experiments.sirm_discover_binary import run as binary
    from . import shrinkage,case_eda,bias_control
    binary.GateSIRM=SpatialBankGateSIRM
    for module in [shrinkage,case_eda,bias_control]:
        module.GateSIRM=SpatialBankGateSIRM;module.DeiTFeaturizer=RawTokenFeaturizer;module.head=spatial_head

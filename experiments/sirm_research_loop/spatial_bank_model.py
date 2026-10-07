"""Matched dual-feature banks; frozen global still consumes original CLS only."""
import math
from pathlib import Path
import torch
from torch import nn
from experiments.sirm_discover.math import center_logits
from .class_coherent_model import ClassCoherentGateSIRM


def pooled_raw_features(featurizer,x):
    captured=[]
    hook=featurizer.vit.norm.register_forward_hook(lambda module,args,result:captured.append(result))
    try:cls=featurizer(x)
    finally:hook.remove()
    assert len(captured)==1;tokens=captured[0]
    assert tokens.ndim==3 and tokens.shape[1]>=3 and torch.equal(cls,tokens[:,0])
    raw=torch.cat([cls,tokens[:,2:].amax(1)],1)
    assert torch.isfinite(raw).all()
    return cls,raw


class SpatialBankGateSIRM(ClassCoherentGateSIRM):
    def __init__(self,featurizer,feature_dim,num_classes,num_domains,config,**kwargs):
        super().__init__(featurizer,feature_dim,num_classes,num_domains,config,**kwargs)
        self.original_feature_dim=feature_dim;self.bank_input_mode=config['bank_input_mode'];assert self.bank_input_mode in ['repeat_cls','cls_patch_max']
        hidden=config['residual_hidden'];self.residuals=nn.ModuleList([nn.Sequential(nn.Linear(2*feature_dim,hidden),nn.GELU(),nn.Linear(hidden,num_classes)) for _ in range(self.num_experts)])
        for expert in self.residuals:nn.init.zeros_(expert[-1].weight);nn.init.zeros_(expert[-1].bias)
        self.router=nn.Linear(2*feature_dim,self.num_experts+1)
        # Rebuild after replacing modules so no discarded parameters remain in Adam.
        self.configure_stage();self.optimizer=torch.optim.Adam(self.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        cache=torch.load(Path(config['center_feature_cache']),map_location='cpu',weights_only=False)
        assert cache['anchor_sha256']==config['strong_anchor_sha256'] and not cache['target_read'] and not cache['label_conditioned']
        assert cache['view']=='original' and cache['features'].shape[1]==2*feature_dim
        raw=self.choose_raw(cache['features']).double()
        self.register_buffer('bank_source_feature_mean',raw.mean(0).float())
        self.register_buffer('bank_source_feature_std',raw.std(0,unbiased=False).clamp_min(1e-8).float())

    def choose_raw(self,raw):
        assert raw.ndim==2 and raw.shape[1]==2*self.original_feature_dim
        return torch.cat([raw[:,:self.original_feature_dim]]*2,1) if self.bank_input_mode=='repeat_cls' else raw

    def bank_features(self,raw):
        return (self.choose_raw(raw)-self.bank_source_feature_mean)/self.bank_source_feature_std/math.sqrt(2)

    def forward(self,x):
        assert self.stage()!='global' and self.config['global_steps']==0
        cls,raw=pooled_raw_features(self.featurizer,x);z=self.bank_features(raw)
        g=self.global_head(cls);delta=self.center_delta(center_logits(torch.stack([e(z) for e in self.residuals],1)));pi=self.router(z).softmax(1)
        f=dict(logits=g+(pi[:,:self.num_experts,None]*delta).sum(1),global_logits=g,delta=delta,pi=pi,z=z)
        if self.training:
            self._assignment_outputs=(g.detach(),delta.detach())
        if self._cap_recording:
            assert self._cap_forward is None;self._cap_forward=f;self._class_outputs=(g.detach(),delta.detach())
        return f

    def residual_logits(self,raw):
        z=self.bank_features(raw)
        return self.center_delta(center_logits(torch.stack([e(z) for e in self.residuals],1)))

    @torch.no_grad()
    def calibrate_features(self,features,domains):
        assert len(features)==len(domains);device=self.expert_source_mean.device;means=[]
        for d in torch.unique(domains,sorted=True):
            total=torch.zeros_like(self.expert_source_mean,dtype=torch.float64,device=device);raw=features[domains.eq(d)];assert len(raw)>0
            for batch in raw.split(128):
                z=self.bank_features(batch.to(device));delta=center_logits(torch.stack([e(z) for e in self.residuals],1));total+=delta.double().sum(0)
            means.append(total/len(raw))
        self.expert_source_mean.copy_(torch.stack(means).mean(0).to(self.expert_source_mean.dtype));self.centering_calibration_step.copy_(self.step)

    def update(self,minibatches):
        row=super().update(minibatches)
        self.last_diagnostics.update(bank_input_mode=self.bank_input_mode,bank_input_dimension=2*self.original_feature_dim,
            global_input_dimension=self.original_feature_dim,feature_normalization='fixed original source-in mean/std; two blocks / sqrt2',spatial_pool='coordinatewise maximum over final normalized patch tokens; excludes both prefix tokens')
        return row

"""Mean-free expert corrections; original GateSIRM objectives consume these logits."""
import torch
from experiments.sirm_discover_gate.model import GateSIRM
from experiments.sirm_discover.math import center_logits

class CenteredGateSIRM(GateSIRM):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.center_residuals=self.config.get('center_residuals',False)
        self.register_buffer('expert_source_mean',torch.zeros(self.num_experts,self.global_head.out_features))
        self.register_buffer('centering_calibration_step',torch.tensor(-1,dtype=torch.long))
        self.runtime_calibration=False
        self._source_features=None;self._source_domains=None

    def center_delta(self,delta):
        if not self.center_residuals:return delta
        if self.training:
            # The DomainBed loader concatenates equal-size source minibatches.
            return delta-delta.mean(0,keepdim=True)
        if int(self.centering_calibration_step)<0:
            raise RuntimeError('Centered inference requires a saved source-in mean')
        return delta-self.expert_source_mean[None]

    def residual_logits(self,z):
        return self.center_delta(center_logits(torch.stack([expert(z) for expert in self.residuals],1)))

    def forward(self,x):
        f=super().forward(x)
        if self.stage()!='global' and self.center_residuals:
            f['delta']=self.center_delta(f['delta'])
            f['logits']=f['global_logits']+(f['pi'][:,:self.num_experts,None]*f['delta']).sum(1)
        return f

    @torch.no_grad()
    def calibrate_features(self,features,domains):
        assert len(features)==len(domains) and features.ndim==2
        device=self.expert_source_mean.device;means=[]
        for d in torch.unique(domains,sorted=True):
            z=features[domains==d];assert len(z)>0
            total=torch.zeros_like(self.expert_source_mean,dtype=torch.float64,device=device)
            for batch in z.split(128):
                batch=batch.to(device)
                raw=center_logits(torch.stack([expert(batch) for expert in self.residuals],1))
                total+=raw.double().sum(0)
            means.append(total/len(z))
        self.expert_source_mean.copy_(torch.stack(means).mean(0).to(self.expert_source_mean.dtype))
        self.centering_calibration_step.copy_(self.step)

    def train(self,mode=True):
        super().train(mode)
        if not mode and self.center_residuals and self.runtime_calibration and int(self.centering_calibration_step)!=int(self.step):
            assert self._source_features is not None and self._source_domains is not None
            self.calibrate_features(self._source_features,self._source_domains)
        return self

def centered_head(model,z):
    """Shared diagnostic head: preserve the saved source mean in replay."""
    delta=model.residual_logits(z) if isinstance(model,CenteredGateSIRM) else center_logits(torch.stack([expert(z) for expert in model.residuals],1))
    return model.global_head(z),delta,model.router(z).softmax(1)

"""Global semantic anchor + residual discovery; independent of legacy SIRM."""
import torch
from torch import nn
import torch.nn.functional as F
from .math import center_logits,residual_utilities,multiple_choice_loss,router_teacher,soft_null_q,component_irm,component_eligibility


class SIRMDiscover(nn.Module):
    def __init__(self,featurizer,feature_dim,num_classes,num_domains,config):
        super().__init__();self.config=dict(config);self.featurizer=featurizer
        self.num_experts=config['num_experts'];self.num_domains=num_domains
        self.global_head=nn.Linear(feature_dim,num_classes)
        self.residuals=nn.ModuleList([nn.Sequential(nn.Linear(feature_dim,config['residual_hidden']),nn.GELU(),nn.Linear(config['residual_hidden'],num_classes)) for _ in range(self.num_experts)])
        # Independent hidden layers, zero corrections preserve the global anchor.
        for expert in self.residuals:
            nn.init.zeros_(expert[-1].weight);nn.init.zeros_(expert[-1].bias)
        self.router=nn.Linear(feature_dim,self.num_experts+1) # final slot is NULL
        self.register_buffer('step',torch.zeros((),dtype=torch.long))
        self.register_buffer('utility_ema',torch.zeros(num_domains,self.num_experts))
        self.register_buffer('utility_valid',torch.zeros(num_domains,self.num_experts,dtype=torch.bool))
        self.register_buffer('utility_updates',torch.zeros((),dtype=torch.long))
        self.register_buffer('q',torch.zeros(num_domains,self.num_experts))
        self.register_buffer('q_null',torch.ones(num_domains))
        self.register_buffer('q_confidence',torch.zeros(num_domains))
        self.register_buffer('utility_interaction',torch.zeros(num_domains,self.num_experts))
        self.last_diagnostics={};self.stage_name=None;self.configure_stage()
        self.optimizer=torch.optim.Adam(self.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])

    def stage(self):
        s=int(self.step)
        if s<self.config['global_steps']:return 'global'
        if s<self.config['global_steps']+self.config['discover_steps']:return 'discover'
        return 'invariance'

    def configure_stage(self):
        stage=self.stage();global_train=stage=='global'
        for module,enabled in [(self.featurizer,global_train),(self.global_head,global_train),(self.residuals,not global_train),(self.router,not global_train)]:
            module.requires_grad_(enabled)
        self.stage_name=stage
        if not global_train:self.featurizer.eval();self.global_head.eval()

    def train(self,mode=True):
        super().train(mode)
        if self.stage()!='global':self.featurizer.eval();self.global_head.eval()
        return self

    def forward(self,x):
        z=self.featurizer(x);global_logits=self.global_head(z)
        if self.stage()=='global':
            delta=global_logits.new_zeros(len(x),self.num_experts,global_logits.shape[-1]);pi=global_logits.new_zeros(len(x),self.num_experts+1);pi[:,-1]=1
            return dict(logits=global_logits,global_logits=global_logits,delta=delta,pi=pi,z=z)
        delta=center_logits(torch.stack([expert(z) for expert in self.residuals],1))
        pi=self.router(z).softmax(1)
        logits=global_logits+(pi[:,:self.num_experts,None]*delta).sum(1)
        return dict(logits=logits,global_logits=global_logits,delta=delta,pi=pi,z=z)

    def predict(self,x):return self(x)['logits']

    @torch.no_grad()
    def update_utility(self,marginal,domains):
        for d in range(self.num_domains):
            mask=domains==d
            if int(mask.sum())<self.config['utility_min_samples']:continue
            value=marginal[mask].mean(0)
            if not torch.isfinite(value).all():raise ValueError('nonfinite utility')
            fresh=~self.utility_valid[d];beta=self.config['utility_ema_beta']
            self.utility_ema[d]=torch.where(fresh,value,beta*self.utility_ema[d]+(1-beta)*value)
            self.utility_valid[d]=True
        self.utility_updates+=1
        q,null,confidence,interaction=soft_null_q(self.utility_ema,self.utility_valid,self.config['q_temperature'],self.config['utility_floor'],self.config['interaction_floor'])
        self.q.copy_(q);self.q_null.copy_(null);self.q_confidence.copy_(confidence);self.utility_interaction.copy_(interaction)

    def effective_lambda(self):
        start=self.config['global_steps']+self.config['discover_steps']
        offset=int(self.step)-start
        if offset<0:return 0.
        return self.config['lambda_sirm']*min(max(offset/self.config['sirm_ramp_steps'],0.),1.)

    def update(self,minibatches):
        self.configure_stage();self.train();stage=self.stage();x=torch.cat([a for a,b in minibatches]);y=torch.cat([b for a,b in minibatches]);domains=torch.cat([torch.full_like(b,i) for i,(a,b) in enumerate(minibatches)])
        f=self(x);cls=F.cross_entropy(f['logits'],y);zero=cls*0
        discover=route=balance=sirm=zero;info=dict(active=False,pair_count=0,G={});lam=self.effective_lambda();refreshed=False;weighted_grad_norm=None;base_grad_norm=None;gradient_changed_elements=None
        if stage!='global':
            _,_,marginal,single=residual_utilities(f['global_logits'],f['delta'],f['pi'],y)
            start=self.config['global_steps']
            if (int(self.step)-start)%self.config['utility_interval']==0:
                self.update_utility(marginal.detach(),domains);refreshed=True
            discover,choice=multiple_choice_loss(f['global_logits'],f['delta'],y,self.config['discovery_temperature'])
            teacher=router_teacher(single,self.config['router_temperature'],self.config['utility_floor'])
            # Auxiliary router loss sees frozen/detached representation and teacher.
            log_pi=F.log_softmax(self.router(f['z'].detach()),1)
            route=F.kl_div(log_pi,teacher,reduction='batchmean')
            loads=choice.mean(0)
            balance=(self.num_experts*loads.square().sum()-1)*(1-teacher[:,-1]).mean().detach()
            if lam>0:sirm,info=component_irm(f['global_logits'],f['delta'],f['pi'],y,domains,self.q,self.config['min_ess'],self.config.get('sirm_weighting','confidence'))
        base_loss=cls+self.config['lambda_discover']*discover+self.config['lambda_router']*route+self.config['lambda_balance']*balance
        loss=base_loss+lam*sirm
        eligibility={}
        if refreshed:
            valid,ess,eligible=component_eligibility(f['pi'],domains,self.q,self.config['min_ess'])
            support=self.q>0
            structural=sum(int((support[i]&support[j]).sum()) for i in range(self.num_domains) for j in range(i+1,self.num_domains))
            eligibility=dict(selected_cells=int(support.sum()),shared_experts=int((support.sum(0)>=2).sum()),structural_pair_expert_count=structural,eligible_pair_expert_count=eligible,eligibility_valid=valid.tolist(),routing_ess=ess.tolist(),utility_valid=self.utility_valid.tolist())
            if info['active'] and self.config.get('log_sirm_gradient_norm',False):
                params=[p for p in self.residuals.parameters() if p.requires_grad]
                grads=torch.autograd.grad(lam*sirm,params,retain_graph=True,allow_unused=True)
                baseline=torch.autograd.grad(base_loss,params,retain_graph=True,allow_unused=True)
                weighted_grad_norm=float(sum(g.detach().double().square().sum() for g in grads if g is not None).sqrt())
                base_grad_norm=float(sum(g.detach().double().square().sum() for g in baseline if g is not None).sqrt())
                gradient_changed_elements=sum(int(((b+g)!=b).sum()) for b,g in zip(baseline,grads) if b is not None and g is not None)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite objective')
        self.optimizer.zero_grad(set_to_none=True);loss.backward();self.optimizer.step()
        self.last_diagnostics=dict(stage=stage,q=self.q.tolist(),q_null=self.q_null.tolist(),q_confidence=self.q_confidence.tolist(),utility_ema=self.utility_ema.tolist(),utility_interaction=self.utility_interaction.tolist(),utility_updates=int(self.utility_updates),sirm_pair_count=info['pair_count'],G=info['G'],q_refreshed=refreshed,weighted_sirm_grad_norm=weighted_grad_norm,base_residual_grad_norm=base_grad_norm,sirm_changes_gradient_elements=gradient_changed_elements,**eligibility)
        row=dict(stage=stage,loss=float(loss.detach()),loss_cls=float(cls.detach()),loss_discover=float(discover.detach()),loss_router=float(route.detach()),loss_balance=float(balance.detach()),loss_sirm=float(sirm.detach()),effective_lambda_sirm=lam,sirm_active=info['active'],sirm_pair_count=info['pair_count'],router_null_mass=float(f['pi'][:,-1].detach().mean()),q_null_mean=float(self.q_null.mean()),residual_rms=float(f['delta'].detach().square().mean().sqrt()))
        self.step+=1
        return row

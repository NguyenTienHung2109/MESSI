"""Logging-only subclass: objective and optimizer remain inherited unchanged."""
import json,math
from pathlib import Path
import torch
from experiments.sirm_h1_h6.model import HypothesesSIRM
from domainbed.subset_irm import subset_irm_penalty

class ReplayComplete(Exception):pass

class DiagnosticSIRM(HypothesesSIRM):
    output=None
    replay=False
    smoke=False

    def _subset_forward(self,x):
        result=super()._subset_forward(x)
        if self.training:self._diagnostic_forward=result
        return result

    def _gradient_diagnostics(self,losses,prefix='grad'):
        # Avoid repeated auxiliary-loss backward passes; log the two relevant gradients.
        if prefix!='grad_raw':return {}
        logits,dense,gamma,features,expert_logits,shared=self._diagnostic_forward
        q=self._assignment(dense,self._diagnostic_domains)
        raw,info=subset_irm_penalty(expert_logits,self._diagnostic_labels,self._diagnostic_domains,gamma,q,
                                  self.min_effective_samples,self.responsibility_detach)
        groups={'backbone':list(self.featurizer.parameters()),'router':list(self.moe_head.router.parameters()),
                'experts':list(self.moe_head.experts.parameters()),'expert_heads':list(self.expert_heads.parameters())}
        parameters=sum(groups.values(),[])
        a=torch.autograd.grad(losses['cls'],parameters,retain_graph=True,allow_unused=True)
        b=torch.autograd.grad(raw,parameters,retain_graph=True,allow_unused=True)
        spans={};start=0
        for name,ps in groups.items():spans[name]=(start,start+len(ps));start+=len(ps)
        spans['all']=(0,len(parameters));metrics={}
        for name,(lo,hi) in spans.items():
            nc=ns=dot=0.
            for gc,gs in zip(a[lo:hi],b[lo:hi]):
                if gc is not None:nc+=float(gc.detach().float().square().sum())
                if gs is not None:ns+=float(gs.detach().float().square().sum())
                if gc is not None and gs is not None:dot+=float((gc.detach().float()*gs.detach().float()).sum())
            metrics[f'grad_raw_cls_{name}']=math.sqrt(nc)
            metrics[f'diagnostic_counterfactual_sirm_norm_{name}']=math.sqrt(ns)
            metrics[f'diagnostic_raw_ratio_{name}']=math.sqrt(ns/nc) if nc>0 else None
            metrics[f'diagnostic_cosine_{name}']=dot/math.sqrt(nc*ns) if nc*ns>1e-30 else None
        metrics['diagnostic_counterfactual_sirm_loss']=float(raw.detach())
        metrics['diagnostic_counterfactual_active_cells']=info['active_cells'].cpu().tolist()
        return metrics

    def update(self,minibatches,unlabeled=None):
        step=int(self.subset_step.item())
        if self.replay and not self.smoke and step>1050:raise ReplayComplete()
        window=400<=step<=650 or 900<=step<=1050
        log_gradient=self.smoke or (490<=step<=510) or (990<=step<=1010) or (window and step%10==0) or step%500==0
        self.gradient_log_interval=1 if log_gradient else 0
        self._diagnostic_labels=torch.cat([y for x,y in minibatches]);self._diagnostic_domains=self._get_domain_ids(minibatches)
        result=super().update(minibatches,unlabeled)
        if self.smoke or window or step%500==0:
            diag=self.last_diagnostics
            q=torch.tensor(diag['q_matrix']);support=q.gt(0);previous=getattr(self,'_previous_diagnostic_q',None)
            gaps=[]
            for values in diag['risk_ema_matrix']:
                ordered=sorted(values);gaps.append(ordered[2]-ordered[1])
            row=dict(step=step,**result,q=diag['q_matrix'],risk_ema=diag['risk_ema_matrix'],
                     risk=diag['risk_matrix'],ess=diag['ess_matrix'],active_sirm_cells=diag['active_sirm_cells'],
                     active_domains_per_expert=diag['active_domains_per_expert'],
                     q_support_churn_since_previous_update=0. if previous is None else float(support.ne(previous).float().mean()),
                     risk_ema_top2_boundary_gap=gaps,
                     class_counts=[torch.bincount(y,minlength=7).cpu().tolist() for x,y in minibatches],
                     responsibility_support=diag['risk_support_matrix'],
                     measurement='actual training minibatch, before optimizer update; Q after scheduled EMA update')
            for group in ['backbone','router','experts','expert_heads','all']:
                raw=row.get(f'diagnostic_raw_ratio_{group}')
                if raw is not None:row[f'diagnostic_weighted_ratio_{group}']=raw*result['effective_lambda_sirm']
            with (Path(self.output)/'dense_diagnostics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        self._previous_diagnostic_q=torch.tensor(self.last_diagnostics['q_matrix']).gt(0)
        self._diagnostic_forward=None;self._diagnostic_labels=None;self._diagnostic_domains=None
        return result

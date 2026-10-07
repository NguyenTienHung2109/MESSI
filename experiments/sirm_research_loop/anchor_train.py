"""Process-local strong-anchor variant; repository training defaults unchanged."""
import hashlib,json,runpy
from pathlib import Path
import torch
from domainbed import sirm_discover

def load_anchor(model,payload):
    state=payload['model_dict']
    for name in ['featurizer','global_head']:
        prefix=name+'.';values={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        getattr(model,name).load_state_dict(values,strict=True)
        actual=getattr(model,name).state_dict()
        assert all(torch.equal(actual[k].cpu(),values[k].cpu()) for k in actual)

class StrongAnchorSIRM(sirm_discover.SIRMDiscoverV2):
    def __init__(self,input_shape,num_classes,num_domains,hparams):
        super().__init__(input_shape,num_classes,num_domains,hparams)
        if self.config.get('strong_anchor_load_at_start',False):
            self.install_anchor()
            self.model.configure_stage()
            assert all(not p.requires_grad for name in ['featurizer','global_head'] for p in getattr(self.model,name).parameters())
            from experiments.sirm_discover.run import weights_hash
            (self.output/'initialization.json').write_text(json.dumps(dict(weights_sha256=weights_hash(self.model),parameters=sum(p.numel() for p in self.model.parameters()),anchor_sha256=self.config['strong_anchor_sha256']),indent=2)+'\n')

    def install_anchor(self):
        path=Path(self.config['strong_anchor_checkpoint']);sha=hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha==self.config['strong_anchor_sha256']
        payload=torch.load(path,map_location='cpu',weights_only=False)
        load_anchor(self.model,payload)
        (self.output/'anchor_loaded.json').write_text(json.dumps(dict(step=int(self.model.step),checkpoint_sha256=sha,anchor_step=payload.get('selected_step',payload.get('step')),copied_modules=['featurizer','global_head'],target_used_for_selection=False),indent=2)+'\n')

    def update(self,minibatches,unlabeled=None):
        if not self.config.get('strong_anchor_load_at_start',False) and int(self.model.step)==self.config['global_steps']:
            self.install_anchor()
        if self.config.get('strong_anchor_load_at_start',False):
            assert all(not p.requires_grad for name in ['featurizer','global_head'] for p in getattr(self.model,name).parameters())
            parameters=[p for name in ['featurizer','global_head'] for p in getattr(self.model,name).parameters()]
            versions=[p._version for p in parameters]
        step=int(self.model.step);result=super().update(minibatches,unlabeled)
        if self.config.get('strong_anchor_load_at_start',False):
            assert versions==[p._version for p in parameters] and all(p.grad is None for p in parameters)
            with (self.output/'frozen_anchor_checks.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,requires_grad_false=True,gradient_absent=True,parameter_versions_unchanged=True))+'\n')
        return result

def main():
    # The CLI still uses SIRMDiscoverV2, preserving its deterministic loaders/hooks.
    sirm_discover.SIRMDiscoverV2=StrongAnchorSIRM
    runpy.run_module('domainbed.scripts.train',run_name='__main__')

if __name__=='__main__':main()

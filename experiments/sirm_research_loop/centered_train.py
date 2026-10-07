"""Process-local centered-residual training on the exact frozen ERM anchor."""
import json,runpy
from pathlib import Path
import torch
from domainbed import sirm_discover
from .anchor_train import StrongAnchorSIRM
from .centered_model import CenteredGateSIRM
from experiments.terra_10pct_factor_diagnosis.run import digest

class StrongCenteredAnchor(StrongAnchorSIRM):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert isinstance(self.model,CenteredGateSIRM) and self.config['center_residuals']
        path=Path(self.config['center_feature_cache']);assert digest(path)==self.config['center_feature_cache_sha256']
        p=torch.load(path,map_location='cpu',weights_only=False)
        assert p['anchor_sha256']==self.config['strong_anchor_sha256'] and p['view']=='original'
        assert set(p['domains'].tolist())=={0,2,3} and not p['target_read'] and not p['label_conditioned']
        assert len(p['features'])==1172 and torch.isfinite(p['features']).all()
        self.model._source_features=p['features'];self.model._source_domains=p['domains'];self.model.runtime_calibration=True
        (self.output/'centering_provenance.json').write_text(json.dumps(dict(cache_sha256=digest(path),source_in_images=len(p['features']),source_domains=[0,2,3],target_read=False,label_conditioned=False,train_mean='differentiable concatenated equal-source minibatch',eval_mean='equal-domain original source-in cached frozen features'),indent=2)+'\n')

    def on_evaluation(self,results,step,selection_checkpoint):
        # misc.accuracy toggles train() after prediction; the mean remains saved.
        assert int(self.model.centering_calibration_step)==int(self.model.step)
        super().on_evaluation(results,step,selection_checkpoint)
        with (self.output/'centering_checks.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=step,mean_calibration_step=int(self.model.centering_calibration_step),mean_finite=bool(torch.isfinite(self.model.expert_source_mean).all()),source_mean_norm=self.model.expert_source_mean.norm(dim=1).tolist(),target_used=False))+'\n')

def main():
    sirm_discover.GateSIRM=CenteredGateSIRM
    sirm_discover.SIRMDiscoverV2=StrongCenteredAnchor
    runpy.run_module('domainbed.scripts.train',run_name='__main__')

if __name__=='__main__':main()

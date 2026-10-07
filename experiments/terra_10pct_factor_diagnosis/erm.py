"""Pure end-to-end ERM using the exact SIRM DeiT and H2 data schedule."""
import torch
import torch.nn.functional as F
from domainbed.algorithms import Algorithm
from domainbed.deit_transformer import DeiTFeaturizer
from domainbed.sirm_discover import SIRMDiscoverV2
from experiments.sirm_h1_h6 import run as base

class ERMH2Diagnostic(Algorithm):
    evaluation_dataset=SIRMDiscoverV2.evaluation_dataset
    on_evaluation=SIRMDiscoverV2.on_evaluation
    extra_checkpoint_steps=SIRMDiscoverV2.extra_checkpoint_steps
    def __init__(self,input_shape,num_classes,num_domains,hparams):
        super().__init__(input_shape,num_classes,num_domains,hparams)
        from pathlib import Path
        self.config=dict(hparams['sirm_discover_config']);self.output=Path(hparams['sirm_output_dir']);self.sources=hparams['sirm_source_envs']
        self.featurizer=DeiTFeaturizer(pretrained=self.config['pretrained'],model_name=self.config['backbone'])
        self.global_head=torch.nn.Linear(self.featurizer.n_outputs,num_classes)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=self.config['lr'],weight_decay=self.config['weight_decay'])
        self.best=-1.;self.best_step=None;self.pending_refresh=None;self.step=0
        base.write(self.output/'initialization.json',dict(pure_ERM=True,parameters=sum(p.numel() for p in self.parameters()),experts=0,router=False))
    def predict(self,x):return self.global_head(self.featurizer(x))
    def update(self,minibatches,unlabeled=None):
        assert unlabeled is None
        self.train();x=torch.cat([a for a,b in minibatches]);y=torch.cat([b for a,b in minibatches]);logits=self.predict(x);loss=F.cross_entropy(logits,y)
        self.optimizer.zero_grad();loss.backward();self.optimizer.step()
        row=dict(step=self.step,loss=float(loss.detach()),loss_cls=float(loss.detach()),train_acc=float(logits.detach().argmax(1).eq(y).float().mean()))
        import json
        with (self.output/'diagnostics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        self.step+=1;return {k:v for k,v in row.items() if k!='step'}

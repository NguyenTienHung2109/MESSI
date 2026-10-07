"""SIRM-Discover v2 adapter for the DomainBed training/evaluation loop."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from domainbed.algorithms import Algorithm
from domainbed.deit_transformer import DeiTFeaturizer
from domainbed.lib.fast_data_loader import InfiniteDataLoader
from experiments.sirm_h1_h6 import run as base
from experiments.sirm_discover_gate.model import GateSIRM
from domainbed.scripts.train_subset_irm_pacs import make_schedule


def training_loader(env,index,args,hparams):
    """Keep the previous paired experiment's exact batch/augmentation schedule."""
    assert not hparams['class_balanced'] and args.task=='domain_generalization'
    schedule=make_schedule(list(range(len(env))),args.steps,hparams['batch_size'],args.seed,index)
    dataset=base.ScheduledImages(env,args.seed,index)
    loader=InfiniteDataLoader(dataset,None,hparams['batch_size'],args.num_workers,batch_sampler=base.ScheduledBatches(schedule,0))
    root=Path(args.output_dir);p=root/'source_schedules.json';metadata=json.loads(p.read_text()) if p.exists() else {}
    metadata[str(index)]=dict(split_keys=env.keys,schedule_sha256=hashlib.sha256(schedule.tobytes()).hexdigest())
    base.write(p,metadata)
    return loader


class SIRMDiscoverV2(Algorithm):
    def __init__(self,input_shape,num_classes,num_domains,hparams):
        super().__init__(input_shape,num_classes,num_domains,hparams)
        self.config=dict(hparams['sirm_discover_config']);self.output=Path(hparams['sirm_output_dir']);self.sources=hparams['sirm_source_envs']
        backbone=DeiTFeaturizer(pretrained=self.config['pretrained'],model_name=self.config['backbone'])
        self.model=GateSIRM(backbone,backbone.n_outputs,num_classes,num_domains,self.config)
        self.best=-1.;self.best_step=None;self.pending_refresh=None
        base.write(self.output/'initialization.json',dict(weights_sha256=__import__('experiments.sirm_discover.run',fromlist=['weights_hash']).weights_hash(self.model),parameters=sum(p.numel() for p in self.model.parameters())))

    def predict(self,x):return self.model.predict(x)

    def update(self,minibatches,unlabeled=None):
        if unlabeled is not None:raise ValueError('SIRM-Discover test is source-only')
        step=int(self.model.step);values=self.model.update(minibatches);d=self.model.last_diagnostics
        row={'step':step,**values,**d}
        with (self.output/'diagnostics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        self.pending_refresh=row if d['q_refreshed'] else None
        return {k:v for k,v in values.items() if isinstance(v,(float,int,bool))}

    def evaluation_dataset(self,ds):
        """Same split keys, deterministic original view; preserve training transform."""
        clone=copy.copy(ds)
        if hasattr(ds,'underlying_dataset'):clone.underlying_dataset=self.evaluation_dataset(ds.underlying_dataset)
        elif hasattr(ds,'dataset'):clone.dataset=self.evaluation_dataset(ds.dataset)
        elif hasattr(ds,'transform'):clone.transform=base.ViewTransform(self.config,'original',False,0,'balanced_views')
        else:raise TypeError('unsupported SIRM evaluation dataset')
        return clone

    def needs_source_evaluation(self):return self.pending_refresh is not None

    def on_evaluation(self,results,step,selection_checkpoint):
        mean=float(np.mean([results[f'env{e}_out_acc'] for e in self.sources]));results['source_val_mean']=mean
        if self.pending_refresh is not None:
            row=dict(self.pending_refresh,source_validation_accuracy=mean,source_validation_per_domain={str(e):results[f'env{e}_out_acc'] for e in self.sources},validation_timing='post-update; Q/utility/loss/ESS are pre-update')
            with (self.output/'q_refresh.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            self.pending_refresh=None
        if selection_checkpoint and step>=self.config['global_steps']+self.config['discover_steps'] and mean>self.best:
            self.best=mean;self.best_step=step
            torch.save(dict(model_dict=self.state_dict(),selected_step=step,source_val_mean=mean,config=self.config),self.output/'source_best.pkl')
        if step==sum(self.config[k] for k in ['global_steps','discover_steps','invariance_steps'])-1:
            torch.save(dict(model_dict=self.state_dict(),step=step,config=self.config),self.output/'last.pkl')
            base.write(self.output/'selection.json',dict(selected_step=self.best_step,source_val_mean=self.best,target_used=False))

    def extra_checkpoint_steps(self):return [self.config['global_steps']-1,self.config['global_steps']+self.config['discover_steps']-1]

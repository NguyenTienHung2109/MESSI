"""Check prior weighted-pool inference/readout under explicitly disabled TF32."""
import json
from pathlib import Path
import torch
import numpy as np
from . import terra_object_pool_source_cache as cache
from . import terra_object_pool_readout as reader
from .shrinkage import digest,write
from .terra_source_object_probe import DEST

def run():
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;out=DEST/'precision_audit';out.mkdir(exist_ok=True);assert not (out/'plan.json').exists();old_cache=cache.RESULT;old_reader=reader.OUTDIR
 write(out/'plan.json',dict(reason='Ranked-control replay check found batch-shape-dependent TF32 deviations in four source-in unboxed images; disable TF32 and quantify old weighted-pool readout impact.',source_only=True,target_read=False,method_changes=False,hyperparameter_changes=False,precision='CUDA matmul and cuDNN TF32 disabled',code_sha256=digest(Path(__file__)),original_code_sha256={str(Path(mod.__file__)):digest(Path(mod.__file__)) for mod in [cache,reader]}));(out/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
 cache.RESULT=out/'cache';cache.run();reader.RESULT=cache.RESULT;reader.OUTDIR=out/'readers';reader.run();results={}
 for split in ['source_in','source_out']:
  old=torch.load(old_cache/f'{split}_features.pt',map_location='cpu',weights_only=False);new=torch.load(cache.RESULT/f'{split}_features.pt',map_location='cpu',weights_only=False);assert old['ids']==new['ids'];assert torch.equal(old['global_logits'],new['global_logits']);e=(old['features']-new['features']).abs();results[split]=dict(max_abs_feature_difference=float(e.max()),images_difference_over1e_minus4=int((e.amax(1)>1e-4).sum()))
 before=json.loads((old_reader/'results.json').read_text());after=json.loads((reader.OUTDIR/'results.json').read_text());comparison={}
 for name in before['states']:
  comparison[name]={}
  for s in ['source_in','source_out']:
   a=before['states'][name]['splits'][s];b=after['states'][name]['splits'][s];comparison[name][s]=dict(before_accuracy=a['accuracy_equal_domain'],after_accuracy=b['accuracy_equal_domain'],prediction_disagreements=sum(x!=y for x,y in zip(a['predictions'],b['predictions'])),before_repairs=a['repairs'],after_repairs=b['repairs'],before_harms=a['harms'],after_harms=b['harms'])
   if name!='CLS_object_pool':assert a==b
 write(out/'results.json',dict(feature_changes=results,readout_changes=comparison,source_only=True,target_read=False,controls_exactly_reproduced=True));print(json.dumps(comparison,indent=2))
if __name__=='__main__':run()

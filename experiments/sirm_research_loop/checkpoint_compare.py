"""Exact content comparison of saved prediction state tensors, excluding Q."""
import argparse,json
from pathlib import Path
import torch
from experiments.terra_10pct_factor_diagnosis.run import digest

def compare(left,right,out):
    torch.set_num_threads(4)
    a=torch.load(left,map_location='cpu',weights_only=False);b=torch.load(right,map_location='cpu',weights_only=False)
    modules={}
    for module in ['featurizer','global_head','residuals','router']:
        prefix='model.'+module+'.';aa={k:v for k,v in a['model_dict'].items() if k.startswith(prefix)};bb={k:v for k,v in b['model_dict'].items() if k.startswith(prefix)}
        assert aa and aa.keys()==bb.keys()
        total=different=0;maximum=0.;square=0.
        for k,v in aa.items():
            w=bb[k];assert v.shape==w.shape and v.dtype==w.dtype
            assert torch.isfinite(v).all() and torch.isfinite(w).all()
            total+=v.numel();different+=int(v.ne(w).sum());d=v.double()-w.double();maximum=max(maximum,float(d.abs().max()));square+=float(d.square().sum())
        modules[module]=dict(elements=total,different_elements=different,max_abs_difference=maximum,RMS_difference=(square/total)**.5)
    prediction_buffers={}
    for key in ['model.expert_source_mean']:
        if key in a['model_dict'] or key in b['model_dict']:
            assert key in a['model_dict'] and key in b['model_dict']
            v=a['model_dict'][key];w=b['model_dict'][key];assert v.shape==w.shape and torch.isfinite(v).all() and torch.isfinite(w).all()
            prediction_buffers[key]=dict(elements=v.numel(),different_elements=int(v.ne(w).sum()),max_abs_difference=float((v.double()-w.double()).abs().max()))
    record=dict(left=str(left),right=str(right),left_sha256=digest(left),right_sha256=digest(right),left_step=a.get('selected_step',a.get('step')),right_step=b.get('selected_step',b.get('step')),scope='Prediction module state tensors and output-affecting centering means by content; excludes Q, calibration-step metadata, configuration and file metadata.',modules=modules,prediction_buffers=prediction_buffers,exact_equal=all(v['different_elements']==0 for v in list(modules.values())+list(prediction_buffers.values())))
    out.write_text(json.dumps(record,indent=2)+'\n');return record

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('left',type=Path);parser.add_argument('right',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args();print(json.dumps(compare(args.left,args.right,args.output)))

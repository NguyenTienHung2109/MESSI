"""Fixed-token ablation motivated by weak frozen-feature transfer.

No backbone updates. Test CLS, distillation token, their mean, concatenation,
and repeated CLS as a matched input-size/parameter-count control.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from domainbed.predictive_compatibility import make_cell,solve_common
from domainbed.compatibility_diagnostics import prediction_metrics
from domainbed.scripts.audit_predictive_compatibility import train_map,write_json
from domainbed.scripts.check_compatibility_mechanisms import save_csv
from domainbed.scripts.analyze_compatibility_failures import load_bank

MODES=['cls','dist','mean_tokens','concat_tokens','repeat_cls']


def extract_dist(args,bank):
    # Instantiate official timm BEFORE the repository imports that register
    # same-named custom models in timm's global model registry.
    import timm
    official=timm.create_model('deit_small_distilled_patch16_224',pretrained=False).eval()
    from domainbed.deit_transformer import DeiTFeaturizer
    from domainbed.datasets import PACS
    from torch.utils.data import DataLoader
    checkpoint=Path(torch.hub.get_dir())/'checkpoints/deit_small_distilled_patch16_224-649709d9.pth'
    pretrained=torch.load(checkpoint,map_location='cpu',weights_only=True)['model']
    official.load_state_dict(pretrained,strict=True)
    custom=DeiTFeaturizer(pretrained=True).eval()
    different=[k for k,v in custom.vit.state_dict().items() if k not in pretrained or not torch.equal(v,pretrained[k])]
    assert not different,'Custom model does not exactly load pretrained weights'
    official.to(args.device);custom.to(args.device)
    dataset=PACS('domainbed/data',[],{'data_augmentation':False})
    output={};verification=[]
    for domain,ds in enumerate(dataset):
        values=[];cls_values=[];labels=[];max_ref=0.
        with torch.inference_mode():
            for batch,(x,y) in enumerate(DataLoader(ds,batch_size=64,shuffle=False,num_workers=4)):
                x=x.to(args.device)
                cls,dist=custom.vit.forward_features(x)
                if batch==0:
                    reference=official.forward_features(x)
                    max_ref=max(float((cls-reference[:,0]).abs().max()),float((dist-reference[:,1]).abs().max()))
                    assert max_ref<1e-4,'Forward mismatch with official timm'
                cls_values.append(cls.cpu());values.append(dist.cpu());labels.append(y)
        recomputed=torch.cat(cls_values);dist=torch.cat(values);y=torch.cat(labels)
        assert torch.equal(y,bank[domain][1])
        cache_error=float((recomputed-bank[domain][0]).abs().max())
        assert cache_error<1e-4,'Existing feature cache does not match extractor'
        verification.append(dict(domain=domain,n=len(y),max_abs_official_token_difference=max_ref,
                                 max_abs_existing_cls_cache_difference=cache_error,all_pretrained_weights_identical=True))
        output[domain]=dist
        torch.save(dict(dist=dist,labels=y),args.output/f'dist_env{domain}.pt')
        print(f'Extracted DIST domain {domain}; official error={max_ref:.3g}; cache error={cache_error:.3g}',flush=True)
    write_json(args.output/'backbone_integrity.json',verification)
    del official,custom
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return output


def assemble(cls,dist,mode):
    if mode=='cls':return cls
    if mode=='dist':return dist
    if mode=='mean_tokens':return (cls+dist)/2
    if mode=='concat_tokens':return torch.cat([cls,dist],1)
    if mode=='repeat_cls':return torch.cat([cls,cls],1)
    raise ValueError(mode)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('predictive_support_outputs/pacs_compatibility_lodo_seed0'))
    parser.add_argument('--output',type=Path,default=Path('predictive_support_outputs/pacs_failure_cycle2_seed0/token_ablation'))
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use fresh token-ablation directory')
    args.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    write_json(args.output/'protocol.json',dict(seed=0,modes=MODES,targets=[0,1,2,3],strict_loso=True,
        frozen_backbone=True,map_steps=500,head_steps=1000,head_class='row radius 1, dimension16, no bias',
        input_size_control='concat_tokens and repeat_cls both 768 dimensions, same MLP parameter count',
        selection='none; fixed controls, no target-based choice',no_new_certificate=True))
    bank=load_bank(args.root);dist=extract_dist(args,bank)
    metrics=[];fits={};proof=[]
    for mode in MODES:
        inputs={d:(assemble(x,dist[d],mode),y) for d,(x,y) in bank.items()}
        for target in range(4):
            for fold in ['main']+[f'loso_env{d}' for d in range(4) if d!=target]:
                original=args.root/f'env{target}_seed0'/fold
                old=torch.load(original/'model.pt',map_location='cpu',weights_only=True)
                config=SimpleNamespace(**old['config'])
                splits={int(d):{k:np.array(v) for k,v in s.items()} for d,s in json.loads((original/'split_manifest.json').read_text()).items()}
                sources=tuple(sorted(splits));held=target if fold=='main' else int(fold.removeprefix('loso_env'))
                assert target not in sources and held not in sources
                key=(mode,sources)
                directory=args.output/mode/f'env{target}_{fold}';directory.mkdir(parents=True)
                if key not in fits:
                    print('FIT',mode,sources,flush=True)
                    model,trace=train_map({d:inputs[d] for d in sources},splits,sources,config,7,0,directory/'training.jsonl',0)
                    if mode=='cls':
                        assert all(torch.equal(v,old['maps'][0][k]) for k,v in model.state_dict().items()),'CLS baseline failed exact reproduction'
                    cells=[]
                    with torch.no_grad():
                        for d in sources:
                            ix=splits[d]['calibration']
                            cells.append(make_cell(model(inputs[d][0][ix]),inputs[d][1][ix]))
                    w,certificate=solve_common(cells,7,1.,1000,1e-5)
                    fits[key]=(model,w,certificate,trace)
                model,w,certificate,trace=fits[key]
                write_json(directory/'adaptation.json',dict(sources=sources,held=held,original_target=target,
                    mode=mode,input_dimension=inputs[sources[0]][0].shape[1],solver=certificate,trace=trace,
                    map_parameters=sum(p.numel() for p in model.parameters()),
                    splits={str(d):{k:v.tolist() for k,v in s.items()} for d,s in splits.items()}))
                torch.save(dict(map=model.state_dict(),head=w,mode=mode),directory/'model.pt')
                proof.append(dict(mode=mode,target=target,fold=fold,sources=sources,held=held))
                eval_sets=[('held',held,*inputs[held])]+[('source_selection',d,inputs[d][0][splits[d]['selection']],inputs[d][1][splits[d]['selection']]) for d in sources]
                for stage,d,x,y in eval_sets:
                    with torch.no_grad():
                        z=model(x).double();logits=z@w.T
                        m=prediction_metrics(logits,y)
                    metrics.append(dict(mode=mode,target=target,fold=fold,stage=stage,domain=d,n=len(y),
                        accuracy=m['accuracy'],ce=m['cross_entropy'],ece=m['ece_15'],confidence=m['mean_confidence'],
                        solver_gap=certificate['gap'],solver_converged=certificate['converged']))
                    write_json(directory/f'{stage}_{d}.json',m)
                    if stage=='held':np.savez_compressed(directory/'held_predictions.npz',labels=y.numpy(),logits=logits.numpy(),features=z.numpy())
                save_csv(args.output/'metrics.csv',metrics)
                write_json(args.output/'status.json',dict(state='running',mode=mode,target=target,fold=fold))
    write_json(args.output/'adaptation_manifest.json',proof)
    (args.output/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.output/'status.json',dict(state='completed',folds=80,seed=0))


if __name__=='__main__':main()

"""Export source-only intervention diagnostics before accessing target pixels."""
import argparse
from pathlib import Path
import torch
from experiments.sirm_h1_h6.run import load, write, digest
from experiments.sirm_h1_h6.controls import metrics, route
from domainbed.subset_irm import per_sample_topk


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--routers',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    routers=load(args.routers)
    results={}
    for path in sorted(Path(args.cache).glob('source_out_*.pt')):
        cache=load(path)
        if cache['metadata']['kind']!='source' or cache['metadata']['split']!='out':
            raise ValueError('source review accepts source validation only')
        for key in ['checkpoint_sha256','split_hash','seed','target_env','cohort']:
            if cache['metadata'][key]!=routers['metadata'][key]:
                raise ValueError('source review provenance mismatch')
        results[cache['metadata']['view']]={name:metrics(cache,pi) for name,pi in {
            'native':cache['router_topk'], 'input':per_sample_topk(route(routers['input_sigma0'],cache),2),
            'global':route(routers['global'],cache)}.items()}
    original=results['original']
    drops={view:{name:original[name]['accuracy']-data[name]['accuracy'] for name in original}
           for view,data in results.items()}
    write(args.output,dict(target_used=False,routers_sha256=digest(args.routers),per_view=results,
                           accuracy_drop_vs_original=drops,
                           interpretation='Check source label recognizability and intervention severity before target access. No PACS intervention is known to preserve true latent mechanism.'))
    print(f'source-only review: {args.output}')


if __name__=='__main__':main()

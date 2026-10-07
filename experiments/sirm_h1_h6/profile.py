"""Measure executed dense inference operations, VRAM and time on source images."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from experiments.sirm_h1_h6.run import selected_model, source_data, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True)
    p.add_argument('--data-dir',default='domainbed/data')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    model,effective,checkpoint_hash=selected_model(args.bank)
    config=effective['config']
    split=json.loads((Path(args.bank)/'split_manifest.json').read_text())
    data=next(iter(source_data(args.data_dir,split,config,'out',cohort=effective['cohort']).values()))
    x,_=next(iter(DataLoader(data,batch_size=min(32,len(data)),num_workers=0)))
    x=x.cuda()
    with torch.no_grad():
        model.predict(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(10): model.predict(x)
        torch.cuda.synchronize()
        elapsed=(time.perf_counter()-start)/10
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],with_flops=True) as prof:
            model.predict(x)
            torch.cuda.synchronize()
    result=dict(checkpoint_sha256=checkpoint_hash,batch_size=len(x),parameters=sum(p.numel() for p in model.parameters()),
                batch_inference_seconds=elapsed,inference_seconds_per_image=elapsed/len(x),
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
                profiled_matmul_conv_flops_per_image=sum(e.flops for e in prof.key_averages())/len(x),
                flop_scope='PyTorch profiler-supported matmul/conv operations only; excludes unsupported ops',
                execution='dense computation of all six experts; Top-k changes routing, not dense compute')
    write(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()

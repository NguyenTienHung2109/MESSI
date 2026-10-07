"""Detached launch after six-arm smoke succeeds."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/sirm_representation_h2/activation');p.add_argument('--smoke-root',default='results/sirm_representation_h2/activation_smoke_final');a=p.parse_args()
    smoke=Path(a.smoke_root);state=json.loads((smoke/'progress.json').read_text());assert state['stage']=='completed' and state['total_cells']==6
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'active.json').exists():
        pid=json.loads((root/'active.json').read_text())['pid']
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise RuntimeError(f'already running PID {pid}')
    command=[sys.executable,'-u','-m','experiments.sirm_h2_activation.run','suite','--output',str(root)]
    env=dict(os.environ,PYTHONWARNINGS='ignore',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0')
    with (root/'full.log').open('a') as stream:process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    result=dict(pid=process.pid,command=command,started_unix=time.time());(root/'active.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()

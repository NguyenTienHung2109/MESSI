"""Launch the full, resumable suite after smoke and reuse audits pass."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/sirm_representation_h2/full')
    parser.add_argument('--smoke-root',default='results/sirm_representation_h2/smoke_final')
    parser.add_argument('--preflight',default='results/sirm_representation_h2/reuse_preflight.json')
    args=parser.parse_args();smoke=Path(args.smoke_root)
    progress=json.loads((smoke/'progress.json').read_text())
    assert progress['status']=='completed' and progress['completed']==4 and progress['failed']==0,progress
    assert len(json.loads((smoke/'paired_controls_audit.json').read_text()))==1
    audit=json.loads(Path(args.preflight).read_text())
    assert audit['passed'] and audit['reused_runs']==24,audit
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    active=root/'active.json'
    if active.exists():
        pid=json.loads(active.read_text())['pid']
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise RuntimeError(f'full suite already running: PID {pid}')
    command=[sys.executable,'-u','-m','experiments.sirm_representation_h2.runner','suite','--output',str(root)]
    env=dict(os.environ,PYTHONWARNINGS='ignore',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0')
    with (root/'full.log').open('a') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    state=dict(pid=process.pid,command=command,started_unix=time.time(),cwd=str(Path.cwd()),output=str(root))
    active.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)

if __name__=='__main__':main()

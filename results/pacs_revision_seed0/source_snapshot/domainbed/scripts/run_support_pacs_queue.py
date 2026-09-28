"""Durable sequential four-target PACS evaluation on one GPU."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def run(root, config):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    if (root/'queue.json').exists():
        raise FileExistsError('queue already exists; choose a fresh directory')
    status={'pid':os.getpid(),'config':str(config),'seed':0,'jobs':[],
            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def save():
        tmp=root/'queue.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(root/'queue.json')
    for target in range(4):
        output=root/f'env{target}_seed0'
        log=root/f'env{target}_seed0.log'
        job={'target_env':target,'state':'queued','output':str(output),'log':str(log)}
        status['jobs'].append(job)
    save()
    for job in status['jobs']:
        command=[sys.executable,'-u','-m','domainbed.scripts.train_predictive_support',
                 '--config',str(config),'--output-dir',job['output'],'--target-env',str(job['target_env']),'--seed','0']
        with open(job['log'],'w') as handle:
            process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT)
            job.update(state='running',pid=process.pid);save()
            code=process.wait()
        job.update(state='completed' if code==0 else 'failed',returncode=code);save()
        if code:
            status['state']='failed';save();return code
    results=[json.loads((Path(j['output'])/'results.json').read_text()) for j in status['jobs']]
    summary={'seed':0,'target_accuracies':[r['target'][0]['accuracy'] for r in results],
             'mean_target_accuracy':sum(r['target'][0]['accuracy'] for r in results)/4,
             'selected_steps':[r['best_step']+1 for r in results],
             'source_selected_accuracies':[r['best_source_accuracy'] for r in results]}
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    status['state']='completed';save();return 0

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',required=True)
    p.add_argument('--config',default='configs/predictive_support_pacs.json')
    a=p.parse_args();sys.exit(run(a.output_root,a.config))

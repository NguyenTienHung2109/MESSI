"""Label-informed source-only exact utility control; never a deployable selector."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from experiments.sirm_discover.math import router_teacher
from .routed_discovery_run import ROOT,DEST
from .source_conditional_accessibility import utility,evaluate
from .shrinkage import digest,write
OUT=DEST/'source_exact_utility_readout'
def run():
 torch.set_num_threads(4);OUT.mkdir(exist_ok=True);assert not (OUT/'results.json').exists()
 fits=json.loads((DEST/'source_conditional_accessibility/results.json').read_text());token=ROOT/'results/sirm_research_loop/iteration_12/frozen_token_source_diagnosis';proof=json.loads((token/'extraction_audit.json').read_text());rows=[]
 plan=dict(source_only=True,target_read=False,primary_training=False,probe_fit=False,utility='Exact standalone CE(g)-CE(g+delta) using known source labels, original router_teacher .1/.002. Label-informed diagnostic, not deployable or an accuracy upper bound.',code_sha256=digest(Path(__file__)))
 write(OUT/'plan.json',plan);(OUT/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
 for state in fits['states']:
  file=DEST/'source_spatial_reliability'/f'{state}_source_replay.npz';assert digest(file)==fits['cache_sha256'][state];bank=np.load(file)
  for split in ['source_in','source_out']:
   file=token/f'{split}_tokens.npz';assert digest(file)==next(r['cache_sha256'] for r in proof['rows'] if r['split']==split);b=np.load(file);assert np.array_equal(b['ids'],bank[split+'_ids'])
   g=b['global_logits'].astype(float);d=bank[split+'_bank_delta'].astype(float);y=b['y'];u=utility(g,d,y);pi=router_teacher(torch.tensor(u),.1,.002).numpy();f=g+np.einsum('nm,nmj->nj',pi[:,:6],d)
   gce=logsumexp(g,axis=1)-g[np.arange(len(g)),y];ce=logsumexp(f,axis=1)-f[np.arange(len(g)),y];assert np.max(ce-gce)<1e-7
   rows.append(dict(state=state,split=split,metric=evaluate(g,d,y,b['domains'],b['ids'],u,u),maximum_CE_increase=float(np.max(ce-gce)),cases=[dict(id=str(b['ids'][i]),label=int(y[i]),global_prediction=int(g[i].argmax()),exact_utility_prediction=int(f[i].argmax()),CE_change=float(ce[i]-gce[i]),pi=pi[i].tolist()) for i in range(len(y)) if (g[i].argmax()==y[i]) != (f[i].argmax()==y[i])]))
 write(OUT/'results.json',dict(plan=plan,rows=rows,source_only=True,target_read=False,bank_updates=False))
if __name__=='__main__':run()

"""Original source image sheets for the preregistered error-balanced diagnostic."""
from pathlib import Path
import json,csv,numpy as np
from experiments.sirm_research_loop.case_eda import make_sheets
from experiments.sirm_research_loop.shrinkage import digest
b=Path('results/sirm_research_loop/iteration_08');out=b/'error_balanced_teacher_probe/eda';out.mkdir(exist_ok=True);cache=b/'source_router_diagnosis/source_out_original.npz';pol=b/'error_balanced_teacher_probe/source_out_policies.npz';a=dict(np.load(cache));p=dict(np.load(pol));assert np.array_equal(a['y'],p['y']) and np.array_equal(a['domains'],p['domains'])
man=json.loads((b/'experiment/subset_manifest.json').read_text());logits=[a['g'],a['full'],a['g']+(a['delta']*p['probe_pi'][:,:6,None]).sum(1)];pred=[l.argmax(1) for l in logits];prob=np.exp(logits[0]-logits[0].max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);rows=[]
for i in range(len(a['y'])):
 rows.append({'split':'source_out','image_id':str(a['ids'][i]),'environment':int(a['domains'][i]),'class_name':man['class_names'][int(a['y'][i])],'pattern':''.join('C' if pr[i]==a['y'][i] else 'W' for pr in pred),'B0_prediction':int(pred[0][i]),'B1_prediction':int(pred[1][i]),'B2_prediction':int(pred[2][i]),'B0_confidence':float(prob[i].max()),'B1_NULL':float(a['pi'][i,-1]),'B2_NULL':float(p['probe_pi'][i,-1])})
specs=[dict(split='source_out',pattern=pattern,cases=[r for r in rows if r['pattern']==pattern]) for pattern in sorted(set(r['pattern'] for r in rows)) if len(set(pattern))>1];make_sheets(out,specs,man)
with (out/'cases.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary={'column_meaning':'B0=ERM, B1=actual375 router, B2=error-balanced fixed-bank teacher probe. B2 here is diagnostic sheet naming only, not primary SIRM ON.','source_cache_hash':digest(cache),'probe_policy_hash':digest(pol),'code_sha256':digest(Path(__file__)),'patterns':{pattern:sum(r['pattern']==pattern for r in rows) for pattern in sorted(set(r['pattern'] for r in rows))},'sheets':[s['pattern'] for s in specs],'target_read':False};(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)

"""Fixed 1NN source teacher transfer and sequence/domain controls; no tuning."""
import json,csv
from pathlib import Path
import numpy as np
import torch
from .fixed_teacher_probe import score
from .shrinkage import digest,write
B=Path('results/sirm_research_loop/iteration_08')
def run():
 torch.set_num_threads(4);out=B/'nearest_teacher_diagnosis';out.mkdir(exist_ok=True);paths={s:B/f'source_router_diagnosis/{s}_original.npz' for s in ['source_in','source_out']};meta=B/'source_annotation_audit/selected_source_results.json';locks={s:digest(p) for s,p in paths.items()};plan={'code_sha256':digest(Path(__file__)),'cache_hashes':locks,'source_metadata_hash':digest(meta),'neighbor_rule':'1 nearest cosine normalized true frozen backbone CLS; ties choose earliest cache index, no label or domain in primary neighbor search','controls':['unrestricted','exclude_same_sequence','exclude_query_source_domain'],'hypothesis':'A nonlinear feature-local teacher control may reveal a linear-router capacity limitation; no gain means that limitation is not demonstrated. Excluding sequences/domains tests event/camera dependence within fixed bank.','scope':'Source-only diagnostic. Source-out labels evaluate only, not neighbor choice. No fitting, grid, target access or new selection. Global, residual bank and source means were trained/calibrated using all source domains; domain-exclusion is router-retrieval control, not independent held-domain training.','primary_method_changed':False};write(out/'plan.json',plan)
 raw={s:dict(np.load(p)) for s,p in paths.items()};train=raw['source_in'];val=raw['source_out'];assert not set(train['ids'])&set(val['ids']);lookup={r['local_id']:r['official_sequence'] for r in json.loads(meta.read_text())['records']};seq={s:np.array([lookup[str(i)] for i in a['ids']]) for s,a in raw.items()};z={s:torch.nn.functional.normalize(torch.tensor(a['z']).double(),dim=1) for s,a in raw.items()};cos=(z['source_out']@z['source_in'].T).numpy();a={k:torch.tensor(val[k]) for k in ['g','delta','y','domains','pi','teacher']};results={};rows=[]
 for mode in plan['controls']:
  similarity=cos.copy()
  if mode=='exclude_same_sequence':similarity[seq['source_out'][:,None]==seq['source_in'][None,:]]=-np.inf
  if mode=='exclude_query_source_domain':similarity[val['domains'][:,None]==train['domains'][None,:]]=-np.inf
  assert np.isfinite(similarity).any(1).all();idx=similarity.argmax(1);pi=torch.tensor(train['teacher'][idx]);results[mode]=score(a,pi);label_good=train['y'][idx]==val['y'];wrong=val['g'].argmax(1)!=val['y'];results[mode]['neighbor_diagnostics']={'same_class_fraction':float(label_good.mean()),'same_domain_fraction':float((train['domains'][idx]==val['domains']).mean()),'same_sequence_fraction':float((seq['source_in'][idx]==seq['source_out']).mean()),'mean_cosine':float(cos[np.arange(len(idx)),idx].mean()),'global_wrong_same_class_fraction':float(label_good[wrong].mean()),'label_1NN_equal_domain_accuracy':float(np.mean([label_good[val['domains']==d].mean() for d in np.unique(val['domains'])]))}
  full=val['g']+(train['teacher'][idx,:6,None]*val['delta']).sum(1);pred=full.argmax(1);correct=pred==val['y'];kl=torch.nn.functional.kl_div(pi.clamp_min(1e-30).log(),a['teacher'],reduction='none').sum(1);results[mode]['global_wrong_teacher_KL']=float(kl[wrong].mean());results[mode]['global_correct_teacher_KL']=float(kl[~wrong].mean())
  for i,j in enumerate(idx):rows.append({'mode':mode,'query_id':str(val['ids'][i]),'neighbor_id':str(train['ids'][j]),'domain':int(val['domains'][i]),'neighbor_domain':int(train['domains'][j]),'y':int(val['y'][i]),'neighbor_y':int(train['y'][j]),'global_pred':int(val['g'][i].argmax()),'actual_pred':int(val['full'][i].argmax()),'neighbor_policy_pred':int(pred[i]),'neighbor_NULL':float(pi[i,-1]),'cosine':float(cos[i,j]),'same_sequence':bool(seq['source_out'][i]==seq['source_in'][j])})
  np.savez_compressed(out/f'{mode}_policies.npz',pi=pi.numpy(),neighbor_index=idx,y=val['y'],ids=val['ids'],domains=val['domains'])
 assert all(digest(p)==locks[s] for s,p in paths.items()) and digest(meta)==plan['source_metadata_hash'];write(out/'results.json',{'plan':plan,'policies':results});
 with (out/'cases.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for k,v in results.items():print(k,v['equal_domain'],v['pooled'],v['neighbor_diagnostics'],flush=True)
if __name__=='__main__':run()

"""Source-only per-instance utility signs; no fitting, target reads or gate changes."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('results/sirm_research_loop/iteration_17')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=ROOT/'source_analysis/results.json';x=json.loads(p.read_text());assert x['source_only'] and not x['target_read']
 report={'source_only':True,'target_read':False,'training':False,'parameter_updates':False,'gate_changes':False,'input_sha256':digest(p),'code_sha256':digest(Path(__file__)),'scope':'Saved per-image routed marginal CE utility; not standalone expert accuracy, causal factor information, or held-out confirmation. Diagnostic thresholds 0 and original0.002; no setting selected.' ,'states':{}}
 for state in ['selected','last']:
  d=x['banks']['OFF/'+state]['splits']['source_out'];rows=d['cases'];u=np.array([r['expert_utility'] for r in rows]);domains=np.array([r['domain'] for r in rows]);records=[]
  for di,domain in enumerate(sorted(set(domains))):
   mask=domains==domain;v=u[mask]
   for e in range(6):
    vv=v[:,e];pos=np.maximum(vv,0).sum();neg=np.maximum(-vv,0).sum()
    records.append(dict(domain=int(domain),expert=e,n=len(vv),mean=float(vv.mean()),saved_mean=float(d['U'][di][e]),beneficial_instances=int((vv>0).sum()),harmful_instances=int((vv<0).sum()),zero_instances=int((vv==0).sum()),positive_utility_mass=float(pos),negative_utility_mass=float(neg),negative_mass_fraction=float(neg/(pos+neg)) if pos+neg else None))
    assert abs(vv.mean()-d['U'][di][e])<1e-6
  cohorts={}
  for name,pred in [('global_correct',lambda r:r['global_prediction']==r['label']),('global_wrong',lambda r:r['global_prediction']!=r['label']),('repaired',lambda r:r['global_prediction']!=r['label'] and r['full_prediction']==r['label']),('harmed',lambda r:r['global_prediction']==r['label'] and r['full_prediction']!=r['label'])]:
   m=np.array([pred(r) for r in rows]);cohorts[name]={'n':int(m.sum()),'expert_mean_utility':u[m].mean(0).tolist() if m.any() else None}
  eligible=[r for r in records if r['mean']>0.002]
  report['states'][state]={'domain_experts':records,'cohorts':cohorts,'cells_above_original_floor':eligible,'mean_negative_instance_fraction_in_above_floor_cells':float(np.mean([r['harmful_instances']/r['n'] for r in eligible])) if eligible else None,'all_correctness_changed_cases':[r for r in rows if (r['global_prediction']==r['label'])!=(r['full_prediction']==r['label'])]}
 dest=ROOT/'conditional_utility_audit.json';dest.write_text(json.dumps(report,indent=2)+'\n')
 lines=['# Conditional source utility: instance reliability audit','','No new fitting, inference, target read, threshold/lambda change or checkpoint selection. This is a descriptive analysis of existing source-out per-image routed marginal CE utility.','','| State | Domain | Expert | Mean U | U>0 images | U<0 images | Negative utility mass / total absolute mass |','|---|---:|---:|---:|---:|---:|---:|']
 for state,z in report['states'].items():
  for r in z['cells_above_original_floor']:lines.append(f"| {state} | {r['domain']} | E{r['expert']} | {r['mean']:.6f} | {r['beneficial_instances']}/{r['n']} | {r['harmful_instances']}/{r['n']} | {r['negative_mass_fraction']:.3f} |")
 lines+=['','A domain-average utility above the original floor does not ensure that an expert helps each image, or preserves each correct classification. Sign-changing instance utility is present in selected source cells. Mean negative fractions are descriptive, unweighted across cells; images overlap across experts, so these are not independent observations. CE benefit does not imply accuracy benefit.','','Next bounded source-only hypothesis: a useful discovery/readout must distinguish beneficial corrections from harmful ones on source holdouts, rather than relying on domain-mean selection or merely reducing NULL. Test conditional reliability against a constant-routing/amplitude-matched control before integrating into another matched400-update OFF/ON run. Existing domain-mean utility/support deficits remain; do not manufacture Q support by lowering thresholds as evidence of discovered invariance.','','This audit does not prove a deployable reliability estimator, camera-independent transfer, new factors or SIRM benefit. Target feedback must not choose reliability settings. The earlier low-dimensional confidence router combined architecture and objective changes; iteration17 tested objective only. A subsequent architecture test must disclose that distinction and use the original total budget.','','[Full signs, cohorts and changed cases](conditional_utility_audit.json).']
 (ROOT/'conditional_utility_interpretation.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({state:{'positive_mean_cells':len(z['cells_above_original_floor']),'negative_instance_fraction_mean':z['mean_negative_instance_fraction_in_above_floor_cells']} for state,z in report['states'].items()},indent=2))
if __name__=='__main__':main()

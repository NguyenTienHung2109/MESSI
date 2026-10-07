"""Independent logged class-loss replay, norm matching and frozen trajectory checks."""
import argparse,json,math
from pathlib import Path
from .protocol_audit import audit as protocol
from .shrinkage import write,digest

def run(dest,arm,complete=False):
 bank=dest/'experiment/runs/L38'/arm;rows=[json.loads(l) for l in (bank/'diagnostics.jsonl').read_text().splitlines()];frozen=[json.loads(l) for l in (bank/'frozen_anchor_checks.jsonl').read_text().splitlines()]
 errors=[];norm_errors=[];fallbacks=[]
 for r in rows:
  info=r['discovery_gradient_norm_match'];blocks=r['discovery_class_blocks'];n=sum(b['count'] for b in blocks);assert n==96
  minima=[min(b['mean_expert_CE']) for b in blocks];orig=sum(v*b['count']/n for v,b in zip(minima,blocks));balanced=sum(minima)/len(blocks)
  errors.extend([abs(orig-info['original_loss']),abs(balanced-info['equal_class_loss']),abs(r['discovery_original_class_loss']-orig),abs(r['discovery_equal_class_unscaled_loss']-balanced)])
  expected=orig if info['fallback'] else balanced*info['scale'];errors.extend([abs(r['loss_discover']-expected),abs(r['discovery_class_loss_replay']-expected)])
  old=info['original_gradient_norm'];matched=info['matched_gradient_norm'];norm_errors.append(abs(old-matched)/old if old else abs(matched));fallbacks.append(bool(info['fallback']))
  assert r['discovery_class_weighting']=='equal_class_original_gradient_norm' and info['scale']>=0
  assert all(math.isfinite(v) for v in info.values() if isinstance(v,(float,int)))
  assert info['fallback']==(info['equal_class_gradient_norm']==0 and old>0)
 frozen_by_step={r['step']:r for r in frozen};assert all(r['step'] in frozen_by_step for r in rows)
 checks={'contiguous_steps':[r['step'] for r in rows]==list(range(len(rows))),'class_formula_replay':max(errors)<2e-6,'norm_match':max(norm_errors)<2e-5,'all_frozen_checks':all(all(frozen_by_step[r['step']][k] for k in ['requires_grad_false','gradient_absent','parameter_versions_unchanged']) for r in rows)}
 if complete:checks['all400_updates']=len(rows)==400 and (bank/'done').exists()
 proto=protocol(bank,complete=complete,schedule='frozen400');checks['protocol']=proto['passed']
 result={'passed':all(checks.values()),'arm':arm,'through_step':rows[-1]['step'],'updates':len(rows),'complete':complete,'checks':checks,'max_class_formula_error':max(errors),'max_relative_gradient_norm_error':max(norm_errors),'fallback_steps':[r['step'] for r,f in zip(rows,fallbacks) if f],'protocol':proto,'code_sha256':digest(Path(__file__)),'target_read':False,'scope':'Direct class-block objective replay and raw discovery norm equality at every observed update, frozen versions/grad flags, independent schedule/source-selection checks. Matching raw gradients does not equate Adam updates.'}
 write(dest/f'{arm}_class_weight_audit.json',result);assert result['passed'];return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dest',type=Path,required=True);p.add_argument('--arm',choices=['OFF','ON'],required=True);p.add_argument('--complete',action='store_true');a=p.parse_args();print(json.dumps(run(a.dest,a.arm,a.complete),indent=2))

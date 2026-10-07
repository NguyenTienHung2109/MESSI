"""Post-hoc scalar IRMv1 violation and source/target rank diagnostics."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def write(p, rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def scalar_cells(logits, labels, domains, gamma):
    z=logits.double();prob=z.softmax(-1)
    # Exact derivative of CE(scale*z,y) at scale=1, per sample/expert.
    derivative=(prob*z).sum(-1)-z.gather(-1,labels[:,None,None].expand(-1,z.shape[1],1)).squeeze(-1)
    ce=F.cross_entropy(z.reshape(-1,z.shape[-1]),labels[:,None].expand(-1,z.shape[1]).reshape(-1),reduction='none').reshape(z.shape[:2])
    result=[]
    for d in sorted(domains.unique().tolist()):
        mask=domains==d;w=gamma[mask].double();total=w.sum(0);ess=total.square()/w.square().sum(0).clamp_min(1e-12)
        signed=derivative[mask].mean(0)
        weighted=(w*derivative[mask]).sum(0)/total.clamp_min(1e-12)
        result.append(dict(domain=d,risk=ce[mask].mean(0),signed=signed,weighted=weighted,ess=ess,total=total))
    return result


def load_cache(run,bank,name,config):
    p=run/'cache'/name
    if not p.exists():p=bank/'cache'/name
    c=torch.load(p,map_location='cpu',weights_only=False);meta=c['metadata']
    for k in ['checkpoint_sha256','split_hash','target_env','seed']:
        assert meta[k]==config[k],(p,k)
    if meta['kind']=='source':assert not c['physical_domains'].eq(config['target_env']).any()
    else:assert c['physical_domains'].eq(config['target_env']).all()
    return c


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/sirm_representation_h2/full');p.add_argument('--output',default='results/sirm_q_followup/diagnosis');args=p.parse_args()
    torch.set_num_threads(4);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cells=[];ranks=[];runrows=[];audit=[]
    for path in sorted((Path(args.root)/'runs').glob('*/env*/seed*/config.json')):
        cfg=json.loads(path.read_text());run=path.parent;bank=Path(cfg['bank']);key={k:cfg[k] for k in ['condition','target_env','seed']}
        assert digest(bank/'checkpoints/best.pkl')==cfg['checkpoint_sha256']
        q=torch.tensor(json.loads((run/'source_diagnostics.json').read_text())['q']['q']);sel=q.gt(0)
        source=load_cache(run,bank,'source_out_original.pt',cfg)
        target=load_cache(run,bank,'target_all_original.pt',cfg)
        stats=scalar_cells(source['expert_logits'],source['labels'],source['domains'],source.get('router_native',source['router_topk']))
        valid=torch.stack([s['ess'].ge(cfg['hparams']['subset_irm_min_effective_samples']) & s['total'].gt(1e-12) for s in stats]);active=valid & sel & (valid & sel).sum(0).ge(2)[None,:]
        source_risk=torch.stack([s['risk'] for s in stats]).mean(0).numpy()
        target_risk=scalar_cells(target['expert_logits'],target['labels'],target['domains'],target['router_topk'])[0]['risk'].numpy()
        rho=float(spearmanr(-source_risk,-target_risk).statistic)
        payload=torch.load(bank/'checkpoints/best.pkl',map_location='cpu',weights_only=False)
        ema=payload['model_dict']['risk_ema.risk'].double().numpy()
        ema_valid=payload['model_dict']['risk_ema.valid'].numpy()
        rho_ema=float(spearmanr(-ema.mean(0),-target_risk).statistic) if ema_valid.all() else None
        del payload
        deltas=[];part=[]
        for s in stats:
            d=s['domain'];selected=sel[d].numpy();delta=float(target_risk[~selected].mean()-target_risk[selected].mean());deltas.append(delta)
            for m in range(q.shape[1]):
                row=dict(**key,source_domain=int(source['physical_domains'][source['domains']==d][0]),expert=m,q=float(q[d,m]),selected=bool(sel[d,m]),G=float(s['signed'][m].abs()),signed_G=float(s['signed'][m]),G_routed=float(s['weighted'][m].abs()) if s['total'][m]>1e-12 else None,signed_G_routed=float(s['weighted'][m]) if s['total'][m]>1e-12 else None,ess=float(s['ess'][m]),valid_routed=bool(valid[d,m]),active_penalty_cell=bool(active[d,m]),source_risk=float(s['risk'][m]))
                cells.append(row);part.append(row)
            ranks.append(dict(**key,source_domain=int(source['physical_domains'][source['domains']==d][0]),delta_target=delta,rho_domain=float(spearmanr(-s['risk'].numpy(),-target_risk).statistic),rho_ema_domain=float(spearmanr(-ema[d],-target_risk).statistic) if ema_valid[d].all() else None))
        for m in range(q.shape[1]):
            audit.append(dict(**key,expert=m,source_risk=float(source_risk[m]),target_risk=float(target_risk[m])))
        row=dict(**key,selected_step=cfg['selected_step'],rho=rho,rho_ema=rho_ema,delta_target=float(np.mean(deltas)))
        for tag,selected in [('Q',True),('nonQ',False)]:
            rs=[r for r in part if r['selected']==selected]
            row['G_'+tag]=float(np.mean([r['G'] for r in rs]));vs=[r['G_routed'] for r in rs if r['valid_routed']]
            row['G_routed_'+tag]=float(np.mean(vs)) if vs else None
            row['valid_'+tag]=len(vs)
        row['active_cells']=int(active.sum());runrows.append(row)
        print('/'.join(map(str,key.values())),flush=True)
    write(out/'irm_cells.csv',cells);write(out/'transfer_by_q_row.csv',ranks);write(out/'expert_risks.csv',audit);write(out/'runs.csv',runrows)
    lines=['# Q / IRMv1 follow-up','', '48 source-selected best checkpoints. Deterministic original view, complete source-out and target domains. No fitting or model changes. Target is post-hoc only.', '', '| Condition | G Q | G non-Q | Routed G Q | Routed G non-Q | Spearman | Target loss gap |','|---|---:|---:|---:|---:|---:|---:|']
    aggregate=[]
    for c in 'ABCD':
        rs=[r for r in runrows if r['condition']==c];a={'condition':c,'n':len(rs)}
        for k in ['G_Q','G_nonQ','G_routed_Q','G_routed_nonQ','rho','rho_ema','delta_target','active_cells']:
            vs=[r[k] for r in rs if r[k] is not None];a[k]=float(np.mean(vs));a[k+'_sd']=float(np.std(vs,ddof=1))
        aggregate.append(a);lines.append('| '+c+' | '+' | '.join(f'{a[k]:.5f}' for k in ['G_Q','G_nonQ','G_routed_Q','G_routed_nonQ','rho','delta_target'])+' |')
    lines.extend(['','G is absolute derivative of domain-mean CE at scalar scale=1. Routed G uses detached native responsibilities and the same weighted-risk normalization as training; means exclude cells below the training ESS threshold. Q membership and actual penalty eligibility are separate columns. Full-domain G is not the mean squared minibatch penalty: derivatives can cancel across samples, and full-domain ESS can exceed minibatch ESS.', '', 'Target gap averages non-selected minus selected expert target CE for each source Q row, then equally across rows. Spearman compares equal-domain source-out CE and target CE over six experts per run; per-domain correlations also exported. Six ranks give limited resolution. These are correlations, not evidence that Q identifies subset-specific invariance.', '', 'Best checkpoints are late snapshots, not pre-activation measurements. B is H2/OFF and has never received SIRM training. No early checkpoint was invented. Near-zero scalar G alone does not establish shared optimal classifier vectors or causal invariance.'])
    write(out/'aggregate.csv',aggregate);(out/'summary.md').write_text('\n'.join(lines)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(target_used_for_selection=False,scope='posthoc',code_sha256=digest(Path(__file__)),runs=len(runrows)),indent=2))

if __name__=='__main__':main()

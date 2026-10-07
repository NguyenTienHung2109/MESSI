"""Source-only selected/endpoint comparisons for the discovery-off auxiliary control."""
from pathlib import Path
import json
import csv
import numpy as np
import torch
from PIL import Image,ImageDraw
from domainbed.deit_transformer import DeiTFeaturizer
from .centered_model import CenteredGateSIRM
from .hard_router_diagnosis import outputs,metrics
from .shrinkage import ROOT,digest,write


def main():
    torch.set_num_threads(4)
    parent=ROOT/'results/sirm_research_loop/iteration_11';dest=parent/'discovery_off_control'
    assert json.loads((dest/'training_audit.json').read_text())['passed']
    models={};locks={};cache=ROOT/'results/sirm_research_loop/iteration_09/source_router_diagnosis'
    for arm,bank in [('control',dest/'run'),('reference',parent/'experiment/runs/L38/OFF')]:
        for state,filename in [('selected','source_best.pkl'),('last','last.pkl')]:
            path=bank/filename;p=torch.load(path,map_location='cpu',weights_only=False);c=p['config']
            model=CenteredGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c)
            model.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()});model.configure_stage();model.eval();model.requires_grad_(False)
            key=arm+'_'+state;models[key]=model;locks[key]=dict(sha256=digest(path),step=int(p['selected_step'] if state=='selected' else p['step']))
    ref=models['reference_selected']
    for model in models.values():
        for module in ['featurizer','global_head']:
            assert all(torch.equal(v,getattr(ref,module).state_dict()[k]) for k,v in getattr(model,module).state_dict().items())
    summaries={};case_records=[];ids_sets={}
    norm_matched=[]
    validated=json.loads((parent/'composed_bias_source_diagnosis/results.json').read_text())
    for split in ['source_in','source_out']:
        path=cache/f'{split}_original.npz'
        assert digest(path)==validated['plan']['source_cache_hashes'][split]
        with np.load(path) as a:z,y,d=[torch.from_numpy(a[k]) for k in ['z','y','domains']];ids=a['ids'].tolist()
        ids_sets[split]=set(ids);summaries[split]={}
        with torch.no_grad():
            for name,model in models.items():
                f=outputs(model,z,y);summary=metrics(f,y,d)
                wrong=f['g'].argmax(1)!=y
                single=(f['g'][:,None]+f['delta']).argmax(2)==y[:,None]
                routed=(f['g'][:,None]+f['pi'][:,:6,None]*f['delta']).argmax(2)==y[:,None]
                flattened=f['delta'].transpose(0,1).flatten(1)
                normalized=torch.nn.functional.normalize(flattened,dim=1)
                cos=normalized@normalized.T
                summary['capacity']=dict(global_wrong=int(wrong.sum()),unscaled_single=int((wrong&single.any(1)).sum()),routed_single=int((wrong&routed.any(1)).sum()),
                    residual_RMS=float(f['delta'].square().mean().sqrt()),
                    correction_RMS=float((f['full']-f['g']).square().mean().sqrt()),
                    output_cosine_mean=float(cos[~torch.eye(6,dtype=torch.bool)].mean()),
                    output_cosine_matrix=cos.tolist())
                summaries[split][name]=summary
                if split=='source_out':
                    for i in range(len(y)):
                        case_records.append(dict(model=name,id=ids[i],domain=int(d[i]),label=int(y[i]),global_prediction=int(f['g'][i].argmax()),prediction=int(f['full'][i].argmax()),
                            single_repair_available=bool(single[i].any()),routed_repair_available=bool(routed[i].any())))
            if split=='source_out':
                for state in ['selected','last']:
                    ref_name='reference_'+state;control_name='control_'+state
                    ref_norm=summaries['source_in'][ref_name]['capacity']['residual_RMS']
                    control_norm=summaries['source_in'][control_name]['capacity']['residual_RMS']
                    for name,alpha in [(ref_name,control_norm/ref_norm),(control_name,ref_norm/control_norm)]:
                        f=outputs(models[name],z,y);delta=alpha*f['delta'];pi=f['pi'];g=f['g']
                        logits=g+(pi[:,:6,None]*delta).sum(1);pred=logits.argmax(1);wrong=g.argmax(1)!=y
                        domain_acc=[float((pred[d==domain]==y[d==domain]).double().mean()) for domain in [0,2,3]]
                        domain_ce=[float(torch.nn.functional.cross_entropy(logits[d==domain],y[d==domain])) for domain in [0,2,3]]
                        norm_matched.append(dict(model=name,alpha_source_in_RMS=alpha,source_out_accuracy=float(np.mean(domain_acc)),source_out_CE=float(np.mean(domain_ce)),
                            repairs=int((wrong&(pred==y)).sum()),harms=int((~wrong&(pred!=y)).sum()),
                            unscaled_single_capacity=int((wrong&((g[:,None]+delta).argmax(2)==y[:,None]).any(1)).sum())))
    assert not ids_sets['source_in']&ids_sets['source_out']
    qrows=[json.loads(l) for l in (dest/'run/q_refresh.jsonl').read_text().splitlines()]
    reference_q={row['step']:row for row in map(json.loads,(parent/'experiment/runs/L38/OFF/q_refresh.jsonl').read_text().splitlines())}
    curves=[]
    for row in qrows:
        previous=reference_q[row['step']]
        curves.append(dict(step=row['step'],control_source_accuracy=row['source_validation_accuracy'],
            reference_source_accuracy=previous['source_validation_accuracy'],
            control_training_CE=row['loss_cls'],reference_training_CE=previous['loss_cls'],
            control_selected_cells=row['selected_cells'],reference_selected_cells=previous['selected_cells'],
            control_shared_experts=row['shared_experts'],reference_shared_experts=previous['shared_experts']))
    with (dest/'source_curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    stage3=[row for row in qrows if row['step']>=200]
    qsummary=dict(refreshes=len(stage3),shared_support_refreshes=sum(row['shared_experts']>0 for row in stage3),
        eligible_refreshes=sum(row['eligible_pair_expert_count']>0 for row in stage3),
        selected_cells_range=[min(row['selected_cells'] for row in stage3),max(row['selected_cells'] for row in stage3)])
    diagnostics=[json.loads(line) for line in (dest/'run/diagnostics.jsonl').read_text().splitlines()]
    objective_error=max(abs(row['loss']-row['loss_cls']-.1*row['loss_router']-.01*row['loss_balance']) for row in diagnostics)
    assert objective_error<1e-6
    manifest=json.loads((parent/'experiment/subset_manifest.json').read_text())
    lookup={entry['id']:entry for domain in manifest['sources'].values() for entry in domain['images']}
    names=manifest['class_names'];eda=dest/'eda';eda.mkdir(exist_ok=True)
    sheets=[]
    for state in ['selected','last']:
        reference={row['id']:row for row in case_records if row['model']=='reference_'+state}
        control=[row for row in case_records if row['model']=='control_'+state]
        groups={state+'_versus_reference':[row for row in control if (row['prediction']==row['label'])!=(reference[row['id']]['prediction']==row['label'])]}
        if state=='selected':
            groups.update(selected_ERM_repairs=[row for row in control if row['global_prediction']!=row['label'] and row['prediction']==row['label']],
                          selected_ERM_harms=[row for row in control if row['global_prediction']==row['label'] and row['prediction']!=row['label']])
        for name,rows in groups.items():
            if not rows:continue
            canvas=Image.new('RGB',(960,260*((len(rows)+2)//3)),'white');draw=ImageDraw.Draw(canvas)
            for j,row in enumerate(rows):
                p=ROOT/'domainbed/data/terra_incognita'/row['id'];assert digest(p)==lookup[row['id']]['sha256']
                image=Image.open(p).convert('RGB');image.thumbnail((314,206));x=j%3*320;yy=j//3*260;canvas.paste(image,(x,yy))
                prev=reference[row['id']]['prediction'] if name.endswith('reference') else row['global_prediction']
                draw.text((x+3,yy+210),f"label {names[row['label']]} | {names[prev]} -> {names[row['prediction']]}",fill='black')
                draw.text((x+3,yy+226),f"env{row['domain']} {row['id'].split('/')[-1][:22]}",fill='black')
            path=eda/(name+'.png');canvas.save(path);sheets.append(dict(path=str(path.relative_to(dest)),cases=rows))
    write(eda/'sheet_index.json',sheets)
    plan=json.loads((dest/'plan.json').read_text())
    write(dest/'source_comparison.json',dict(code_sha256=digest(Path(__file__)),training_plan_sha256=digest(dest/'plan.json'),
        checkpoints=locks,scope='Source-only saved selected/common399 endpoint bank replay. No target reads, fitting or selection changes.',
        summaries=summaries,cases=case_records,Q_refreshes=qrows,Q_summary=qsummary,
        actual_objective_max_abs_error=objective_error,source_in_RMS_matched_controls=norm_matched))
    lines=['# Discovery-off auxiliary control: source comparison','','The sole training change is lambda_discover0.1→0; router, balance, cap, centering and frozen global remain identical. Source schedules/initialization and frozen tensor audits passed. No target read or target result. This auxiliary control is not a new SIRM method. Original-case EDA remains required before choosing an intervention.','','| Split | Bank/state | Step | Equal-domain accuracy | CE | Repairs/harms | Unscaled/routed capacity |','|---|---|---:|---:|---:|---:|---:|']
    for split,rows in summaries.items():
        for name,v in rows.items():
            a=v['equal_domain']['actual'];pooled=v['all']['actual'];cap=v['capacity']
            lines.append(f"|{split}|{name}|{locks[name]['step']}|{100*a['accuracy']:.6f}%|{a['CE']:.8f}|{pooled['repairs']}/{pooled['harms']}|{cap['unscaled_single']}/{cap['routed_single']}|")
    lines+=['','[Full source metrics and cases](source_comparison.json). No conclusion about subset invariance or OOD benefit follows from this control alone.']
    (dest/'source_report.md').write_text('\n'.join(lines)+'\n')
    write(dest/'progress.json',dict(status='source_replay_complete',remaining='original-case EDA and hypothesis verdict'))


if __name__=='__main__':main()

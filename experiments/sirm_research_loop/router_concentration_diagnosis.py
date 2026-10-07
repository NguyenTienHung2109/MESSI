"""Predeclared source-only label-free routing concentration controls, saved bank."""
from pathlib import Path
import numpy as np
import torch
from domainbed.deit_transformer import DeiTFeaturizer
from .centered_model import CenteredGateSIRM
from .hard_router_diagnosis import outputs
from .bias_control import scores
from .shrinkage import ROOT,digest,write


def main():
    torch.set_num_threads(4)
    dest=ROOT/'results/sirm_research_loop/iteration_11'
    out=dest/'router_concentration_diagnosis';out.mkdir(exist_ok=True)
    path=dest/'experiment/runs/L38/OFF/source_best.pkl'
    p=torch.load(path,map_location='cpu',weights_only=False);assert p['selected_step']==399
    c=p['config'];model=CenteredGateSIRM(DeiTFeaturizer(pretrained=False,model_name=c['backbone']),384,10,3,c)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in p['model_dict'].items()})
    model.configure_stage();model.requires_grad_(False);model.eval()
    plan=dict(code_sha256=digest(Path(__file__)),checkpoint_sha256=digest(path),
        target_read=False,fitting=False,training=False,selection=False,
        policies=['deployed','argmax_including_NULL','argmax_expert_unit_mass',
                  'argmax_expert_preserve_NULL_mass','renormalize_nonNULL'],
        scope='Fixed saved-bank source-only controls; no new routing parameters or temperature fitting. '
              'These inference interventions are not trained variants; true labels only evaluate outcomes.')
    write(out/'plan.json',plan)
    result={};cases=[]
    for split in ['source_in','source_out']:
        cache=ROOT/'results/sirm_research_loop/iteration_09/source_router_diagnosis'/f'{split}_original.npz'
        with np.load(cache) as a:
            z,y,d=[torch.from_numpy(a[k]) for k in ['z','y','domains']];ids=a['ids'].tolist()
        assert set(d.tolist())=={0,2,3}
        if split=='source_in':
            anchor=torch.load(dest/'source_feature_cache.pt',map_location='cpu',weights_only=False)
            assert anchor['ids']==ids and torch.equal(anchor['features'],z)
        with torch.no_grad():f=outputs(model,z,y)
        g,delta,pi=[f[k].numpy() for k in ['g','delta','pi']]; yy=y.numpy();dd=d.numpy()
        nonnull=pi[:,:6];winner=nonnull.argmax(1);one=np.eye(6)[winner]
        choice7=pi.argmax(1)
        policies=dict(deployed=nonnull,
            argmax_including_NULL=one*(choice7!=6)[:,None],
            argmax_expert_unit_mass=one,
            argmax_expert_preserve_NULL_mass=one*nonnull.sum(1)[:,None],
            renormalize_nonNULL=nonnull/nonnull.sum(1)[:,None])
        globalpred=g.argmax(1);globalwrong=globalpred!=yy
        singlepred=(g[:,None]+delta).argmax(2)
        available=(singlepred==yy[:,None]).any(1)&globalwrong
        chosenrepairs=(singlepred[np.arange(len(yy)),winner]==yy)&globalwrong
        summary=dict(n=len(yy),global_wrong=int(globalwrong.sum()),
            single_repairable=int(available.sum()),top_expert_repairs=int(chosenrepairs.sum()),
            repairable_top_expert_correct_fraction=float(chosenrepairs[available].mean()),
            repairable_NULL_argmax_fraction=float((choice7[available]==6).mean()))
        summary['policies']={}
        for name,w in policies.items():
            assert np.all(w>=0) and np.all(w.sum(1)<=1+1e-6)
            logits=g+(w[:,:,None]*delta).sum(1)
            sc,pred=scores(logits,yy,dd)
            sc.update(repairs=int((globalwrong&(pred==yy)).sum()),harms=int((~globalwrong&(pred!=yy)).sum()))
            summary['policies'][name]=sc
            for i in range(len(yy)):
                if globalwrong[i] or pred[i]!=globalpred[i]:
                    cases.append(dict(split=split,id=ids[i],domain=int(dd[i]),label=int(yy[i]),
                        policy=name,prediction=int(pred[i]),global_prediction=int(globalpred[i]),
                        top_expert=int(winner[i]),NULL_argmax=bool(choice7[i]==6),
                        single_repairable=bool(available[i]),top_expert_can_repair=bool(chosenrepairs[i])))
        plan.setdefault('cache_sha256',{})[split]=digest(cache)
        result[split]=summary
    assert digest(path)==plan['checkpoint_sha256']
    write(out/'results.json',dict(plan=plan,summaries=result,cases=cases))
    print({s:{k:(v['accuracy'],v['CE'],v['repairs'],v['harms']) for k,v in r['policies'].items()} for s,r in result.items()})


if __name__=='__main__':main()

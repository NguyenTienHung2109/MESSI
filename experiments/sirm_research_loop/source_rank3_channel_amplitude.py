exec(open('/home/ntduc/Desktop/Dat/MESSI/experiments/sirm_research_loop/source_rank3_channel_payoff.py').read().split("for fit in ['pooled'")[0].replace("assert not (out/'results.json').exists()",''))
# Same immutable source-in norm, no source-out-selected amplitude.
results={}
for fit in ['pooled','held0','held2','held3']:
 results[fit]={};held=None if fit=='pooled' else int(fit[-1])
 for policy in ['learned','fixed']:
  a=torch.load(base/'source_rank3_risk_subspace'/f'{fit}_{policy}_risk_axis.pt',map_location='cpu',weights_only=False);w=(a['b']-a['b'].mean(0,keepdim=True))@a['a'];norms=None;replay={}
  for split in ['source_in','source_out']:
   d=dict(np.load(cache/f'{split}_tokens.npz'));mask=np.ones(len(d['y']),bool) if held is None else (d['domains']!=held if split=='source_in' else d['domains']==held);raw=torch.tensor(np.concatenate([d['cls'],d['patch_max']],1)[mask]);z=(raw-a['mean'])/a['std']/2**.5;g=torch.tensor(d['global_logits'][mask]);y=torch.tensor(d['y'][mask]);dom=torch.tensor(d['domains'][mask]);parts={'CLS':F.linear(z[:,:384],w[:,:384]),'patch_max':F.linear(z[:,384:],w[:,384:])};total=parts['CLS']+parts['patch_max']
   if split=='source_in':norms={k:float(v.square().mean().sqrt()) for k,v in dict(parts,full=total).items()}
   else:
    for name,corr in parts.items():
     scale=norms['full']/norms[name];replay[name]={'scale_to_full_source_in_RMS':scale,'score':score(g+scale*corr,g,y,dom)}
  results[fit][policy]={'source_in_RMS':norms,'source_out':replay}
save(out/'amplitude_controls.json',{'source_only':True,'target_read':False,'rule':'Each single channel scaled to its same saved head full source-in RMS, neither score nor scale selected on source-out. Zero in normalized feature space.','records':results})
for f,v in results.items():print(f,{p:{k:x['score'] for k,x in r['source_out'].items()} for p,r in v.items()})

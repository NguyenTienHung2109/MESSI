"""Post-hoc t-SNE and explicitly approximate 2D decision-region diagnostics.

No refitting of the predictive model, no target-driven selection. The 2D
surrogate interpolates frozen logits, never the ground-truth labels.
"""
import argparse
import base64
import csv
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from scipy.spatial import Delaunay
import torch

from domainbed.predictive_compatibility import CandidateMap
from domainbed.scripts.audit_predictive_compatibility import write_json

DOMAINS = ['Art', 'Cartoon', 'Photo', 'Sketch']
COLORS = ['#377eb8', '#ff7f00', '#4daf4a', '#984ea3', '#e41a1c', '#a65628', '#f781bf']


def sample_indices(labels, eligible, seed, per_class):
    rng = np.random.default_rng(seed)
    result = []
    for c in np.unique(labels):
        ids = eligible[labels[eligible] == c]
        result.extend(rng.choice(ids, min(len(ids), per_class), replace=False).tolist())
    return np.array(sorted(result), dtype=int)


def surrogate_regions(embedding, logits, domains, target, seed):
    train, test = train_test_split(np.arange(len(embedding)), test_size=.25,
                                   stratify=domains, random_state=seed)
    knn = KNeighborsRegressor(n_neighbors=15, weights='distance')
    knn.fit(embedding[train], logits[train])
    prediction = knn.predict(embedding[test]).argmax(1)
    actual = logits[test].argmax(1)
    fidelity = dict(all=float((prediction == actual).mean()),
                    source=float((prediction[domains[test]!=target] == actual[domains[test]!=target]).mean()),
                    target=float((prediction[domains[test]==target] == actual[domains[test]==target]).mean()),
                    n_test=len(test))
    # Fit all displayed points only AFTER recording held-out surrogate fidelity.
    knn.fit(embedding, logits)
    low, high = embedding.min(0), embedding.max(0)
    pad = (high-low)*.04
    low, high = low-pad, high+pad
    gx, gy = np.meshgrid(np.linspace(low[0], high[0], 180), np.linspace(low[1], high[1], 180))
    grid = np.c_[gx.ravel(), gy.ravel()]
    decision = knn.predict(grid).argmax(1).reshape(gx.shape)
    near = NearestNeighbors(n_neighbors=2).fit(embedding)
    local_dist = near.kneighbors(embedding)[0][:, 1]
    nearest = near.kneighbors(grid, n_neighbors=1)[0][:, 0]
    covered = (nearest <= 2*np.quantile(local_dist, .95)) & (Delaunay(embedding).find_simplex(grid) >= 0)
    masked = np.ma.array(decision, mask=~covered.reshape(gx.shape))
    return masked, [low[0], high[0], low[1], high[1]], fidelity


def scatter(ax, xy, color, domains, target, cmap, vmax):
    for held, marker, size in [(False, '.', 7), (True, '^', 15)]:
        mask = (domains == target) if held else (domains != target)
        ax.scatter(*xy[mask].T, c=color[mask], cmap=cmap, vmin=0, vmax=vmax,
                   marker=marker, s=size, alpha=.7, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])


def run_target(root, target, args, output):
    folder = root/f'env{target}_seed0'
    config = json.loads((folder/'config.json').read_text())
    manifest = json.loads((folder/'features.json').read_text())
    splits = json.loads((folder/'main/split_manifest.json').read_text())
    payload = torch.load(folder/'main/model.pt', map_location='cpu', weights_only=True)
    xs, ys, ds, original_ids = [], [], [], []
    for d in range(4):
        cache = Path(config['cache_dir'])/f'env{d}_{manifest[str(d)]["fingerprint"][:16]}.pt'
        data = torch.load(cache, map_location='cpu', weights_only=True)
        if data['fingerprint'] != manifest[str(d)]['fingerprint']:
            raise ValueError('Cache fingerprint mismatch')
        labels = data['y'].numpy()
        # Source points are unseen during map/head/router fitting; use selection
        # split, which did participate in the deployment-choice rule.
        eligible = np.arange(len(labels)) if d == target else np.array(splits[str(d)]['selection'])
        indices = sample_indices(labels, eligible, args.seed+1009*d, args.per_class)
        xs.append(data['x'][indices]); ys.append(labels[indices])
        ds.extend([d]*len(indices)); original_ids.extend(indices.tolist())
    x = torch.cat(xs)
    y, domains = np.concatenate(ys), np.array(ds)
    maps = []
    for state in payload['maps']:
        model = CandidateMap(x.shape[1], config['dimension'], config['hidden'])
        model.load_state_dict(state)
        maps.append(model.eval())
    router = torch.nn.Linear(x.shape[1], len(maps))
    router.load_state_dict(payload['all_router'])
    with torch.no_grad():
        features = [m(x) for m in maps]
        logits = torch.stack([z@w.T for z,w in zip(features, payload['heads'])], 1)
        weights = router(x).softmax(1)
        mixture = (weights[:,:,None]*logits).sum(1).numpy()
    representations = [('backbone_router', x.numpy(), mixture)] + [
        (f'expert{i}', z.numpy(), logits[:,i].numpy()) for i,z in enumerate(features)]
    classes = manifest['0']['classes']
    fig_dist, axes_dist = plt.subplots(6, 2, figsize=(12, 24), layout='constrained')
    fig_bound, axes_bound = plt.subplots(6, 2, figsize=(12, 24), layout='constrained')
    cmap = ListedColormap(COLORS)
    rows, views = [], []
    for row, (name, representation, output_logits) in enumerate(representations):
        print(f'Target {target}: {name}, n={len(y)}', flush=True)
        # PCA is an unsupervised preprocessing step, not a model adaptation.
        inputs = PCA(n_components=50, random_state=args.seed).fit_transform(representation) if representation.shape[1] > 50 else representation
        model = TSNE(n_components=2, perplexity=args.perplexity, init='pca',
                     learning_rate='auto', max_iter=1000, random_state=args.seed)
        xy = model.fit_transform(inputs)
        regions, extent, fidelity = surrogate_regions(xy, output_logits, domains, target, args.seed)
        trust = float(trustworthiness(representation, xy, n_neighbors=15))
        prediction = output_logits.argmax(1)
        row_data = dict(target=target, representation=name, n=len(y), n_target=int((domains==target).sum()),
                        kl_divergence=float(model.kl_divergence_), trustworthiness_15=trust,
                        surrogate_fidelity=fidelity['all'], surrogate_fidelity_source=fidelity['source'],
                        surrogate_fidelity_target=fidelity['target'],
                        sampled_target_accuracy=float((prediction[domains==target]==y[domains==target]).mean()))
        rows.append(row_data)
        np.savez_compressed(output/f'env{target}_{name}.npz', xy=xy, labels=y, domains=domains,
                            sample_indices=np.array(original_ids), logits=output_logits,
                            router_weights=weights.numpy())
        scatter(axes_dist[row,0], xy, domains, domains, target, ListedColormap(COLORS[:4]), 3)
        scatter(axes_dist[row,1], xy, y, domains, target, cmap, 6)
        axes_dist[row,0].set_title(f'{name}: domain | trustworthiness={trust:.3f}')
        axes_dist[row,1].set_title(f'{name}: true class')
        scatter(axes_bound[row,0], xy, prediction, domains, target, cmap, 6)
        wrong = (domains==target) & (prediction!=y)
        axes_bound[row,0].scatter(*xy[wrong].T, s=24, facecolors='none', edgecolors='black', linewidths=.45)
        axes_bound[row,0].set_title(f'{name}: actual predictions (ring = target error)')
        axes_bound[row,1].imshow(regions, extent=extent, origin='lower', cmap=cmap, vmin=0, vmax=6, alpha=.22, aspect='auto')
        scatter(axes_bound[row,1], xy, y, domains, target, cmap, 6)
        axes_bound[row,1].set_xlim(extent[:2]); axes_bound[row,1].set_ylim(extent[2:])
        axes_bound[row,1].set_title(f'2D surrogate regions | held-point fidelity={fidelity["all"]:.1%}')
        rgba = cmap(np.asarray(regions.filled(0))/6)
        rgba[:,:,3] = np.where(np.ma.getmaskarray(regions), 0, .2)
        buffer = io.BytesIO()
        plt.imsave(buffer, rgba, format='png', origin='lower')
        views.append(dict(target=target, name=name, extent=list(map(float, extent)),
                          fidelity=fidelity, trust=trust,
                          image=base64.b64encode(buffer.getvalue()).decode(),
                          points=[[round(float(a),4), round(float(b),4), int(d), int(label), int(pred), int(idx),
                                   [round(float(w),4) for w in mass]]
                                  for (a,b),d,label,pred,idx,mass in zip(xy,domains,y,prediction,original_ids,weights)]))
    domain_legend = [Line2D([],[], marker='o', linestyle='', color=COLORS[d], label=DOMAINS[d]) for d in range(4)]
    class_legend = [Line2D([],[], marker='o', linestyle='', color=COLORS[c], label=classes[c]) for c in range(7)]
    axes_dist[0,0].legend(handles=domain_legend, fontsize=7, loc='lower left')
    axes_dist[0,1].legend(handles=class_legend, fontsize=7, loc='lower left')
    axes_bound[0,0].legend(handles=class_legend, fontsize=7, loc='lower left')
    fig_dist.suptitle(f'Target {DOMAINS[target]} | circles: source selection; triangles: target\nSeparate t-SNE per row; cross-row distances are not comparable', fontsize=14)
    fig_bound.suptitle(f'Target {DOMAINS[target]} | actual predictions vs 2D logit interpolation\nSurrogate regions are NOT the original network boundary; blank = unsupported region', fontsize=14)
    fig_dist.savefig(output/f'env{target}_distributions.png', dpi=140)
    fig_bound.savefig(output/f'env{target}_decision_regions.png', dpi=140)
    plt.close(fig_dist); plt.close(fig_bound)
    return rows, views, classes


def html_viewer(views, classes, output):
    html = '''<!doctype html><meta charset="utf-8"><title>PACS t-SNE explorer</title>
<style>body{font:15px system-ui;margin:24px;background:#fafafa;color:#222}select,label{margin-right:16px}svg{background:white;border:1px solid #ddd;max-width:100%}#tip{position:fixed;display:none;background:#222;color:white;padding:10px;pointer-events:none;white-space:pre-line}p{max-width:1100px}</style>
<h1>PACS t-SNE explorer — seed 0</h1>
<p>Points: source selection + sampled target. Labels and target features are used for visualization only. The background is a 2D surrogate interpolating frozen logits, not the original decision boundary. Distances across embeddings are not comparable.</p>
<select id="view"></select><label>Color <select id="color"><option value="domain">Domain</option><option value="label">True class</option><option value="prediction">Actual prediction</option><option value="route">Top-1 router expert</option></select></label>
<label><input id="boundary" type="checkbox">2D surrogate background</label><label><input id="errors" type="checkbox">Target errors only</label>
<p id="stats"></p><p id="legend"></p><svg id="plot" viewBox="0 0 1100 700" width="1100" height="700"></svg><div id="tip"></div>
<script>const views=__VIEWS__, classes=__CLASSES__, domains=['Art','Cartoon','Photo','Sketch'], colors=__COLORS__;
const sel=document.querySelector('#view'),svg=document.querySelector('#plot'),tip=document.querySelector('#tip');
views.forEach((v,i)=>sel.add(new Option('Target '+domains[v.target]+' / '+v.name,i)));sel.value=String(views.findIndex(v=>v.target===3));
function draw(){const v=views[+sel.value],mode=document.querySelector('#color').value;svg.replaceChildren();
document.querySelector('#stats').textContent=`Neighborhood trustworthiness: ${v.trust.toFixed(3)}; held-point surrogate fidelity: ${(100*v.fidelity.all).toFixed(1)}% (target ${(100*v.fidelity.target).toFixed(1)}%). Hover points for details. Triangle = target.`;
const names=mode==='domain'?domains:mode==='route'?['expert0','expert1','expert2','expert3','expert4']:classes;
document.querySelector('#legend').innerHTML=names.map((n,i)=>`<span style="color:${colors[i]};margin-right:16px">● ${n}</span>`).join('');
if(document.querySelector('#boundary').checked){const im=document.createElementNS('http://www.w3.org/2000/svg','image');im.setAttribute('href','data:image/png;base64,'+v.image);im.setAttribute('width',1100);im.setAttribute('height',700);im.setAttribute('preserveAspectRatio','none');svg.append(im);}
v.points.forEach(p=>{if(document.querySelector('#errors').checked&&!(p[2]===v.target&&p[3]!==p[4]))return;
const x=(p[0]-v.extent[0])/(v.extent[1]-v.extent[0])*1100,y=700-(p[1]-v.extent[2])/(v.extent[3]-v.extent[2])*700;
const route=p[6].indexOf(Math.max(...p[6])), c=mode==='domain'?p[2]:mode==='label'?p[3]:mode==='prediction'?p[4]:route;
const el=document.createElementNS('http://www.w3.org/2000/svg',p[2]===v.target?'polygon':'circle');
if(p[2]===v.target)el.setAttribute('points',`${x},${y-4} ${x-4},${y+3} ${x+4},${y+3}`);else{el.setAttribute('cx',x);el.setAttribute('cy',y);el.setAttribute('r',3);}
el.setAttribute('fill',colors[c]);el.setAttribute('opacity','.8');if(p[2]===v.target&&p[3]!==p[4]){el.setAttribute('stroke','#222');el.setAttribute('stroke-width','.6');}
el.onmousemove=e=>{tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';tip.textContent=`${domains[p[2]]}, sample ${p[5]}\nLabel: ${classes[p[3]]}\nPrediction: ${classes[p[4]]}\nRouting: ${p[6].join(', ')}`};el.onmouseleave=()=>tip.style.display='none';svg.append(el);});}
document.querySelectorAll('select,input').forEach(el=>el.onchange=draw);draw();</script>'''
    html = html.replace('__VIEWS__', json.dumps(views)).replace('__CLASSES__', json.dumps(classes)).replace('__COLORS__',json.dumps(COLORS))
    (output/'explorer.html').write_text(html)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('predictive_support_outputs/pacs_compatibility_lodo_seed0'))
    parser.add_argument('--output', type=Path, default=Path('predictive_support_outputs/pacs_compatibility_tsne_seed0'))
    parser.add_argument('--targets', type=int, nargs='+', default=[0,1,2,3])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--per-class', type=int, default=40)
    parser.add_argument('--perplexity', type=float, default=30.)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh visualization output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    rows, views = [], []
    for target in args.targets:
        r,v,classes=run_target(args.root,target,args,args.output)
        rows.extend(r);views.extend(v)
        write_json(args.output/'metrics.json', rows)
    html_viewer(views,classes,args.output)
    with (args.output/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(args.output/'config.json',dict(seed=args.seed, targets=args.targets, per_class=args.per_class,
        perplexity=args.perplexity, iterations=1000, sklearn=sklearn.__version__,
        source_points='selection split', target_points='up to 40/class, sampled with labels for display only',
        surrogate='15-neighbor distance-weighted logit regression; 25% held-point fidelity; fit all for display',
        caveat='Transductive t-SNE; fidelity is conditional on this embedding, not unseen-image validation'))
    (args.output/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    lines=['# PACS t-SNE: data distribution and approximate decision regions','',
           '[Open interactive explorer](explorer.html) — hover for labels/predictions/routing; switch domain/class colors and error filtering.','',
           'Each run contains six separately fitted embeddings: frozen backbone and five expert maps.',
           'Circles are source selection samples; triangles are held-target samples. Sampling is capped at 40 images/class/domain.',
           'Source selection data were used for deployment selection, though not map/head/router optimization.',
           'The plotted sample is not population-proportional; reported full-run accuracies remain authoritative.','',
           '**Boundary limitation:** t-SNE has no unique inverse. Colored regions interpolate frozen model logits in 2D.',
           'They are not exact network boundaries. Unsupported gaps outside the hull or far from samples are blank.',
           'Held-point fidelity uses 25% of points withheld from surrogate fitting; t-SNE still uses all features.',
           'This measures visualization fidelity, not generalization. Ground-truth labels never train the surrogate.',
           'Global distances, cluster sizes and separation across different panels cannot establish invariance.','',
           '| Target | Representation | Neighborhood trustworthiness | 2D fidelity | Target 2D fidelity |',
           '|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {DOMAINS[r["target"]]} | {r["representation"]} | {r["trustworthiness_15"]:.3f} | {r["surrogate_fidelity"]:.1%} | {r["surrogate_fidelity_target"]:.1%} |')
    for target in args.targets:
        lines += ['',f'## Target {DOMAINS[target]}','',f'![Distributions](env{target}_distributions.png)','',f'![Decision regions](env{target}_decision_regions.png)']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()

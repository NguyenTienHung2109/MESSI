"""Read-only source case/official metadata join; benchmark untouched."""
import csv,json
from pathlib import Path
from .routed_discovery_run import ROOT,DEST
from .shrinkage import digest,write

def run():
 meta=ROOT/'results/sirm_research_loop/iteration_08/source_annotation_audit';p=meta/'selected_source_results.json';r=json.loads(p.read_text());assert digest(meta/'caltech_images_20210113.json')==r['metadata_sha256'];manifest=json.loads((DEST/'experiment/subset_manifest.json').read_text());lookup={a['local_id']:a for a in r['records']};trainseq={a['official_sequence'] for a in r['records'] if a['split']=='source_in'};rows=list(csv.DictReader((DEST/'eda/cases.csv').open()));out=[];names=manifest['class_names']
 for c in rows:
  if c['split']!='source_out':continue
  a=lookup[c['image_id']];assert a['split']=='source_out' and a['label_matches'] and a['domain_matches'] and a['image_content_matches'];assert digest(ROOT/'domainbed/data/terra_incognita'/c['image_id'])==a['expected_image_sha256']
  out.append(dict(image_id=c['image_id'],environment=int(c['environment']),folder_label=c['class_name'],official_labels=a['official_labels'],official_sequence=a['official_sequence'],sequence_seen_in_source_train=a['official_sequence'] in trainseq,pattern=c['pattern'],predictions={b:names[int(c[b+'_prediction'])] for b in ['B0','B1','B2']},correct={b:c[b+'_correct']=='True' for b in ['B0','B1','B2']}))
 summary={}
 for label,mask in [('all',lambda a:True),('sequence_seen',lambda a:a['sequence_seen_in_source_train']),('sequence_unseen',lambda a:not a['sequence_seen_in_source_train']),('multi_official_labels',lambda a:len(a['official_labels'])>1),('single_official_label',lambda a:len(a['official_labels'])==1)]:
  a=[x for x in out if mask(x)];summary[label]=dict(n=len(a),image_mean_accuracy={b:sum(x['correct'][b] for x in a)/len(a) if a else None for b in ['B0','B1','B2']},B1_repairs=sum(not x['correct']['B0'] and x['correct']['B1'] for x in a),B1_harms=sum(x['correct']['B0'] and not x['correct']['B1'] for x in a),global_other_official_label_matches=[x['image_id'] for x in a if not x['correct']['B0'] and x['predictions']['B0'] in x['official_labels']])
 write(DEST/'eda/source_annotation_join.json',dict(source_only=True,target_metadata_read=False,benchmark_unchanged=True,training=False,selected_source_audit_sha256=digest(p),metadata_sha256=r['metadata_sha256'],rows=out,summary=summary,scope='All304 source validation records matched with existing official metadata. Sequence-unseen is descriptive and does not undo backbone/bank selection on the full source-out split. Official labels are not human annotation verification; alternate-label matches do not alter single-label accuracy.'))
if __name__=='__main__':run()

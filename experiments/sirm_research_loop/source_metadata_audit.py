"""Verify selected source labels and event identities against official CCT metadata.

No relabeling or split changes. Official full-dataset metadata is filtered to
source manifest records; target records are never used for the audit.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .shrinkage import ROOT, digest, write


def run(dest):
    out = dest / 'source_annotation_audit'
    path = out / 'caltech_images_20210113.json'
    official = json.loads(path.read_text())
    manifest = json.loads((dest / 'experiment/subset_manifest.json').read_text())
    categories = {x['id']: x['name'] for x in official['categories']}
    local = []
    for domain in [0, 2, 3]:
        source = manifest['sources'][str(domain)]
        for split in ['in', 'out']:
            assert source[split + '_ids'] == [source['images'][i]['id'] for i in source[split]]
            for index in source[split]:
                item = source['images'][index]
                local.append(dict(environment=domain, split='source_' + split,
                                  local_id=item['id'], local_class=manifest['class_names'][item['class']],
                                  expected_image_sha256=item['sha256']))
    ids = {Path(x['local_id']).stem for x in local}
    images = {x['id']: x for x in official['images'] if x['id'] in ids}
    annotations = defaultdict(list)
    for x in official['annotations']:
        if x['image_id'] in ids:
            annotations[x['image_id']].append(categories[x['category_id']])
    records = []
    for row in local:
        image_id = Path(row['local_id']).stem
        metadata = images.get(image_id, {})
        labels = sorted(set(annotations[image_id]))
        records.append(dict(row, original_image_id=image_id,
                            official_labels=labels,
                            official_sequence=metadata.get('seq_id'),
                            official_datetime=metadata.get('date_captured'),
                            official_location=metadata.get('location'),
                            found=image_id in images,
                            label_matches=row['local_class'] in labels,
                            domain_matches=metadata.get('location') == str({0:100, 2:43, 3:46}[row['environment']]),
                            image_content_matches=digest(ROOT / 'domainbed/data/terra_incognita' / row['local_id']) == row['expected_image_sha256']))
    by_split = {s:[r for r in records if r['split'] == s] for s in ['source_in', 'source_out']}
    in_images = {r['original_image_id'] for r in by_split['source_in']}
    in_sequences = {r['official_sequence'] for r in by_split['source_in'] if r['official_sequence'] is not None}
    out_images = {r['original_image_id'] for r in by_split['source_out']}
    out_sequences = {r['official_sequence'] for r in by_split['source_out'] if r['official_sequence'] is not None}
    ambiguity = []
    with np.load(dest / 'source_router_diagnosis/source_out_original.npz') as a:
        saved_ids, saved_y, pred = a['ids'], a['y'], a['g'].argmax(1)
        lookup = {r['local_id']:r for r in by_split['source_out']}
        for id, y, prediction in zip(saved_ids, saved_y, pred):
            if y != prediction and manifest['class_names'][prediction] in lookup[str(id)]['official_labels']:
                ambiguity.append(dict(local_id=str(id), folder_label=manifest['class_names'][y],
                                      global_prediction=manifest['class_names'][prediction],
                                      official_labels=lookup[str(id)]['official_labels']))
    result = dict(
        metadata_sha256=digest(path), metadata_version=official['info']['version'],
        official_page='https://lila.science/datasets/caltech-camera-traps',
        download_provenance=json.loads((out / 'download_provenance.json').read_text()),
        scope='Fixed10% selected source-in/out records only, preserving multi-class local copies. '
              'Full official archive contains other locations, but no target metadata record is used here. '
              'No labels, images, splits, training or scores changed. Official label agreement is not a human correctness audit.',
        selected_source_rows=len(records), unique_original_source_images=len(ids),
        missing=sum(not r['found'] for r in records),
        label_mismatches=sum(not r['label_matches'] for r in records),
        domain_mismatches=sum(not r['domain_matches'] for r in records),
        image_hash_mismatches=sum(not r['image_content_matches'] for r in records),
        multi_distinct_label_rows=sum(len(r['official_labels']) > 1 for r in records),
        original_image_ID_overlap=sorted(in_images & out_images),
        sequence_overlap_count=len(in_sequences & out_sequences),
        source_out_rows_with_source_in_sequence=sum(r['official_sequence'] in in_sequences for r in by_split['source_out']),
        per_split={s:dict(rows=len(rs), unique_original_images=len({r['original_image_id'] for r in rs}),
                         unique_sequences=len({r['official_sequence'] for r in rs if r['official_sequence'] is not None}),
                         missing_sequence=sum(r['official_sequence'] is None for r in rs)) for s,rs in by_split.items()},
        global_benchmark_wrong_but_other_official_label_matches=ambiguity,
        records=records)
    write(out / 'selected_source_results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('destination', type=Path)
    run(parser.parse_args().destination.resolve())

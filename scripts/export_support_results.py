"""Create a Git-friendly point-in-time export without model checkpoints."""
import argparse
import datetime
import json
from pathlib import Path
import shutil


def export(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'export destination already exists: {destination}')
    destination.mkdir(parents=True)
    queue = json.loads((source / 'queue.json').read_text())
    completed, running = [], []
    for job in queue['jobs']:
        if job['state'] in ('completed', 'running'):
            (completed if job['state'] == 'completed' else running).append(Path(job['output']).name)
        # Published metadata must be portable across machines.
        job['output'] = Path(job['output']).name
        job['log'] = Path(job['log']).name
    excluded = []
    for path in sorted(source.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        relative = path.relative_to(source)
        if path.suffix in ('.pt', '.pth', '.pkl', '.pyc'):
            excluded.append({'path': str(relative), 'bytes': path.stat().st_size})
            continue
        if path.name in ('queue.json', 'queue.tmp'):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in ('.jsonl', '.log'):
            data = path.read_bytes()
            # A running writer may be part-way through its final record.
            end = data.rfind(b'\n')
            target.write_bytes(data[:end + 1] if end >= 0 else b'')
        else:
            shutil.copy2(path, target)
    (destination / 'queue.json').write_text(json.dumps(queue, indent=2) + '\n')
    manifest = {'exported_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'seeds': queue['seeds'], 'completed_runs': completed,
                'partial_runs': running, 'excluded_checkpoints': excluded,
                'note': 'Point-in-time export; partial run metrics are not final target results.'}
    (destination / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    from domainbed.scripts.run_support_revision_suite import summarize
    local_jobs = [dict(j, output=str(destination / j['output'])) for j in queue['jobs']]
    summarize(destination, local_jobs)
    summary = json.loads((destination / 'summary.json').read_text())
    text = ['# PACS v2 — seed 0 results', '',
            'Snapshot exported at ' + manifest['exported_utc'] + '.', '',
            'Seeds 1 and 2 were cancelled at the user\'s request. Seed-0 ablations continue locally.', '',
            '| Arm | Target | Accuracy | Selected step | EDA |', '|---|---|---:|---:|---|']
    names = ['Art painting', 'Cartoon', 'Photo', 'Sketch']
    for arm, group in summary.items():
        for seed, targets in group['seeds'].items():
            for target, result in targets.items():
                name = f'{arm}_env{target}_seed{seed}'
                text.append(f'| {arm} | {names[int(target)]} | {100 * result["target_accuracy"]:.2f}% | {result["selected_step"]} | [report]({name}/eda/report.md) |')
        if group['mean_accuracy'] is not None:
            text.extend(['', f'{arm}: mean target accuracy **{100 * group["mean_accuracy"]:.2f}%**.', ''])
    text.extend(['', 'Completed runs: ' + ', '.join(completed) + '.',
                 'Partial runs: ' + (', '.join(running) or 'none') + '.', '',
                 'Logs, metrics, prediction arrays, configs, EDA and the frozen source snapshot are included. Model checkpoints remain local; see export_manifest.json for excluded files.'])
    (destination / 'README.md').write_text('\n'.join(text) + '\n')
    # Replace the copied live overview with an export-scoped report so its
    # local absolute paths and moving progress do not misrepresent this snapshot.
    (destination / 'report.md').write_text('\n'.join(text) + '\n')
    print(json.dumps({'destination': str(destination), 'completed': len(completed),
                      'partial': len(running), 'excluded_checkpoints': len(excluded)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    export(args.source, args.destination)

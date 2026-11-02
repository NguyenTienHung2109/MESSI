"""Prepare the frozen sampling plan and public ImageNet-A/V2 target sets."""
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

BASE = Path('results/reproduction_ood_table2')
PLAN = Path(__file__).with_name('imagenet_sampling_plan.json')
SETS = (
    ('https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar', 'imagenet-a.tar',
     'imagenet-adversarial', 'imagenet-a', 7500),
    ('https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz',
     'imagenetv2-matched-frequency.tar.gz', 'imagenetv2', 'imagenetv2-matched-frequency-format-val', 10000),
)


def main():
    index = BASE / 'imagenet_mirror_index'
    index.mkdir(parents=True, exist_ok=True)
    target_plan = index / 'sampling_plan.json'
    if target_plan.exists() and target_plan.read_bytes() != PLAN.read_bytes():
        raise RuntimeError('Existing sampling plan differs from the published plan')
    if not target_plan.exists():
        shutil.copyfile(PLAN, target_plan)
    downloads = BASE / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    for url, filename, folder, image_dir, count in SETS:
        target = BASE / 'data' / folder
        marker = target / 'REPRODUCTION_DATA_COMPLETE.json'
        if marker.exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        archive = downloads / filename
        subprocess.run(['curl', '-fL', '--retry', '8', '--retry-delay', '5', '-C', '-', '-o', str(archive), url], check=True)
        with tarfile.open(archive, 'r:*') as tar:
            for member in tar.getmembers():
                p = Path(member.name)
                if p.is_absolute() or '..' in p.parts or not (member.isfile() or member.isdir()):
                    raise RuntimeError(f'Unsafe archive member {member.name}')
            tar.extractall(target)
        images = target / image_dir
        alternate = target / 'imagenetv2-matched-frequency'
        if folder == 'imagenetv2' and not images.exists() and alternate.exists():
            alternate.rename(images)
        items = [p for p in images.glob('*/*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        if len(items) != count:
            raise RuntimeError(f'{folder}: expected {count} images, got {len(items)}')
        marker.write_text(json.dumps({'images': count, 'source_url': url}) + '\n')
        archive.unlink()
        print(folder, 'ready:', count, 'images', flush=True)


if __name__ == '__main__':
    main()

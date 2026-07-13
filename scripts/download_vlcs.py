"""Download VLCS from the JigenDG MediaFire mirror and re-format it to the
DomainBed `MultipleEnvironmentImageFolder` layout.

Source (working as of May 2026):
    https://download1514.mediafire.com/...vlcs.tar.gz
    (192 MB, mirrored from JigenDG/CVPR'19)

The raw archive has:
    VLCS/{CALTECH,LABELME,PASCAL,SUN}/{train,test,crossval,full}/{0..4}/*.jpg
where `full == train + crossval` and `test` is disjoint. We take `full + test`
(union) and write the DomainBed layout:
    VLCS/{Caltech101,LabelMe,SUN09,VOC2007}/{bird,car,chair,dog,person}/*.jpg

The DomainBed VLCS class sorts environments alphabetically, so picking names
that begin with C/L/S/V preserves the canonical ENVIRONMENTS=['C','L','S','V']
ordering.

Usage:
    python -m scripts.download_vlcs               # default → domainbed/data/
    python -m scripts.download_vlcs --data_dir X  # custom destination
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile

CLASS_MAP = {"0": "bird", "1": "car", "2": "chair", "3": "dog", "4": "person"}
ENV_MAP = {"CALTECH": "Caltech101", "LABELME": "LabelMe",
           "PASCAL":  "VOC2007",    "SUN":     "SUN09"}
KEEP_SPLITS = ("full", "test")   # union = all unique images, no duplicates
MIRROR_URL = (
    "https://download1514.mediafire.com/lfg4y6hln4tgkWeDEP5gOSzC0DR4qrp0-26pX"
    "8GwcrC69q6FLSUFPcSWqFyaxKabmkmR7axvXjhtRkm9-ynjgl5Y5yY4qCLfxzLe8-RTuVOm"
    "EbKwYiGT_d7batoTIjVmQpCZgFnw400i79SfCjkm39Uhq5fmXA3u-HClvMoO_t61ZYs/"
    "7yv132lgn1v267r/vlcs.tar.gz"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./domainbed/data")
    p.add_argument("--keep_archive", action="store_true",
                   help="don't delete vlcs.tar.gz after extraction")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "vlcs.tar.gz")
    raw_dir  = os.path.join(data_dir, "VLCS_raw")
    out_dir  = os.path.join(data_dir, "VLCS")

    if os.path.isdir(out_dir):
        print(f"[skip] {out_dir} already exists. Delete it first to re-download.")
        return

    # --- 1. download
    if not os.path.exists(tar_path):
        print(f"[download] {MIRROR_URL}\n           → {tar_path}")
        subprocess.run(["curl", "-L", "-o", tar_path, MIRROR_URL], check=True)
    else:
        print(f"[skip] {tar_path} already downloaded ({os.path.getsize(tar_path)/1e6:.1f} MB)")

    # --- 2. extract to a temp staging dir
    if os.path.isdir(raw_dir):
        shutil.rmtree(raw_dir)
    os.makedirs(raw_dir)
    print(f"[extract] → {raw_dir}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(raw_dir)

    src_root = os.path.join(raw_dir, "VLCS")
    if not os.path.isdir(src_root):
        sys.exit(f"[error] expected {src_root} inside archive, not found")

    # --- 3. restructure into DomainBed layout
    n_copied = {env: 0 for env in ENV_MAP.values()}
    for src_env, dst_env in ENV_MAP.items():
        for split in KEEP_SPLITS:
            split_dir = os.path.join(src_root, src_env, split)
            if not os.path.isdir(split_dir):
                continue
            for class_idx, class_name in CLASS_MAP.items():
                src_cls = os.path.join(split_dir, class_idx)
                if not os.path.isdir(src_cls):
                    continue
                dst_cls = os.path.join(out_dir, dst_env, class_name)
                os.makedirs(dst_cls, exist_ok=True)
                for fname in os.listdir(src_cls):
                    # Prefix split to disambiguate (e.g. train_imgs_9.jpg in
                    # both full/ and test/ would otherwise collide).
                    new_name = f"{split}_{fname}"
                    shutil.copy2(
                        os.path.join(src_cls, fname),
                        os.path.join(dst_cls, new_name),
                    )
                    n_copied[dst_env] += 1

    # --- 4. cleanup
    shutil.rmtree(raw_dir)
    if not args.keep_archive:
        os.remove(tar_path)

    # --- 5. report
    print("\n[done] DomainBed-formatted VLCS at:", out_dir)
    print(f"{'env':<12} {'#images':>8}")
    for env, n in n_copied.items():
        print(f"  {env:<10} {n:>8}")
    total = sum(n_copied.values())
    print(f"{'  TOTAL':<12} {total:>8}")
    expected = 10729
    print(f"\n(reference: standard VLCS has ~{expected} images)")


if __name__ == "__main__":
    main()

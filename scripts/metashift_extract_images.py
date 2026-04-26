"""Extract only the GQA images referenced by existing MetaShift_K split CSVs.

Reads:
    data/metashift/raw/gqa_images.zip                       (21.8 GB)
    data/metashift/splits/<class_set>/*.csv                 (manifests)

Writes:
    data/metashift/raw/images/<vg_image_id>.jpg             (extracted)
    data/metashift/raw/extraction_summary.json
"""
from __future__ import annotations

import json
import zipfile
from glob import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "data" / "metashift" / "raw" / "gqa_images.zip"
SPLITS_DIR = ROOT / "data" / "metashift" / "splits"
OUT_DIR = ROOT / "data" / "metashift" / "raw" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    csvs = sorted(glob(str(SPLITS_DIR / "*" / "*.csv")))
    needed: set[int] = set()
    for p in csvs:
        df = pd.read_csv(p, usecols=["vg_image_id"])
        needed.update(df.vg_image_id.astype(int).tolist())
    print(f"Need {len(needed)} unique image IDs across {len(csvs)} CSVs.")

    have = {int(p.stem) for p in OUT_DIR.glob("*.jpg")}
    missing = sorted(needed - have)
    print(f"Already have {len(have & needed)}; missing {len(missing)}")
    if not missing:
        print("Nothing to extract.")
        return

    extracted, not_found = 0, []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = set(zf.namelist())
        for vid in missing:
            arc = f"images/{vid}.jpg"
            if arc not in names:
                not_found.append(vid)
                continue
            with zf.open(arc) as src, (OUT_DIR / f"{vid}.jpg").open("wb") as dst:
                dst.write(src.read())
            extracted += 1
            if extracted % 500 == 0:
                print(f"  extracted {extracted}/{len(missing)}")
    print(f"Extracted {extracted}; missing in zip: {len(not_found)}")
    summary = {
        "needed": len(needed),
        "already_had": len(have & needed),
        "extracted": extracted,
        "not_found_in_zip": not_found,
    }
    (OUT_DIR.parent / "extraction_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

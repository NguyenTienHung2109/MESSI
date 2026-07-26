"""Evaluate a DomainBed checkpoint on one complete environment.

Example:
    python -m domainbed.scripts.eval \
        --env 0 \
        --dir_dataset /path/to/PACS \
        --dir_ckpt /path/to/model.pkl

The dataset name, model architecture, and hparams may be embedded in the
checkpoint or stored in a ``metadata.json`` sidecar beside weights-only
checkpoints. The command line intentionally only exposes the three inputs
needed at eval time. ``dir_dataset`` may point either to the dataset directory
itself (for example ``.../data/PACS``) or to its parent (``.../data``).
"""

import argparse
import json
import os
import sys
import warnings

# Keep the CLI result readable despite deprecations/registry notices emitted by
# the repository's pinned timm compatibility layer.
warnings.filterwarnings("ignore", message="Importing from timm.models")
warnings.filterwarnings("ignore", message="Overwriting .* in registry")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Some legacy model modules use top-level imports such as ``vit_helpers``.
# Match train.py's import bootstrap so ``python -m`` works from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domainbed import datasets
from domainbed.deit_transformer import DeiTFeaturizer, ExplicitMoEHead


_METADATA_FILENAME = "metadata.json"
_CHECKPOINT_FIELDS = {
    "args",
    "model_input_shape",
    "model_num_classes",
    "model_num_domains",
    "model_hparams",
}


class _MESSIInferenceModel(nn.Module):
    """Inference-only reconstruction for MESSI and legacy GMOE variants."""

    def __init__(self, input_shape, num_classes, hparams):
        super().__init__()
        if tuple(input_shape) != (3, 224, 224):
            raise ValueError(
                "Legacy MESSI/GMOE image checkpoints currently require "
                f"input_shape=(3, 224, 224), got {tuple(input_shape)}"
            )

        model_name = hparams.get("model", "deit_small_patch16_224")
        self.featurizer = DeiTFeaturizer(
            model_name=model_name,
            pretrained=False,
        )
        self.moe_head = ExplicitMoEHead(
            in_dim=self.featurizer.n_outputs,
            expert_dim=hparams.get("expert_dim", self.featurizer.n_outputs),
            num_experts=hparams.get("num_experts", 6),
            num_classes=num_classes,
            mlp_ratio=hparams.get(
                "expert_mlp_ratio", hparams.get("mlp_ratio", 4.0)
            ),
            prune_ratio=hparams.get("expert_prune_ratio", 0.0),
            expert_depth=hparams.get("expert_depth", 2),
        )

    def predict(self, x):
        features = self.featurizer(x)
        logits, _, _ = self.moe_head(features)
        return logits


def _same_metadata_value(field, left, right):
    if field == "model_input_shape":
        return tuple(left) == tuple(right)
    return left == right


def _merge_sidecar_metadata(checkpoint, metadata, metadata_path):
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be a JSON object: {metadata_path}")
    if metadata.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported or missing schema_version in {metadata_path}"
        )

    allowed = _CHECKPOINT_FIELDS | {"schema_version"}
    unexpected = sorted(set(metadata) - allowed)
    if unexpected:
        raise ValueError(
            f"Unexpected metadata fields in {metadata_path}: {unexpected}"
        )

    for field in ("args", "model_hparams"):
        sidecar_value = metadata.get(field)
        if sidecar_value is None:
            continue
        if not isinstance(sidecar_value, dict):
            raise ValueError(
                f"metadata[{field!r}] must be an object: {metadata_path}"
            )
        embedded_value = checkpoint.get(field)
        if embedded_value is None:
            checkpoint[field] = dict(sidecar_value)
            continue
        if not isinstance(embedded_value, dict):
            raise ValueError(f"checkpoint[{field!r}] must be a mapping")

        merged = dict(sidecar_value)
        for key, value in embedded_value.items():
            if key in sidecar_value and sidecar_value[key] != value:
                raise ValueError(
                    f"Checkpoint/metadata conflict for {field}.{key}: "
                    f"{value!r} != {sidecar_value[key]!r}"
                )
            merged[key] = value
        checkpoint[field] = merged

    for field in (
        "model_input_shape",
        "model_num_classes",
        "model_num_domains",
    ):
        if field not in metadata:
            continue
        if field in checkpoint and not _same_metadata_value(
            field, checkpoint[field], metadata[field]
        ):
            raise ValueError(
                f"Checkpoint/metadata conflict for {field}: "
                f"{checkpoint[field]!r} != {metadata[field]!r}"
            )
        checkpoint.setdefault(field, metadata[field])


def _load_checkpoint(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a mapping")
    if "model_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing required field: model_dict")

    metadata_path = os.path.join(
        os.path.dirname(os.path.abspath(path)), _METADATA_FILENAME
    )
    if os.path.isfile(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        _merge_sidecar_metadata(checkpoint, metadata, metadata_path)

    required = _CHECKPOINT_FIELDS | {"model_dict"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        detail = (
            f" Add {_METADATA_FILENAME} beside weights-only checkpoints."
            if not os.path.isfile(metadata_path)
            else ""
        )
        raise ValueError(
            f"Checkpoint is missing required fields: {missing}.{detail}"
        )
    return checkpoint


def _is_messi_style(state_dict):
    keys = state_dict.keys()
    return (
        any(key.startswith("featurizer.vit.") for key in keys)
        and any(key.startswith("moe_head.experts.") for key in keys)
        and any(key.startswith("moe_head.classifier.") for key in keys)
    )


def _build_model(checkpoint):
    ckpt_args = checkpoint["args"]
    state_dict = checkpoint["model_dict"]
    algorithm_name = ckpt_args.get("algorithm", "unknown")

    if _is_messi_style(state_dict):
        model = _MESSIInferenceModel(
            checkpoint["model_input_shape"],
            checkpoint["model_num_classes"],
            checkpoint["model_hparams"],
        )
        inference_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(("featurizer.", "moe_head."))
        }
        # Strict loading is important: a partial model could produce plausible
        # but invalid accuracy numbers.
        model.load_state_dict(inference_state, strict=True)
        ignored = sorted(set(state_dict) - set(inference_state))
        return model, algorithm_name, ignored

    # Keep the heavyweight algorithm/Tutel imports out of the common
    # MESSI-compatible path.
    import domainbed.tutel_patch  # noqa: F401
    from domainbed import algorithms

    algorithm_class = algorithms.get_algorithm_class(algorithm_name)
    hparams = dict(checkpoint["model_hparams"])
    hparams["pretrained"] = False
    model = algorithm_class(
        checkpoint["model_input_shape"],
        checkpoint["model_num_classes"],
        checkpoint["model_num_domains"],
        hparams,
    )
    model.load_state_dict(state_dict, strict=True)
    return model, algorithm_name, []


def _load_dataset(checkpoint, dir_dataset, env):
    dataset_name = checkpoint["args"].get("dataset")
    if not dataset_name:
        raise ValueError("Checkpoint args do not contain a dataset name")

    dataset_class = datasets.get_dataset_class(dataset_name)
    dataset_hparams = dict(checkpoint["model_hparams"])
    # Data augmentation is a training concern. Some inference-only checkpoints
    # intentionally store only architecture hparams, while DomainBed's image
    # dataset constructors still expect this key to exist.
    dataset_hparams.setdefault("data_augmentation", False)
    supplied_path = os.path.abspath(os.path.expanduser(dir_dataset))
    if not os.path.isdir(supplied_path):
        raise NotADirectoryError(f"Dataset directory not found: {supplied_path}")

    # DomainBed image datasets traditionally receive the parent data directory,
    # while this eval CLI also accepts the more natural direct dataset path.
    candidates = [supplied_path, os.path.dirname(supplied_path)]
    errors = []
    dataset = None
    for root in dict.fromkeys(candidates):
        try:
            dataset = dataset_class(
                root,
                [env],
                dataset_hparams,
            )
            break
        except (FileNotFoundError, NotADirectoryError) as exc:
            errors.append(f"{root}: {exc}")

    if dataset is None:
        detail = "\n  ".join(errors)
        raise FileNotFoundError(
            f"Could not load {dataset_name} from {supplied_path}.\n  {detail}"
        )
    if env < 0 or env >= len(dataset):
        raise ValueError(
            f"--env must be in [0, {len(dataset) - 1}] for {dataset_name}, "
            f"got {env}"
        )
    if dataset.num_classes != checkpoint["model_num_classes"]:
        raise ValueError(
            "Dataset/checkpoint class-count mismatch: "
            f"{dataset.num_classes} != {checkpoint['model_num_classes']}"
        )
    return dataset_name, dataset


def _evaluate(model, env_dataset, device):
    loader = DataLoader(
        env_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    model.to(device)
    model.eval()

    correct = 0
    loss_sum = 0.0
    count = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model.predict(x)
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            correct += logits.argmax(dim=1).eq(y).sum().item()
            count += y.numel()

    if count == 0:
        raise ValueError("Selected environment contains no samples")
    return correct / count, loss_sum / count, count


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on one complete dataset environment"
    )
    parser.add_argument("--env", type=int, required=True)
    parser.add_argument("--dir_dataset", required=True)
    parser.add_argument("--dir_ckpt", required=True)
    args = parser.parse_args()

    checkpoint = _load_checkpoint(args.dir_ckpt)
    model, algorithm_name, ignored_keys = _build_model(checkpoint)
    dataset_name, dataset = _load_dataset(
        checkpoint, args.dir_dataset, args.env
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy, loss, sample_count = _evaluate(
        model, dataset[args.env], device
    )

    env_label = (
        dataset.ENVIRONMENTS[args.env]
        if dataset.ENVIRONMENTS and args.env < len(dataset.ENVIRONMENTS)
        else str(args.env)
    )
    env_path = getattr(dataset[args.env], "root", "n/a")
    print(f"checkpoint : {os.path.abspath(args.dir_ckpt)}")
    print(f"algorithm  : {algorithm_name}")
    print(f"dataset    : {dataset_name}")
    print(f"environment: {args.env} ({env_label})")
    print(f"env_path   : {env_path}")
    print(f"device     : {device}")
    print(f"samples    : {sample_count}")
    print(f"loss       : {loss:.6f}")
    print(f"accuracy   : {accuracy:.6f} ({accuracy * 100:.2f}%)")
    if ignored_keys:
        print(f"note       : ignored non-inference state: {', '.join(ignored_keys)}")


if __name__ == "__main__":
    main()

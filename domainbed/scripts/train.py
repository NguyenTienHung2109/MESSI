# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import PIL
import numpy as np
import torch
import torch.utils.data
import torchvision

# Patch Tutel CUDA kernels with pure-PyTorch fallbacks BEFORE importing
# vision_transformer / algorithms (which import tutel at module level).
# Required when Tutel's compiled extensions don't support the current GPU.
import domainbed.tutel_patch  # noqa: F401

from domainbed import algorithms
from domainbed import datasets
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import (
    InfiniteDataLoader, FastDataLoader, LocationGroupedBatchSampler)
from domainbed.lib.sweep_logger import SweepLogger


IWILDCAM_DATASETS = {"WILDSIWildCam", "WILDSIWildCamERM"}


def _default_eval_metric(dataset_name):
    return "f1" if dataset_name in IWILDCAM_DATASETS else "acc"


def _parameter_counts(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default='./domainbed/data')
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--drop_out', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers for training. Default: dataset.N_WORKERS.')
    parser.add_argument('--eval_num_workers', type=int, default=None,
                        help='DataLoader workers for evaluation. Default: --num_workers.')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='Evaluation batch size. Default: 64.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--sweep_log_dir', type=str, default=None,
                        help='If set, write structured sweep logs to this directory.')
    parser.add_argument('--sweep_run_id', type=str, default=None,
                        help='Human-readable run ID for sweep logging (e.g. gmoe_N6_K2_PR00).')
    parser.add_argument('--max_samples_per_env', type=int, default=None,
                        help='Truncate each TRAINING env in_split to this many samples '
                             '(after holdout split). Used for K-sweep experiments to keep '
                             'total training data fixed when varying #source domains. '
                             'Test envs (those in --test_envs) are not truncated.')
    parser.add_argument('--stratified_subsample',
                        type=lambda s: s.lower() not in ('false', '0', 'no'),
                        default=True,
                        help='When --max_samples_per_env is set, subsample with class '
                             'proportions preserved (largest-remainder per-class quota) '
                             'instead of taking the first N. Pass --stratified_subsample false '
                             'to reproduce the old sequential behavior.')
    parser.add_argument('--source_envs', type=int, nargs='+', default=None,
                        help='Whitelist: env indices to use as TRAINING sources. '
                             'If set, only these envs are used for training (other non-test '
                             'envs are ignored). Useful for datasets with many envs (e.g. iWildCam '
                             '323 locations) where listing every excluded env in --test_envs '
                             'would be unwieldy.')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to a train.py checkpoint, e.g. model.pkl, to resume model weights from.')
    parser.add_argument('--resume_step', type=int, default=None,
                        help='Global step to resume at. If unset, infer from output_dir/results.jsonl '
                             'or the checkpoint args.')
    args = parser.parse_args()

    start_step = 0
    algorithm_dict = None
    resume_checkpoint = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    if args.batch_size is not None:
        hparams['batch_size'] = args.batch_size
    if args.drop_out is not None:
        hparams['drop_out'] = args.drop_out
    if args.lr is not None:
        hparams['lr'] = args.lr
    if args.weight_decay is not None:
        hparams['weight_decay'] = args.weight_decay

    def _infer_resume_step(checkpoint, output_dir):
        if args.resume_step is not None:
            return args.resume_step

        results_path = os.path.join(output_dir, 'results.jsonl')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                for line in f:
                    if line.strip():
                        last_line = line
                if 'last_line' in locals():
                    return int(json.loads(last_line)['step']) + 1

        ckpt_args = checkpoint.get('args', {}) if isinstance(checkpoint, dict) else {}
        if 'steps' in ckpt_args and ckpt_args['steps'] is not None:
            return int(ckpt_args['steps'])
        if 'step' in checkpoint:
            return int(checkpoint['step'])

        raise ValueError(
            'Could not infer resume step. Pass --resume_step explicitly.'
        )

    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f'--resume_from checkpoint not found: {args.resume_from}')
        resume_checkpoint = torch.load(args.resume_from, map_location='cpu')
        if not isinstance(resume_checkpoint, dict) or 'model_dict' not in resume_checkpoint:
            raise ValueError(
                '--resume_from must point to a train.py checkpoint containing model_dict'
            )
        algorithm_dict = resume_checkpoint['model_dict']
        start_step = _infer_resume_step(resume_checkpoint, args.output_dir)
        if args.steps is not None and start_step >= args.steps:
            raise ValueError(
                f'Resume step {start_step} is >= requested --steps {args.steps}; '
                'increase --steps to continue training.'
            )
        print(f'Resuming from {args.resume_from} at step {start_step}')

    default_eval_metric = _default_eval_metric(args.dataset)
    eval_metric = hparams.get('eval_metric', default_eval_metric)
    if eval_metric not in ('acc', 'f1', 'recall'):
        raise ValueError(
            "hparams['eval_metric'] must be one of: 'acc', 'f1', 'recall'"
        )
    hparams['eval_metric'] = eval_metric
    print('\teval_metric: {}'.format(eval_metric))

    # print('HParams:')
    # for k, v in sorted(hparams.items()):
    #     print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError

    train_num_workers = dataset.N_WORKERS if args.num_workers is None else args.num_workers
    # Eval falls back to dataset.EVAL_N_WORKERS (if defined) before train_num_workers,
    # so datasets like WILDSIWildCam that force N_WORKERS=0 for training (243
    # forked train_loaders OOM) can still parallelise eval (only ~10 eval loaders).
    if args.eval_num_workers is not None:
        eval_num_workers = args.eval_num_workers
    elif hasattr(dataset, 'EVAL_N_WORKERS'):
        eval_num_workers = dataset.EVAL_N_WORKERS
    else:
        eval_num_workers = train_num_workers
    if train_num_workers < 0:
        raise ValueError("--num_workers must be >= 0")
    if eval_num_workers < 0:
        raise ValueError("--eval_num_workers must be >= 0")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval_batch_size must be > 0")
    print("Loader config:")
    print("\ttrain_num_workers: {}".format(train_num_workers))
    print("\teval_num_workers: {}".format(eval_num_workers))
    print("\teval_batch_size: {}".format(args.eval_batch_size))


    if 'Debug' not in args.dataset:
        # Rely on `wandb login` / ~/.netrc / $WANDB_API_KEY for credentials.
        _NEVER_SHOW = {
            'data_augmentation', 'resnet18', 'resnet_dropout',
            'nonlinear_classifier', 'class_balanced',
            'val_augment', 'freeze_bn', 'pretrained', 'optimizer',
        }
        relevant_keys = {k for k in hparams if k not in _NEVER_SHOW}

        test_env_names = [dataset.ENVIRONMENTS[i] for i in args.test_envs]
        test_env_str = '+'.join(test_env_names)
        hparam_str = '_'.join(f'{k}={hparams[k]}' for k in sorted(relevant_keys))
        run_name = f'{args.algorithm}_{args.dataset}_test[{test_env_str}]_{hparam_str}'
        if len(run_name) > 128:
            run_name = run_name[:125] + '...'

        wandb.init(
            project=os.environ.get('WANDB_PROJECT', 'PACS_sweep'),
            entity=os.environ.get('WANDB_ENTITY', 'hunghn2003'),
            name=run_name,
            config={
                'dataset': args.dataset,
                'algorithm': args.algorithm,
                'test_envs': args.test_envs,
                'test_env_names': test_env_names,
                'seed': args.seed,
                'trial_seed': args.trial_seed,
                'hparams_seed': args.hparams_seed,
                **{f'hp/{k}': hparams[k] for k in sorted(relevant_keys)},
            },
            settings=wandb.Settings(start_method='thread'),
        )


    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env, int(len(env) * args.holdout_fraction), misc.seed_hash(args.trial_seed, env_i))
        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_, int(len(in_) * args.uda_holdout_fraction), misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    # K-sweep support: truncate each TRAINING env's in_split to max_samples_per_env.
    # Non-training envs (test envs, or envs not in --source_envs whitelist) are
    # left full so evaluation metrics are unbiased.
    if args.max_samples_per_env is not None:
        def _is_training_env(i):
            if args.source_envs is not None:
                return i in args.source_envs
            return i not in args.test_envs
        truncated_in_splits = []
        for env_i, (in_, in_weights) in enumerate(in_splits):
            if not _is_training_env(env_i):
                truncated_in_splits.append((in_, in_weights))
                continue
            if len(in_) > args.max_samples_per_env:
                if args.stratified_subsample:
                    env_underlying = in_.underlying_dataset
                    if hasattr(env_underlying, 'get_labels'):
                        env_labels = env_underlying.get_labels().numpy()
                        split_labels = env_labels[np.asarray(in_.keys)]
                    else:
                        split_labels = np.array(
                            [int(in_[i][1]) for i in range(len(in_))])
                    keep = misc.stratified_subsample_indices(
                        split_labels, args.max_samples_per_env,
                        seed=misc.seed_hash(args.trial_seed, env_i, 'subsample'))
                    in_ = torch.utils.data.Subset(in_, keep)
                    cls_counts = np.bincount(split_labels[keep])
                    top1 = 100.0 * cls_counts.max() / cls_counts.sum()
                    print(f'[max_samples_per_env] env{env_i}: size={len(in_)} '
                          f'#classes={int((cls_counts > 0).sum())} '
                          f'top-1={top1:.1f}% (stratified)')
                else:
                    in_ = torch.utils.data.Subset(
                        in_, list(range(args.max_samples_per_env)))
                    print(f'[max_samples_per_env] env{env_i}: '
                          f'size={len(in_)} (sequential)')
            else:
                print(f'[max_samples_per_env] env{env_i}: '
                      f'size={len(in_)} (no truncation)')
            truncated_in_splits.append((in_, in_weights))
        in_splits = truncated_in_splits

    # Decide which envs are training:
    #   - If --source_envs is set: whitelist (only listed envs are training).
    #   - Else: blacklist (any env not in --test_envs is training).
    def _is_training_env(i):
        if args.source_envs is not None:
            return i in args.source_envs
        return i not in args.test_envs

    # Optional location-grouped batch sampler (e.g. WILDSIWildCam): a single
    # train env yields batches whose first K_BATCH samples come from one source
    # camera location, next K_BATCH from another, etc — so domain-invariance
    # losses see num_domains=K source domains per step. Only activates when
    # `dataset.GROUP_SAMPLER_K` is set AND the underlying training env exposes
    # `location_per_idx_in`. Other datasets fall back to per-env loaders.
    group_K = getattr(dataset, 'GROUP_SAMPLER_K', None)
    group_K_BATCH = getattr(dataset, 'GROUP_SAMPLER_K_BATCH', None)
    grouped_train_loader_idx = None  # position in train_loaders that yields grouped batches

    train_loaders = []
    for i, (env, env_weights) in enumerate(in_splits):
        if not _is_training_env(i):
            continue
        # `env` is a _SplitDataset wrapping a base dataset. Walk to the base
        # to look up location metadata.
        base = getattr(env, 'underlying_dataset', env)
        loc_per_idx_in = getattr(base, 'location_per_idx_in', None)
        if (group_K is not None and group_K_BATCH is not None
                and loc_per_idx_in is not None):
            # Map _SplitDataset.keys (== pre_split['in']) → base in indices,
            # then pick out their locations for the grouped sampler.
            split_keys = np.asarray(env.keys, dtype=np.int64)
            loc_per_split_idx = loc_per_idx_in[
                np.searchsorted(np.asarray(base.pre_split['in']), split_keys)]
            sampler = LocationGroupedBatchSampler(
                loc_per_split_idx, K=group_K,
                bs_per_group=group_K_BATCH, seed=args.seed)
            train_loaders.append(InfiniteDataLoader(
                dataset=env, weights=None,
                batch_size=group_K * group_K_BATCH,
                num_workers=train_num_workers,
                batch_sampler=sampler))
            grouped_train_loader_idx = len(train_loaders) - 1
            print(f"[group-sampler] env_{i}: K={group_K} locations × "
                  f"{group_K_BATCH} samples = batch_size {group_K*group_K_BATCH} "
                  f"({len(sampler.qualified_locations)} qualified locations)",
                  flush=True)
        else:
            train_loaders.append(InfiniteDataLoader(
                dataset=env,
                weights=env_weights,
                batch_size=hparams['batch_size'],
                num_workers=train_num_workers))

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=train_num_workers)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    # Decide which envs to evaluate. With many-domain datasets (e.g. iWildCam, 323
    # envs), evaluating ALL envs spawns thousands of worker processes (323 × 2 ×
    # N_WORKERS) and crashes the machine. We filter to relevant envs only.
    eval_loader_specs = []
    if args.source_envs is not None:
        eval_env_indices = sorted(set(args.source_envs) | set(args.test_envs))
    elif len(in_splits) > 50:
        # Heuristic: dataset has many envs but no whitelist → eval test + first
        # 5 training envs as a sample (avoids OOM on iWildCam-like cases).
        train_sample = [i for i in range(len(in_splits)) if i not in args.test_envs][:5]
        eval_env_indices = sorted(set(args.test_envs) | set(train_sample))
        print(f'[eval-filter] dataset has {len(in_splits)} envs > 50; '
              f'evaluating only test_envs + first 5 training envs: {eval_env_indices}')
    else:
        eval_env_indices = list(range(len(in_splits)))

    # in_splits
    for i in eval_env_indices:
        eval_loader_specs.append((f'env{i}_in', in_splits[i][0], None))
    # out_splits
    for i in eval_env_indices:
        eval_loader_specs.append((f'env{i}_out', out_splits[i][0], None))

    # uda_splits (only test envs typically have these)
    for i, (env, _) in enumerate(uda_splits):
        eval_loader_specs.append((f'env{i}_uda', env, None))

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    # When a group sampler emits K location-groups per batch, the algorithm
    # sees `minibatches` of length K (after train.py reshape). `num_domains`
    # must match so domain-invariance losses (e.g. loss_inv_MMD) iterate over
    # exactly the K positional domain ids. Otherwise default = #training envs.
    n_train_envs = len(dataset) - len(args.test_envs)
    if grouped_train_loader_idx is not None:
        num_domains_for_algo = group_K
    else:
        num_domains_for_algo = n_train_envs
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                num_domains_for_algo, hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    model_total_params, model_trainable_params = _parameter_counts(algorithm)
    print("Model parameters:")
    print("\ttotal: {}".format(model_total_params))
    print("\ttrainable: {}".format(model_trainable_params))
    if wandb.run:
        wandb.config.update({
            'model/total_params': model_total_params,
            'model/trainable_params': model_trainable_params,
        }, allow_val_change=True)

    # Sweep logger — activated only when --sweep_log_dir is provided
    sweep_logger = None
    if args.sweep_log_dir:
        run_id = args.sweep_run_id or os.path.basename(args.sweep_log_dir)
        sweep_logger = SweepLogger(
            log_dir=args.sweep_log_dir,
            run_id=run_id,
            hparams=hparams,
            args=args,
            model=algorithm.model if hasattr(algorithm, 'model') else None,
        )
        sweep_logger.write_hparams_json()
        sweep_logger.start_run_meta()

    def _train_minibatches_iter():
        iters = [iter(l) for l in train_loaders]
        if grouped_train_loader_idx is not None:
            # The grouped loader yields one batch of (K * K_BATCH, ...) per
            # step; reshape into K minibatches so the algorithm sees K source
            # domains via positional domain_ids.
            gi = grouped_train_loader_idx
            while True:
                x, y = next(iters[gi])
                x = x.view(group_K, group_K_BATCH, *x.shape[1:])
                y = y.view(group_K, group_K_BATCH)
                yield [(x[i], y[i]) for i in range(group_K)]
        else:
            while True:
                yield [next(it) for it in iters]

    train_minibatches_iterator = _train_minibatches_iter()
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env) / hparams['batch_size'] for env, _ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ


    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_total_params": model_total_params,
            "model_trainable_params": model_trainable_params,
            "model_hparams": hparams,
            "step": start_step,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        # non_blocking=True pairs with pin_memory=True in DataLoader for async H2D transfer.
        batch_fetch_start_time = time.time()
        minibatches = next(train_minibatches_iterator)
        batch_fetch_time = time.time() - batch_fetch_start_time

        batch_transfer_start_time = time.time()
        minibatches_device = [(x.to(device, non_blocking=True), y.to(device, non_blocking=True))
                              for x, y in minibatches]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device, non_blocking=True) for x, _ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        batch_transfer_time = time.time() - batch_transfer_start_time

        update_start_time = time.time()
        step_vals = algorithm.update(minibatches_device)
        update_time = time.time() - update_start_time
        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        checkpoint_vals['batch_fetch_time'].append(batch_fetch_time)
        checkpoint_vals['batch_transfer_time'].append(batch_transfer_time)
        checkpoint_vals['update_time'].append(update_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            eval_start_time = time.time()
            for name, eval_dataset, weights in eval_loader_specs:
                loader = FastDataLoader(
                    dataset=eval_dataset,
                    batch_size=args.eval_batch_size,
                    num_workers=eval_num_workers)
                m = misc.accuracy_metrics(algorithm, loader, weights, device)
                results[name + '_' + eval_metric] = m[eval_metric]
                del loader
            results['eval_time'] = time.time() - eval_start_time

            results['algorithm'] = args.algorithm
            results['dataset'] = args.dataset
            results['eval_metric'] = eval_metric
            results['model_total_params'] = model_total_params
            results['model_trainable_params'] = model_trainable_params
            results['test_envs'] = args.test_envs
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024. * 1024. * 1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                           colwidth=12)

            if wandb.run:
                wandb.log(results)
            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            if sweep_logger is not None:
                avg_step_vals = {k: float(np.mean(v)) for k, v in checkpoint_vals.items()}
                sweep_logger.append_train_log(step, avg_step_vals, results.get('mem_gb'))
                sweep_logger.append_eval_log(step, results)
                sweep_logger.append_expert_stats(step, algorithm)

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    save_checkpoint('model.pkl')

    if sweep_logger is not None:
        sweep_logger.write_final_summary()
        sweep_logger.close_run_meta(exit_code=0)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np

from domainbed.lib import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST', 'ColoredMNIST_E', 'ColoredMNIST_K']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        # assert (name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    # Backbone selector — 'resnet50' (default) or a ViT factory name like
    # 'deit_small_patch16_224'. Read by networks.Featurizer().
    _hparam('model', 'resnet50', lambda r: 'resnet50')
    _hparam('image_size', 224, lambda r: 224)
    _hparam('resolution', None, lambda r: None)
    _hparam('class_balanced', False, lambda r: False)
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))
    hparams["optimizer"] = ("adam", "adam")

    hparams["val_augment"] = (False, False)  # augmentation for in-domain validation set
    hparams["freeze_bn"] = (True, True)
    hparams["pretrained"] = (True, True)  # only for ResNet
    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.

    if algorithm in ['DANN', 'CDANN']:
        _hparam('lambda', 1.0, lambda r: 10 ** r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10 ** r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2 ** r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10 ** r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    elif algorithm == 'Fish':
        _hparam('meta_lr', 0.5, lambda r: r.choice([0.05, 0.1, 0.5]))

    elif algorithm == "RSC":
        _hparam('rsc_f_drop_factor', 1 / 3, lambda r: r.uniform(0, 0.5))
        _hparam('rsc_b_drop_factor', 1 / 3, lambda r: r.uniform(0, 0.5))

    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10 ** r.uniform(-2, 1))

    elif algorithm == "IRM" or algorithm == 'IRM_IN21k':
        _hparam('irm_lambda', 1e2, lambda r: 10 ** r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10 ** r.uniform(0, 4)))

    elif algorithm == "Mixup":
        _hparam('mixup_alpha', 0.2, lambda r: 10 ** r.uniform(-1, -1))

    elif algorithm == "GroupDRO":
        _hparam('groupdro_eta', 1e-2, lambda r: 10 ** r.uniform(-3, -1))

    elif algorithm == "MMD" or algorithm == "CORAL":
        _hparam('mmd_gamma', 1., lambda r: 10 ** r.uniform(-1, 1))

    elif algorithm == "MLDG":
        _hparam('mldg_beta', 1., lambda r: 10 ** r.uniform(-1, 1))

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    elif algorithm == "VREx":
        _hparam('vrex_lambda', 1e1, lambda r: 10 ** r.uniform(-1, 5))
        _hparam('vrex_penalty_anneal_iters', 500,
                lambda r: int(10 ** r.uniform(0, 4)))

    elif algorithm == "SD":
        _hparam('sd_reg', 0.1, lambda r: 10 ** r.uniform(-5, -1))

    elif algorithm == "ANDMask":
        _hparam('tau', 1, lambda r: r.uniform(0.5, 1.))

    elif algorithm == "IGA":
        _hparam('penalty', 1000, lambda r: 10 ** r.uniform(1, 5))

    elif algorithm == "SANDMask":
        _hparam('tau', 1.0, lambda r: r.uniform(0.0, 1.))
        _hparam('k', 1e+1, lambda r: 10 ** r.uniform(-3, 5))

    elif algorithm == "Fishr":
        _hparam('lambda', 1000., lambda r: 10 ** r.uniform(1., 4.))
        _hparam('penalty_anneal_iters', 1500, lambda r: int(r.uniform(0., 5000.)))
        _hparam('ema', 0.95, lambda r: r.uniform(0.90, 0.99))

    elif algorithm == "TRM":
        _hparam('cos_lambda', 1e-4, lambda r: 10 ** r.uniform(-5, 0))
        _hparam('iters', 200, lambda r: int(10 ** r.uniform(0, 4)))
        _hparam('groupdro_eta', 1e-2, lambda r: 10 ** r.uniform(-3, -1))

    elif algorithm == "IB_ERM":
        _hparam('ib_lambda', 1e2, lambda r: 10 ** r.uniform(-1, 5))
        _hparam('ib_penalty_anneal_iters', 500,
                lambda r: int(10 ** r.uniform(0, 4)))

    elif algorithm == "IB_IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10 ** r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10 ** r.uniform(0, 4)))
        _hparam('ib_lambda', 1e2, lambda r: 10 ** r.uniform(-1, 5))
        _hparam('ib_penalty_anneal_iters', 500,
                lambda r: int(10 ** r.uniform(0, 4)))

    elif algorithm == "CAD" or algorithm == "CondCAD":
        _hparam('lmbda', 1e-1, lambda r: r.choice([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]))
        _hparam('temperature', 0.1, lambda r: r.choice([0.05, 0.1]))
        _hparam('is_normalized', False, lambda r: False)
        _hparam('is_project', False, lambda r: False)
        _hparam('is_flipped', True, lambda r: True)

    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.

    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10 ** r.uniform(-4.5, -2.5))
    else:
        _hparam('lr', 3e-5, lambda r: 10 ** r.uniform(-5, -3.5))

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)
    else:
        _hparam('weight_decay', 0., lambda r: 10 ** r.uniform(-6, -2))

    if dataset in SMALL_IMAGES:
        _hparam('batch_size', 64, lambda r: int(2 ** r.uniform(3, 9)))
    elif algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2 ** r.uniform(3, 5)))
    else:
        _hparam('batch_size', 32, lambda r: int(2 ** r.uniform(3, 5.5)))

    if dataset == "ColoredMNIST_E":
        _hparam('num_environments', 8,
                lambda r: int(r.choice([2, 4, 8, 16, 32, 64])))

    if dataset == "ColoredMNIST_K":
        _hparam('num_source_domains', 9, lambda r: 9)
        _hparam('test_size', 10000, lambda r: 10000)
        _hparam('patch_size', 4, lambda r: int(r.choice([4, 7])))

    if dataset == "MetaShift_K":
        _hparam('class_set', 'cat_dog_horse_elephant_bird', lambda r: 'cat_dog_horse_elephant_bird')
        _hparam('K', 8, lambda r: int(r.choice([4, 6, 8])))
        _hparam('split_seed', 0, lambda r: 0)
        _hparam('total_per_class', 240, lambda r: 240)
        _hparam('single_test', True, lambda r: True)

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_g', 1e-3, lambda r: 10 ** r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_g', 5e-5, lambda r: 10 ** r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_d', 1e-3, lambda r: 10 ** r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_d', 5e-5, lambda r: 10 ** r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('weight_decay_g', 0., lambda r: 0.)
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('weight_decay_g', 0., lambda r: 10 ** r.uniform(-6, -2))

    if algorithm in ('GMOE', 'GMoEOMoE'):
        _hparam('num_experts',        6,                        lambda r: 6)
        _hparam('gate_k',             1,                        lambda r: 1)
        _hparam('mlp_ratio',          4.0,                      lambda r: 4.0)
        _hparam('expert_depth',       2,                        lambda r: 2)
        _hparam('expert_prune_ratio', 0.0,                      lambda r: 0.0)
        _hparam('model',              'deit_small_patch16_224', lambda r: 'deit_small_patch16_224')
        _hparam('use_omoe',           False,                    lambda r: False)
        _hparam('use_balance_loss',   False,                    lambda r: False)

        if dataset == 'VLCS':
            _hparam('lr', 3e-5, lambda r: 10 ** r.uniform(-4.5, -2.5))
            _hparam('resnet_dropout', 0.5, lambda r: r.choice([0., 0.1, 0.5]))
            _hparam('weight_decay', 1e-6, lambda r: 0.)

        if dataset == 'PACS':
            _hparam('lr', 3e-5, lambda r: 10 ** r.uniform(-4.5, -2.5))
            _hparam('resnet_dropout', 0.0, lambda r: r.choice([0., 0.1, 0.5]))
            _hparam('weight_decay', 1e-6, lambda r: 0.)

        if dataset == 'OfficeHome':
            _hparam('lr', 1e-5, lambda r: 10 ** r.uniform(-4.5, -2.5))
            _hparam('resnet_dropout', 0.1, lambda r: r.choice([0., 0.1, 0.5]))
            _hparam('weight_decay', 1e-6, lambda r: 0.)

        if dataset == 'TerraIncognita':
            _hparam('lr', 5e-5, lambda r: 10 ** r.uniform(-4.5, -2.5))
            _hparam('resnet_dropout', 0.0, lambda r: r.choice([0., 0.1, 0.5]))
            _hparam('weight_decay', 1e-4, lambda r: 0.)

        if dataset == 'DomainNet':
            _hparam('lr', 5e-5, lambda r: 10 ** r.uniform(-4.5, -2.5))
            _hparam('resnet_dropout', 0.1, lambda r: r.choice([0., 0.1, 0.5]))
            _hparam('weight_decay', 0, lambda r: 0.)

        if dataset == 'CUB':
            _hparam('lr', 5e-5, lambda r: 10 ** r.uniform(-4.5, -2.5))
            _hparam('resnet_dropout', 0.1, lambda r: r.choice([0., 0.1, 0.5]))
            _hparam('weight_decay', 0, lambda r: 0.)

        # New losses from Advanced MoE (NeurIPS 2025); default 0 preserves baseline
        _hparam('ortho_loss_weight',    0.0, lambda r: r.choice([0., 1e-4, 1e-3]))
        _hparam('variance_loss_weight', 0.0, lambda r: r.choice([0., 1e-4, 1e-3]))
        _hparam('moe_top_k',            1,   lambda r: r.choice([1, 2]))
        _hparam('num_experts',          6,   lambda r: r.choice([6, 8, 12, 16]))


    if algorithm == 'GMOE_InvOT':
        _hparam('model',              'deit_small_patch16_224', lambda r: 'deit_small_patch16_224')
        _hparam('moe_dim',            'auto',                   lambda r: 'auto')
        _hparam('num_experts',        6,                        lambda r: 6)
        _hparam('lambda_inv',         0.1,                      lambda r: 0.1)
        _hparam('lambda_sp',          0.01,                     lambda r: 0.01)
        _hparam('lambda_bal',         0.01,                     lambda r: 0.01)
        _hparam('lambda_div',         0.01,                     lambda r: 0.01)
        _hparam('alpha',              4.0,                      lambda r: 4.0)
        _hparam('ot_epsilon',         0.1,                      lambda r: 0.1)
        _hparam('sinkhorn_iters',     50,                       lambda r: 50)

    if algorithm == 'MESSI_SubsetIRM':
        # Explicit opt-in surface. Every new coefficient is zero by default;
        # the legacy MESSI coefficients retain their established defaults.
        _hparam('model',              'deit_small_patch16_224', lambda r: 'deit_small_patch16_224')
        _hparam('moe_dim',            'auto',                   lambda r: 'auto')
        _hparam('num_experts',        6,                        lambda r: 6)
        _hparam('lambda_inv',         0.1,                      lambda r: 0.1)
        _hparam('lambda_sp',          0.01,                     lambda r: 0.01)
        _hparam('lambda_bal',         0.01,                     lambda r: 0.01)
        _hparam('lambda_div',         0.01,                     lambda r: 0.01)
        _hparam('alpha',              4.0,                      lambda r: 4.0)
        _hparam('subset_irm_enabled', False,                    lambda r: False)
        _hparam('subset_irm_prediction_mode', 'shared_feature_mix', lambda r: 'shared_feature_mix')
        _hparam('subset_irm_router_topk', None,                  lambda r: None)
        _hparam('subset_irm_topk_warmup_steps', 0,              lambda r: 0)
        _hparam('subset_irm_use_expert_heads', False,           lambda r: False)
        _hparam('subset_irm_encoder_skip_enabled', False,       lambda r: False)
        _hparam('subset_irm_encoder_skip_scale', 1.0,           lambda r: 1.0)
        _hparam('subset_irm_lambda_expert', 0.0,                 lambda r: 0.0)
        _hparam('subset_irm_lambda_route', 0.0,                  lambda r: 0.0)
        _hparam('subset_irm_lambda_sirm', 0.0,                   lambda r: 0.0)
        _hparam('subset_irm_responsibility_detach', True,       lambda r: True)
        _hparam('subset_irm_assignment_mode', 'risk_ema',       lambda r: 'risk_ema')
        _hparam('subset_irm_assignment_topr', 2,                lambda r: 2)
        _hparam('subset_irm_assignment_temperature', 1.0,       lambda r: 1.0)
        _hparam('subset_irm_assignment_ema', 0.9,               lambda r: 0.9)
        _hparam('subset_irm_probe_interval', 100,                lambda r: 100)
        _hparam('subset_irm_min_effective_samples', 2,          lambda r: 2)
        _hparam('subset_irm_sirm_anneal_steps', 0,              lambda r: 0)

    if algorithm == 'rMESSI_InvOT':
        _hparam('model',              'deit_small_patch16_224', lambda r: 'deit_small_patch16_224')
        _hparam('moe_dim',            'auto',                   lambda r: 'auto')
        _hparam('num_experts',        6,                        lambda r: 6)
        _hparam('lambda_inv',         0.1,                      lambda r: 0.1)
        _hparam('lambda_sp',          0.01,                     lambda r: 0.01)
        _hparam('lambda_bal',         0.01,                     lambda r: 0.01)
        _hparam('lambda_div',         0.01,                     lambda r: 0.01)
        _hparam('alpha',              4.0,                      lambda r: 4.0)
        _hparam('ot_epsilon',         0.1,                      lambda r: 0.1)
        _hparam('sinkhorn_iters',     50,                       lambda r: 50)
        _hparam('reliability_mode',   'none',                   lambda r: 'none')
        _hparam('r_n_min',            2,                        lambda r: int(r.choice([1, 2, 3])))
        _hparam('r_tau',              4.0,                      lambda r: float(r.choice([2.0, 4.0, 8.0])))
        _hparam('r_ema_beta',         0.9,                      lambda r: float(r.choice([0.8, 0.9, 0.95])))
        _hparam('r_emin',             4.0,                      lambda r: float(r.choice([2.0, 4.0, 8.0])))
        _hparam('r_detach',           True,                     lambda r: True)
        _hparam('reliable_log_detail', 'compact',                lambda r: 'compact')
        _hparam('r_conf_floor',       0.2,                      lambda r: 0.2)
        _hparam('r_conf_warmup_steps', 5000,                    lambda r: 5000)
        _hparam('r_stab_sigma',       0.1,                      lambda r: 0.1)

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}

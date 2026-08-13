# Subset-IRM pre-implementation repository audit

This note records the worktree's MESSI implementation before the Subset-IRM
extension. It treats the current worktree (including its pre-existing changes)
as authoritative.

## Forward path

- `domainbed.algorithms.MESSI` is an inference-compatible subclass of
  `GMOE_InvMMD`, whose common model is `GMoEVariantBase`.
- The shared image backbone is `DeiTFeaturizer` (DeiT-Small by default). Its
  CLS feature is passed through the MoE head's `input_proj`, which is an
  identity whenever `moe_dim` equals the backbone width.
- `domainbed.deit_transformer.ExplicitMoEHead` owns the six independent expert
  MLPs in `experts`, the dense linear `router`, and the single shared linear
  `classifier`.
- For a backbone feature `z`, the head computes all expert features
  `h_m = E_m(input_proj(z))`, dense routing probabilities
  `pi = softmax(router(input_proj(z)))`, the feature mixture
  `h = sum_m pi_m h_m`, and `logits = classifier(h)`.
- This path is dense: every expert is evaluated for every sample. The routing
  probabilities multiply expert outputs, so a sample normally sends gradients
  to every expert, the router, the input projection, and the backbone. There is
  no per-sample Top-k masking in this explicit MESSI head.

## Training objective

`GMOE_InvMMD.update` concatenates one minibatch from every source environment,
constructs local source-domain IDs from minibatch position, and computes:

`L = L_cls + lambda_inv L_ssi + lambda_sp L_sp + lambda_bal L_bal + lambda_div L_div`.

- `L_cls` is cross entropy on the shared classifier's feature-mixture logits.
- The MESSI `L_ssi` term is named `loss_inv` in code and is implemented by
  `loss_inv_MMD`: routing-weighted, class-conditional, expert-wise MMD over
  pairs of source environments. Pair responsibilities are derived from mean
  router mass and detached before weighting the discrepancy.
- `L_sp` (`loss_sparse`) is mean routing entropy.
- `L_bal` (`loss_balance`) is squared deviation of mean expert load from the
  uniform distribution.
- `L_div` (`loss_diversity`) sums squared cross-expert batch correlations.
- All terms have independent scalar hparams. The current MESSI defaults are
  inherited from `GMOE_InvMMD`; experiment scripts commonly override them.

`optimizer.zero_grad(); loss.backward(); optimizer.step()` determines the
gradient path. Since all expert features enter both the dense mixture and the
existing auxiliary losses, all expert parameters can receive gradients on a
normal update. Routing weights are detached only inside the MMD pair selector,
not in classification, sparsity, or balancing.

## PACS protocol and selection

- `domainbed.datasets.PACS.ENVIRONMENTS` is `["A", "C", "P", "S"]`, backed by
  the `art_painting`, `cartoon`, `photo`, and `sketch` folders in sorted order.
  Therefore held-out index 1 is **C (cartoon)**.
- PACS inherits the repository default `N_STEPS = 5001` and uses
  `CHECKPOINT_FREQ = 300`. The standard hparam registry supplies the existing
  augmentations, per-environment batch construction, Adam optimizer settings,
  and deterministic DomainBed in/out split.
- The generic `domainbed.scripts.train` evaluates all in/out splits at every
  checkpoint and saves the final model. `IIDAccuracySelectionMethod` can later
  select the step maximizing mean source `env*_out_acc`, but the generic loop
  has already evaluated the target along the way.
- `domainbed.scripts.train_messi_class_only_rebuttal` contains the stricter
  source-only pattern needed here: it evaluates only source out-splits during
  training, saves `best.pkl` on mean source-validation accuracy, reloads that
  checkpoint, and constructs/evaluates target views only afterward.

The Subset-IRM runner must follow the latter selection pattern and must never
place PACS environment 1 in its training minibatches, risk matrices, EMA, or
assignment matrix.

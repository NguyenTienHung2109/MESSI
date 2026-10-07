"""Pair discovery with detached auxiliary router norm bounded by classification."""
from unittest.mock import patch
import torch
import torch.nn.functional as F
from .pair_shared_model import PairSharedGateSIRM, norm


def bounded_auxiliary(classification, weighted_auxiliary, parameters, enabled=True):
    cg = torch.autograd.grad(classification, parameters, retain_graph=True)
    ag = torch.autograd.grad(weighted_auxiliary, parameters, retain_graph=True)
    cn, an = norm(cg), norm(ag)
    scale = cn.new_tensor(1.) if not enabled or an == 0 else (cn / an).clamp(max=1.)
    scale = scale.detach().to(weighted_auxiliary.dtype)
    bounded = scale * weighted_auxiliary
    bn = norm(torch.autograd.grad(bounded, parameters, retain_graph=True))
    assert torch.isfinite(scale) and torch.isfinite(bn)
    if enabled:
        assert bn <= cn * (1 + 1e-5) + 1e-10
    return bounded, scale, cn, an, bn


class RouterCapGateSIRM(PairSharedGateSIRM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cap_recording = False
        self._cap_forward = None

    def forward(self, x):
        f = super().forward(x)
        if self._cap_recording:
            assert self._cap_forward is None
            self._cap_forward = f
        return f

    def update(self, minibatches):
        assert self.config['router_strength_control'] in ['cap_aux_to_classification', 'none']
        y = torch.cat([b for _, b in minibatches]); observed = {}; calls = 0
        original_kl = F.kl_div
        weight = self.config['lambda_router']; assert weight > 0
        def route(input, target, **kwargs):
            nonlocal calls
            calls += 1; assert calls == 1 and kwargs == {'reduction': 'batchmean'}
            raw = original_kl(input, target, **kwargs)
            cls = F.cross_entropy(self._cap_forward['logits'], y)
            _, scale, cn, an, bn = bounded_auxiliary(
                cls, weight * raw, list(self.router.parameters()),
                self.config['router_strength_control'] == 'cap_aux_to_classification')
            # Preserve the uncapped expression/order exactly; avoid multiply/divide by lambda.
            scaled = raw if scale == 1 else scale * raw
            bn = norm(torch.autograd.grad(weight * scaled, list(self.router.parameters()), retain_graph=True))
            if self.config['router_strength_control'] == 'cap_aux_to_classification':
                assert bn <= cn * (1 + 1e-5) + 1e-10
            observed.update(router_strength_control=self.config['router_strength_control'],
                            router_aux_detached_scale=float(scale),
                            router_cls_parameter_grad_norm=float(cn),
                            router_aux_pre_cap_parameter_grad_norm=float(an),
                            router_aux_post_cap_parameter_grad_norm=float(bn),
                            router_aux_raw_loss=float(raw.detach()),
                            router_aux_capped=bool(scale < 1),
                            router_norm_scope='router parameters; classification only reference, excludes SIRM')
            return scaled
        self._cap_recording = True
        try:
            # One process-local KL call; base update and its single optimizer step remain intact.
            with patch('experiments.sirm_discover.model.F.kl_div', route):
                row = super().update(minibatches)
            assert calls == 1
            self.last_diagnostics.update(observed)
            return row
        finally:
            self._cap_recording = False; self._cap_forward = None

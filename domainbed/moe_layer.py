"""A Tutel-compatible MoE layer that exposes routing intermediates."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal



class CosineGate(nn.Module):
    """Compute cosine-routing logits with a learnable temperature."""

    def __init__(self, model_dim: int, num_experts: int,
                 proj_dim: int = 256, init_t: float = 0.5,
                 fp32_gate: bool = True):
        super().__init__()
        self.fp32_gate = fp32_gate
        self.clamp_max = math.log(1.0 / 0.01)

        self.temperature = nn.Parameter(
            torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True
        )
        self.cosine_projector = nn.Linear(model_dim, proj_dim)  # with bias (Tutel default)
        self.sim_matrix = nn.Parameter(torch.randn(proj_dim, num_experts))
        nn.init.normal_(self.sim_matrix, 0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[N, D]`` inputs to ``[N, E]`` routing logits."""
        if self.fp32_gate:
            x = x.float()
            cp = self.cosine_projector.float()
            sm = self.sim_matrix.float()
        else:
            cp = self.cosine_projector
            sm = self.sim_matrix

        logits = torch.matmul(
            F.normalize(cp(x), dim=1),   # [N, proj_dim]
            F.normalize(sm, dim=0),       # [proj_dim, E]
        )                                  # [N, E]
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        return logits * logit_scale        # [N, E]



def _load_importance_loss(scores_no_noise: torch.Tensor,
                           topk_logits_noisy: torch.Tensor,
                           num_experts: int,
                           gate_noise: float) -> torch.Tensor:
    """Compute Tutel-compatible load and importance loss."""
    Impi = scores_no_noise.float().sum(dim=0)            # [E]
    l_imp = Impi.var() / (Impi.mean() ** 2 + 1e-10)

    normal = Normal(
        torch.tensor([0.0], device=scores_no_noise.device),
        torch.tensor([gate_noise / num_experts], device=scores_no_noise.device),
    )
    threshold = topk_logits_noisy[:, -1].view(-1, 1).float()  # [N, 1] — min top-k noisy logit
    diff = scores_no_noise.float() - threshold                  # [N, E]
    prob = normal.cdf(diff)                                     # [N, E]
    Load = prob.sum(dim=0)                                      # [E]
    l_load = Load.var() / (Load.mean() ** 2 + 1e-10)

    return (l_imp + l_load) / 2.0



class TransparentMoELayer(nn.Module):
    """Sparse MoE FFN that exposes auxiliary loss and routing tensors."""

    def __init__(self,
                 model_dim: int,
                 n_experts: int,
                 hidden_size_per_expert: int,
                 top_k: int = 1,
                 capacity_factor: float = 1.5,
                 gate_noise: float = 1.0,
                 proj_dim: int = 256,
                 init_temperature: float = 0.5,
                 fp32_gate: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.gate_noise = gate_noise
        self.model_dim = model_dim
        self.hidden_size = hidden_size_per_expert

        E, H, D = n_experts, hidden_size_per_expert, model_dim

        self.gate = CosineGate(D, E, proj_dim, init_temperature, fp32_gate)

        # Batched expert weights.
        self.fc1_weight = nn.Parameter(torch.empty(E, H, D))
        self.fc1_bias   = nn.Parameter(torch.zeros(E, H))
        self.fc2_weight = nn.Parameter(torch.empty(E, D, H))
        self.fc2_bias   = nn.Parameter(torch.zeros(E, D))

        for e in range(E):
            nn.init.kaiming_uniform_(self.fc1_weight[e], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.fc2_weight[e], a=math.sqrt(5))
            bound1 = 1.0 / math.sqrt(D)
            nn.init.uniform_(self.fc1_bias[e], -bound1, bound1)
            bound2 = 1.0 / math.sqrt(H)
            nn.init.uniform_(self.fc2_bias[e], -bound2, bound2)

        self.act_dropout = nn.Dropout(dropout)

        # Forward-pass outputs.
        self.l_aux: torch.Tensor = None
        self.routing_scores: torch.Tensor = None
        self.expert_outputs: torch.Tensor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route a ``[B, T, D]`` sequence through the experts."""
        orig_dtype = x.dtype
        B, T, D = x.shape
        N = B * T
        E = self.n_experts

        x_flat = x.reshape(N, D)   # [N, D]

        # Compute clean routing scores before noisy top-k selection.
        logits = self.gate(x_flat)
        scores_no_noise = F.softmax(logits, dim=1)
        self.routing_scores = scores_no_noise
        if self.training and self.gate_noise > 0:
            noise = self.gate_noise * torch.randn_like(logits) / E
            logits_noisy = logits + noise
        else:
            logits_noisy = logits
        scores_noisy = F.softmax(logits_noisy, dim=1)   # [N, E], fp32

        # Select the top experts for each token.
        top_vals, top_indices = torch.topk(scores_noisy, self.top_k, dim=1)
        top_logits_noisy = logits_noisy.gather(1, top_indices)   # [N, k]

        # Compute the auxiliary routing loss.
        self.l_aux = _load_importance_loss(
            scores_no_noise, top_logits_noisy, E, self.gate_noise
        )

        # Enforce expert capacity.
        gate_weights = torch.zeros(N, E, dtype=scores_noisy.dtype,
                                   device=x.device)
        gate_weights.scatter_(1, top_indices, top_vals)   # [N, E]

        # Keep the highest-scoring tokens within each expert capacity.
        capacity = max(1, int(math.ceil(self.capacity_factor * N / E)))
        for j in range(E):
            col = gate_weights[:, j]              # [N]
            nonzero_count = int((col > 0).sum().item())
            if nonzero_count > capacity:
                # Sort descending; zero out tokens beyond capacity
                sorted_idx = torch.argsort(col, descending=True)
                overflow_idx = sorted_idx[capacity:]
                gate_weights[overflow_idx, j] = 0.0

        # Evaluate every expert in one vectorized pass.
        x_f = x_flat.float()   # ensure fp32 for expert computation

        # hidden[n, e, h] = sum_d x_f[n, d] * fc1_weight[e, h, d] + fc1_bias[e, h]
        hidden = torch.einsum('nd,ehd->neh', x_f, self.fc1_weight.float()) \
                 + self.fc1_bias.float().unsqueeze(0)          # [N, E, H]
        hidden = self.act_dropout(F.gelu(hidden))

        # all_expert_out[n, e, d] = sum_h hidden[n,e,h]*fc2_weight[e,d,h] + fc2_bias[e,d]
        all_expert_out = torch.einsum('neh,edh->ned', hidden, self.fc2_weight.float()) \
                         + self.fc2_bias.float().unsqueeze(0)  # [N, E, D]

        # Block gate gradients from the orthogonality loss.
        selection_mask = (gate_weights > 0).float()
        self.expert_outputs = all_expert_out * selection_mask.unsqueeze(-1)  # [N, E, D]

        # Aggregate expert outputs with routing weights.
        y = (all_expert_out * gate_weights.unsqueeze(-1)).sum(dim=1)   # [N, D]
        y = y.to(orig_dtype).reshape(B, T, D)
        return y

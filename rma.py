"""RMA stage 2 (decisions.md P6/P7): privileged teacher with an environment-factor encoder.

The teacher is pi(x_t, z_t) with z_t = mu(e_t), where e_t is the env's "privileged" observation group
(delta_mass_kg, mu_static, motor_strength) and z_t is the extrinsics latent. mu and pi are trained jointly by PPO
(RMA, Kumar et al. 2021, arXiv 2107.04034, Sec. III-A and Algorithm 1). The critic gets its own copy of the same
structure (ASSUMED: rsl-rl keeps actor and critic as separate models, so the encoders are not shared).

Warm start from stage 1 (P7): the first linear layer of actor and critic is zero-padded for the z columns
(`pad_stage1_checkpoint`), so at the first iteration the teacher acts exactly like the stage-1 policy, whatever mu
outputs. The encoder starts at the default rsl-rl MLP init.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP

# e = (delta_mass_kg, mu_static, motor_strength). Fixed affine scaling to about [-1, 1] over the training DR ranges
# (decisions.md D1-D3: mass +-0.75 kg, friction 0.5-1.25, strength 0.8-1.2). ASSUMED convention (scale inputs to
# unit range); a fixed map keeps out-of-range test values (payload +2.5 kg, s 0.5) visible as |e| > 1.
PRIV_CENTER = (0.0, 0.875, 1.0)
PRIV_HALF_RANGE = (0.75, 0.375, 0.2)


class RMATeacherModel(MLPModel):
    """MLPModel whose input is [non-privileged groups, mu(privileged group)]; z is appended last."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        priv_group: str = "privileged",
        z_dim: int = 8,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        priv_center: tuple[float, ...] | list[float] = PRIV_CENTER,
        priv_half_range: tuple[float, ...] | list[float] = PRIV_HALF_RANGE,
        **kwargs,
    ) -> None:
        self._priv_group = priv_group
        self._z_dim = int(z_dim)
        if priv_group not in obs_groups[obs_set]:
            raise ValueError(f"RMATeacherModel: obs set '{obs_set}' must include the '{priv_group}' group")
        priv_dim = obs[priv_group].shape[-1]
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        activation = kwargs.get("activation", "elu")
        self.encoder = MLP(priv_dim, self._z_dim, list(encoder_hidden_dims), activation)
        self.register_buffer("priv_center", torch.tensor(list(priv_center), dtype=torch.float32))
        self.register_buffer("priv_half_range", torch.tensor(list(priv_half_range), dtype=torch.float32))
        if self.priv_center.numel() != priv_dim or self.priv_half_range.numel() != priv_dim:
            raise ValueError(f"priv_center/priv_half_range must have {priv_dim} entries")

    # the base class sizes the normalizer and the MLP from these two hooks
    def _get_obs_dim(self, obs, obs_groups, obs_set):
        groups = [g for g in obs_groups[obs_set] if g != self._priv_group]
        return super()._get_obs_dim(obs, {obs_set: groups}, obs_set)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self._z_dim

    def encode(self, obs: TensorDict) -> torch.Tensor:
        e = (obs[self._priv_group] - self.priv_center) / self.priv_half_range
        return self.encoder(e)

    def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
        base = super().get_latent(obs, masks, hidden_state)
        return torch.cat([base, self.encode(obs)], dim=-1)

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("RMATeacherModel export: use hexapod.rma.load_policy instead")

    def as_onnx(self, verbose: bool) -> nn.Module:
        raise NotImplementedError("RMATeacherModel export: use hexapod.rma.load_policy instead")


# ----------------------------------------------------------------------------------------------------------------
# Checkpoint helpers (pure torch, used by the conversion tool and the eval/long-run tools)
# ----------------------------------------------------------------------------------------------------------------


def _mlp_from_sd(sd: dict, prefix: str) -> nn.Sequential:
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith(prefix + ".") and k.endswith(".weight")})
    mods: list[nn.Module] = []
    for i, li in enumerate(layers):
        w = sd[f"{prefix}.{li}.weight"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(sd[f"{prefix}.{li}.bias"])
        mods.append(lin)
        if i < len(layers) - 1:
            mods.append(nn.ELU())
    return nn.Sequential(*mods)


class CheckpointPolicy(nn.Module):
    """Deterministic actor rebuilt from a checkpoint; forward takes the env's observation dict.

    Plain stage-1 checkpoints use obs["policy"] only. Teacher checkpoints (encoder.* keys) also use
    obs["privileged"] through the encoder, with the scaling stored in the checkpoint.
    """

    def __init__(self, sd: dict):
        super().__init__()
        for k in sd:
            if "normalizer" in k:
                raise RuntimeError(f"checkpoint has an observation normalizer ({k}); not supported here")
        self.mlp = _mlp_from_sd(sd, "mlp")
        self.is_teacher = any(k.startswith("encoder.") for k in sd)
        if self.is_teacher:
            self.encoder = _mlp_from_sd(sd, "encoder")
            self.register_buffer("priv_center", sd["priv_center"].clone().float())
            self.register_buffer("priv_half_range", sd["priv_half_range"].clone().float())

    def z(self, priv: torch.Tensor) -> torch.Tensor:
        return self.encoder((priv - self.priv_center) / self.priv_half_range)

    def forward(self, obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            if self.is_teacher:
                raise ValueError("teacher policy needs the observation dict (policy + privileged)")
            return self.mlp(obs)
        x = obs["policy"]
        if self.is_teacher:
            x = torch.cat([x, self.z(obs["privileged"])], dim=-1)
        return self.mlp(x)


def load_policy(path: str, device) -> CheckpointPolicy:
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["actor_state_dict"] if "actor_state_dict" in ck else ck
    return CheckpointPolicy(sd).to(device).eval()


def pad_stage1_checkpoint(
    src: dict,
    priv_dim: int = 3,
    z_dim: int = 8,
    encoder_hidden_dims=(256, 128),
    priv_center=PRIV_CENTER,
    priv_half_range=PRIV_HALF_RANGE,
    seed: int = 0,
) -> dict:
    """Stage-1 checkpoint dict -> teacher init dict (same iteration counter).

    First-layer weights of actor and critic get z_dim zero columns appended (z is the last input block), each model
    gets a freshly initialised encoder, and the optimizer state is replaced by an empty Adam state with the same
    hyper-parameters (the parameter list changed, so the old moments cannot be mapped by index; ASSUMED harmless
    because the padded policy is identical at the switch).
    """
    g = torch.Generator().manual_seed(seed)
    out = dict(src)
    n_params = 0
    for key in ("actor_state_dict", "critic_state_dict"):
        sd = {k: v.clone() for k, v in src[key].items()}
        w0 = sd["mlp.0.weight"]
        sd["mlp.0.weight"] = torch.cat([w0, torch.zeros(w0.shape[0], z_dim, dtype=w0.dtype)], dim=1)
        dims = [priv_dim, *encoder_hidden_dims, z_dim]
        for j in range(len(dims) - 1):
            lin = nn.Linear(dims[j], dims[j + 1])
            # nn.Linear default init (same as rsl-rl's MLP), drawn from a seeded generator for reproducibility
            bound = 1.0 / dims[j] ** 0.5
            lin.weight.data.uniform_(-bound, bound, generator=g)
            lin.bias.data.uniform_(-bound, bound, generator=g)
            sd[f"encoder.{2 * j}.weight"] = lin.weight.data.clone()
            sd[f"encoder.{2 * j}.bias"] = lin.bias.data.clone()
        sd["priv_center"] = torch.tensor(list(priv_center), dtype=torch.float32)
        sd["priv_half_range"] = torch.tensor(list(priv_half_range), dtype=torch.float32)
        n_params += sum(1 for k in sd if k not in ("priv_center", "priv_half_range"))
        out[key] = sd
    opt = src["optimizer_state_dict"]
    group = {k: v for k, v in opt["param_groups"][0].items() if k != "params"}
    group["params"] = list(range(n_params))
    out["optimizer_state_dict"] = {"state": {}, "param_groups": [group]}
    out["infos"] = {"rma_init_from_iter": src.get("iter"), "z_dim": z_dim}
    return out

from typing import Dict, List, Sequence

import torch
from torch import Tensor, nn


def _build_mlp_body(in_dim: int, hidden_channels: int, n_layers: int, p_drop: float) -> nn.Sequential:
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    layers: List[nn.Module] = [nn.Linear(in_dim, hidden_channels), nn.ReLU()]
    if p_drop > 0:
        layers.append(nn.Dropout(p_drop))
    width = hidden_channels
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(width, hidden_channels))
        layers.append(nn.ReLU())
        if p_drop > 0:
            layers.append(nn.Dropout(p_drop))
    return nn.Sequential(*layers)


def _build_operator_stack(
    in_dim: int,
    embed_dim: int,
    hidden_channels: int,
    n_layers: int,
    out_dim: int,
    p_drop: float,
) -> nn.Sequential:
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    layers: List[nn.Module] = [nn.Linear(in_dim, embed_dim), nn.GELU()]
    if p_drop > 0:
        layers.append(nn.Dropout(p_drop))
    width = embed_dim
    for _ in range(max(1, n_layers - 1)):
        layers.append(nn.Linear(width, hidden_channels))
        layers.append(nn.GELU())
        if p_drop > 0:
            layers.append(nn.Dropout(p_drop))
        width = hidden_channels
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_channels: int, n_layers: int, p_drop: float = 0.0) -> None:
        super().__init__()
        body = _build_mlp_body(in_dim, hidden_channels, n_layers, p_drop)
        self.net = nn.Sequential(body, nn.Linear(hidden_channels, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiHeadMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_channels: int,
        n_layers: int,
        head_names: Sequence[str],
        head_dim: int = 1,
        p_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if not head_names:
            raise ValueError("head_names must not be empty")
        self.trunk = _build_mlp_body(in_dim, hidden_channels, n_layers, p_drop)
        self.head_names = list(head_names)
        self.head_dim = head_dim
        self.heads = nn.ModuleList([nn.Linear(hidden_channels, head_dim) for _ in self.head_names])

    def forward(self, x: Tensor) -> Tensor:
        shared = self.trunk(x)
        outputs = [head(shared).unsqueeze(1) for head in self.heads]
        return torch.cat(outputs, dim=1)


class DeepONet(nn.Module):
    def __init__(
        self,
        n_params: int,
        param_embedding_dim: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int,
        n_basis: int,
        p_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.n_basis = n_basis
        self.branch = _build_operator_stack(
            n_params,
            param_embedding_dim,
            hidden_channels,
            n_layers,
            out_channels * (n_basis + 1),
            p_drop,
        )
        self.trunk = _build_operator_stack(1, param_embedding_dim, hidden_channels, n_layers, n_basis, p_drop)

    def forward(self, params: Tensor, grid: Tensor) -> Tensor:
        batch, n_points, gdim = grid.shape
        if gdim != 1:
            raise ValueError("grid must have size 1 on the last dimension")
        coeff = self.branch(params).view(batch, self.out_channels, self.n_basis + 1)
        phi = self.trunk(grid.reshape(batch * n_points, gdim)).view(batch, n_points, self.n_basis)
        ones = torch.ones(batch, n_points, 1, dtype=phi.dtype, device=phi.device)
        phi = torch.cat([phi, ones], dim=-1)
        return torch.einsum("bcn,bln->bcl", coeff, phi)


class MultiHeadDeepONet(DeepONet):
    def __init__(
        self,
        n_params: int,
        param_embedding_dim: int,
        hidden_channels: int,
        n_heads: int,
        n_layers: int,
        n_basis: int,
        p_drop: float = 0.0,
    ) -> None:
        super().__init__(
            n_params=n_params,
            param_embedding_dim=param_embedding_dim,
            hidden_channels=hidden_channels,
            out_channels=n_heads,
            n_layers=n_layers,
            n_basis=n_basis,
            p_drop=p_drop,
        )
        self.n_heads = n_heads


MODEL_REGISTRY: Dict[str, nn.Module] = {
    "mlp": SimpleMLP,
    "multihead_mlp": MultiHeadMLP,
    "deeponet": MultiHeadDeepONet,
    "deeponet_single": DeepONet,
}


def build_model(name: str, **kwargs) -> nn.Module:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return MODEL_REGISTRY[key](**kwargs)

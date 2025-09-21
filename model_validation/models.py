from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn

ACTIVATIONS: Dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}

def count_parameters(module: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(param.numel() for param in module.parameters() if param.requires_grad)

def _make_activation(name: str) -> nn.Module:
    if name.lower() not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {name}")
    return ACTIVATIONS[name.lower()]()

class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Sequence[int],
        activation: str = "relu",
        dropout: float = 0.0,
        layernorm: bool = False,
    ) -> None:
        super().__init__()
        body, last_dim = _build_MLP(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
            layernorm=layernorm,
        )
        layers = list(body)
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={count_parameters(self)})"


class MultiHeadMLP(nn.Module):
    """
        Shared trunk with per-target linear heads.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int],
        head_names: Sequence[str],
        head_dim: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        layernorm: bool = False,
    ) -> None:
        super().__init__()
        if not head_names:
            raise ValueError("head_names must be provided")
        self.head_names = list(head_names)
        self.trunk, trunk_out = _build_MLP(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
            layernorm=layernorm,
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(trunk_out, head_dim)
            for name in self.head_names
        })

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        shared = self.trunk(x)
        outputs = {
            name: head(shared)
            for name, head in self.heads.items()
        }
        return outputs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={count_parameters(self)})"


class DeepONet(nn.Module):
    """
        Deep operator network implementation.
        branch multi-head
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        param_embd_dim: int,
        branch_layers: Sequence[int],
        trunk_layers: Sequence[int],
        latent_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        trunk_input_dim: int = 1,
    ) -> None:
        super().__init__()
        self.output_dim = max(1, output_dim)
        self.latent_dim = latent_dim
        self.branch = _build_operator_stack(
            input_dim=input_dim,
            param_embd_dim= param_embd_dim,
            hidden_layers=branch_layers,
            output_dim=latent_dim * self.output_dim,
            activation=activation,
            dropout=dropout,
        )
        self.trunk = _build_operator_stack(
            input_dim=trunk_input_dim,
            hidden_layers=trunk_layers,
            output_dim=latent_dim,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, branch_input: Tensor, trunk_input: Tensor) -> Tensor:
        batch, n_points, w_dim = trunk_input.shape
        if self.trunk[0].in_features != w_dim:
            raise ValueError(
                f"Expected trunk input dim {self.trunk[0].in_features}, received {w_dim}"
            )
        branch_out = self.branch(branch_input)
        branch_out = branch_out.view(batch, self.output_dim, self.latent_dim)
        trunk_flat = trunk_input.reshape(batch * n_points, w_dim)
        trunk_features = self.trunk(trunk_flat)
        trunk_features = trunk_features.view(batch, n_points, self.latent_dim)
        branch_exp = branch_out.unsqueeze(2)
        trunk_exp = trunk_features.unsqueeze(1)
        values = torch.sum(branch_exp * trunk_exp, dim=-1)
        return values

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={count_parameters(self)})"


def _build_operator_stack(
    input_dim: int,
    hidden_layers: Sequence[int],
    output_dim: int,
    activation: str,
    dropout: float,
    param_embd_dim: Optional[int] = None,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = input_dim
    if param_embd_dim is not None:
        layers.append(nn.Linear(prev, param_embd_dim))
        prev = param_embd_dim
        
    for width in hidden_layers:
        layers.append(nn.Linear(prev, width))
        layers.append(_make_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = width
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def _build_MLP(
    input_dim: int,
    hidden_layers: Sequence[int],
    activation: str,
    dropout: float,
    layernorm: bool,
) -> tuple[nn.Sequential, int]:
    modules: List[nn.Module] = []
    prev = input_dim
    for width in hidden_layers:
        modules.append(nn.Linear(prev, width))
        if layernorm:
            modules.append(nn.LayerNorm(width))
        modules.append(_make_activation(activation))
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        prev = width
    return nn.Sequential(*modules), prev

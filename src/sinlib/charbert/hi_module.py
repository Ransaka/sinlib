"""
Heterogeneous Interaction (HI) Module for layer-by-layer Token and Character Channel Fusion.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
from transformers.activations import get_activation

from sinlib.charbert.config import SinhalaCharBERTConfig


class HeterogeneousInteractionModule(nn.Module):
    """
    Heterogeneous Interaction (HI) Module.
    Executes a 2-step interaction:
    1. Step 1 (Fusion): Multi-Window 1D Convolutions over concatenated token and char projections.
    2. Step 2 (Divide): GELU-activated feedforward projections with residual connections and LayerNorm.
    """

    def __init__(self, config: SinhalaCharBERTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.kernel_sizes = config.hi_kernel_sizes

        # Step 1: Linear Projections for Fusion
        self.proj_token = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_char = nn.Linear(config.hidden_size, config.hidden_size)

        # Multi-Window 1D Convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=2 * config.hidden_size,
                out_channels=config.hidden_size,
                kernel_size=k,
                padding=k // 2,
            )
            for k in self.kernel_sizes
        ])

        # Convolution output projection to unify multi-window contexts
        self.conv_proj = nn.Linear(len(self.kernel_sizes) * config.hidden_size, config.hidden_size)
        self.act = get_activation(config.hidden_act)

        # Step 2: Divide Step Projections
        self.token_divide_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.char_divide_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.token_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.char_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        token_repr: torch.Tensor,
        char_repr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Heterogeneous Interaction Module.

        Parameters
        ----------
        token_repr : torch.Tensor
            Current token channel representation of shape (batch_size, m, hidden_size).
        char_repr : torch.Tensor
            Current character channel representation of shape (batch_size, m, hidden_size).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Updated (token_repr, char_repr), each of shape (batch_size, m, hidden_size).
        """
        # ----------------------------------------------------
        # Step 1: Fusion
        # ----------------------------------------------------
        t_proj = self.proj_token(token_repr)  # (batch, m, hidden_size)
        c_proj = self.proj_char(char_repr)    # (batch, m, hidden_size)

        # Concatenate along hidden dimension: (batch, m, 2 * hidden_size)
        fused_input = torch.cat([t_proj, c_proj], dim=-1)

        # Permute for 1D convolution: (batch, 2 * hidden_size, m)
        fused_conv_in = fused_input.transpose(1, 2)

        # Apply multi-window convolutions
        conv_outputs: List[torch.Tensor] = []
        for conv in self.convs:
            conv_out = conv(fused_conv_in)  # (batch, hidden_size, m)
            conv_out = self.act(conv_out)
            conv_outputs.append(conv_out.transpose(1, 2))  # (batch, m, hidden_size)

        # Concatenate multi-window outputs: (batch, m, num_kernels * hidden_size)
        multi_conv_cat = torch.cat(conv_outputs, dim=-1)
        fused_C = self.conv_proj(multi_conv_cat)  # (batch, m, hidden_size)
        fused_C = self.act(fused_C)

        # ----------------------------------------------------
        # Step 2: Divide
        # ----------------------------------------------------
        delta_token = self.dropout(self.token_divide_proj(fused_C))
        delta_char = self.dropout(self.char_divide_proj(fused_C))

        updated_token = self.token_layer_norm(token_repr + delta_token)
        updated_char = self.char_layer_norm(char_repr + delta_char)

        return updated_token, updated_char

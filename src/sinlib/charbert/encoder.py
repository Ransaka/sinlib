"""
Transformer Encoder Stack interleaved with Heterogeneous Interaction (HI) modules.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer

from sinlib.charbert.config import SinhalaCharBERTConfig
from sinlib.charbert.hi_module import HeterogeneousInteractionModule


class SinhalaCharBERTLayer(nn.Module):
    """
    A single dual-channel Transformer Encoder layer consisting of:
    1. Standard Transformer Self-Attention & Feed-Forward layer for Token Channel.
    2. Heterogeneous Interaction (HI) Module for layer-by-layer Token-Char interaction.
    """

    def __init__(self, config: SinhalaCharBERTConfig):
        super().__init__()
        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps,
            pad_token_id=config.pad_token_id,
        )
        bert_config._attn_implementation = "eager"
        self.transformer_layer = BertLayer(bert_config)
        self.hi_module = HeterogeneousInteractionModule(config)

    def forward(
        self,
        token_hidden_states: torch.Tensor,
        char_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        token_hidden_states : torch.Tensor
            Token channel hidden state of shape (batch_size, m, hidden_size).
        char_hidden_states : torch.Tensor
            Character channel hidden state of shape (batch_size, m, hidden_size).
        attention_mask : Optional[torch.Tensor]
            Attention mask for token channel (batch_size, 1, 1, m).
        head_mask : Optional[torch.Tensor]
            Head mask tensor.
        output_attentions : bool
            Whether to output attention weights.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            (updated_token_hidden, updated_char_hidden, attention_weights)
        """
        layer_outputs = self.transformer_layer(
            token_hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )

        if isinstance(layer_outputs, tuple):
            updated_token_states = layer_outputs[0]
            attentions = layer_outputs[1] if len(layer_outputs) > 1 else None
        else:
            updated_token_states = layer_outputs
            attentions = None

        # Execute Heterogeneous Interaction (HI) step
        updated_token_states, updated_char_states = self.hi_module(
            token_repr=updated_token_states,
            char_repr=char_hidden_states,
        )

        return updated_token_states, updated_char_states, attentions


class SinhalaCharBERTEncoder(nn.Module):
    """
    Full Sinhala-CharBERT Encoder stack with N interleaved Transformer + HI layers.
    """

    def __init__(self, config: SinhalaCharBERTConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            SinhalaCharBERTLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        token_hidden_states: torch.Tensor,
        char_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        all_token_hidden_states: List[torch.Tensor] = []
        all_char_hidden_states: List[torch.Tensor] = []
        all_attentions: List[torch.Tensor] = []

        curr_token = token_hidden_states
        curr_char = char_hidden_states

        for layer in self.layers:
            if output_hidden_states:
                all_token_hidden_states.append(curr_token)
                all_char_hidden_states.append(curr_char)

            curr_token, curr_char, attns = layer(
                token_hidden_states=curr_token,
                char_hidden_states=curr_char,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            if output_attentions and attns is not None:
                all_attentions.append(attns)

        if output_hidden_states:
            all_token_hidden_states.append(curr_token)
            all_char_hidden_states.append(curr_char)

        return (
            curr_token,
            curr_char,
            all_token_hidden_states if output_hidden_states else None,
            all_char_hidden_states if output_hidden_states else None,
            all_attentions if output_attentions else None,
        )

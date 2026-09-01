"""
Sinhala-CharBERT dual-channel encoder backbone (inference-only vendored copy).

State-dict key compatibility is preserved with the original training
repository (Ransaka/sinhala-charbert): module attribute names are identical.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from sinlib.charbert.char_encoder import CharacterBiGRUEncoder
from sinlib.charbert.config import SinhalaCharBERTConfig
from sinlib.charbert.embeddings import SinhalaCharEmbeddings, SinhalaTokenEmbeddings
from sinlib.charbert.encoder import SinhalaCharBERTEncoder


@dataclass
class SinhalaCharBERTOutput:
    """Output container for the Sinhala-CharBERT encoder backbone."""

    last_hidden_state: torch.Tensor = None
    last_char_hidden_state: torch.Tensor = None
    fused_hidden_state: torch.Tensor = None
    all_token_hidden_states: Optional[List[torch.Tensor]] = None
    all_char_hidden_states: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None


class SinhalaCharBERTModel(nn.Module):
    """
    Dual-Channel Transformer Backbone for Sinhala-CharBERT (inference only).
    Processes subword tokens in parallel with phonological character units,
    bridging them with boundary pooling and Heterogeneous Interaction modules.
    """

    def __init__(self, config: SinhalaCharBERTConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = SinhalaTokenEmbeddings(config)
        self.char_embeddings = SinhalaCharEmbeddings(config)
        self.char_encoder = CharacterBiGRUEncoder(config)
        self.encoder = SinhalaCharBERTEncoder(config)

        # Final fusion projection
        self.fused_proj = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        char_input_ids: torch.Tensor,
        start_char_idx: torch.Tensor,
        end_char_idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        char_attention_mask: Optional[torch.Tensor] = None,
    ) -> SinhalaCharBERTOutput:
        token_embeds = self.token_embeddings(input_ids)
        char_embeds = self.char_embeddings(char_input_ids)

        token_aligned_char = self.char_encoder(
            char_embeds, start_char_idx, end_char_idx, char_attention_mask
        )

        token_hidden, char_hidden, _, _, _ = self.encoder(
            token_hidden_states=token_embeds,
            char_hidden_states=token_aligned_char,
            attention_mask=attention_mask,
        )

        fused = self.fused_proj(torch.cat([token_hidden, char_hidden], dim=-1))
        fused = self.layer_norm(fused)

        return SinhalaCharBERTOutput(
            last_hidden_state=token_hidden,
            last_char_hidden_state=char_hidden,
            fused_hidden_state=fused,
        )

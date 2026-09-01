"""
Character Channel Bi-GRU Encoder and Boundary Concatenation Pooling Module.
"""

from typing import Optional
import torch
import torch.nn as nn

from sinlib.charbert.config import SinhalaCharBERTConfig


class CharacterBiGRUEncoder(nn.Module):
    """
    Bidirectional GRU encoder for phonological character units with boundary pooling.
    Compresses phonological unit sequences of length N into token-aligned representations of length m.
    """

    def __init__(self, config: SinhalaCharBERTConfig):
        super().__init__()
        self.char_gru_hidden_size = config.char_gru_hidden_size
        self.hidden_size = config.hidden_size

        self.gru = nn.GRU(
            input_size=config.char_embedding_dim,
            hidden_size=config.char_gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        char_embeddings: torch.Tensor,
        start_char_idx: torch.Tensor,
        end_char_idx: torch.Tensor,
        char_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of Character Bi-GRU with boundary concatenation pooling.

        Parameters
        ----------
        char_embeddings : torch.Tensor
            Character embeddings of shape (batch_size, N, char_embedding_dim).
        start_char_idx : torch.Tensor
            Start character indices for each subword token of shape (batch_size, m).
        end_char_idx : torch.Tensor
            End character indices for each subword token of shape (batch_size, m).
        char_attention_mask : Optional[torch.Tensor]
            Character attention mask of shape (batch_size, N).

        Returns
        -------
        torch.Tensor
            Token-aligned character representation of shape (batch_size, m, hidden_size).
        """
        batch_size, n_chars, _ = char_embeddings.shape
        _, m_tokens = start_char_idx.shape

        # Optional zero-out of padded character embeddings
        if char_attention_mask is not None:
            char_embeddings = char_embeddings * char_attention_mask.unsqueeze(-1).to(char_embeddings.dtype)

        # Run Bidirectional GRU
        gru_output, _ = self.gru(char_embeddings)  # (batch_size, N, 2 * char_gru_hidden_size)

        # Split into forward and backward hidden representations
        fwd_states = gru_output[:, :, :self.char_gru_hidden_size]  # (batch_size, N, char_gru_hidden_size)
        bwd_states = gru_output[:, :, self.char_gru_hidden_size:]  # (batch_size, N, char_gru_hidden_size)

        # Clamp boundary indices to valid sequence bounds for safety
        clamped_start = torch.clamp(start_char_idx, min=0, max=n_chars - 1)
        clamped_end = torch.clamp(end_char_idx, min=0, max=n_chars - 1)

        # Gather forward state at start_char_idx
        start_expanded = clamped_start.unsqueeze(-1).expand(-1, -1, self.char_gru_hidden_size)
        fwd_gathered = torch.gather(fwd_states, dim=1, index=start_expanded)  # (batch_size, m, d_fwd)

        # Gather backward state at end_char_idx
        end_expanded = clamped_end.unsqueeze(-1).expand(-1, -1, self.char_gru_hidden_size)
        bwd_gathered = torch.gather(bwd_states, dim=1, index=end_expanded)  # (batch_size, m, d_bwd)

        # Concatenate boundary states: [h_start_fwd ; h_end_bwd]
        token_aligned_char = torch.cat([fwd_gathered, bwd_gathered], dim=-1)  # (batch_size, m, hidden_size)

        token_aligned_char = self.layer_norm(token_aligned_char)
        token_aligned_char = self.dropout(token_aligned_char)

        return token_aligned_char

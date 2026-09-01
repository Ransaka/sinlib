"""
Embedding modules for Sinhala-CharBERT dual-channel architecture.
Implements Token Embeddings (subwords) and Character Embeddings (phonological Akshara units).
"""

from typing import Optional
import torch
import torch.nn as nn

from sinlib.charbert.config import SinhalaCharBERTConfig


class SinhalaTokenEmbeddings(nn.Module):
    """
    Subword Token Channel Embeddings.
    Constructs embeddings from subword token IDs, position IDs, and token type IDs.
    """

    def __init__(self, config: SinhalaCharBERTConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Register position_ids buffer
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Enforce max position sequence length
        if input_ids.size(1) > self.position_embeddings.num_embeddings:
            input_ids = input_ids[:, : self.position_embeddings.num_embeddings]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, : self.position_embeddings.num_embeddings]

        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SinhalaCharEmbeddings(nn.Module):
    """
    Phonological Akshara Character Channel Embeddings.
    Maps sinlib phonological unit IDs into dense character embedding vectors.
    """

    def __init__(self, config: SinhalaCharBERTConfig):
        super().__init__()
        self.char_embeddings = nn.Embedding(
            config.char_vocab_size,
            config.char_embedding_dim,
            padding_idx=config.char_pad_token_id,
        )
        self.layer_norm = nn.LayerNorm(
            config.char_embedding_dim, eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, char_input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.char_embeddings(char_input_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

"""
Mode B: Open-Vocabulary Seq2Seq Character-Level Transformer Decoder for Sinhala-CharBERT.
Autoregressively decodes sinlib phonological Akshara units conditioned on CharBERT's fused sequence states.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.file_utils import ModelOutput

from sinlib.charbert.config import SinhalaCharBERTConfig
from sinlib.charbert.modeling import SinhalaCharBERTModel, SinhalaCharBERTOutput


@dataclass
class Seq2SeqCorrectionOutput(ModelOutput):
    """Output container for SinhalaCharBERTSeq2SeqModel."""
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    encoder_hidden_states: Optional[SinhalaCharBERTOutput] = None
    decoder_hidden_states: Optional[torch.Tensor] = None


class SinhalaCharBERTDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer with:
    1. Masked Causal Self-Attention
    2. Cross-Attention over CharBERT fused encoder sequence states
    3. GELU Position-wise Feed-Forward Network with residual LayerNorm
    """

    def __init__(self, config: SinhalaCharBERTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # 1. Causal Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 2. Cross-Attention over CharBERT Encoder
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.cross_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 3. Feed-Forward Network
        self.ffn_dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_act = nn.GELU() if config.hidden_act == "gelu" else nn.ReLU()
        self.ffn_dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        encoder_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Step 1: Masked Causal Self-Attention
        residual = hidden_states
        attn_out, _ = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=causal_mask,
            # need_weights=True forces the math code path of
            # nn.MultiheadAttention; the fused fast path has shown
            # device-dependent failures with 2-D float causal masks
            # on some CUDA builds.
            need_weights=True,
        )
        hidden_states = self.self_attn_layer_norm(residual + self.dropout(attn_out))

        # Step 2: Cross-Attention over CharBERT Fused State
        residual = hidden_states
        cross_out, _ = self.cross_attn(
            query=hidden_states,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            key_padding_mask=encoder_key_padding_mask,
            need_weights=False,
        )
        hidden_states = self.cross_attn_layer_norm(residual + self.dropout(cross_out))

        # Step 3: Feed-Forward Network
        residual = hidden_states
        ffn_out = self.ffn_dense2(self.dropout(self.ffn_act(self.ffn_dense1(hidden_states))))
        hidden_states = self.ffn_layer_norm(residual + self.dropout(ffn_out))

        return hidden_states


class SinhalaCharBERTSeq2SeqModel(nn.Module):
    """
    Open-Vocabulary Seq2Seq Corrector (Mode B).
    Pairs the Sinhala-CharBERT dual-channel encoder with an autoregressive
    Transformer Decoder producing sinlib phonological Akshara units.
    """

    def __init__(
        self,
        config: SinhalaCharBERTConfig,
        num_decoder_layers: int = 4,
        max_target_positions: int = 512,
    ):
        super().__init__()
        self.config = config
        self.max_target_positions = max_target_positions

        # Encoder: Sinhala-CharBERT Dual-Channel Backbone
        self.encoder = SinhalaCharBERTModel(config)

        # Decoder Target Character Embeddings & Positional Embeddings
        self.target_char_embeddings = nn.Embedding(
            config.char_vocab_size,
            config.hidden_size,
            padding_idx=config.char_pad_token_id,
        )
        self.position_embeddings = nn.Embedding(max_target_positions, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Decoder Transformer Layers
        self.decoder_layers = nn.ModuleList(
            [SinhalaCharBERTDecoderLayer(config) for _ in range(num_decoder_layers)]
        )

        # Output LM Head over Character Vocabulary
        self.lm_head = nn.Linear(config.hidden_size, config.char_vocab_size, bias=False)

        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_target_positions, max_target_positions), float("-inf")), diagonal=1),
            persistent=False,
        )
        self.register_buffer(
            "position_ids",
            torch.arange(max_target_positions).expand((1, -1)),
            persistent=False,
        )

        self.init_decoder_weights()

    def init_decoder_weights(self):
        """Initializes decoder embeddings and projections."""
        for module in [self.target_char_embeddings, self.position_embeddings, self.lm_head]:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if hasattr(module, "padding_idx") and module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        char_input_ids: torch.Tensor,
        start_char_idx: torch.Tensor,
        end_char_idx: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        char_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        label_weights: Optional[torch.Tensor] = None,
    ) -> Seq2SeqCorrectionOutput:
        """
        Forward pass through Seq2Seq Encoder-Decoder.

        ``label_weights`` (B, T) per-position loss weights. Positions flagged as
        corrupted (noisy != clean) are upweighted so the correction signal is
        not drowned by the copy-majority tokens. When ``None``, unweighted
        mean cross-entropy is used (backward compatible).
        """
        # 1. Encode with Sinhala-CharBERT Dual-Channel Backbone
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            char_input_ids=char_input_ids,
            start_char_idx=start_char_idx,
            end_char_idx=end_char_idx,
            attention_mask=attention_mask,
            char_attention_mask=char_attention_mask,
        )
        fused_hidden = encoder_outputs.fused_hidden_state  # (batch, seq_len, hidden_size)

        # 2. Embed Decoder Target Inputs
        target_seq_len = decoder_input_ids.size(1)
        if target_seq_len > self.max_target_positions:
            decoder_input_ids = decoder_input_ids[:, : self.max_target_positions]
            target_seq_len = self.max_target_positions

        target_pos_ids = self.position_ids[:, :target_seq_len]
        char_embeds = self.target_char_embeddings(decoder_input_ids)
        pos_embeds = self.position_embeddings(target_pos_ids)

        decoder_hidden = self.embed_layer_norm(char_embeds + pos_embeds)
        decoder_hidden = self.embed_dropout(decoder_hidden)

        # Construct causal mask
        causal_mask = self.causal_mask[:target_seq_len, :target_seq_len]

        # Encoder key padding mask (True for padding positions)
        enc_key_padding = None
        if attention_mask is not None:
            enc_key_padding = (attention_mask == 0)

        # 3. Pass through Decoder Stack
        for layer in self.decoder_layers:
            decoder_hidden = layer(
                hidden_states=decoder_hidden,
                encoder_hidden_states=fused_hidden,
                causal_mask=causal_mask,
                encoder_key_padding_mask=enc_key_padding,
            )

        # 4. Predict Character Logits
        logits = self.lm_head(decoder_hidden)  # (batch, target_seq_len, char_vocab_size)

        loss = None
        if labels is not None:
            if labels.size(1) > target_seq_len:
                labels = labels[:, :target_seq_len]
            if label_weights is not None and label_weights.size(1) > target_seq_len:
                label_weights = label_weights[:, :target_seq_len]

            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=label_smoothing,
                reduction="none",
            )
            flat_logits = logits.reshape(-1, self.config.char_vocab_size)
            flat_labels = labels.reshape(-1)
            per_token_loss = loss_fct(flat_logits, flat_labels)  # (B*T,)
            # CrossEntropyLoss with ignore_index returns 0 for ignored positions
            # (reduction="none" zeroes them), so multiply-and-sum is safe.
            valid = (flat_labels != -100).to(per_token_loss.dtype)

            if label_weights is not None:
                flat_weights = label_weights.reshape(-1).to(per_token_loss.dtype) * valid
                denom = flat_weights.sum().clamp(min=1.0)
                loss = (per_token_loss * flat_weights).sum() / denom
            else:
                denom = valid.sum().clamp(min=1.0)
                loss = (per_token_loss * valid).sum() / denom

        return Seq2SeqCorrectionOutput(
            loss=loss,
            logits=logits,
            encoder_hidden_states=encoder_outputs,
            decoder_hidden_states=decoder_hidden,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        char_input_ids: torch.Tensor,
        start_char_idx: torch.Tensor,
        end_char_idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        char_attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        num_beams: int = 1,
        temperature: float = 1.0,
        length_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Generates target phonological character units via Greedy Search or Beam Search.
        Returns generated character ID tensor of shape (batch, generated_len).
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Encode input sequence once
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            char_input_ids=char_input_ids,
            start_char_idx=start_char_idx,
            end_char_idx=end_char_idx,
            attention_mask=attention_mask,
            char_attention_mask=char_attention_mask,
        )
        fused_hidden = encoder_outputs.fused_hidden_state

        enc_key_padding = None
        if attention_mask is not None:
            enc_key_padding = (attention_mask == 0)

        # Fast path: Greedy Search (num_beams == 1)
        if num_beams <= 1:
            generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
            is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for step in range(max_length):
                curr_seq_len = generated.size(1)
                target_pos_ids = self.position_ids[:, :curr_seq_len]
                char_embeds = self.target_char_embeddings(generated)
                pos_embeds = self.position_embeddings(target_pos_ids)

                dec_hidden = self.embed_layer_norm(char_embeds + pos_embeds)
                causal_mask = self.causal_mask[:curr_seq_len, :curr_seq_len]

                for layer in self.decoder_layers:
                    dec_hidden = layer(
                        hidden_states=dec_hidden,
                        encoder_hidden_states=fused_hidden,
                        causal_mask=causal_mask,
                        encoder_key_padding_mask=enc_key_padding,
                    )

                logits = self.lm_head(dec_hidden[:, -1, :])  # (batch, char_vocab_size)

                if temperature <= 0.0 or temperature == 1.0:
                    next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)

                is_finished |= (next_tokens.squeeze(-1) == eos_token_id)
                generated = torch.cat([generated, next_tokens], dim=-1)

                if is_finished.all():
                    break

            return generated

        # Beam Search path (batch_size == 1 or loop per sample)
        final_batch_results: List[torch.Tensor] = []
        for b_idx in range(batch_size):
            sample_fused = fused_hidden[b_idx: b_idx + 1]  # (1, M, H)
            sample_padding = enc_key_padding[b_idx: b_idx + 1] if enc_key_padding is not None else None

            # Beam hypothesis container: (log_prob_score, token_tensor)
            beams = [(0.0, torch.tensor([[bos_token_id]], dtype=torch.long, device=device))]
            completed_beams: List[Tuple[float, torch.Tensor]] = []

            for step in range(max_length):
                candidates: List[Tuple[float, torch.Tensor]] = []

                for score, seq in beams:
                    if seq[0, -1].item() == eos_token_id:
                        norm_score = score / ((seq.size(1) ** length_penalty) or 1.0)
                        completed_beams.append((norm_score, seq))
                        continue

                    curr_len = seq.size(1)
                    target_pos_ids = self.position_ids[:, :curr_len]
                    char_embeds = self.target_char_embeddings(seq)
                    pos_embeds = self.position_embeddings(target_pos_ids)
                    dec_hidden = self.embed_layer_norm(char_embeds + pos_embeds)
                    causal_mask = self.causal_mask[:curr_len, :curr_len]

                    for layer in self.decoder_layers:
                        dec_hidden = layer(
                            hidden_states=dec_hidden,
                            encoder_hidden_states=sample_fused,
                            causal_mask=causal_mask,
                            encoder_key_padding_mask=sample_padding,
                        )

                    logits = self.lm_head(dec_hidden[:, -1, :])  # (1, char_vocab_size)
                    log_probs = F.log_softmax(logits, dim=-1)
                    topk_log_probs, topk_indices = torch.topk(log_probs, k=num_beams, dim=-1)

                    for k in range(num_beams):
                        cand_score = score + topk_log_probs[0, k].item()
                        cand_token = topk_indices[0, k].unsqueeze(0).unsqueeze(0)
                        cand_seq = torch.cat([seq, cand_token], dim=-1)
                        candidates.append((cand_score, cand_seq))

                if not candidates:
                    break

                # Sort and retain top `num_beams`
                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:num_beams]

                if len(completed_beams) >= num_beams:
                    break

            if completed_beams:
                completed_beams.sort(key=lambda x: x[0], reverse=True)
                best_seq = completed_beams[0][1]
            else:
                best_seq = beams[0][1]

            final_batch_results.append(best_seq)

        max_res_len = max(res.size(1) for res in final_batch_results)
        padded_batch = torch.full((batch_size, max_res_len), eos_token_id, dtype=torch.long, device=device)
        for b_idx, res in enumerate(final_batch_results):
            padded_batch[b_idx, :res.size(1)] = res[0]

        return padded_batch

    @classmethod
    def from_pretrained_charbert(
        cls,
        checkpoint_path_or_dir: Union[str, Path],
        config: Optional[SinhalaCharBERTConfig] = None,
        num_decoder_layers: int = 4,
        max_target_positions: int = 512,
        char_vocab_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> "SinhalaCharBERTSeq2SeqModel":
        """
        Loads pre-trained Sinhala-CharBERT encoder weights (embeddings, Bi-GRU, HI module, Transformer layers)
        from a pre-training checkpoint (e.g. 'checkpoints/sinhala_charbert/final_model').
        Initializes the causal Transformer Decoder layers randomly.
        """
        ckpt_path = Path(checkpoint_path_or_dir)
        weight_path = None
        if ckpt_path.is_file():
            weight_path = ckpt_path
        elif ckpt_path.is_dir():
            for candidate_name in ["pytorch_model.bin", "model.safetensors", "final_model/pytorch_model.bin"]:
                candidate = ckpt_path / candidate_name
                if candidate.exists():
                    weight_path = candidate
                    break

        if weight_path is None or not weight_path.exists():
            # Check if user passed directory containing pytorch_model.bin directly
            bin_candidate = ckpt_path / "pytorch_model.bin"
            if bin_candidate.exists():
                weight_path = bin_candidate
            else:
                raise FileNotFoundError(
                    f"Checkpoint file 'pytorch_model.bin' not found in path: '{ckpt_path}'. "
                    f"Please verify the directory exists and contains 'pytorch_model.bin'."
                )

        map_loc = device if device is not None else "cpu"
        state_dict = torch.load(weight_path, map_location=map_loc, weights_only=True)

        # Detect char_vocab_size from state_dict if not provided
        if char_vocab_size is None and "charbert.char_embeddings.char_embeddings.weight" in state_dict:
            char_vocab_size = state_dict["charbert.char_embeddings.char_embeddings.weight"].shape[0]

        if config is None:
            subword_vocab_size = 32000
            if "charbert.token_embeddings.word_embeddings.weight" in state_dict:
                subword_vocab_size = state_dict["charbert.token_embeddings.word_embeddings.weight"].shape[0]

            config = SinhalaCharBERTConfig(
                vocab_size=subword_vocab_size,
                char_vocab_size=char_vocab_size or 1500,
            )
        elif char_vocab_size is not None:
            config.char_vocab_size = char_vocab_size

        model = cls(
            config=config,
            num_decoder_layers=num_decoder_layers,
            max_target_positions=max_target_positions,
        )

        # Extract and map encoder weights: 'charbert.X' -> 'encoder.X' or 'encoder.X' -> 'encoder.X'
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("charbert."):
                mapped_key = "encoder." + k[len("charbert."):]
                encoder_state_dict[mapped_key] = v
            elif k.startswith("encoder."):
                encoder_state_dict[k] = v

        model.load_state_dict(encoder_state_dict, strict=False)
        print(f"Successfully transferred {len(encoder_state_dict)} pre-trained encoder weights into Seq2Seq model.")
        return model

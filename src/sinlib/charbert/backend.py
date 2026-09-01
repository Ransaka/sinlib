"""
High-level CharBERT neural correction backend for sinlib.

Loads a published Sinhala-CharBERT seq2seq checkpoint (HF Hub repository or
a local checkpoint directory containing ``pytorch_model.bin`` and
``char_vocab.json``) and exposes sentence / word level neural correction.

All heavy imports (torch, transformers) are performed lazily so that the
base sinlib package remains installable without them.
"""

import re
import warnings
from pathlib import Path
from typing import Optional

_DEFAULT_CHARBERT_REPO = "Ransaka/sinhala-charbert-seq2seq"
_DEFAULT_SUBWORD_TOKENIZER = "Ransaka/sinhala-bert-medium-v2"

# Latin script detection (Singlish code-mixed inputs)
_LATIN_RE = re.compile(r"[A-Za-z]")


class CharBERTBackend:
    """
    Neural correction backend backed by a Sinhala-CharBERT seq2seq checkpoint.

    Parameters
    ----------
    model_id : str, optional
        HF Hub repository id (default ``Ransaka/sinhala-charbert-seq2seq``)
        or a local path to a checkpoint directory containing
        ``pytorch_model.bin`` and ``char_vocab.json``.
    device : str, optional
        Torch device string (e.g. ``"mps"``, ``"cuda"``, ``"cpu"``).
        Defaults to cuda > mps > cpu.
    revision : str, optional
        Optional HF Hub revision to pin.
    num_beams : int, optional
        Beam width for sentence generation. Default ``4``.
    max_length : int, optional
        Maximum generated akshara sequence length. Default ``128``.
    subword_tokenizer : str, optional
        Subword tokenizer name/path. Default ``Ransaka/sinhala-bert-medium-v2``.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_CHARBERT_REPO,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        num_beams: int = 4,
        max_length: int = 128,
        subword_tokenizer: str = _DEFAULT_SUBWORD_TOKENIZER,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.num_beams = num_beams
        self.max_length = max_length
        self.subword_tokenizer_name = subword_tokenizer

        if device is None:
            device = self._auto_device()
        self.device = device

        self._model = None
        self._char_tokenizer = None
        self._alignment_engine = None

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    @property
    def is_available(self) -> bool:
        """Returns True when the backend model has been loaded successfully."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _resolve_local_dir(self) -> Optional[Path]:
        path = Path(self.model_id).expanduser()
        if path.is_dir() and (path / "pytorch_model.bin").exists():
            return path
        return None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        try:
            return self._load()
        except Exception as exc:
            warnings.warn(
                f"CharBERT backend could not be loaded from '{self.model_id}': {exc}. "
                "Falling back to statistical-only correction.",
                ImportWarning,
            )
            return False

    def _load(self) -> bool:
        import torch

        from huggingface_hub import hf_hub_download

        # 1. Locate weights + char vocabulary (local dir or HF Hub)
        local_dir = self._resolve_local_dir()
        weight_path: Path
        char_vocab_path: Optional[Path] = None

        if local_dir is not None:
            weight_path = local_dir / "pytorch_model.bin"
            candidate = local_dir / "char_vocab.json"
            char_vocab_path = candidate if candidate.exists() else None
        else:
            weight_path = Path(
                hf_hub_download(
                    repo_id=self.model_id,
                    filename="pytorch_model.bin",
                    revision=self.revision,
                )
            )
            try:
                char_vocab_path = Path(
                    hf_hub_download(
                        repo_id=self.model_id,
                        filename="char_vocab.json",
                        revision=self.revision,
                    )
                )
            except Exception:
                char_vocab_path = None

        if char_vocab_path is None:
            raise FileNotFoundError(
                "char_vocab.json not found alongside the CharBERT checkpoint. "
                "The character-channel vocabulary is required for decoding."
            )

        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        return self._build_model(state_dict, char_vocab_path)

    def _build_model(self, state_dict, char_vocab_path: Path) -> bool:
        from transformers import AutoTokenizer

        from sinlib.charbert.alignment import SequenceAlignmentEngine
        from sinlib.charbert.char_tokenization import SinhalaCharTokenizer
        from sinlib.charbert.config import SinhalaCharBERTConfig
        from sinlib.charbert.seq2seq_decoder import SinhalaCharBERTSeq2SeqModel

        char_tokenizer = SinhalaCharTokenizer.load(char_vocab_path)
        subword_tokenizer = AutoTokenizer.from_pretrained(self.subword_tokenizer_name)

        char_vocab_size = char_tokenizer.vocab_size
        for key in (
            "charbert.char_embeddings.char_embeddings.weight",
            "encoder.char_embeddings.char_embeddings.weight",
        ):
            if key in state_dict:
                char_vocab_size = state_dict[key].shape[0]
                break

        subword_vocab_size = subword_tokenizer.vocab_size
        for key in (
            "charbert.token_embeddings.word_embeddings.weight",
            "encoder.token_embeddings.word_embeddings.weight",
        ):
            if key in state_dict:
                subword_vocab_size = state_dict[key].shape[0]
                break

        # Infer architecture dimensions from checkpoint tensor shapes so that
        # any checkpoint variant loads without hard-coded assumptions.
        import re as _re

        def _shape(key):
            return tuple(state_dict[key].shape) if key in state_dict else None

        w_emb = (
            _shape("charbert.token_embeddings.word_embeddings.weight")
            or _shape("encoder.token_embeddings.word_embeddings.weight")
        )
        c_emb = (
            _shape("charbert.char_embeddings.char_embeddings.weight")
            or _shape("encoder.char_embeddings.char_embeddings.weight")
        )
        gru_ih = (
            _shape("charbert.char_encoder.gru.weight_ih_l0")
            or _shape("encoder.char_encoder.gru.weight_ih_l0")
        )
        ffn1 = None
        pos_emb = None
        dec_pos = None
        num_enc_layers = 0
        num_dec_layers = 0
        for key in state_dict:
            if ffn1 is None and key.endswith("decoder_layers.0.ffn_dense1.weight"):
                ffn1 = state_dict[key].shape
            if pos_emb is None and (
                key.endswith("token_embeddings.position_embeddings.weight")
            ):
                pos_emb = state_dict[key].shape
            if dec_pos is None and key == "position_embeddings.weight":
                dec_pos = state_dict[key].shape
            m = _re.search(r"encoder\.layers\.(\d+)\.", key)
            if m and "transformer_layer" in key:
                num_enc_layers = max(num_enc_layers, int(m.group(1)) + 1)
            m = _re.search(r"decoder_layers\.(\d+)\.", key)
            if m and key.endswith("in_proj_weight"):
                num_dec_layers = max(num_dec_layers, int(m.group(1)) + 1)

        config = SinhalaCharBERTConfig(
            vocab_size=subword_vocab_size,
            char_vocab_size=char_vocab_size,
            hidden_size=w_emb[1] if w_emb else 786,
            char_embedding_dim=c_emb[1] if c_emb else 128,
            char_gru_hidden_size=(gru_ih[0] // 3) if gru_ih else 393,
            num_hidden_layers=num_enc_layers or 6,
            num_attention_heads=6,
            intermediate_size=ffn1[0] if ffn1 else 1024,
            max_position_embeddings=pos_emb[0] if pos_emb else 256,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )

        # Build model and load weights (encoder + decoder in one state dict)
        model = SinhalaCharBERTSeq2SeqModel(
            config,
            num_decoder_layers=num_dec_layers or 4,
            max_target_positions=dec_pos[0] if dec_pos else 512,
        )
        missing, _unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            warnings.warn(
                f"CharBERT checkpoint is missing {len(missing)} weights; "
                "those modules fall back to random initialization.",
                RuntimeWarning,
            )
        model.to(self.device)
        model.eval()

        self._model = model
        self._char_tokenizer = char_tokenizer
        self._alignment_engine = SequenceAlignmentEngine(
            subword_tokenizer=subword_tokenizer, char_tokenizer=char_tokenizer
        )
        return True

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------
    def correct_sentence(
        self,
        text: str,
        num_beams: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> str:
        """
        Corrects a single sentence of Sinhala text using open-vocabulary
        seq2seq decoding over phonological akshara units.

        Returns the corrected text; raises RuntimeError if the backend
        could not be loaded.
        """
        import torch

        if not self._ensure_loaded():
            raise RuntimeError("CharBERT backend is not available.")

        text = text.strip()
        if not text:
            return text

        num_beams = num_beams or self.num_beams
        max_length = max_length or self.max_length
        # Cap generation to the decoder's positional capacity
        max_length = min(max_length, self._model.max_target_positions - 1)

        aligned = self._alignment_engine.align(text)
        device = next(self._model.parameters()).device

        generated = self._model.generate(
            input_ids=torch.tensor([aligned.input_ids], dtype=torch.long, device=device),
            char_input_ids=torch.tensor(
                [aligned.char_input_ids], dtype=torch.long, device=device
            ),
            start_char_idx=torch.tensor(
                [aligned.start_char_idx], dtype=torch.long, device=device
            ),
            end_char_idx=torch.tensor(
                [aligned.end_char_idx], dtype=torch.long, device=device
            ),
            attention_mask=torch.tensor(
                [aligned.attention_mask], dtype=torch.long, device=device
            ),
            char_attention_mask=torch.tensor(
                [aligned.char_attention_mask], dtype=torch.long, device=device
            ),
            max_length=max_length,
            bos_token_id=self._char_tokenizer.bos_token_id,
            eos_token_id=self._char_tokenizer.eos_token_id,
            num_beams=num_beams,
        )

        gen_ids = generated[0].tolist()
        decoded = self._char_tokenizer.decode(gen_ids, skip_special_tokens=True)
        return decoded.strip()

    def correct_word(self, word: str, **kwargs) -> str:
        """
        Corrects a single word (bounded, word-level denoising). Uses a
        shortened generation budget since word akshara sequences are short.
        """
        word = word.strip()
        if not word:
            return word
        max_length = kwargs.pop("max_length", None) or max(16, 4 * len(word) + 4)
        return self.correct_sentence(word, max_length=max_length, **kwargs)

    @staticmethod
    def has_structural_noise(text: str) -> bool:
        """
        Cheap heuristic detector for noise classes the statistical word-level
        pipeline cannot fix: Singlish/Latin code-mixing, ZWJ-damaged ligature
        clusters, and split/fusion whitespace artifacts.
        """
        if not text:
            return False

        # 1. Latin script (Singlish / code-switched tokens)
        if _LATIN_RE.search(text):
            return True

        # 2. ZWJ-damaged ligatures: virama directly followed by a rakaranshaya
        #    (්‍ර), yanshaya (්‍ය), or bandima (්‍ව) consonant without a
        #    zero-width joiner between them.
        if re.search(r"[\u0DCA](?![\u200D])[\u0DBB\u0DBA\u0DC0]", text):
            return True

        # 3. Whitespace between a consonant and its vowel sign (split artifact)
        if re.search(r"[\u0D85-\u0D9F]\s+[\u0DCF-\u0DDF\u0DCA]", text):
            return True

        return False

    def __repr__(self) -> str:
        return (
            f"CharBERTBackend(model_id='{self.model_id}', device='{self.device}', "
            f"num_beams={self.num_beams}, loaded={self.is_available})"
        )


__all__ = ["CharBERTBackend", "_DEFAULT_CHARBERT_REPO"]

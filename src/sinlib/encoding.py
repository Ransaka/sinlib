"""
BatchEncoding — a dict-like container for tokenizer outputs.

Mirrors the HuggingFace ``BatchEncoding`` interface so that downstream code
can access fields either by attribute (``enc.input_ids``) or by key
(``enc["input_ids"]``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional


class BatchEncoding:
    """
    A dict-like container that holds the output of the :class:`~sinlib.Tokenizer`.

    Supports attribute-style access (``enc.input_ids``), dict-style access
    (``enc["input_ids"]``), and iteration over keys.  Mirrors the
    ``BatchEncoding`` interface from HuggingFace ``transformers``.

    Parameters
    ----------
    data : dict
        A dictionary whose values are lists of integers.  Expected keys are
        ``"input_ids"`` and optionally ``"attention_mask"``.
    tensor_type : str, optional
        If set, converts lists to tensors. Supported values: ``"pt"`` (PyTorch),
        ``"tf"`` (TensorFlow), ``"np"`` (NumPy).

    Examples
    --------
    >>> enc = BatchEncoding({"input_ids": [3, 10, 11], "attention_mask": [1, 1, 1]})
    >>> enc.input_ids
    [3, 10, 11]
    >>> enc["attention_mask"]
    [1, 1, 1]
    >>> "input_ids" in enc
    True
    """

    def __init__(self, data: Dict[str, Any], tensor_type: Optional[str] = None) -> None:
        self._data: Dict[str, Any] = data
        if tensor_type is not None:
            self.convert_to_tensors(tensor_type)

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    @property
    def input_ids(self) -> Any:
        """
        Token IDs (list or tensor depending on return_tensors).
        """
        return self._data["input_ids"]

    @property
    def attention_mask(self) -> Any:
        """
        Attention mask (1 for real tokens, 0 for padding).
        """
        return self._data.get("attention_mask")

    # ------------------------------------------------------------------
    # Mapping protocol
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def keys(self):
        """Return the keys of the underlying data dictionary."""
        return self._data.keys()

    def values(self):
        """Return the values of the underlying data dictionary."""
        return self._data.values()

    def items(self):
        """Return ``(key, value)`` pairs of the underlying data dictionary."""
        return self._data.items()

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def convert_to_tensors(self, tensor_type: str) -> None:
        """
        Convert the internal input lists to tensors.
        """
        if tensor_type == "pt":
            import torch
            for key, val in self._data.items():
                if val is not None:
                    try:
                        self._data[key] = torch.tensor(val)
                    except ValueError as e:
                        raise ValueError(
                            f"Failed to convert {key} to PyTorch tensor. "
                            f"Ensure sequences are rectangular (uniform length) or padding is enabled. "
                            f"Error: {e}"
                        ) from e
        elif tensor_type == "tf":
            import tensorflow as tf
            for key, val in self._data.items():
                if val is not None:
                    try:
                        self._data[key] = tf.convert_to_tensor(val)
                    except (ValueError, TypeError) as e:
                        raise ValueError(
                            f"Failed to convert {key} to TensorFlow tensor. "
                            f"Ensure sequences are rectangular (uniform length) or padding is enabled. "
                            f"Error: {e}"
                        ) from e
        elif tensor_type == "np":
            import numpy as np
            for key, val in self._data.items():
                if val is not None:
                    try:
                        self._data[key] = np.array(val)
                    except ValueError as e:
                        raise ValueError(
                            f"Failed to convert {key} to NumPy array. "
                            f"Ensure sequences are rectangular (uniform length) or padding is enabled. "
                            f"Error: {e}"
                        ) from e
        else:
            raise ValueError(
                f"Unsupported tensor_type '{tensor_type}'. Use 'pt', 'tf', or 'np'."
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a plain Python dictionary.

        Returns
        -------
        dict
            A copy of the internal data as a plain ``dict``.

        Examples
        --------
        >>> enc = BatchEncoding({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]})
        >>> enc.to_dict()
        {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]}
        """
        return dict(self._data)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for key, val in self._data.items():
            if isinstance(val, list):
                parts.append(f"{key}={val}")
            else:
                parts.append(f"{key}={val!r}")
        return f"BatchEncoding({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BatchEncoding):
            return self._data == other._data
        if isinstance(other, dict):
            return self._data == other
        return NotImplemented

---
title: Sinhala Text Visualization
description: Example of Matplotlib Sinhala text visualization with sinlib.
---

`sinlib` provides a zero-configuration utility `setup_matplotlib()` to resolve and configure Sinhala text rendering in Matplotlib.

## Quickstart

```python
import matplotlib.pyplot as plt
from sinlib import setup_matplotlib

# Fetch and prepare font properties (Noto Sans Sinhala)
fp = setup_matplotlib()

# Create a plot with mixed English and Sinhala text
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_title("OCR Model Attention Visualizer", fontsize=14, fontproperties=fp)
ax.text(
    0.5, 0.5,
    "Predicted Text: 'ඇසළ පුර පසලොස්වක පොහෝ දිනය අදයි'",
    fontsize=16,
    ha="center",
    va="center",
    fontproperties=fp
)
ax.axis("off")
plt.savefig("attention_vis.png", dpi=200, bbox_inches="tight")
```

## Features

- **Mixed-Script Support:** Noto Sans Sinhala covers both Latin (ASCII/English) and Sinhala codepoints, so strings containing both languages render cleanly in a single `plt.text()` call.
- **Automatic Caching:** Downloads the font from Hugging Face Hub on first run and caches it locally via `huggingface_hub`.
- **No Global Mutation:** Returns a clean `FontProperties` instance instead of mutating Matplotlib's global `rcParams`.

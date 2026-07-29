import matplotlib.font_manager as fm
from huggingface_hub import hf_hub_download


def setup_matplotlib() -> fm.FontProperties:
    """Download Noto Sans Sinhala font and return a Matplotlib FontProperties object.

    This font supports both Latin (English) and Sinhala Unicode blocks natively,
    allowing mixed-script strings to be rendered cleanly in a single text element.

    Returns:
        matplotlib.font_manager.FontProperties bound to the cached Noto Sans Sinhala font.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from sinlib import setup_matplotlib
        >>> fp = setup_matplotlib()
        >>> plt.text(0.5, 0.5, "Target Token: 'සිංහල'", fontproperties=fp)
    """
    try:
        font_path = hf_hub_download(
            repo_id="Ransaka/sinlib",
            filename="fonts/NotoSansSinhala-Regular.ttf",
            repo_type="model",
        )
    except Exception as e:
        raise RuntimeError(
            "Could not download Noto Sans Sinhala font from Hugging Face Hub. "
            f"Error: {e}"
        )

    return fm.FontProperties(fname=font_path)

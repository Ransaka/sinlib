import os
import matplotlib.font_manager as fm
from sinlib import setup_matplotlib
from sinlib.utils import setup_matplotlib as utils_setup_matplotlib


def test_setup_matplotlib_import() -> None:
    assert setup_matplotlib is utils_setup_matplotlib


def test_setup_matplotlib_returns_font_properties() -> None:
    fp = setup_matplotlib()
    assert isinstance(fp, fm.FontProperties)
    assert fp.get_name() == "Noto Sans Sinhala"
    assert os.path.exists(fp.get_file())

"""
test_plots.py
notebooks/lib/plots.py: ısı haritası tablonun etiketlerini kullanır ve yalnız dolu hücrelere değer yazar. Ekrana bir şey
çizilmez (pyplot'suz Figure). Gerçek veri okunmaz.
"""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from lib import plots


def test_heatmap_writes_filled_cells_only():
    table = pd.DataFrame({"C": [1.234, np.nan], "D": [0.5, 2.0], "E": [np.nan, 3.0]}, index=["3 Serisi", "A4"])
    ax = Figure().add_subplot()
    plots.heatmap(ax, table, cmap="YlGnBu", digits=1)
    assert sorted(t.get_text() for t in ax.texts) == ["0.5", "1.2", "2.0", "3.0"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["C", "D", "E"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["3 Serisi", "A4"]

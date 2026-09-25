"""
test_tool_charts.py
EN: The explorer's numbers and figures on the made-up data: every figure builds for raw, silver and gold and says
    "Veri yok" on empty rows instead of failing, the KPIs are the medians of the rows, the city is the part after the
    last comma, and the damage matrix shows unknown panels in silver that gold reports as original.
TR: Gezginin sayıları ve figürleri uydurma veride: her figür ham, silver ve gold için kurulur ve boş satırda hata
    vermek yerine "Veri yok" der, KPI'lar satırların medyanlarıdır, il son virgülden sonraki kısımdır ve hasar matrisi
    silver'da bilinmeyen panelleri gösterir, gold bunları orijinal sayar.
"""
import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from internal_tool import charts as CH, sources as S  # noqa: E402


@pytest.mark.parametrize("source", ["raw", "silver", "gold"])
def test_every_figure_builds(tool_data, source):
    """EN: All figures, full and empty. / TR: Bütün figürler, dolu ve boş."""
    df, _ = S.load(source, tool_data)
    for fn in CH.FIGURES:
        assert isinstance(fn(df, source), go.Figure), fn.__name__
        empty = fn(df.iloc[:0], source)
        assert [a.text for a in empty.layout.annotations] == ["Veri yok"], fn.__name__


def test_kpis(tool_data):
    """EN: Medians of the rows; age = snapshot year − model year. / TR: Satırların medyanları; yaş = tarama − model yılı."""
    raw, _ = S.load("raw", tool_data)
    k = CH.kpis(raw, "raw")
    assert k["rows"] == 5 and k["ads"] == 3 and k["price"] == 1_225_000 and k["age"] == 2
    silver, _ = S.load("silver", tool_data)
    assert CH.kpis(silver, "silver") == {"rows": 4, "ads": 3, "price": 1_550_000, "km": 50_000, "age": 6}
    assert CH.kpis(silver.iloc[:0], "silver")["price"] is None


def test_city():
    """EN: Last comma part; empty → unknown. / TR: Son virgül parçası; boş → bilinmiyor."""
    assert CH.city("Göztepe Mh. Bağcılar, İstanbul") == "İstanbul" and CH.city(None) == CH.UNKNOWN


def test_damage_matrix(tool_data):
    """EN: Silver has unknown panels, gold does not; raw states as written. / TR: Silver'da bilinmeyen var, gold'da yok."""
    silver = CH.damage_matrix(S.load("silver", tool_data)[0], "silver")
    gold = CH.damage_matrix(S.load("gold", tool_data)[0], "gold")
    assert silver.loc["Tavan"].to_dict() == {CH.UNKNOWN: 1, "değişen": 1, "orijinal": 1, "boyalı": 1}
    assert CH.UNKNOWN not in gold.columns and gold.loc["Tavan", "orijinal"] == 2
    raw = CH.damage_matrix(S.load("raw", tool_data)[0], "raw")
    assert {k: v for k, v in raw.loc["Motor Kaputu"].items() if v} == {"Boyalı": 2, "Orjinal": 1, "Belirtilmemiş": 1,
                                                                       CH.UNKNOWN: 1}

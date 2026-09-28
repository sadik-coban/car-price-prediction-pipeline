"""
test_03_association.py
EN: Invariants of metrics/03_association.json (technical report §3 dependence): the Theil's U matrix is square,
    every value is a share in [0, 1] with a unit diagonal; the Spearman matrix is symmetric with a unit diagonal, and
    every flagged high pair is above 0.5 and equals its Spearman cell.
TR: metrics/03_association.json'un değişmezleri (teknik rapor §3 bağıntı): Theil's U matrisi kare, her değer
    [0, 1] aralığında bir pay, köşegeni 1; Spearman matrisi simetrik, köşegeni 1, işaretlenen her yüksek çift 0,5'in
    üzerinde ve Spearman hücresine eşit.
"""
import pytest


@pytest.fixture(scope="module")
def doc(metrics):
    """EN: The published file. / TR: Yayımlanan dosya."""
    return metrics("03_association")


def square(m, n):
    """EN: True if m is an n × n matrix. / TR: m n × n bir matrisse True."""
    return len(m) == n and all(len(r) == n for r in m)


def test_theils_is_a_share(doc):
    """EN: Square; every value in [0, 1]; U[i][i] = 1. / TR: Kare; her değer [0, 1]; U[i][i] = 1."""
    tm = doc["methodology"]["theils_matrix"]
    m = tm["matrix"]
    assert square(m, len(tm["labels"]))
    assert all(0 <= x <= 1 for r in m for x in r)
    assert all(m[i][i] == 1 for i in range(len(m)))


def test_numeric_correlation(doc):
    """
    EN: Spearman is symmetric with a unit diagonal in [-1, 1]; high pairs are the |ρ| > 0.5 cells, largest first.
    TR: Spearman simetrik, köşegeni 1, [-1, 1] aralığında; yüksek çiftler |ρ| > 0,5 hücreleri, büyükten küçüğe.
    """
    nc = doc["domain"]["numeric_correlation"]
    labels, m = nc["labels"], nc["spearman"]
    n = len(m)
    assert square(m, len(labels))
    assert all(m[i][i] == pytest.approx(1) for i in range(n))
    assert all(-1 <= x <= 1 for r in m for x in r)
    assert all(m[i][j] == pytest.approx(m[j][i], abs=1e-3) for i in range(n) for j in range(n))
    cells = {(labels[i], labels[j]) for i in range(n) for j in range(i + 1, n) if abs(m[i][j]) > 0.5}
    assert {(a, b) for a, b, _r in nc["high_pairs"]} == cells
    for a, b, r in nc["high_pairs"]:
        assert r == pytest.approx(m[labels.index(a)][labels.index(b)], abs=1e-3)
    assert [abs(r) for *_ab, r in nc["high_pairs"]] == sorted((abs(r) for *_ab, r in nc["high_pairs"]), reverse=True)

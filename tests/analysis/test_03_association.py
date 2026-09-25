"""
test_03_association.py
EN: Invariants of metrics/03_association.json (technical report §3 dependence): the four categorical matrices
    (Cramér's V, Theil's U and their permutation floors) share one label order and are square; Cramér's V is
    symmetric with a unit diagonal; every value is a share in [0, 1]; the numeric correlations are symmetric with a
    unit diagonal, and every flagged high pair is above 0.5 and equals its Pearson cell.
TR: metrics/03_association.json'un değişmezleri (teknik rapor §3 bağıntı): dört kategorik matris (Cramér's V,
    Theil's U ve permütasyon tabanları) tek etiket sırası paylaşıyor ve kare; Cramér's V simetrik, köşegeni 1; her
    değer [0, 1] aralığında bir pay; sayısal korelasyonlar simetrik, köşegeni 1, işaretlenen her yüksek çift 0,5'in
    üzerinde ve Pearson hücresine eşit.
"""
import pytest

MATRICES = ("cramers_matrix", "theils_matrix", "cramers_null", "theils_null")


@pytest.fixture(scope="module")
def doc(metrics):
    """EN: The published file. / TR: Yayımlanan dosya."""
    return metrics("03_association")


def square(m, n):
    """EN: True if m is an n × n matrix. / TR: m n × n bir matrisse True."""
    return len(m) == n and all(len(r) == n for r in m)


def test_matrices_share_labels(doc):
    """EN: Same labels, square matrices, P = 5 floors. / TR: Aynı etiketler, kare matrisler, P = 5 tabanlar."""
    met = doc["methodology"]
    labels = met["cramers_matrix"]["labels"]
    for k in MATRICES:
        assert met[k]["labels"] == labels and square(met[k]["matrix"], len(labels))
    assert met["cramers_null"]["P"] == met["theils_null"]["P"] == 5


def test_cramers_is_symmetric(doc):
    """EN: V[i][j] = V[j][i], V[i][i] = 1. / TR: V[i][j] = V[j][i], V[i][i] = 1."""
    m = doc["methodology"]["cramers_matrix"]["matrix"]
    n = len(m)
    assert all(m[i][i] == 1 for i in range(n))
    assert all(m[i][j] == pytest.approx(m[j][i], abs=1e-3) for i in range(n) for j in range(n))


def test_theils_is_a_share(doc):
    """EN: Every value in [0, 1]; U[i][i] = 1. / TR: Her değer [0, 1]; U[i][i] = 1."""
    met = doc["methodology"]
    for k in MATRICES:
        assert all(0 <= x <= 1 for r in met[k]["matrix"] for x in r), k
    m = met["theils_matrix"]["matrix"]
    assert all(m[i][i] == 1 for i in range(len(m)))


def test_numeric_correlation(doc):
    """
    EN: Pearson and Spearman are symmetric with a unit diagonal in [-1, 1]; high pairs are |r| > 0.5 Pearson cells.
    TR: Pearson ve Spearman simetrik, köşegeni 1, [-1, 1] aralığında; yüksek çiftler |r| > 0,5 Pearson hücreleri.
    """
    nc = doc["domain"]["numeric_correlation"]
    labels = nc["labels"]
    for k in ("pearson", "spearman"):
        m = nc[k]
        assert square(m, len(labels))
        assert all(m[i][i] == pytest.approx(1) for i in range(len(m)))
        assert all(-1 <= x <= 1 for r in m for x in r)
        assert all(m[i][j] == pytest.approx(m[j][i], abs=1e-3) for i in range(len(m)) for j in range(len(m)))
    for a, b, r in nc["high_pairs"]:
        assert abs(r) > 0.5 and r == pytest.approx(nc["pearson"][labels.index(a)][labels.index(b)], abs=1e-3)

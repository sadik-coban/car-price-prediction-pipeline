"""
test_02_missingness.py
EN: Invariants of metrics/02_missingness.json (technical report §1 feature tables and §2 missingness): every raw
    column is in exactly one class and the classes add up to the raw columns the table holds; the kept features
    match their missing list; the missing list is sorted, over 2% and made of dropped columns; the co-missing
    blocks hold together (≥ 90%, the script's stop rule); the unspecified columns are counted apart and their
    panel flags equal 01_unspecified_panels' flags; the kb/gb twins keep one side at most.
TR: metrics/02_missingness.json'un değişmezleri (teknik rapor §1 öznitelik tabloları ve §2 eksiklik): her ham
    kolon tam bir sınıfta ve sınıflar tablonun tuttuğu ham kolonlara toplanıyor; tutulan öznitelikler eksiklik
    listeleriyle eşleşiyor; eksik listesi sıralı, %2'nin üzerinde ve atılan kolonlardan; birlikte-eksik bloklar
    bir arada (≥ %90, betiğin durma kuralı); belirtilmemiş kolonlar ayrı sayılıyor ve panel bayrakları
    01_unspecified_panels'in bayraklarına eşit; kb/gb ikizlerinde en çok bir taraf tutuluyor.
"""
import pytest


@pytest.fixture(scope="module")
def met(metrics):
    """EN: The methodology section. / TR: Methodology bölümü."""
    return metrics("02_missingness")["methodology"]


def test_column_accounting(met, metrics):
    """
    EN: raw = in model + target + dropped = the table's columns minus id; dropped = the drop classes.
    TR: ham = modelde + hedef + atılan = tablonun kolonları eksi id; atılan = atılan sınıflar.
    """
    acct = met["column_accounting"]
    raw_cols = metrics("02_missingness")["error_drivers"]["raw_columns"]
    assert acct["raw"] == acct["in_model"] + acct["target"] + acct["dropped"] == raw_cols["raw"]
    assert raw_cols["table_columns"] == raw_cols["raw"] + 1 and acct["target"] == 1
    assert acct["dropped"] == sum(r[2] for r in met["feature_drop"])
    assert all(r[2] == len(r[3]) for r in met["feature_drop"])
    assert all(len(v) == 2 for v in acct["twin_reasons"].values())


def test_features(met, metrics):
    """
    EN: One missing share per kept feature; the count matches meta.n_features.
    TR: Tutulan her özniteliğin bir eksik payı var; sayı meta.n_features ile aynı.
    """
    assert metrics("02_missingness")["meta"]["n_features"] == len(met["feature_kept"]) == len(met["column_missing"])
    assert {c for c, _ in met["column_missing"]} == set(met["feature_kept"])
    assert all(0 <= p <= 100 for _, p in met["column_missing"])


def test_missing_list(met):
    """
    EN: The missing list is sorted, every share is over 2%, it repeats under systematic_missing and holds only
        dropped columns.
    TR: Eksik listesi sıralı, her pay %2'nin üzerinde, systematic_missing altında aynen tekrarlanıyor ve yalnız
        atılan kolonları tutuyor.
    """
    rates = [p for _, p in met["column_missing_all"]]
    assert rates == sorted(rates, reverse=True) and min(rates) > 2
    assert met["systematic_missing"]["column_missing_all"] == met["column_missing_all"]
    assert not {c for c, _ in met["column_missing_all"]} & set(met["feature_kept"])


def test_blocks(met):
    """
    EN: Every block is at least two columns missing together ≥ 90%, drawn from the missing list; its sample is
        its first eight columns.
    TR: Her blok en az iki kolon, ≥ %90 birlikte eksik, eksik listesinden; örneği ilk sekiz kolonu.
    """
    missing = {c for c, _ in met["column_missing_all"]}
    for g in met["systematic_missing"]["systematic_groups"]:
        assert g["n_columns"] == len(g["columns"]) >= 2
        assert g["co_missing_pct"] >= 90
        assert g["sample_columns"] == g["columns"][:8]
        assert set(g["columns"]) <= missing


def test_unspecified(met, metrics):
    """
    EN: The unspecified columns are counted apart from the missing list; their panel flags equal
        01_unspecified_panels' flag count; the heavy-damage share is the listed one.
    TR: Belirtilmemiş kolonlar eksik listesinden ayrı sayılıyor; panel bayrakları 01_unspecified_panels'in bayrak
        sayısına eşit; ağır hasar payı listedekiyle aynı.
    """
    u = met["systematic_missing"]["unspecified"]
    cols = dict(u["columns"])
    assert u["n_columns"] == len(cols)
    assert not set(cols) & {c for c, _ in met["column_missing_all"]}
    assert u["panel_flags"] == metrics("01_unspecified_panels")["error_drivers"]["unspecified"]["structure"]["flags"]
    assert u["panel_min_pct"] <= u["panel_max_pct"]
    assert u["heavy_damage_pct"] == cols["is_heavy_damaged"]


def test_twins(met):
    """
    EN: A kb/gb pair keeps its kb side, its gb side or neither; shares are percentages.
    TR: Bir kb/gb çifti kb tarafını, gb tarafını ya da hiçbirini tutuyor; paylar yüzde.
    """
    for t in met["kb_gb_twins"]:
        assert t["kept"] in (t["kb"], t["gb"], None)
        assert all(0 <= t[k] <= 100 for k in ("kb_missing_pct", "gb_missing_pct", "same_pct"))
        assert t["kb_unique"] >= 0 and t["gb_unique"] >= 0

"""
test_metric_renames.py
EN: The translator of tools/metric_renames.py on small made-up documents — the traps the real map has to get
    right: the same word with two meanings (alt/ust as candidates vs range bounds), enum values by row position,
    data-valued subtrees, names from analysis/lib that change only with the LIB group, group gating, key order and
    clashes — plus the real map (every metrics file has exactly one group; the map is valid against the P0 copy when
    it exists) and snapshot_metrics' --rename-exemption on temp folders.
TR: tools/metric_renames.py'nin çevirmeni küçük uydurma belgeler üzerinde — gerçek eşlemenin doğru yapması gereken
    tuzaklar: iki anlamlı aynı sözcük (aday olarak alt/ust ile aralık sınırı), satır konumuna göre değerler, veri
    değerli alt ağaçlar, yalnız LIB grubuyla değişen analysis/lib adları, grup kapısı, anahtar sırası ve çakışma —
    ayrıca gerçek eşleme (her metrik dosyasının tek grubu var; P0 kopyası varsa eşleme ona göre geçerli) ve
    snapshot_metrics'in --rename-exemption'ı geçici klasörlerde.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import metric_renames as MR  # noqa: E402
import snapshot_metrics as SM  # noqa: E402

MAP = {"keys": {"alt": "lower", "ust": "upper", "medyan": "median", "not": "note", "yol": "category",
                "harita": "series_map", "kova": "range", "lower": "lower"},
       "overrides": {"a.*.ornek.alt": "low", "a.*.ornek.ust": "up", "b.yol": "paths"},
       "keep": ["en"], "data_valued": ["a.data"],
       "values": {"a.*.secilen": {"ust": "upper", "orta": "mid"}, "a.rows[][1]": {"alt": "lower"},
                  "b.worst[][0]": {"harita": "series_map"}},
       "site_only": {}, "groups": {"G": {"prefixes": ["x"], "status": "todo"}}, "lib": ["harita"]}


def tr(doc, ok=True, lib=False):
    """EN: Translates doc with the made-up map. / TR: doc'u uydurma eşlemeyle çevirir."""
    return MR.translate(doc, MR.compile_map(json.loads(json.dumps(MAP))), lambda p: ok, lib)


def test_same_word_two_meanings():
    """
    EN: alt/ust are candidates (lower/upper) in one place and range bounds (low/up) under ornek.
    TR: alt/ust bir yerde aday (lower/upper), ornek altında aralık sınırı (low/up).
    """
    doc = {"a": {"cc": {"adaylar": {"alt": 1, "ust": 2}, "ornek": {"alt": 3, "ust": 4}}}}
    assert tr(doc) == {"a": {"cc": {"adaylar": {"lower": 1, "upper": 2}, "ornek": {"low": 3, "up": 4}}}}


def test_values_by_site_and_position():
    """
    EN: A value changes only at its site and row position; other strings stay.
    TR: Değer yalnız kendi yerinde ve satır konumunda değişir; öteki metinler kalır.
    """
    doc = {"a": {"cc": {"secilen": "ust", "other": "ust"}, "rows": [[1, "alt"], ["alt", "ust"]]}}
    assert tr(doc) == {"a": {"cc": {"secilen": "upper", "other": "ust"}, "rows": [[1, "lower"], ["alt", "ust"]]}}


def test_data_valued_and_keep():
    """EN: Keys under a data-valued root and kept keys stay. / TR: Veri değerli kök altı ve kalan anahtarlar kalır."""
    doc = {"a": {"data": {"alt": 1, "medyan": 2}, "en": {"not": "x"}}}
    assert tr(doc) == {"a": {"data": {"alt": 1, "medyan": 2}, "en": {"note": "x"}}}


def test_lib_names_only_with_lib():
    """
    EN: segment_rule's names (keys and values) change only when LIB is done; overrides still apply before.
    TR: segment_rule adları (anahtar ve değer) yalnız LIB bitince değişir; istisnalar önceden de uygulanır.
    """
    doc = {"b": {"yol": {"harita": 5}, "worst": [["harita", 1]]}}
    assert tr(doc) == {"b": {"paths": {"harita": 5}, "worst": [["harita", 1]]}}
    assert tr(doc, lib=True) == {"b": {"paths": {"series_map": 5}, "worst": [["series_map", 1]]}}


def test_group_gating():
    """EN: Nothing changes while the owner is not done. / TR: Sahibi bitmedikçe hiçbir şey değişmez."""
    doc = {"a": {"cc": {"secilen": "ust", "medyan": 1}}}
    assert tr(doc, ok=False) == doc


def test_order_kept_and_clash_stops():
    """EN: Key order is kept; two keys ending with one name stop. / TR: Sıra korunur; aynı ada düşen iki anahtar durur."""
    assert list(tr({"z": 1, "medyan": 2, "a": 3})) == ["z", "median", "a"]
    with pytest.raises(SystemExit):
        tr({"x": {"alt": 1, "lower": 2}})


def test_prose_quotes_follow_the_rename():
    """
    EN: A prose value at a "prose" site gets the quoted old path replaced; elsewhere prose stays.
    TR: "prose" yerindeki düzyazı değerde anılan eski yol değişir; başka yerde düzyazı kalır.
    """
    m = json.loads(json.dumps(MAP))
    m["prose"] = {"a.note_here": {"domain.kova": "domain.range"}}
    m = MR.compile_map(m)
    doc = {"a": {"note_here": "see domain.kova", "elsewhere": "see domain.kova"}}
    assert MR.translate(doc, m, lambda p: True, False) == {"a": {"note_here": "see domain.range",
                                                                 "elsewhere": "see domain.kova"}}


def test_translate_pattern_keeps_list_suffix():
    """EN: '[][3]' survives a pattern translation. / TR: '[][3]' kalıp çevirisinde kalır."""
    assert MR.translate_pattern("a.kova[][3]", MR.compile_map(json.loads(json.dumps(MAP)))) == "a.range[][3]"


def test_guard_helpers():
    """
    EN: A retired key and an old enum value are found in a done file; a stale dotted path is found in text.
    TR: Bitmiş dosyada emekli anahtar ve eski değer bulunur; metinde bayat noktalı yol bulunur.
    """
    m = MR.compile_map(json.loads(json.dumps(MAP)))
    retired = MR.retired_names(m, {"G"})
    assert {"alt", "medyan", "not"} <= retired and "lower" not in retired and "harita" not in retired
    assert MR.key_violations({"a": {"cc": {"medyan": 1}, "data": {"alt": 2}}}, m, retired) == ["a.cc.medyan"]
    assert MR.value_violations({"a": {"cc": {"secilen": "ust"}}}, m, {"G"}) == ["a.cc.secilen"]
    owners = {"domain.a": {"G"}}
    assert MR.stale_refs("see domain.a.medyan and domain.a.median", m, {"G"}, owners, retired) == ["domain.a.medyan"]
    assert MR.stale_refs("domain.a.medyan", m, set(), owners, retired) == []


def test_every_metrics_file_has_one_group():
    """EN: The real map assigns every metrics file to exactly one group. / TR: Her metrik dosyasının tek grubu var."""
    m = MR.load_map()
    for name in MR.metrics_docs(ROOT / "metrics"):
        assert MR.group_of(name, m) in m["groups"]


def test_real_map_is_valid():
    """
    EN: Every map entry matches the P0 metrics and the fully translated metrics still merge (skipped without P0).
    TR: Eşlemenin her girdisi P0 metrikleriyle eşleşir ve tamamen çevrilmiş metrikler birleşir (P0 yoksa atlanır).
    """
    if not MR.P0_DIR.exists():
        pytest.skip("no P0 copy in this clone | bu klonda P0 yok")
    assert MR.cmd_validate(MR.load_map(), MR.P0_DIR) == []


@pytest.fixture
def folders(tmp_path):
    """
    EN: A temp metrics folder whose key was already renamed, and a baseline folder with the old exemption.
    TR: Anahtarı çoktan yeniden adlandırılmış geçici metrik klasörü ve eski istisnalı referans klasörü.
    """
    metrics, base = tmp_path / "metrics", tmp_path / "base"
    metrics.mkdir()
    base.mkdir()
    (metrics / "x.json").write_text(json.dumps({"_meta": {}, "domain": {"setup": {"seconds": 1.5}}}), encoding="utf-8")
    (base / "exemptions.json").write_text(json.dumps({"exemptions": [
        {"file": "x", "path": "domain.ayar.sure_sn", "reason": {"en": "time", "tr": "süre"}}]}), encoding="utf-8")
    return metrics, base


def test_rename_exemption_moves_it(folders):
    """EN: The exemption moves to the new path with its reasons. / TR: İstisna gerekçeleriyle yeni yola taşınır."""
    metrics, base = folders
    assert SM.rename_exemption("x", "domain.ayar.sure_sn", "domain.setup.seconds", metrics, base) == \
        ("domain.ayar.sure_sn", "domain.setup.seconds")
    e = json.loads((base / "exemptions.json").read_text(encoding="utf-8"))["exemptions"][0]
    assert e["path"] == "domain.setup.seconds" and e["reason"]["tr"] == "süre"


def test_rename_exemption_refuses(folders):
    """
    EN: Refused for an unknown exemption or a new path that matches nothing.
    TR: Bilinmeyen istisnada ya da hiçbir şeye uymayan yeni yolda reddedilir.
    """
    metrics, base = folders
    with pytest.raises(ValueError):
        SM.rename_exemption("x", "domain.nope", "domain.setup.seconds", metrics, base)
    with pytest.raises(ValueError):
        SM.rename_exemption("x", "domain.ayar.sure_sn", "domain.setup.nothing", metrics, base)

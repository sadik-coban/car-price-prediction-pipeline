"""
test_metric_renames.py
EN: The helpers of tools/metric_renames.py that the name guard uses, on a small made-up map (retired names, old
    enum values, stale dotted paths, list suffixes in patterns), the real map (every metrics file has one group at
    most; none = a new file, always checked) and snapshot_metrics' --rename-exemption on temp folders. The
    translator's tests and the checks against the archived P0 copy went with them on 2026-09-27 (nothing from the
    archive reaches the live chain).
TR: tools/metric_renames.py'nin ad bekçisinin kullandığı yardımcıları küçük uydurma bir eşleme üzerinde (emekli
    adlar, eski değerler, bayat noktalı yollar, kalıplarda liste ekleri), gerçek eşleme (her metrik dosyasının en çok
    bir grubu var; hiç yoksa yeni dosyadır ve hep sınanır) ve snapshot_metrics'in --rename-exemption'ı geçici
    klasörlerde. Çevirmenin testleri ve arşivdeki P0 kopyasına karşı denetimler 2026-09-27'de onlarla birlikte
    kalktı (arşivden canlı zincire hiçbir şey girmez).
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
    """
    EN: The real map gives every metrics file one group at most (group_of stops on two); a file with none is new.
    TR: Gerçek eşleme her metrik dosyasına en çok bir grup verir (group_of ikide durur); hiç grubu yoksa yenidir.
    """
    m = MR.load_map()
    for name in MR.metrics_docs(ROOT / "metrics"):
        assert MR.group_of(name, m) in set(m["groups"]) | {MR.NEW_GROUP}


def test_new_file_is_checked():
    """
    EN: A metrics file no group owns (a script added later) is in the always-done "new" group.
    TR: Hiçbir grubun sahiplenmediği metrik dosyası (sonradan eklenen betik) her zaman bitmiş "new" grubunda.
    """
    m = MR.load_map()
    assert MR.group_of("11_price_history", m) == MR.NEW_GROUP
    assert MR.NEW_GROUP in MR.done_groups(m)


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

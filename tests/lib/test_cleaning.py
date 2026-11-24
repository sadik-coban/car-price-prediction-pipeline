"""
test_cleaning.py
notebooks/lib/cleaning.py: plaka türü, ham sayfalardan hasar listesi okuma ve bayrakların yeniden kurulması küçük,
elle kurulmuş verilerle; plaka ve hasar cevabı eşlemelerinin gözlenen değerler kaydıyla (db/observed_values.json)
iki yönlü uyumu; yeniden kurulan bayrak adlarının veritabanı kolunun bayrak listesiyle (db/gold_rules.json) aynı
olması. Gerçek veri okunmaz.
"""
import json

import pandas as pd
import pytest

from lib import cleaning, data

REGISTER = json.loads((data.ROOT / "db" / "observed_values.json").read_text(encoding="utf-8"))
MAPS = json.loads(data.DAMAGE_MAPPINGS.read_text(encoding="utf-8"))
PANELS, STATUSES = MAPS["panels"], MAPS["statuses"]


def test_plate_kind():
    plates = pd.Series([data.TR_PLATE, "Mavi plakalı", None, data.TR_PLATE])
    assert cleaning.plate_kind(plates).tolist() == ["tr", "blue", "empty", "tr"]


def test_plate_kind_stops_on_an_unseen_value():
    # durmanın kendisini sınar: "Yabancı plakalı" veride hiç görülmedi
    with pytest.raises(ValueError, match="Yabancı plakalı"):
        cleaning.plate_kind(pd.Series([data.TR_PLATE, "Yabancı plakalı"]))


def test_plate_kinds_both_ways():
    """(a) eşlemedeki her plaka değeri veride geçiyor; (b) veride görülen her dolu plaka değeri eşlemede."""
    observed = {v for v, _ in REGISTER["silver"]["columns"]["gb_plate_origin"]["values"] if v is not None}
    assert set(cleaning.PLATE_KINDS) == observed


def test_damage_answers_both_ways():
    """Bayrak kuran cevaplar ile bayraksız iki cevap (orijinal, belirtilmemiş) birlikte gözlenen cevapların tamamı."""
    observed = {v for v, _ in REGISTER["raw"]["damage"]["states"]}
    assert set(cleaning.FLAG_OF_ANSWER) <= observed
    assert observed == set(cleaning.FLAG_OF_ANSWER) | {"Orjinal", "Belirtilmemiş"}
    assert observed == set(STATUSES)


def test_flag_prefix():
    assert cleaning.flag_prefix("roof_status") == "tavan"
    assert cleaning.flag_prefix("trunk_lid_status") == "bagaj"
    assert cleaning.flag_prefix("door_fl_status") == "door_fl"


def test_flags_from_statuses():
    statuses = pd.DataFrame({"engine_hood_status": [STATUSES["Değişmiş"], STATUSES["Belirtilmemiş"]],
                             "door_fl_status": [STATUSES["Lokal boyalı"], STATUSES["Orjinal"]]})
    flags = cleaning.flags_from_statuses(statuses, STATUSES)
    assert list(flags.columns) == ["kaput_degisen", "kaput_boyali", "kaput_lokal",
                                   "door_fl_degisen", "door_fl_boyali", "door_fl_lokal"]
    assert flags.iloc[0].tolist() == [1, 0, 0, 0, 0, 1]
    assert flags.iloc[1].tolist() == [0, 0, 0, 0, 0, 0]


def test_rebuilt_flags_are_the_database_flags():
    """13 parçadan kurulan 39 bayrağın adları, veritabanı kolunun parça bayrağı listesiyle aynı."""
    gold = json.loads(data.FILL_RULES.read_text(encoding="utf-8"))
    db_flags = next(rule["columns"] for rule in gold["fill"] if rule["name"] == "panel_flags")
    statuses = pd.DataFrame({column: [STATUSES["Orjinal"]] for column in PANELS.values()})
    assert sorted(cleaning.flags_from_statuses(statuses, STATUSES).columns) == sorted(db_flags)


def damage_list(**answers):
    """Sitenin biçiminde bir hasar listesi; verilmeyen parça "Orjinal"."""
    return [f"{panel}: {answers.get(panel, 'Orjinal')}" for panel in PANELS]


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """data/raw yerine geçici bir klasör; sayfalar sahte (ilan numaraları uydurma), etiketler gözlenen değerler."""
    monkeypatch.setattr(data, "RAW_DIR", tmp_path)

    def write(pages, brand="bmw", snapshot="2026-01-18_19-56"):
        folder = tmp_path / brand / snapshot
        folder.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(page, ensure_ascii=False) for page in pages]
        (folder / "details.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return write


def test_read_damage_lists(raw_dir):
    raw_dir([{"KısaBilgi - İlan No": "101", "search_date": "2026-01-18_19-56",
              "Hasar_Listesi": damage_list(Tavan="Belirtilmemiş")},
             {"KısaBilgi - İlan No": "102", "search_date": "2026-01-18_19-56"}])          # listesiz: atlanır
    raw_dir([{"KısaBilgi - İlan No": "103", "search_date": "2026-01-27_02-10",
              "Hasar_Listesi": damage_list(**{"Motor Kaputu": "Boyalı"})}], brand="audi", snapshot="2026-01-27_02-10")
    damage = cleaning.read_damage_lists(PANELS, STATUSES)
    assert sorted(damage["ad_id"]) == [101, 103]
    assert set(damage.columns) == {"ad_id", "snap", *PANELS.values()}
    row = damage.set_index("ad_id").loc[101]
    assert row["snap"] == "2026-01-18" and row["roof_status"] == STATUSES["Belirtilmemiş"]
    assert damage.set_index("ad_id").loc[103, "engine_hood_status"] == STATUSES["Boyalı"]


def test_read_damage_lists_stops_on_a_malformed_entry(raw_dir):
    # durmanın kendisini sınar: ":" içermeyen kayıt veride hiç görülmedi
    raw_dir([{"KısaBilgi - İlan No": "101", "search_date": "2026-01-18_19-56",
              "Hasar_Listesi": damage_list()[:-1] + ["Tavan"]}])
    with pytest.raises(ValueError, match="beklenmeyen"):
        cleaning.read_damage_lists(PANELS, STATUSES)


def test_read_damage_lists_stops_on_an_unknown_answer(raw_dir):
    # durmanın kendisini sınar: "Sökülmüş" cevabı veride hiç görülmedi
    raw_dir([{"KısaBilgi - İlan No": "101", "search_date": "2026-01-18_19-56",
              "Hasar_Listesi": damage_list(Tavan="Sökülmüş")}])
    with pytest.raises(KeyError, match="Sökülmüş"):
        cleaning.read_damage_lists(PANELS, STATUSES)

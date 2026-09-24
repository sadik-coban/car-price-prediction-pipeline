"""
test_build_duckdb.py
EN: Tests of db/build_duckdb.py — the column list, the semi-raw rules in the built DB (unknown = NULL),
    an end-to-end build from a small fake raw tree (two brands, two snapshots, every real engine / power
    format), and the safe build: a failure leaves the old DB byte for byte, a .wal or a locked DB stops before
    anything is touched, and incomplete raw data stops with the path. Everything is built in temp folders.
TR: db/build_duckdb.py testleri — kolon listesi, kurulan DB'deki yarı ham kurallar (bilinmeyen = NULL),
    küçük sahte bir ham ağaçtan uçtan uca kurulum (iki marka, iki tarama, gerçek her motor / güç biçimi) ve
    güvenli kurulum: düşen kurulum eski DB'yi bayt bayt bırakır, .wal ya da kilitli DB hiçbir şeye dokunmadan
    durdurur, eksik ham veri yolu söyleyerek durdurur. Her şey geçici klasörlerde kurulur.
"""
import hashlib

import duckdb
import pandas as pd
import pytest
from conftest import S1, S2, make_old_db, raw_record, small_tree, write_raw_tree

import build_duckdb as BD
from lib.process_for_db import DAMAGE_PART_MAP

def sha(path):
    """EN: sha256 of a file. / TR: Dosyanın sha256'sı."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """
    EN: One build of the small tree over an old DB with caches; returns (summary, rows by ad_id and snapshot, out).
    TR: Önbellekli eski bir DB'nin üzerine küçük ağacın tek kurulumu; (özet, ad_id ve taramaya göre satırlar, out).
    """
    root = tmp_path_factory.mktemp("build")
    out = root / "db" / "cars.duckdb"
    out.parent.mkdir()
    make_old_db(out)
    summary = BD.build(out, data_dir=small_tree(root / "raw"))
    con = duckdb.connect(str(out), read_only=True)
    df = con.execute("SELECT * FROM car_listings ORDER BY id").df()
    con.close()
    rows = {(int(r.ad_id), str(r.search_date)[:10]): r for r in df.itertuples()}
    return summary, rows, out


def query(out, sql):
    """EN: Runs sql on the built DB (read-only). / TR: sql'i kurulan DB'de (salt okunur) koşar."""
    con = duckdb.connect(str(out), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


# ---- schema | şema ----

def test_db_columns():
    """
    EN: 117 unique columns (+ id), legacy head, 39 flags in panel order, the new summary after the counts.
    TR: 117 tekil kolon (+ id), eski baş, panel sırasıyla 39 bayrak, yeni özet sayaçların arkasında.
    """
    names = [n for n, _, _ in BD.DB_COLUMNS]
    assert len(names) == 117 == len(set(names))
    assert names[:8] == ["ad_id", "listing_date", "ad_title", "brand", "series", "model", "location", "price"]
    assert "description_clean" not in names and names[-1] == "search_date"
    assert names.index("kb_paint_change_summary") == names.index("count_local_painted") + 1
    flags = [n for n in names if n.endswith(("_degisen", "_boyali", "_lokal"))]
    assert flags == [f"{p}_{s}" for p in BD.DAMAGE_PREFIX for s in BD.DAMAGE_SUFFIX] and len(flags) == 39


def test_damage_prefix_matches_mappings():
    """EN: Every panel of the JSON has a DB prefix. / TR: JSON'daki her parçanın bir DB öneki var."""
    assert sorted(BD.DAMAGE_PREFIX.values()) == sorted(DAMAGE_PART_MAP.values())
    assert BD.KNOWN_STATUSES == ("original", "painted", "local_painted", "changed")


def test_duplicate_report():
    """EN: Only ad_ids seen more than once, most frequent first. / TR: Yalnız birden çok görülen ad_id'ler."""
    df = pd.DataFrame({"ad_id": [1, 2, 2, 3, 3, 3]})
    assert BD.duplicate_report(df).values.tolist() == [[3, 3], [2, 2]]


# ---- end to end | uçtan uca ----

def test_rows_and_plates(built):
    """
    EN: Blue plates and empty pages are dropped; a listing without a plate field is kept.
    TR: Mavi plaka ve boş sayfa atılır; plaka alanı olmayan ilan tutulur.
    """
    summary, rows, _ = built
    assert summary["rows"] == 6 and sorted({a for a, _ in rows}) == [10000001, 10000002, 10000004, 10000005, 10000006]
    assert pd.isna(rows[(10000004, "2026-01-18")].gb_plate_origin)


@pytest.mark.parametrize("key, cc, hp", [
    ((10000001, "2026-01-18"), (1968, 1968, 1968, False), (150, 150, 150, False)),
    ((10000002, "2026-01-18"), (1401, 1600, 1500.5, True), (101, 125, 113, True)),
    ((10000005, "2026-01-27"), (None, 1200, 1200, True), (None, 50, 50, True)),
    ((10000006, "2026-01-18"), (1968, 1968, 1968, False), (601, None, 601, True)),
])
def test_engine_and_power_columns(built, key, cc, hp):
    """
    EN: Every real format lands in low / up / val as documented (open bucket: val = the known bound).
    TR: Gerçek her biçim anlatıldığı gibi low / up / val'e gider (açık kova: val = bilinen sınır).
    """
    r = built[1][key]

    def num(v):
        """EN: NaN / None → None. / TR: NaN / None → None."""
        return None if v is None or pd.isna(v) else v
    assert (num(r.engine_cc_low), num(r.engine_cc_up), num(r.engine_cc_val), bool(r.engine_cc_is_range)) == cc
    assert (num(r.power_hp_low), num(r.power_hp_up), num(r.power_hp_val), bool(r.power_hp_is_range)) == hp


@pytest.mark.parametrize("key, expected", [((10000001, "2026-01-18"), False), ((10000002, "2026-01-18"), True),
                                           ((10000004, "2026-01-18"), None), ((10000001, "2026-01-27"), None)])
def test_heavy_damage_semi_raw(built, key, expected):
    """
    EN: Hayır → False, Evet → True, missing / Belirtilmemiş → NULL, in both twin columns.
    TR: Hayır → False, Evet → True, yok / Belirtilmemiş → NULL, iki ikiz kolonda da.
    """
    r = built[1][key]
    for v in (r.is_heavy_damaged, r.kb_is_heavy_damaged):
        assert (None if pd.isna(v) else bool(v)) is expected


def test_panel_flags_semi_raw(built):
    """
    EN: "Belirtilmemiş" → all three flags NULL; a known status → 1 on its own flag, 0 on the others.
    TR: "Belirtilmemiş" → üç bayrak da NULL; bilinen durum → kendi bayrağı 1, ötekiler 0.
    """
    _, _, out = built
    got = query(out, "SELECT tavan_degisen, tavan_boyali, tavan_lokal, kaput_degisen, kaput_boyali, kaput_lokal, "
                     "door_fl_boyali, door_fr_lokal, bagaj_degisen FROM car_listings WHERE id = 1")[0]
    assert got == (None, None, None, 1, 0, 0, 1, 1, 0)


def test_first_owner_description_summary(built):
    """
    EN: Missing first owner → NULL; the description without the heading (heading only → NULL); raw summary.
    TR: İlk sahip yoksa NULL; başlıksız açıklama (yalnız başlık → NULL); ham özet.
    """
    rows = built[1]
    assert pd.isna(rows[(10000002, "2026-01-18")].gb_is_first_owner)
    assert bool(rows[(10000001, "2026-01-18")].gb_is_first_owner) is False
    assert rows[(10000001, "2026-01-18")].description_text == "Araç sorunsuzdur."
    assert pd.isna(rows[(10000006, "2026-01-18")].description_text)
    assert rows[(10000001, "2026-01-18")].kb_paint_change_summary == "Tamamı orjinal"


def test_duplicates_price_history_ids(built):
    """
    EN: The re-seen ad_id is reported; price_history carries the change; id is 1…N.
    TR: Yeniden görülen ad_id raporlanır; price_history değişimi taşır; id 1…N.
    """
    _, _, out = built
    assert query(out, "SELECT ad_id, occurrence_count FROM duplicate_ad_ids") == [(10000001, 2)]
    assert query(out, "SELECT price_delta, snapshot_idx FROM price_history WHERE ad_id = 10000001 "
                      "ORDER BY snapshot_idx") == [(None, 1), (-50000.0, 2)]
    assert query(out, "SELECT min(id), max(id), count(DISTINCT id) FROM car_listings") == [(1, 6, 6)]


def test_csv_next_to_db_and_caches(built):
    """EN: The CSV lands beside the DB; caches are carried over. / TR: CSV DB'nin yanında; önbellekler taşınır."""
    summary, _, out = built
    assert summary["csv"] == out.parent / "duplicate_ad_ids.csv"
    assert "10000001,2" in summary["csv"].read_text(encoding="utf-8-sig")
    assert summary["caches"] == {"dashboard_cache": 2, "options_cache": 1}
    assert query(out, "SELECT count(*) FROM dashboard_cache") == [(2,)]
    assert not list(out.parent.glob("*.tmp"))


# ---- safe build | güvenli kurulum ----

@pytest.fixture
def tree_and_db(tmp_path):
    """
    EN: A fresh raw tree and a finished DB built from it; returns (data_dir, out).
    TR: Taze bir ham ağaç ve ondan kurulmuş bir DB; (data_dir, out) döndürür.
    """
    data_dir = small_tree(tmp_path / "raw")
    out = tmp_path / "cars.duckdb"
    BD.build(out, data_dir=data_dir)
    return data_dir, out


def test_failure_leaves_old_db(tree_and_db, monkeypatch):
    """
    EN: A write that fails half-way leaves the old DB byte for byte and no .tmp behind.
    TR: Yarıda düşen yazma eski DB'yi bayt bayt bırakır, geride .tmp kalmaz.
    """
    data_dir, out = tree_and_db
    before = sha(out)

    def half_write(df, dups, path, caches):
        """EN: Starts the file, then fails. / TR: Dosyayı başlatır, sonra düşer."""
        path.write_bytes(b"half")
        raise RuntimeError("disk full")
    monkeypatch.setattr(BD, "write_duckdb", half_write)
    with pytest.raises(RuntimeError, match="disk full"):
        BD.build(out, data_dir=data_dir)
    assert sha(out) == before and not (out.parent / "cars.duckdb.tmp").exists()


def test_leftover_tmp_is_replaced(tree_and_db):
    """EN: A .tmp left by an earlier crash does not block a build. / TR: Önceki çöküşten kalan .tmp engel olmaz."""
    data_dir, out = tree_and_db
    (out.parent / "cars.duckdb.tmp").write_bytes(b"old junk")
    assert BD.build(out, data_dir=data_dir)["rows"] == 6
    assert not (out.parent / "cars.duckdb.tmp").exists()


def test_wal_stops_before_anything(tree_and_db, monkeypatch):
    """
    EN: A .wal next to the DB stops the build before the raw data is even read.
    TR: DB'nin yanında .wal varsa kurulum ham veri okunmadan durur.
    """
    data_dir, out = tree_and_db
    (out.parent / "cars.duckdb.wal").write_bytes(b"")
    before = sha(out)

    def no_read(*_a, **_k):
        """EN: Fails if called. / TR: Çağrılırsa düşer."""
        raise AssertionError("raw data read despite .wal | .wal'a rağmen ham veri okundu")
    monkeypatch.setattr(BD, "read_all_silver", no_read)
    with pytest.raises(RuntimeError, match=r"\.wal"):
        BD.build(out, data_dir=data_dir)
    assert sha(out) == before


def test_locked_db_stops(tree_and_db):
    """
    EN: A DB that another connection holds open stops the build with a clear message; it stays unchanged.
    TR: Başka bir bağlantının açık tuttuğu DB kurulumu anlaşılır bir mesajla durdurur; DB değişmez.
    """
    data_dir, out = tree_and_db
    before = sha(out)
    holder = duckdb.connect(str(out))
    try:
        with pytest.raises(RuntimeError, match="another program"):
            BD.build(out, data_dir=data_dir)
    finally:
        holder.close()
    assert sha(out) == before


def test_replace_blocked_stops(tree_and_db, monkeypatch):
    """
    EN: When the finished DB cannot be moved over the old one (the file is open elsewhere on Windows), the build
        stops with a clear message, the old DB stays and the .tmp is removed.
    TR: Biten DB eskisinin yerine taşınamazsa (Windows'ta dosya başka yerde açık), kurulum anlaşılır bir mesajla
        durur, eski DB kalır ve .tmp silinir.
    """
    data_dir, out = tree_and_db
    before = sha(out)

    def blocked(src, dst):
        """EN: Fails like a file in use. / TR: Kullanımdaki dosya gibi düşer."""
        raise PermissionError("in use")
    monkeypatch.setattr(BD.os, "replace", blocked)
    with pytest.raises(RuntimeError, match="another program"):
        BD.build(out, data_dir=data_dir)
    assert sha(out) == before and not (out.parent / "cars.duckdb.tmp").exists()


def test_no_foreign_plates(tmp_path):
    """EN: A tree without foreign plates keeps every row. / TR: Yabancı plakasız ağaçta her satır kalır."""
    data_dir = write_raw_tree(tmp_path / "raw", {("audi", S1): [raw_record(10000001)],
                                                 ("bmw", S1): [raw_record(10000002, brand="bmw")]})
    assert BD.build(tmp_path / "cars.duckdb", data_dir=data_dir)["rows"] == 2


def test_missing_brand_folder(tmp_path):
    """EN: A missing brand folder stops with its path. / TR: Eksik marka klasörü yoluyla durdurur."""
    data_dir = write_raw_tree(tmp_path / "raw", {("audi", S1): [raw_record()]})
    with pytest.raises(FileNotFoundError, match="bmw"):
        BD.build(tmp_path / "cars.duckdb", data_dir=data_dir)


def test_snapshot_without_details(tmp_path):
    """EN: A snapshot folder without details.jsonl stops. / TR: details.jsonl'suz tarama klasörü durdurur."""
    data_dir = write_raw_tree(tmp_path / "raw", {("audi", S1): [raw_record()], ("bmw", S1): [raw_record()]})
    (data_dir / "bmw" / S2).mkdir()
    with pytest.raises(FileNotFoundError, match=S2):
        BD.build(tmp_path / "cars.duckdb", data_dir=data_dir)


def test_broken_line_stops(tmp_path):
    """EN: An unreadable line stops with file and line. / TR: Okunamayan satır dosya ve satırla durdurur."""
    data_dir = write_raw_tree(tmp_path / "raw", {("audi", S1): [raw_record()], ("bmw", S1): [raw_record()]})
    with open(data_dir / "bmw" / S1 / "details.jsonl", "a", encoding="utf-8") as fh:
        fh.write("{bozuk\n")
    with pytest.raises(ValueError, match=r"details\.jsonl:2"):
        BD.build(tmp_path / "cars.duckdb", data_dir=data_dir)
    assert not (tmp_path / "cars.duckdb").exists()


def test_no_data_exits(tmp_path):
    """EN: Only empty pages → SystemExit, nothing written. / TR: Yalnız boş sayfalar → SystemExit, hiçbir şey yazılmaz."""
    empty = {"url": "https://www.arabam.com/ilan/bos", "search_date": S1}
    data_dir = write_raw_tree(tmp_path / "raw", {("audi", S1): [empty], ("bmw", S1): [empty]})
    with pytest.raises(SystemExit):
        BD.build(tmp_path / "cars.duckdb", data_dir=data_dir)
    assert not (tmp_path / "cars.duckdb").exists()


def test_main(tmp_path, monkeypatch, capsys):
    """
    EN: main passes --out to build; a failure returns 1 with the reason on stderr.
    TR: main --out'u build'e verir; hata 1 döndürür ve sebebi stderr'e yazar.
    """
    seen = []
    monkeypatch.setattr(BD, "build", lambda out: seen.append(out) or {})
    assert BD.main(["--out", str(tmp_path / "x.duckdb")]) == 0 and seen == [tmp_path / "x.duckdb"]

    def fail(out):
        """EN: Fails like a missing folder. / TR: Eksik klasör gibi düşer."""
        raise FileNotFoundError("brand folder missing")
    monkeypatch.setattr(BD, "build", fail)
    assert BD.main([]) == 1 and "FAILED: brand folder missing" in capsys.readouterr().err

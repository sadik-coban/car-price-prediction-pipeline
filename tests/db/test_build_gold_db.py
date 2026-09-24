"""
test_build_gold_db.py
EN: Tests of db/build_gold_db.py and gold_rules.json. A semi-raw DB is built from the small fake raw tree (the same
    one test_build_duckdb uses, with "Belirtilmemiş" panels and heavy damage) and turned into gold; then:
      - the rule file matches the DB columns (39 panel flags in build_duckdb's order, 3 + 3 other columns, one
        dropped column) and a broken rule file is refused;
      - every gold rule column has no NULL left, a NULL became false / 0 and a known value did not change;
      - every other column, the ids, the row order and the four other tables are cell for cell the semi-raw ones;
      - the schema is id + GOLD_COLUMNS with the semi-raw types (no description_clean, no kb_paint_change_summary);
      - the input check refuses an old-contract or a gold file; the safe write leaves an old gold DB byte for byte;
      - publish_data_to_s3 accepts the built gold file.
    Everything runs in temp folders.
TR: db/build_gold_db.py ve gold_rules.json testleri. Küçük sahte ham ağaçtan (test_build_duckdb'nin kullandığı,
    "Belirtilmemiş" panelli ve ağır hasarlı ağaç) yarı ham DB kurulur ve gold'a çevrilir; sonra:
      - kural dosyası DB kolonlarıyla uyuşuyor (build_duckdb sırasıyla 39 panel bayrağı, 3 + 3 öteki kolon, bir
        alınmayan kolon) ve bozuk kural dosyası reddediliyor;
      - her gold kural kolonunda NULL kalmıyor, NULL false / 0 oluyor, bilinen değer değişmiyor;
      - öteki her kolon, id'ler, satır sırası ve öteki dört tablo hücre hücre yarı hamdaki gibi;
      - şema id + GOLD_COLUMNS, yarı hamdaki tiplerle (description_clean yok, kb_paint_change_summary yok);
      - girdi denetimi eski sözleşmeli ya da gold bir dosyayı reddediyor; güvenli yazma eski gold DB'yi bayt bayt
        bırakıyor;
      - publish_data_to_s3 kurulan gold dosyayı kabul ediyor.
    Her şey geçici klasörlerde koşar.
"""
import hashlib
import json

import duckdb
import pandas as pd
import pytest
from conftest import make_old_db, small_tree

import build_duckdb as BD
import build_gold_db as G
import publish_data_to_s3 as P

DB_NAMES = [n for n, _, _ in BD.DB_COLUMNS]
FILL = G.RULES["fill"]
OTHER = [n for n, _ in G.GOLD_COLUMNS if n not in FILL]


def sha(path):
    """EN: sha256 of a file. / TR: Dosyanın sha256'sı."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame(path, table="car_listings", order="id"):
    """EN: A whole table of a DuckDB file (read-only). / TR: Bir DuckDB dosyasının bütün tablosu (salt okunur)."""
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(f"SELECT * FROM {table}" + (f" ORDER BY {order}" if order else "")).df()
    finally:
        con.close()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """
    EN: The semi-raw DB of the small tree (over an old DB with caches) and its gold DB; returns (summary, semi, gold).
    TR: Küçük ağacın yarı ham DB'si (önbellekli eski bir DB'nin üzerine) ve gold DB'si; (özet, yarı ham, gold).
    """
    root = tmp_path_factory.mktemp("gold")
    semi = root / "cars.duckdb"
    make_old_db(semi)
    BD.build(semi, data_dir=small_tree(root / "raw"))
    gold = root / "cars_gold.duckdb"
    return G.build(semi, gold), semi, gold


# ---- rules and schema | kurallar ve şema ----

def test_rules_file_matches_db_columns():
    """
    EN: 3 flags → false, the 39 panel flags in build_duckdb's order and 3 counters → 0, one dropped column; every
        rule has an en/tr reason.
    TR: 3 bayrak → false, build_duckdb sırasıyla 39 panel bayrağı ve 3 sayaç → 0, bir alınmayan kolon; her kuralın
        en/tr gerekçesi var.
    """
    groups = {g["name"]: g for g in G.RULES["groups"]}
    assert groups["heavy_damage_first_owner"]["columns"] == ["is_heavy_damaged", "kb_is_heavy_damaged",
                                                            "gb_is_first_owner"]
    assert groups["heavy_damage_first_owner"]["value"] is False
    assert groups["panel_flags"]["columns"] == [n for n, _, _ in BD.damage_flag_columns()]
    assert groups["damage_counts"]["columns"] == ["count_changed", "count_painted", "count_local_painted"]
    assert groups["panel_flags"]["value"] == groups["damage_counts"]["value"] == 0 and len(FILL) == 45
    assert G.RULES["drop"] == ["kb_paint_change_summary"]
    raw = json.loads(G.RULES_PATH.read_text(encoding="utf-8"))
    assert all(set(r["reason"]) == {"en", "tr"} for r in raw["fill"] + raw["drop"])


def test_gold_columns():
    """
    EN: 116 columns: DB_COLUMNS in order without kb_paint_change_summary, same types; no description_clean.
    TR: 116 kolon: kb_paint_change_summary hariç DB_COLUMNS sırasıyla, aynı tipler; description_clean yok.
    """
    assert [n for n, _ in G.GOLD_COLUMNS] == [n for n in DB_NAMES if n != "kb_paint_change_summary"]
    assert len(G.GOLD_COLUMNS) == 116
    types = {n: t for n, t, _ in BD.DB_COLUMNS}
    assert all(types[n] == t for n, t in G.GOLD_COLUMNS)
    assert "description_clean" not in dict(G.GOLD_COLUMNS) and "description_text" in dict(G.GOLD_COLUMNS)


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r["fill"][0]["columns"].append("no_such_column"), "not in DB_COLUMNS"),
    (lambda r: r["fill"][2]["columns"].append("is_heavy_damaged"), "two gold rules"),
    (lambda r: r["drop"].append({"column": "price", "reason": {}}) or r["fill"][2]["columns"].append("price"),
     "two gold rules|bad dropped"),
])
def test_broken_rules_refused(tmp_path, mutate, message):
    """EN: A rule file that does not fit the DB is refused. / TR: DB'ye uymayan kural dosyası reddedilir."""
    raw = json.loads(G.RULES_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    bad = tmp_path / "rules.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        G.load_rules(bad)


# ---- the gold rules on a real build | gerçek bir kurulumda gold kuralları ----

def test_schema_and_types(built):
    """
    EN: id + GOLD_COLUMNS with the semi-raw DB's types, rows in the same order.
    TR: id + GOLD_COLUMNS, yarı ham DB'nin tipleriyle; satırlar aynı sırada.
    """
    _, semi, gold = built
    con_g, con_s = duckdb.connect(str(gold), read_only=True), duckdb.connect(str(semi), read_only=True)
    try:
        g = con_g.execute("DESCRIBE car_listings").fetchall()
        s = {r[0]: r[1] for r in con_s.execute("DESCRIBE car_listings").fetchall()}
    finally:
        con_g.close()
        con_s.close()
    assert [r[0] for r in g] == ["id"] + [n for n, _ in G.GOLD_COLUMNS]
    assert all(r[1] == s[r[0]] for r in g)


def test_rule_columns_filled(built):
    """
    EN: No NULL left in a rule column; NULL became the rule value, a known value stayed. The small tree has
        "Belirtilmemiş" panels, unknown heavy damage and a missing first owner, so each group really fills cells.
    TR: Kural kolonunda NULL kalmadı; NULL kural değeri oldu, bilinen değer kaldı. Küçük ağaçta "Belirtilmemiş"
        panel, bilinmeyen ağır hasar ve olmayan ilk sahip var; her grup gerçekten hücre dolduruyor.
    """
    summary, semi, gold = built
    s, g = frame(semi), frame(gold)
    assert list(g["id"]) == list(s["id"])
    for col, value in FILL.items():
        was_null = s[col].isna()
        assert g[col].notna().all(), col
        assert (g.loc[was_null, col] == value).all(), col
        assert (g.loc[~was_null, col] == s.loc[~was_null, col]).all(), col
        assert summary["filled"][col] == int(was_null.sum()), col
    for group in G.RULES["groups"][:2]:
        assert sum(summary["filled"][c] for c in group["columns"]) > 0, group["name"]


def test_other_columns_identical(built):
    """
    EN: Every column outside the rules is cell for cell the semi-raw one (NULLs included).
    TR: Kural dışındaki her kolon hücre hücre yarı hamdaki gibi (NULL'lar dahil).
    """
    _, semi, gold = built
    pd.testing.assert_frame_equal(frame(gold)[["id"] + OTHER], frame(semi)[["id"] + OTHER])


@pytest.mark.parametrize("table, order", [("duplicate_ad_ids", "ad_id"), ("price_history", "ad_id, snapshot_idx"),
                                          ("dashboard_cache", "scope_brand"), ("options_cache", "scope_brand")])
def test_other_tables_copied(built, table, order):
    """EN: The four other tables are copied unchanged. / TR: Öteki dört tablo aynen kopyalanır."""
    _, semi, gold = built
    pd.testing.assert_frame_equal(frame(gold, table, order), frame(semi, table, order))


def test_publish_accepts_built_gold(built):
    """
    EN: The built gold file passes publish's validation; the semi-raw file it came from does not.
    TR: Kurulan gold dosya yayının doğrulamasından geçer; kaynağı olan yarı ham dosya geçmez.
    """
    _, semi, gold = built
    assert P.validate_duckdb(gold) == 6
    with pytest.raises(ValueError, match="not a gold DB"):
        P.validate_duckdb(semi)


# ---- input check | girdi denetimi ----

def test_old_contract_input_refused(tmp_path):
    """
    EN: An old-contract DB (description_clean, no kb_paint_change_summary) is not turned into gold.
    TR: Eski sözleşmeli DB (description_clean var, kb_paint_change_summary yok) gold'a çevrilmez.
    """
    old = tmp_path / "old.duckdb"
    cols = ["id BIGINT"] + [f"{n} {t}" for n, t, _ in BD.DB_COLUMNS if n != "kb_paint_change_summary"]
    con = duckdb.connect(str(old))
    con.execute(f"CREATE TABLE car_listings ({', '.join(cols)}, description_clean VARCHAR)")
    con.close()
    with pytest.raises(ValueError, match="not the semi-raw DB"):
        G.build(old, tmp_path / "gold.duckdb")
    assert not (tmp_path / "gold.duckdb").exists()


def test_gold_input_refused(built, tmp_path):
    """EN: A gold file is not taken as input again. / TR: Gold bir dosya yeniden girdi olarak alınmaz."""
    with pytest.raises(ValueError, match="not the semi-raw DB"):
        G.build(built[2], tmp_path / "again.duckdb")


def test_missing_or_same_file(built, tmp_path):
    """EN: A missing input or input == output stops. / TR: Olmayan girdi ya da girdi == çıktı durdurur."""
    with pytest.raises(FileNotFoundError, match="build_duckdb"):
        G.build(tmp_path / "none.duckdb", tmp_path / "gold.duckdb")
    with pytest.raises(ValueError, match="same file"):
        G.build(built[1], built[1])


# ---- safe write | güvenli yazma ----

@pytest.fixture
def semi_and_gold(built, tmp_path):
    """
    EN: A copy of the semi-raw DB and a finished gold DB built from it, in a fresh folder; returns (semi, gold).
    TR: Yarı ham DB'nin bir kopyası ve ondan kurulmuş bir gold DB, taze bir klasörde; (yarı ham, gold) döndürür.
    """
    semi = tmp_path / "cars.duckdb"
    semi.write_bytes(built[1].read_bytes())
    gold = tmp_path / "cars_gold.duckdb"
    G.build(semi, gold)
    return semi, gold


def test_failure_leaves_old_gold(semi_and_gold, monkeypatch):
    """
    EN: A write that fails half-way leaves the old gold DB byte for byte and no .tmp behind.
    TR: Yarıda düşen yazma eski gold DB'yi bayt bayt bırakır, geride .tmp kalmaz.
    """
    semi, gold = semi_and_gold
    before = sha(gold)

    def half_write(src, path):
        """EN: Starts the file, then fails. / TR: Dosyayı başlatır, sonra düşer."""
        path.write_bytes(b"half")
        raise RuntimeError("disk full")
    monkeypatch.setattr(G, "write_gold", half_write)
    with pytest.raises(RuntimeError, match="disk full"):
        G.build(semi, gold)
    assert sha(gold) == before and not gold.with_name("cars_gold.duckdb.tmp").exists()


def test_wal_stops_before_anything(semi_and_gold, monkeypatch):
    """EN: A .wal next to the gold DB stops before any write. / TR: Gold DB'nin yanında .wal varsa hiç yazmadan durur."""
    semi, gold = semi_and_gold
    gold.with_name("cars_gold.duckdb.wal").write_bytes(b"")
    before = sha(gold)

    def no_write(*_a):
        """EN: Fails if called. / TR: Çağrılırsa düşer."""
        raise AssertionError("written despite .wal | .wal'a rağmen yazıldı")
    monkeypatch.setattr(G, "write_gold", no_write)
    with pytest.raises(RuntimeError, match=r"\.wal"):
        G.build(semi, gold)
    assert sha(gold) == before


def test_leftover_tmp_is_replaced(semi_and_gold):
    """EN: A .tmp left by an earlier crash does not block a build. / TR: Önceki çöküşten kalan .tmp engel olmaz."""
    semi, gold = semi_and_gold
    gold.with_name("cars_gold.duckdb.tmp").write_bytes(b"old junk")
    assert G.build(semi, gold)["rows"] == 6
    assert not gold.with_name("cars_gold.duckdb.tmp").exists()


def test_replace_blocked_stops(semi_and_gold, monkeypatch):
    """
    EN: When the new gold DB cannot be moved over the old one (open elsewhere on Windows), the build stops with a
        clear message, the old gold DB stays and the .tmp is removed.
    TR: Yeni gold DB eskisinin yerine taşınamazsa (Windows'ta başka yerde açık), kurulum anlaşılır bir mesajla
        durur, eski gold DB kalır ve .tmp silinir.
    """
    semi, gold = semi_and_gold
    before = sha(gold)

    def blocked(src, dst):
        """EN: Fails like a file in use. / TR: Kullanımdaki dosya gibi düşer."""
        raise PermissionError("in use")
    monkeypatch.setattr(G.os, "replace", blocked)
    with pytest.raises(RuntimeError, match="another program"):
        G.build(semi, gold)
    assert sha(gold) == before and not gold.with_name("cars_gold.duckdb.tmp").exists()


def test_main(semi_and_gold, tmp_path, capsys):
    """EN: 0 on success, 1 with the reason on failure. / TR: Başarıda 0, hatada sebebiyle 1."""
    semi, _ = semi_and_gold
    assert G.main(["--in", str(semi), "--out", str(tmp_path / "g2.duckdb")]) == 0
    assert G.main(["--in", str(tmp_path / "none.duckdb"), "--out", str(tmp_path / "g3.duckdb")]) == 1
    assert "FAILED" in capsys.readouterr().err

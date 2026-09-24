"""
test_publish_data_to_s3.py
EN: Tests of db/publish_data_to_s3.py — validation, hashing, versioning, the upload order (database first,
    manifest after), the manifest's content, nothing uploaded on a bad file, and --dry-run never connecting to
    S3. Small DuckDB files are built in a temp folder; S3 is a fake store. No network, no .env.
TR: db/publish_data_to_s3.py testleri — doğrulama, hash, sürüm, yükleme sırası (önce veritabanı, sonra
    manifest), manifestin içeriği, bozuk dosyada hiçbir şey yüklenmemesi ve --dry-run'ın S3'e hiç bağlanmaması.
    Küçük DuckDB dosyaları geçici klasörde kurulur; S3 sahte bir depo. Ağ yok, .env yok.
"""
import hashlib
from datetime import datetime, timezone

import duckdb
import pytest
from conftest import FakeStore

import publish_data_to_s3 as P

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def test_validate_counts_rows(make_duckdb):
    """EN: A valid file gives its car_listings row count. / TR: Geçerli dosya car_listings satır sayısını verir."""
    assert P.validate_duckdb(make_duckdb(rows=3)) == 3


def test_validate_missing_table(make_duckdb):
    """EN: No car_listings → ValueError. / TR: car_listings yok → ValueError."""
    with pytest.raises(ValueError, match="car_listings"):
        P.validate_duckdb(make_duckdb(table="other_table"))


def test_validate_empty_table(make_duckdb):
    """EN: An empty car_listings → ValueError. / TR: Boş car_listings → ValueError."""
    with pytest.raises(ValueError, match="empty"):
        P.validate_duckdb(make_duckdb(rows=0))


def test_default_input_is_gold():
    """
    EN: The default file is the gold DB; the S3 object keeps the name the API polls.
    TR: Varsayılan dosya gold DB; S3 nesnesi API'nin yokladığı adı korur.
    """
    assert P.DEFAULT_DUCKDB.name == "cars_gold.duckdb" and P.DATA_KEY == "data/cars.duckdb"


def test_validate_refuses_non_gold_table(make_duckdb):
    """
    EN: A car_listings that is not the gold contract (here a bare ad_id + price table) is refused.
    TR: Gold sözleşmesinde olmayan bir car_listings (burada yalın ad_id + price tablosu) reddedilir.
    """
    with pytest.raises(ValueError, match="not a gold DB.*columns"):
        P.validate_duckdb(make_duckdb(gold=False))


def test_validate_refuses_null_in_rule_column(make_duckdb):
    """
    EN: A NULL left in a gold rule column (an unknown panel) is refused, so the API never gets it.
    TR: Bir gold kural kolonunda kalan NULL (bilinmeyen panel) reddedilir; API'ye asla gitmez.
    """
    path = make_duckdb(rows=3)
    con = duckdb.connect(str(path))
    con.execute("UPDATE car_listings SET tavan_boyali = NULL WHERE id = 2")
    con.close()
    with pytest.raises(ValueError, match="NULL left.*tavan_boyali"):
        P.validate_duckdb(path)


def test_validate_not_duckdb(tmp_path):
    """EN: A file that is not DuckDB → ValueError. / TR: DuckDB olmayan dosya → ValueError."""
    bad = tmp_path / "bad.duckdb"
    bad.write_bytes(b"this is not a database")
    with pytest.raises(ValueError, match="DuckDB"):
        P.validate_duckdb(bad)


def test_sha256_matches_hashlib(tmp_path):
    """EN: The hash equals hashlib's on the same bytes. / TR: Hash, aynı baytlarda hashlib'inkiyle aynı."""
    f = tmp_path / "blob.bin"
    data = bytes(range(256)) * 10_000
    f.write_bytes(data)
    assert P.file_sha256(f) == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("manifest, expected", [(None, 1), ({}, 1), ({"version": 4}, 5), ({"version": "7"}, 8),
                                                ({"version": "bozuk"}, 1), ({"version": None}, 1)])
def test_next_version(manifest, expected):
    """EN: S3 manifest + 1; missing or broken → 1. / TR: S3 manifest + 1; yoksa ya da bozuksa → 1."""
    assert P.next_version(manifest) == expected


def test_publish_uploads_database_before_manifest(make_duckdb):
    """
    EN: The order is read the current manifest → upload the database → write the manifest.
    TR: Sıra: güncel manifesti oku → veritabanını yükle → manifesti yaz.
    """
    store = FakeStore(manifest={"version": 4})
    P.publish(make_duckdb(rows=3), store, now=NOW)
    assert store.calls == [("read_json", P.MANIFEST_KEY), ("upload_file", P.DATA_KEY), ("put_json", P.MANIFEST_KEY)]


def test_publish_manifest_content(make_duckdb):
    """EN: The manifest carries version, sha256, time and rows. / TR: Manifest sürüm, sha256, zaman ve satır taşır."""
    path = make_duckdb(rows=3)
    store = FakeStore(manifest={"version": 4})
    result = P.publish(path, store, now=NOW)
    expected = {"version": 5, "sha256": P.file_sha256(path), "built_at": NOW.isoformat(), "car_listings_rows": 3}
    assert store.written[P.MANIFEST_KEY] == expected and result["manifest"] == expected
    assert result["key"] == "data/cars.duckdb"


def test_publish_explicit_version_skips_s3_read(make_duckdb):
    """EN: A given version is used and S3 is not asked. / TR: Verilen sürüm kullanılır, S3'e sorulmaz."""
    store = FakeStore(manifest={"version": 99})
    result = P.publish(make_duckdb(), store, version=12, now=NOW)
    assert result["manifest"]["version"] == 12 and ("read_json", P.MANIFEST_KEY) not in store.calls


def test_publish_bad_file_uploads_nothing(make_duckdb):
    """EN: Validation fails before any S3 call. / TR: Doğrulama S3'e hiç dokunmadan düşer."""
    store = FakeStore()
    with pytest.raises(ValueError):
        P.publish(make_duckdb(rows=0), store, now=NOW)
    assert store.calls == []


def test_publish_missing_file(tmp_path):
    """EN: A missing file stops with FileNotFoundError. / TR: Olmayan dosya FileNotFoundError ile durur."""
    store = FakeStore()
    with pytest.raises(FileNotFoundError):
        P.publish(tmp_path / "yok.duckdb", store)
    assert store.calls == []


def test_dry_run_never_connects(make_duckdb, capsys):
    """
    EN: --dry-run prints the manifest and never builds an S3 connection (connect would fail the test).
    TR: --dry-run manifesti basar ve S3 bağlantısı hiç kurmaz (connect çağrılırsa test düşer).
    """
    def no_connect():
        """EN: Fails if called. / TR: Çağrılırsa düşer."""
        raise AssertionError("dry-run connected to S3 | dry-run S3'e bağlandı")
    path = make_duckdb(rows=3)
    assert P.main(["--duckdb", str(path), "--dry-run"], connect=no_connect) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "rows=3" in out and P.file_sha256(path) in out


def test_main_publishes_through_connect(make_duckdb, capsys):
    """EN: Without --dry-run, main uploads through the store connect returns. / TR: --dry-run'sız main connect'in deposuyla yükler."""
    store = FakeStore(manifest=None)
    assert P.main(["--duckdb", str(make_duckdb(rows=2))], connect=lambda: store) == 0
    assert [c[0] for c in store.calls] == ["read_json", "upload_file", "put_json"]
    assert "s3://test-bucket/data/cars.duckdb" in capsys.readouterr().out


def test_main_reports_failure(tmp_path, capsys):
    """EN: A failure returns 1 and explains on stderr. / TR: Hata 1 döndürür ve stderr'de açıklar."""
    assert P.main(["--duckdb", str(tmp_path / "yok.duckdb")], connect=FakeStore) == 1
    assert "FAILED" in capsys.readouterr().err


def test_validate_opens_read_only(make_duckdb):
    """EN: Validation leaves the file unchanged. / TR: Doğrulama dosyayı değiştirmez."""
    path = make_duckdb(rows=3)
    before = P.file_sha256(path)
    P.validate_duckdb(path)
    assert P.file_sha256(path) == before
    duckdb.connect(str(path), read_only=True).close()

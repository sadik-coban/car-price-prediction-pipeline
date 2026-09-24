"""
test_s3_publish.py
EN: Tests of db/lib/s3_publish.py — settings from a temporary .env (never the real one), the environment
    winning over the file, no secret in repr or error messages, no side effect at import, and S3Store's three
    operations against a fake client. No network.
TR: db/lib/s3_publish.py testleri — geçici bir .env'den ayarlar (asla gerçeği değil), ortamın dosyaya göre
    öncelikli olması, repr'de ve hata mesajında sır olmaması, import'ta yan etki olmaması ve S3Store'un üç işlemi
    sahte istemciyle. Ağ yok.
"""
import importlib
import json
import os

import pytest
from botocore.exceptions import ClientError
from conftest import FakeS3Client

from lib import s3_publish

SECRET = "secret-value-123"


def write_env(tmp_path, **values):
    """
    EN: Writes a temporary .env with the given RAILWAY_S3_* values; returns its path.
    TR: Verilen RAILWAY_S3_* değerleriyle geçici bir .env yazar; yolunu döndürür.
    """
    path = tmp_path / ".env"
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
    return path


def test_load_settings_reads_env_file(tmp_path):
    """EN: All four settings come from the file. / TR: Dört ayar da dosyadan gelir."""
    env = write_env(tmp_path, RAILWAY_S3_ENDPOINT="http://s3.test", RAILWAY_S3_ACCESS_KEY="ak",
                    RAILWAY_S3_SECRET_KEY=SECRET, RAILWAY_S3_BUCKET="test-bucket")
    s = s3_publish.load_settings(env, environ={})
    assert (s.endpoint, s.access_key, s.secret_key, s.bucket) == ("http://s3.test", "ak", SECRET, "test-bucket")


def test_environment_wins_over_file(tmp_path):
    """EN: An environment variable overrides the .env value. / TR: Ortam değişkeni .env değerini geçer."""
    env = write_env(tmp_path, RAILWAY_S3_BUCKET="from-file")
    s = s3_publish.load_settings(env, environ={"RAILWAY_S3_BUCKET": "from-env"})
    assert s.bucket == "from-env"


def test_missing_file_counts_as_empty(tmp_path):
    """EN: A missing .env is fine if the environment has the values. / TR: .env yoksa ortam yeterli."""
    s = s3_publish.load_settings(tmp_path / "nope.env", environ={"RAILWAY_S3_BUCKET": "b"})
    assert s.bucket == "b" and s.endpoint is None


def test_missing_bucket_stops_without_leaking(tmp_path):
    """
    EN: No bucket → RuntimeError naming the variable; the secret is not in the message.
    TR: Bucket yok → değişkeni adlandıran RuntimeError; sır mesajda yok.
    """
    env = write_env(tmp_path, RAILWAY_S3_SECRET_KEY=SECRET)
    with pytest.raises(RuntimeError) as err:
        s3_publish.load_settings(env, environ={})
    assert "RAILWAY_S3_BUCKET" in str(err.value) and SECRET not in str(err.value)


def test_repr_hides_keys(tmp_path):
    """EN: Printing the settings shows no key. / TR: Ayarları basmak anahtar göstermez."""
    env = write_env(tmp_path, RAILWAY_S3_ACCESS_KEY="access-abc", RAILWAY_S3_SECRET_KEY=SECRET,
                    RAILWAY_S3_BUCKET="test-bucket")
    text = repr(s3_publish.load_settings(env, environ={}))
    assert SECRET not in text and "access-abc" not in text and "test-bucket" in text


def test_os_environ_is_not_modified(tmp_path, monkeypatch):
    """EN: Reading the .env does not export it. / TR: .env okumak onu ortama yazmaz."""
    for var in s3_publish.ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    env = write_env(tmp_path, RAILWAY_S3_BUCKET="test-bucket", RAILWAY_S3_SECRET_KEY=SECRET)
    s3_publish.load_settings(env)
    assert all(var not in os.environ for var in s3_publish.ENV_VARS.values())


def test_import_reads_nothing(monkeypatch):
    """EN: Importing the module does not read any .env. / TR: Modülü import etmek hiçbir .env okumaz."""
    import dotenv

    def boom(*_a, **_k):
        """EN: Fails if called. / TR: Çağrılırsa düşer."""
        raise AssertionError(".env read at import | import'ta .env okundu")
    monkeypatch.setattr(dotenv, "dotenv_values", boom)
    importlib.reload(s3_publish)


def test_make_client_uses_endpoint():
    """EN: The client targets the configured endpoint (no connection is made). / TR: İstemci ayarlı endpoint'e bakar."""
    s = s3_publish.S3Settings(endpoint="http://localhost:9", access_key="a", secret_key="b", bucket="x")
    assert s3_publish.make_client(s).meta.endpoint_url == "http://localhost:9"


def test_upload_file_goes_to_bucket_and_key(tmp_path):
    """EN: upload_file passes the path, bucket and key. / TR: upload_file yolu, bucket'ı ve anahtarı iletir."""
    client = FakeS3Client()
    s3_publish.S3Store(client, "test-bucket").upload_file(tmp_path / "f.bin", "data/f.bin")
    assert client.calls == [("upload_file", str(tmp_path / "f.bin"), "test-bucket", "data/f.bin")]


def test_put_json_writes_utf8():
    """EN: Non-ASCII text survives as UTF-8. / TR: ASCII dışı metin UTF-8 olarak bozulmadan kalır."""
    client = FakeS3Client()
    s3_publish.S3Store(client, "test-bucket").put_json("m.json", {"ülke": "Türkiye"})
    body = client.objects["m.json"]
    assert "Türkiye".encode("utf-8") in body and json.loads(body.decode("utf-8")) == {"ülke": "Türkiye"}


def test_read_json_round_trip():
    """EN: read_json returns what is stored. / TR: read_json saklananı döndürür."""
    client = FakeS3Client(objects={"m.json": b'{"version": 7}'})
    assert s3_publish.S3Store(client, "test-bucket").read_json("m.json") == {"version": 7}


@pytest.mark.parametrize("code", ["NoSuchKey", "404"])
def test_read_json_missing_is_none(code):
    """EN: A missing object reads as None. / TR: Olmayan nesne None olarak okunur."""
    assert s3_publish.S3Store(FakeS3Client(error_code=code), "test-bucket").read_json("m.json") is None


def test_read_json_other_errors_raise():
    """EN: Any other S3 error is not swallowed. / TR: Başka bir S3 hatası yutulmaz."""
    with pytest.raises(ClientError):
        s3_publish.S3Store(FakeS3Client(error_code="AccessDenied"), "test-bucket").read_json("m.json")

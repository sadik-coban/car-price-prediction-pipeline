"""
s3_publish.py
EN: How to talk to S3 — a small library, never run on its own (the scripts decide WHAT to upload; this module
    only knows HOW). Settings come from the RAILWAY_S3_* variables: an environment variable wins over the
    repo-root .env, which is read only when a connection is first needed (importing this module reads nothing).
    Secrets are never printed: the settings object hides the keys in its repr. The S3 client is passed into
    S3Store, so tests can use a fake one.
      settings = load_settings()                  RAILWAY_S3_* from the environment / repo-root .env
      store = S3Store(make_client(settings), settings.bucket)      or simply: store = connect()
      store.upload_file(path, key) · store.read_json(key) · store.put_json(key, obj)
TR: S3 ile nasıl konuşulur — tek başına koşulmayan küçük bir kütüphane (NEYİN yükleneceğine betikler karar verir;
    bu modül yalnız NASIL'ı bilir). Ayarlar RAILWAY_S3_* değişkenlerinden gelir: ortam değişkeni repo kökündeki
    .env'e göre önceliklidir; .env yalnız bağlantı ilk gerektiğinde okunur (bu modülü import etmek hiçbir şey
    okumaz). Sırlar asla basılmaz: ayar nesnesi repr'inde anahtarları gizler. S3 istemcisi S3Store'a dışarıdan
    verilir; testler sahte istemci kullanabilir.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import boto3
import dotenv
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[2]                 # db/lib -> repo root | depo kökü
ENV_FILE = ROOT / ".env"
# EN: settings field → environment variable | TR: ayar alanı → ortam değişkeni
ENV_VARS = {"endpoint": "RAILWAY_S3_ENDPOINT", "access_key": "RAILWAY_S3_ACCESS_KEY",
            "secret_key": "RAILWAY_S3_SECRET_KEY", "bucket": "RAILWAY_S3_BUCKET"}
NOT_FOUND_CODES = ("NoSuchKey", "404")


@dataclass(frozen=True)
class S3Settings:
    """
    EN: Where and how to connect. The two keys are left out of repr, so printing the settings shows no secret.
    TR: Nereye ve nasıl bağlanılacağı. İki anahtar repr'de yok; ayarları basmak hiçbir sır göstermez.
    """
    endpoint: str | None
    access_key: str | None = field(repr=False)
    secret_key: str | None = field(repr=False)
    bucket: str | None


def load_settings(env_file=ENV_FILE, environ=None):
    """
    EN: Reads RAILWAY_S3_* — an environment variable wins over env_file (the repo-root .env by default);
        os.environ is never modified. Stops if no bucket is set; the message names the variable, never a value.
        env_file: path of the .env to read (a missing file counts as empty); environ: mapping to read the
        environment from (os.environ by default). Returns: S3Settings.
    TR: RAILWAY_S3_* değişkenlerini okur — ortam değişkeni env_file'a (varsayılan repo kökündeki .env) göre
        önceliklidir; os.environ asla değiştirilmez. Bucket yoksa durur; mesaj değişkenin adını söyler, değeri
        asla. env_file: okunacak .env yolu (olmayan dosya boş sayılır); environ: ortamın okunacağı eşleme
        (varsayılan os.environ). Döndürür: S3Settings.
    """
    environ = os.environ if environ is None else environ
    from_file = dotenv.dotenv_values(env_file) if Path(env_file).exists() else {}
    values = {name: environ[var] if var in environ else from_file.get(var) for name, var in ENV_VARS.items()}
    if not values["bucket"]:
        raise RuntimeError(f"{ENV_VARS['bucket']} is not set — put the RAILWAY_S3_* variables in the repo-root .env "
                           f"| ayarlı değil — RAILWAY_S3_* değişkenlerini repo kökündeki .env'e yazın")
    return S3Settings(**values)


def make_client(settings):
    """
    EN: A boto3 S3 client for the settings (creating it opens no connection).
    TR: Ayarlar için bir boto3 S3 istemcisi (kurmak bağlantı açmaz).
    """
    return boto3.client("s3", endpoint_url=settings.endpoint, aws_access_key_id=settings.access_key,
                        aws_secret_access_key=settings.secret_key)


class S3Store:
    """
    EN: The three operations the publish scripts need, on one bucket. The client is passed in (tests use a fake).
    TR: Yayın betiklerinin ihtiyaç duyduğu üç işlem, tek bir bucket üzerinde. İstemci dışarıdan verilir (testler
        sahte istemci kullanır).
    """

    def __init__(self, client, bucket):
        """
        EN: client: a boto3 S3 client (or a fake with the same methods); bucket: the bucket name.
        TR: client: bir boto3 S3 istemcisi (ya da aynı metotlara sahip sahtesi); bucket: bucket adı.
        """
        self.client, self.bucket = client, bucket

    def upload_file(self, local_path, key):
        """
        EN: Uploads a local file to key, overwriting what is there.
        TR: Yerel bir dosyayı key adına yükler; oradakinin üzerine yazar.
        """
        self.client.upload_file(str(local_path), self.bucket, key)

    def read_json(self, key):
        """
        EN: The JSON object at key, or None if there is no such object; any other error is raised.
        TR: key'deki JSON nesnesi; öyle bir nesne yoksa None. Başka her hata yükseltilir.
        """
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in NOT_FOUND_CODES:
                return None
            raise
        return json.loads(resp["Body"].read())

    def put_json(self, key, obj):
        """
        EN: Writes obj as UTF-8 JSON (non-ASCII kept as is, indent 2) to key, overwriting what is there.
        TR: obj'yi UTF-8 JSON olarak (ASCII dışı karakterler olduğu gibi, girinti 2) key'e yazar; üzerine yazar.
        """
        body = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body)


def connect(env_file=ENV_FILE):
    """
    EN: Shortcut: settings from the environment / env_file → client → S3Store. Returns: S3Store.
    TR: Kısa yol: ortam / env_file'dan ayarlar → istemci → S3Store. Döndürür: S3Store.
    """
    settings = load_settings(env_file)
    return S3Store(make_client(settings), settings.bucket)

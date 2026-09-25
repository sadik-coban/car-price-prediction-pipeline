"""
conftest.py
EN: Shared test helpers for db/: puts db/ on the import path, a fake S3 client (boto3's methods,
    in memory), a fake store (S3Store's methods, recording the calls), a factory for small DuckDB files, and
    fake raw arabam.com records written in the real formats (every value made up; no real listing, no ad_id
    from the data). Nothing here touches the network, the real .env or data/.
TR: db/ için ortak test yardımcıları: db/'yi import yoluna ekler, sahte bir S3 istemcisi (boto3'ün
    metotları, bellekte), sahte bir depo (S3Store'un metotları, çağrıları kaydeder), küçük DuckDB dosyaları
    kuran bir fabrika ve gerçek biçimlerle yazılmış sahte ham arabam.com kayıtları (her değer uydurma; gerçek
    ilan yok, veriden ad_id yok). Burada hiçbir şey ağa, gerçek .env'e ya da data/'ya dokunmaz.
"""
import io
import json
import sys
from pathlib import Path

import duckdb
import pytest
from botocore.exceptions import ClientError

DB_DIR = Path(__file__).resolve().parents[2] / "db"
sys.path.insert(0, str(DB_DIR))

# EN: the 13 panel labels of the site's damage diagram | TR: sitenin hasar şemasının 13 parça etiketi
PANELS = list(json.loads((DB_DIR / "lib" / "damage_mappings.json").read_text(encoding="utf-8"))["panels"])
MISSING = object()          # raw_record(key=MISSING) leaves the key out | anahtarı kayıttan çıkarır


def damage_list(**statuses):
    """
    EN: A Hasar_Listesi in the scraper's format ("Panel: Status"); every panel "Orjinal" unless given, e.g.
        damage_list(Tavan="Belirtilmemiş"). Keyword names use "_" for spaces ("Motor_Kaputu").
    TR: Scraper'ın biçiminde bir Hasar_Listesi ("Parça: Durum"); verilmeyen her parça "Orjinal". Ör.
        damage_list(Tavan="Belirtilmemiş"). Anahtar adlarında boşluk yerine "_" ("Motor_Kaputu").
    """
    given = {k.replace("_", " "): v for k, v in statuses.items()}
    return [f"{p}: {given.get(p, 'Orjinal')}" for p in PANELS]


def raw_record(ad_no=10000001, **overrides):
    """
    EN: One raw details.jsonl record with every key the parser reads, in the real formats (values made up).
        Change values with a dict of raw keys: raw_record(**{"KısaBilgi - Motor Hacmi": "1401 - 1600 cm3"});
        MISSING removes a key.
    TR: Ayrıştırıcının okuduğu her anahtarla tek ham details.jsonl kaydı, gerçek biçimlerle (değerler uydurma).
        Değiştirmek için ham anahtarlı dict verin: raw_record(**{"KısaBilgi - Motor Hacmi": "1401 - 1600 cm3"});
        MISSING anahtarı çıkarır.
    """
    rec = {
        "Fiyat": "1.250.000 TL",
        "Hasar_Listesi": damage_list(),
        "KısaBilgi - İlan No": f"Kopyalandı\n\r\n                                                    {ad_no}",
        "brand": "audi",
        "KısaBilgi - Seri": "A4",
        "KısaBilgi - Model": "A4 Sedan 2.0 TDI",
        "Ilan_Basligi": "Deneme ilanı",
        "url": f"https://www.arabam.com/ilan/deneme/{ad_no}",
        "EIDS_Model": None,
        "Konum": "Deneme",
        "KısaBilgi - İlan Tarihi": "26 Kasım 2025",
        "scraped_at": "2026-01-18T16:59:10.760527+00:00",
        "search_date": "2026-01-18_19-56",
        "KısaBilgi - Yıl": "2015", "Genel Bakış - Yıl": "2015",
        "KısaBilgi - Kilometre": "120.000 km", "Genel Bakış - Kilometre": "120.000 km",
        "KısaBilgi - Vites Tipi": "Otomatik", "Genel Bakış - Vites Tipi": "Otomatik",
        "KısaBilgi - Yakıt Tipi": "Dizel", "Genel Bakış - Yakıt Tipi": "Dizel",
        "KısaBilgi - Kasa Tipi": "Sedan", "Genel Bakış - Kasa Tipi": "Sedan",
        "KısaBilgi - Renk": "Beyaz", "Genel Bakış - Renk": "Beyaz",
        "KısaBilgi - Çekiş": "Önden Çekiş",
        "KısaBilgi - Araç Durumu": "İkinci El", "Genel Bakış - Araç Durumu": "İkinci El",
        "KısaBilgi - Kimden": "Galeriden", "Genel Bakış - Kimden": "Galeriden",
        "KısaBilgi - Takasa Uygun": "Takasa Uygun", "Genel Bakış - Takasa Uygun": "Takasa Uygun",
        "Genel Bakış - Şanzıman": "S-Tronic",
        "Genel Bakış - Garanti Durumu": "Garantisi Yok",
        "Genel Bakış - Araç Türü": "Bireysel",
        "Genel Bakış - Aracın ilk sahibiyim": "İlk Sahibi Değilim",
        "Genel Bakış - Sınıfı": "D Segment",
        "Genel Bakış - Plaka Uyruğu": "(TR) Türkiye",
        "KısaBilgi - Motor Hacmi": "1968 cc",
        "KısaBilgi - Motor Gücü": "150 hp",
        "Motor ve Performans - Tork": "320 nm",
        "Motor ve Performans - Silindir Sayısı": "4",
        "Motor ve Performans - Maksimum Hız": "220 km/s",
        "Motor ve Performans - Hızlanma (0-100)": "8,6 sn",
        "Motor ve Performans - Maksimum Güç": "4000 rpm",
        "Motor ve Performans - Minimum Güç": "3500 rpm",
        "KısaBilgi - Ort. Yakıt Tüketimi": "4,9 lt",
        "Yakıt Tüketimi - Ortalama Yakıt Tüketimi": "4,9 lt",
        "Yakıt Tüketimi - Şehir İçi Yakıt Tüketimi": "5,8 lt",
        "Yakıt Tüketimi - Şehir Dışı Yakıt Tüketimi": "4,3 lt",
        "KısaBilgi - Yakıt Deposu": "54 lt",
        "Boyut ve Kapasite - Uzunluk": "4726 mm",
        "Boyut ve Kapasite - Genişlik": "1842 mm",
        "Boyut ve Kapasite - Yükseklik": "1427 mm",
        "Boyut ve Kapasite - Ağırlık": "1505 kg",
        "Boyut ve Kapasite - Boş Ağırlığı": "1505 kg",
        "Boyut ve Kapasite - Bagaj Hacmi": "480 lt",
        "Boyut ve Kapasite - Aks Aralığı": "2820 mm",
        "Boyut ve Kapasite - Koltuk Sayısı": "5",
        "Boyut ve Kapasite - Ön Lastik": "225/50 R17",
        "Genel Bakış - Yıllık MTV": "1.198 TL",
        "Genel Bakış - Ortalama Kasko": "25.000 TL",
        "Genel Bakış - Ortalama Trafik Sigortası": "7.000 TL",
        "Genel Bakış - Üretim Yılı (İlk/Son)": "2015 - 2019",
        "KısaBilgi - Ağır Hasarlı": "Hayır",
        "Agir_Hasar": False,
        "Tramer_Tutari": None,
        "Degisen_Parca_Sayisi": 0,
        "Boyali_Parca_Sayisi": 0,
        "Lokal_Boyali_Parca_Sayisi": 0,
        "KısaBilgi - Boya-değişen": "Tamamı orjinal",
        "Aciklama_HTML": "<h5>Açıklama</h5><div><p>Araç sorunsuzdur.</p></div>",
    }
    rec.update(overrides)
    return {k: v for k, v in rec.items() if v is not MISSING}


def write_raw_tree(data_dir, snapshots):
    """
    EN: Writes data_dir/<brand>/<snapshot>/details.jsonl for {(brand, snapshot): [records]}; returns data_dir.
    TR: {(marka, tarama): [kayıtlar]} için data_dir/<marka>/<tarama>/details.jsonl yazar; data_dir'i döndürür.
    """
    for (brand, snapshot), records in snapshots.items():
        folder = Path(data_dir) / brand / snapshot
        folder.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(r, ensure_ascii=False) for r in records]
        (folder / "details.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(data_dir)


class FakeS3Client:
    """
    EN: In-memory stand-in for a boto3 S3 client; records every call. error_code makes get_object fail with it.
    TR: boto3 S3 istemcisinin bellekteki yerine geçeni; her çağrıyı kaydeder. error_code get_object'i o kodla düşürür.
    """

    def __init__(self, objects=None, error_code=None):
        """EN: objects: {key: bytes} already on "S3". / TR: objects: "S3"te zaten duran {anahtar: bayt}."""
        self.objects, self.error_code, self.calls = dict(objects or {}), error_code, []

    def upload_file(self, filename, bucket, key):
        """EN: Records the upload. / TR: Yüklemeyi kaydeder."""
        self.calls.append(("upload_file", filename, bucket, key))

    def put_object(self, Bucket, Key, Body):                                   # noqa: N803 (boto3 names)
        """EN: Stores Body under Key. / TR: Body'yi Key altında saklar."""
        self.calls.append(("put_object", Bucket, Key))
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):                                         # noqa: N803 (boto3 names)
        """EN: Returns the stored object or raises ClientError like S3. / TR: Saklananı döndürür ya da S3 gibi ClientError."""
        self.calls.append(("get_object", Bucket, Key))
        if self.error_code:
            raise ClientError({"Error": {"Code": self.error_code}}, "GetObject")
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}


class FakeStore:
    """
    EN: Stand-in for s3_publish.S3Store used by publish(); records the order of calls.
    TR: publish()'in kullandığı s3_publish.S3Store'un yerine geçen; çağrıların sırasını kaydeder.
    """
    bucket = "test-bucket"

    def __init__(self, manifest=None):
        """EN: manifest: what read_json returns (the manifest "on S3"). / TR: manifest: read_json'ın döndürdüğü."""
        self.manifest, self.calls, self.written = manifest, [], {}

    def upload_file(self, local_path, key):
        """EN: Records the upload. / TR: Yüklemeyi kaydeder."""
        self.calls.append(("upload_file", key))

    def read_json(self, key):
        """EN: Returns the preset manifest. / TR: Önceden verilen manifesti döndürür."""
        self.calls.append(("read_json", key))
        return self.manifest

    def put_json(self, key, obj):
        """EN: Keeps the written object. / TR: Yazılan nesneyi saklar."""
        self.calls.append(("put_json", key))
        self.written[key] = obj


@pytest.fixture
def make_duckdb(tmp_path):
    """
    EN: Factory: make_duckdb(rows=3, table="car_listings", gold=True) builds a small DuckDB file in a temp folder.
        gold=True: car_listings in the gold contract (id + GOLD_COLUMNS, every gold rule column filled, the other
        columns NULL except ad_id and price); gold=False: a bare (ad_id, price) table.
    TR: Fabrika: make_duckdb(rows=3, table="car_listings", gold=True) geçici klasörde küçük bir DuckDB dosyası kurar.
        gold=True: gold sözleşmesinde car_listings (id + GOLD_COLUMNS, her kural kolonu dolu, ad_id ve fiyat
        dışındaki öteki kolonlar NULL); gold=False: yalın bir (ad_id, price) tablosu.
    """
    from build_gold_db import GOLD_COLUMNS, RULES

    def _make(rows=3, table="car_listings", name="cars.duckdb", gold=True):
        """EN: Builds the file and returns its path. / TR: Dosyayı kurar ve yolunu döndürür."""
        path = tmp_path / name
        con = duckdb.connect(str(path))
        if gold:
            cols = [("id", "BIGINT")] + GOLD_COLUMNS
            con.execute(f"CREATE TABLE {table} ({', '.join(f'{n} {t}' for n, t in cols)})")
            filled = list(RULES["fill"])
            for i in range(rows):
                values = {"id": i + 1, "ad_id": i, "price": 1_000_000 + i, **RULES["fill"]}
                names = ["id", "ad_id", "price"] + filled
                con.execute(f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
                            [values[n] for n in names])
        else:
            con.execute(f"CREATE TABLE {table} (ad_id BIGINT, price DOUBLE)")
            for i in range(rows):
                con.execute(f"INSERT INTO {table} VALUES (?, ?)", [i, 1_000_000 + i])
        con.close()
        return path
    return _make


S1, S2 = "2026-01-18_19-56", "2026-01-27_02-10"
SECOND = {"search_date": S2, "scraped_at": "2026-01-26T23:14:02.146290+00:00"}


def small_tree(data_dir):
    """
    EN: The fake raw tree of the end-to-end test: 8 records, 6 kept (one blue plate, one empty page dropped).
    TR: Uçtan uca testin sahte ham ağacı: 8 kayıt, 6'sı tutulur (bir mavi plaka, bir boş sayfa atılır).
    """
    return write_raw_tree(data_dir, {
        ("audi", S1): [
            raw_record(10000001, Hasar_Listesi=damage_list(Tavan="Belirtilmemiş", Motor_Kaputu="Değişmiş",
                                                           Sol_Ön_Kapı="Boyalı", Sağ_Ön_Kapı="Lokal boyalı")),
            raw_record(10000002, **{"KısaBilgi - Motor Hacmi": "1401 - 1600 cm3", "KısaBilgi - Motor Gücü": "101 - 125 HP",
                                    "KısaBilgi - Ağır Hasarlı": "Evet", "Agir_Hasar": True,
                                    "Genel Bakış - Aracın ilk sahibiyim": "-"}),
            raw_record(10000003, **{"Genel Bakış - Plaka Uyruğu": "Mavi plakalı"}),
            raw_record(10000004, **{"Genel Bakış - Plaka Uyruğu": MISSING, "KısaBilgi - Ağır Hasarlı": MISSING}),
        ],
        ("audi", S2): [
            raw_record(10000001, Fiyat="1.200.000 TL", **SECOND, **{"KısaBilgi - Ağır Hasarlı": "Belirtilmemiş"}),
            raw_record(10000005, **SECOND, **{"KısaBilgi - Motor Hacmi": "1200 cm3' e kadar",
                                              "KısaBilgi - Motor Gücü": "50 HP'ye kadar"}),
        ],
        ("bmw", S1): [
            raw_record(10000006, brand="bmw", **{"KısaBilgi - Motor Gücü": "601 HP ve üzeri",
                                                 "Aciklama_HTML": "<h5>Açıklama</h5><div></div>"}),
            {"url": "https://www.arabam.com/ilan/bos", "search_date": S1},          # empty page | boş sayfa
        ],
    })


def make_old_db(path):
    """
    EN: A previous DB holding the two archived cache tables (to be carried over).
    TR: İki arşiv önbellek tablosunu taşıyan önceki bir DB (taşınmak üzere).
    """
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE dashboard_cache (scope_brand VARCHAR, payload VARCHAR)")
    con.execute("INSERT INTO dashboard_cache VALUES ('__ALL__', '{}'), ('audi', '{}')")
    con.execute("CREATE TABLE options_cache (scope_brand VARCHAR, payload VARCHAR)")
    con.execute("INSERT INTO options_cache VALUES ('__ALL__', '{}')")
    con.close()

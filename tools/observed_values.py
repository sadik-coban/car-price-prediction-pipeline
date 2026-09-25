"""
observed_values.py (tools)
EN: Writes or checks the register of what the data holds, db/observed_values.json (see db/lib/observed_values.py):
    raw field values or formats, damage labels, silver columns with few values, series → models. Data is only read.
    Update the register only on purpose: after new data showed a value and the code was made to handle it (the DB
    build stops on anything not in the register; tests/db/test_observed_values.py checks the code against it).
    --check compares the file with the data (the generation stamp left out) and exits 1 on any difference.
TR: Verinin ne tuttuğunun kaydını, db/observed_values.json'u yazar ya da sınar (bkz. db/lib/observed_values.py): ham
    alan değerleri ya da biçimleri, hasar etiketleri, az değerli silver kolonları, seri → modeller. Veri yalnız okunur.
    Kaydı yalnız bilerek güncelleyin: yeni veri bir değer gösterdikten ve kod o değeri ele alacak hâle getirildikten
    sonra (DB kurulumu kayıtta olmayan her şeyde durur; tests/db/test_observed_values.py kodu ona göre sınar).
    --check dosyayı veriyle karşılaştırır (üretim damgası hariç) ve herhangi bir farkta 1 ile çıkar.
Run / Koşum:
    python tools/observed_values.py [--check]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "db"))
from lib import observed_values as OV  # noqa: E402


def main(argv=None):
    """EN: Command line. Returns: exit code. / TR: Komut satırı. Döndürür: çıkış kodu."""
    ap = argparse.ArgumentParser(description="Register of observed data values | gözlenen veri değerleri kaydı")
    ap.add_argument("--check", action="store_true", help="compare with the data, write nothing | karşılaştır, yazma")
    args = ap.parse_args(argv)
    fresh = OV.collect()
    if args.check:
        diff = OV.differences(OV.load(), fresh)
        print("register matches the data | kayıt veriyle aynı" if not diff
              else f"register differs from the data | kayıt veriden farklı: {', '.join(diff)}")
        return 1 if diff else 0
    OV.write(fresh)
    raw = fresh["raw"]
    print(f"written | yazıldı: {OV.REGISTRY_PATH.relative_to(ROOT)} · {raw['records']:,} raw records · "
          f"{len(raw['fields'])} fields · {fresh['silver']['rows']:,} silver rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())

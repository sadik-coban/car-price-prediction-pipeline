"""
new_analysis.py
EN: Scaffolds a new analysis so that it cannot be born without a test and a card:
      analysis/<name>.py                  the three-layer template (pure functions → to_metrics → load / compute /
                                          save cells), bilingual docstrings, # %% cells, save_metrics once;
      tests/analysis/test_<name>.py       a stub marked `template` that FAILS until a real invariant replaces it;
      analysis/cards/<name>.json          the card with the question filled in and status "todo".
    It prints the line to add to analysis/run_all.py ORDER (it does not edit run_all.py). Until the card, the test
    and ORDER are done, the coverage gate (tests/analysis/test_coverage.py) stays red. Existing files are never
    overwritten.
TR: Yeni bir analizi, testsiz ve kartsız doğamayacak şekilde kurar:
      analysis/<ad>.py                    üç katmanlı şablon (saf fonksiyonlar → to_metrics → yükle / hesapla /
                                          kaydet hücreleri), iki dilli docstring, # %% hücreleri, save_metrics bir kez;
      tests/analysis/test_<ad>.py         `template` işaretli, gerçek bir invariant yerini alana dek DÜŞEN taslak;
      analysis/cards/<ad>.json            sorusu dolu, durumu "todo" kart.
    analysis/run_all.py ORDER'a eklenecek satırı basar (run_all.py'yi düzenlemez). Kart, test ve ORDER bitene dek
    kapsam kapısı (tests/analysis/test_coverage.py) kırmızı kalır. Var olan dosyanın üzerine asla yazmaz.
Run / Koşum:
    python tools/new_analysis.py 11_price_history "İlan fiyatları taramalar arasında nasıl değişiyor?" "How do asking prices change between snapshots?"
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"^(shap/)?\d\d_[a-z0-9_]+$")

SCRIPT = '''"""
{file}
EN: Technical report §{sec} — {q_en}
    TODO: the data slice, the method, what is published.
TR: Teknik rapor §{sec} — {q_tr}
    TODO: veri kesiti, yöntem, ne yayımlanır.
Output / Çıktı: metrics/{name}.json
"""

# %% [1] Setup | Kurulum
from lib.common import load_clean, save_metrics


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def compute(listings):
    """
    EN: TODO — the analysis. Returns: raw values (no rounding here).
    TR: TODO — analiz. Döndürür: ham değerler (burada yuvarlama yok).
    """
    raise NotImplementedError("TODO: write the analysis | analizi yaz")


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published under report.<key> (TODO: choose the section and the key names). Keys and the values code compares
        are English snake_case; Turkish only in text the report shows ({{"tr": ..., "en": ...}}).
    TR: report.<anahtar> altında yayımlanır (TODO: bölümü ve anahtar adlarını seç). Anahtarlar ve kodun
        karşılaştırdığı değerler İngilizce snake_case; Türkçe yalnız raporda görünen metinde ({{"tr": ..., "en": ...}}).
    """
    return {{"report": {{"TODO": res}}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
res = compute(listings)

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("{name}", to_metrics(res)))
'''

TEST = '''"""
{file}
EN: Invariants of metrics/{name}.json. SCAFFOLD: replace the stub below with at least one real invariant (a range,
    a total, a count that must match another script, a claim the report makes); the stub fails on purpose.
TR: metrics/{name}.json'un değişmezleri. İSKELET: aşağıdaki taslağı en az bir gerçek invariantla değiştir (aralık,
    toplam, başka bir betikle tutması gereken sayım, raporun bir iddiası); taslak bilerek düşer.
"""
import pytest


@pytest.mark.template
def test_invariant_not_written_yet():
    """EN: Fails until a real invariant replaces it. / TR: Gerçek bir invariant yerini alana dek düşer."""
    pytest.fail("TODO: write a real invariant for metrics/{name}.json | gerçek bir invariant yaz")
'''


def card(name, q_tr, q_en):
    """
    EN: The card of a new analysis: the question filled in, everything else TODO, status "todo".
    TR: Yeni bir analizin kartı: soru dolu, geri kalan her şey TODO, durum "todo".
    """
    todo = {"tr": "TODO", "en": "TODO"}
    unknown = {"status": "unknown", "note": {"tr": "TODO", "en": "TODO"}}
    return {"script": f"analysis/{name}.py", "question": {"tr": q_tr, "en": q_en}, "data": todo, "method": todo,
            "split": "TODO", "leakage": {k: unknown for k in ("duplicate_ad_id", "target_in_features",
                                                             "fit_on_full_data", "temporal")},
            "evidence": {"tests": [], "report_keys": []}, "status": "todo"}


def targets(name, root):
    """EN: The three files of a new analysis under root. / TR: root altında yeni analizin üç dosyası."""
    return {"script": root / "analysis" / f"{name}.py",
            "test": root / "tests" / "analysis" / f"test_{name.replace('/', '_')}.py",
            "card": root / "analysis" / "cards" / f"{name}.json"}


def scaffold(name, q_tr, q_en, root=ROOT):
    """
    EN: Writes the three files. Raises ValueError on a bad name or when any of them exists.
        Returns: {"script", "test", "card"} paths.
    TR: Üç dosyayı yazar. Ad hatalıysa ya da biri varsa ValueError. Döndürür: {"script", "test", "card"} yolları.
    """
    if not NAME.match(name):
        raise ValueError(f"name must look like 11_price_history or shap/07_x | ad 11_price_history gibi olmalı: {name}")
    paths = targets(name, Path(root))
    existing = [p for p in paths.values() if p.exists()]
    if existing:
        raise ValueError(f"already exists, nothing written | zaten var, hiçbir şey yazılmadı: {existing}")
    sec = int(name.split("/")[-1][:2])
    fields = {"name": name, "sec": sec, "q_tr": q_tr, "q_en": q_en}
    for key, text in (("script", SCRIPT), ("test", TEST)):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths[key].write_text(text.format(file=paths[key].name, **fields), encoding="utf-8", newline="\n")
    paths["card"].parent.mkdir(parents=True, exist_ok=True)
    paths["card"].write_text(json.dumps(card(name, q_tr, q_en), ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8", newline="\n")
    return paths


def main(argv=None):
    """
    EN: Command line. Returns: 0 on success, 1 on a refused name / existing file.
    TR: Komut satırı. Döndürür: başarıda 0, reddedilen ad / var olan dosyada 1.
    """
    ap = argparse.ArgumentParser(description="Scaffold a new analysis | yeni analiz iskeleti")
    ap.add_argument("name", help="e.g. 11_price_history")
    ap.add_argument("question_tr")
    ap.add_argument("question_en")
    ap.add_argument("--root", default=str(ROOT), help="repository root (tests use a temp folder)")
    args = ap.parse_args(argv)
    try:
        paths = scaffold(args.name, args.question_tr, args.question_en, Path(args.root))
    except ValueError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    for key, p in paths.items():
        print(f"written | yazıldı ({key}): {p}")
    print(f'next | sonra: add "{args.name}.py" to ORDER in analysis/run_all.py (numeric order) | ORDER\'a ekle; '
          f"then write the analysis, a real test and the card; python tools/analysis_coverage.py shows what is left")
    return 0


if __name__ == "__main__":
    sys.exit(main())

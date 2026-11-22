"""
records.py
EN: The notebooks' side of the analysis scripts (notebooks/). It never writes a file.
      - run(name): a light script runs in the notebook. Its cells [1]–[5] execute: setup, functions, load (the data
        and the recorded artefacts such as the OOF predictions) and compute. The save cell [6] is under
        `if __name__ == "__main__"`, so it does not run. The analysis code is the script's own; the notebook holds no
        copy.
      - HEAVY: the scripts that train models or run cross-validation. A notebook reads their record (metrics/), and
        run() refuses them.
      - check(fresh, recorded): whether the result computed in the notebook equals the recorded metrics file.
      - card, flat, show, runs: a script's card (analysis/cards/<name>.json), the scalar leaves of a metrics section,
        a titled table, the run stamps of the metrics files.
    The notebooks load the recorded metrics with the builders' reader (builders/report_lib/metrics_view.py), so they
    pass the same consistency check as the reports: one database, one model run. Nothing at module level imports
    another lib module, so tests can load this file by its path (in a pytest session the name `lib` means db/lib).
TR: Analiz betiklerinin defter tarafı (notebooks/). Hiçbir dosya yazmaz.
      - run(name): hafif bir betik defterde koşar. [1]–[5] hücreleri çalışır: kurulum, fonksiyonlar, yükleme (veri ve
        OOF tahminleri gibi kayıtlı artefaktlar) ve hesap. Kaydet hücresi [6] `if __name__ == "__main__"` altında
        olduğu için çalışmaz. Analiz kodu betiğin kendisidir; defterde kopyası yok.
      - HEAVY: model eğiten ya da çapraz doğrulama koşan betikler. Defter onların kaydını (metrics/) okur, run() onları
        reddeder.
      - check(fresh, recorded): defterde hesaplanan sonucun kayıtlı metrik dosyasıyla aynı olup olmadığı.
      - card, flat, show, runs: bir betiğin kartı (analysis/cards/<ad>.json), bir metrik bölümünün skaler yaprakları,
        başlıklı tablo, metrik dosyalarının koşum damgaları.
    Defterler kayıtlı metrikleri derleyicilerin okuyucusuyla (builders/report_lib/metrics_view.py) yükler; böylece
    raporlarla aynı tutarlılık denetiminden geçer: tek veritabanı, tek model koşumu. Modül düzeyinde başka bir lib
    modülü import edilmez; testler bu dosyayı yolundan yükleyebilir (pytest oturumunda `lib` adı db/lib demek).
"""
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis"
CARDS_DIR = ANALYSIS / "cards"
# EN: Scripts that train models or run cross-validation, with the reason (Turkish, shown in the notebook). A notebook
#     reads their record; run() refuses them. Every other script guards its save cell (tests/repo/test_notebooks.py).
# TR: Model eğiten ya da çapraz doğrulama koşan betikler ve nedeni (Türkçe, defterde görünür). Defter onların kaydını
#     okur; run() onları reddeder. Diğer her betik kaydet hücresini korur (tests/repo/test_notebooks.py).
HEAVY = {"03_brand_ablation": "üç kolda 5-fold çapraz doğrulama",
         "07_lofo": "her öznitelik çıkarımı için 5-fold çapraz doğrulama (~15 dk)",
         "07_model_comparison": "LightGBM ve iki CatBoost ile 5-fold çapraz doğrulama; OOF kaydını yazar",
         "07_final_model": "servis modellerini bütün veriyle eğitir ve yazar",
         "09_backtest": "her tarama ve eğitim kümesi için model eğitimi",
         "10_free_text": "açıklama metniyle 5-fold çapraz doğrulama",
         "shap/02_oof_shap": "fold modellerini yeniden kurar ve SHAP hesaplar; oof_shap.npz'yi yazar",
         "shap/04_variants": "servis modelleriyle bütün veride SHAP"}
_ABSENT = object()
# EN: Turkish names of the card's four leakage types, keyed by the card's ids. An unknown id stops.
# TR: Kartın dört sızıntı tipinin Türkçe adları, kartın kimlikleriyle. Bilinmeyen kimlik durdurur.
LEAKAGE_TR = {"duplicate_ad_id": "aynı ilan iki tarafta", "target_in_features": "hedef öznitelikte",
              "fit_on_full_data": "bütün veriye uydurma", "temporal": "zamansal"}


def run(name):
    """
    EN: Runs a light script in the notebook: cells [1]–[5] execute, the guarded save cell [6] does not. The script
        reads the data and the recorded artefacts; nothing is trained or written. A heavy script stops with its
        reason. The notebook must have analysis/ on sys.path (the scripts import lib).
        Returns: the script as a module; res, the loaded frames and every function of the script are its attributes.
    TR: Hafif bir betiği defterde koşar: [1]–[5] hücreleri çalışır, korunan kaydet hücresi [6] çalışmaz. Betik veriyi
        ve kayıtlı artefaktları okur; hiçbir şey eğitilmez ya da yazılmaz. Ağır bir betik nedeniyle durur. Defterin
        sys.path'inde analysis/ olmalı (betikler lib'i import eder).
        Döndürür: modül olarak betik; res, yüklenen tablolar ve betiğin bütün fonksiyonları onun öznitelikleri.
    """
    if name in HEAVY:
        raise ValueError(f"{name}: heavy, read its record | ağır, kaydını okuyun: docs[{name!r}] — {HEAVY[name]}")
    path = ANALYSIS / f"{name}.py"
    if not path.exists():
        raise ValueError(f"no such script | böyle bir betik yok: {path.relative_to(ROOT).as_posix()}")
    spec = importlib.util.spec_from_file_location("script_" + name.replace("/", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recorded(name, docs):
    """
    EN: The record of a heavy script, with a line saying why the notebook does not run it.
        docs: metrics_view.load_metrics output. Returns: the metrics dict.
    TR: Ağır bir betiğin kaydı; defterin onu neden koşmadığını söyleyen bir satırla.
        docs: metrics_view.load_metrics çıktısı. Döndürür: metrik sözlüğü.
    """
    from IPython.display import Markdown, display
    display(Markdown(f"**Kayıttan okunuyor** (ağır betik: {HEAVY[name]})."))
    return docs[name]


def differences(fresh, recorded_doc):
    """
    EN: Where a result computed in the notebook differs from the recorded metrics file (_meta left out). The fresh
        values go through the same JSON conversion as save_metrics, so an unchanged script on unchanged data gives
        no row. Returns: DataFrame with the columns "yol", "defter", "kayıt" (empty = same as the record).
    TR: Defterde hesaplanan bir sonucun kayıtlı metrik dosyasından nerede farklı olduğu (_meta hariç). Taze değerler
        save_metrics'teki JSON dönüşümünden geçer; değişmemiş betik değişmemiş veride hiç satır vermez.
        Döndürür: "yol", "defter", "kayıt" sütunlu DataFrame (boş = kayıtla aynı).
    """
    from .common import _to_jsonable
    a = json.loads(json.dumps(_to_jsonable(fresh), ensure_ascii=False))
    b = {k: v for k, v in recorded_doc.items() if k != "_meta"}
    rows = []

    def short(x):
        """EN: A value cut to 80 characters for the table. / TR: Tablo için 80 karaktere kesilmiş değer."""
        return "(yok)" if x is _ABSENT else (s if len(s := json.dumps(x, ensure_ascii=False)) <= 80 else s[:77] + "...")

    def walk(x, y, path):
        """EN: Collects the differing leaves into rows. / TR: Farklı yaprakları rows'a toplar."""
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k, _ABSENT), y.get(k, _ABSENT), f"{path}.{k}" if path else k)
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
        elif x is _ABSENT or y is _ABSENT or x != y:
            rows.append({"yol": path, "defter": short(x), "kayıt": short(y)})

    walk(a, b, "")
    return pd.DataFrame(rows, columns=["yol", "defter", "kayıt"])


def check(fresh, recorded_doc):
    """
    EN: Shows whether the result computed in the notebook equals the recorded metrics file; if not, the differing
        leaves (differences). A difference means the script, its lib or the data changed after the record was written.
    TR: Defterde hesaplanan sonucun kayıtlı metrik dosyasıyla aynı olup olmadığını gösterir; değilse farklı
        yaprakları (differences). Fark, kayıt yazıldıktan sonra betiğin, lib'inin ya da verinin değiştiği anlamına gelir.
    """
    from IPython.display import Markdown, display
    d = differences(fresh, recorded_doc)
    if d.empty:
        display(Markdown("**Kayıtla aynı:** defterde hesaplanan sonuç `metrics/` kaydıyla birebir aynı."))
    else:
        display(Markdown(f"**Kayıttan farklı:** {len(d)} değer. Betik, lib ya da veri kayıttan sonra değişmiş."), d)


def card(name):
    """
    EN: A script's card as Markdown text: question, data, method, split and the leakage types with their notes
        (Turkish fields). name: "09_drift" or "shap/02_oof_shap". Returns: str.
    TR: Bir betiğin kartı, Markdown metni olarak: soru, veri, yöntem, bölme ve notlarıyla sızıntı tipleri (Türkçe
        alanlar). name: "09_drift" ya da "shap/02_oof_shap". Döndürür: str.
    """
    c = json.loads((CARDS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    lines = [f"**Soru:** {c['question']['tr']}", f"**Veri:** {c['data']['tr']}", f"**Yöntem:** {c['method']['tr']}",
             f"**Bölme:** `{c['split']}`", "**Sızıntı:**"]
    for kind, v in c["leakage"].items():
        note = f" — {v['note']['tr']}" if v.get("note") else ""
        lines.append(f"- {LEAKAGE_TR[kind]}: `{v['status']}`{note}")
    return "\n\n".join(lines[:5]) + "\n" + "\n".join(lines[5:])


def flat(section, max_list=12):
    """
    EN: The scalar leaves of a metrics section as a table, one row per dotted key. A short list of scalars is shown
        as one value; nested tables and long lists are left out (a notebook shows them with their own columns).
        Returns: DataFrame with the column "değer", indexed by the dotted key.
    TR: Bir metrik bölümünün skaler yaprakları, noktalı anahtar başına bir satırlık tablo olarak. Kısa bir skaler
        listesi tek değer olarak gösterilir; iç içe tablolar ve uzun listeler dışarıda kalır (defter onları kendi
        sütunlarıyla gösterir). Döndürür: "değer" sütunlu, noktalı anahtarla indekslenmiş DataFrame.
    """
    rows = {}

    def walk(o, path):
        """
        EN: Collects the scalar leaves under o into rows. / TR: o altındaki skaler yaprakları rows'a toplar.
        """
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(o, list):
            if len(o) <= max_list and all(not isinstance(x, (dict, list)) for x in o):
                rows[path] = ", ".join(str(x) for x in o)
        else:
            rows[path] = o

    walk(section, "")
    return pd.DataFrame({"değer": pd.Series(rows, dtype=object)})


def show(title, table):
    """
    EN: Shows a table under a bold title in a notebook (IPython is imported only here, so the scripts never need it).
    TR: Bir tabloyu defterde kalın bir başlığın altında gösterir (IPython yalnız burada import edilir; betikler ona
        hiç ihtiyaç duymaz).
    """
    from IPython.display import Markdown, display
    display(Markdown(f"**{title}**"), table)


def runs(docs, names):
    """
    EN: When each metrics file was written and from what: generated_at, data_until, the first 12 characters of the
        database fingerprint and the model run_id. docs: metrics_view.load_metrics output; names: the scripts a
        notebook shows. Returns: DataFrame indexed by script.
    TR: Her metrik dosyasının ne zaman ve neyden yazıldığı: generated_at, data_until, veritabanı parmak izinin ilk 12
        karakteri ve model run_id'si. docs: metrics_view.load_metrics çıktısı; names: defterin gösterdiği betikler.
        Döndürür: betikle indekslenmiş DataFrame.
    """
    return pd.DataFrame([{"betik": n, "yazıldı": docs[n]["_meta"]["generated_at"],
                          "veri sonu": docs[n]["_meta"]["data_until"],
                          "DB": docs[n]["_meta"]["db"]["sha256"][:12],
                          "run_id": docs[n]["_meta"].get("run_id")} for n in names]).set_index("betik")

"""
metrics_view.py
EN: The single reader the builders use. Loads every metrics/**/*.json, checks that they belong together — the
    same database fingerprint everywhere, and the same run_id on everything derived from the model (07's OOF
    artefact) — and deep-merges their sections into one view:
      meta · domain · methodology   the site tree (site_data.json is assembled from these)
      error_drivers                 report inputs (the keys of the former metrics/error_drivers.json)
      oof_shap · shap · shap_final · shap_case   SHAP report inputs
      report                        report-only numbers
    A key written by two scripts stops the build (no silent overwrite). No analysis here.
TR: Derleyicilerin kullandığı tek okuyucu. Bütün metrics/**/*.json dosyalarını okur, birbirine ait olduklarını
    sınar — her yerde aynı veritabanı parmak izi, modelden türeyen her şeyde aynı run_id (07'nin OOF artefaktı)
    — ve bölümlerini tek bir görünümde derin birleştirir:
      meta · domain · methodology   site ağacı (site_data.json bunlardan derlenir)
      error_drivers                 rapor girdileri (eski metrics/error_drivers.json'ın anahtarları)
      oof_shap · shap · shap_final · shap_case   SHAP raporu girdileri
      report                        yalnız raporun kullandığı sayılar
    İki betiğin yazdığı aynı anahtar derlemeyi durdurur (sessiz üzerine yazma yok). Burada analiz yok.
"""
import json
import math
from pathlib import Path

SECTIONS = ("meta", "domain", "methodology", "error_drivers", "oof_shap", "shap", "shap_final", "shap_case", "report")


def load_metrics(metrics_dir):
    """
    EN: Every metrics JSON under metrics_dir as {name: document}; name is the path without .json
        (e.g. "08_residuals", "shap/02_oof_shap"). Stops on a file without _meta (not written by save_metrics).
    TR: metrics_dir altındaki her metrik JSON'u {ad: belge} olarak; ad .json'suz yol ("08_residuals",
        "shap/02_oof_shap"). _meta'sı olmayan dosyada durur (save_metrics yazmamış).
    """
    docs = {}
    for f in sorted(Path(metrics_dir).rglob("*.json")):
        name = f.relative_to(metrics_dir).with_suffix("").as_posix()
        doc = json.loads(f.read_text(encoding="utf-8"))
        if "_meta" not in doc:
            raise SystemExit(f"{f} has no _meta — not written by analysis/common.save_metrics | _meta yok")
        docs[name] = doc
    if not docs:
        raise SystemExit(f"no metrics in {metrics_dir} — run | koşun: python analysis/run_all.py")
    return docs


def check_consistency(docs):
    """
    EN: Stops unless every file has the same database fingerprint and every model-derived file (run_id set)
        the same run_id; the message names the scripts to rerun.
    TR: Her dosyada aynı veritabanı parmak izi ve modelden türeyen her dosyada (run_id dolu) aynı run_id yoksa
        durur; mesaj yeniden koşulacak betikleri söyler.
    """
    dbs = {}
    for name, doc in docs.items():
        dbs.setdefault(doc["_meta"]["db"]["sha256"], []).append(name)
    if len(dbs) > 1:
        stale = sorted(min(dbs.values(), key=len))
        raise SystemExit(f"metrics come from different databases | farklı veritabanlarından — rerun | yeniden koşun: {stale}")
    runs = {}
    for name, doc in docs.items():
        if doc["_meta"].get("run_id"):
            runs.setdefault(doc["_meta"]["run_id"], []).append(name)
    if len(runs) > 1:
        stale = sorted(min(runs.values(), key=len))
        raise SystemExit(f"model-derived metrics come from different runs | farklı koşumlardan — rerun | yeniden koşun: {stale}")


def deep_merge(dst, src, where, owners):
    """
    EN: Merges src into dst recursively; stops if the same leaf key comes from two scripts.
    TR: src'yi dst'ye özyinelemeli birleştirir; aynı yaprak anahtar iki betikten gelirse durur.
    """
    for k, v in src.items():
        path = f"{where}.{k}"
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            deep_merge(dst[k], v, path, owners)
        elif k in dst:
            raise SystemExit(f"{path} is written by both {owners.get(path)} and {owners['_current']} | iki betik yazıyor")
        else:
            dst[k] = v
            owners[path] = owners["_current"]


def load_view(root):
    """
    EN: The merged view of all metrics under root/metrics, with _meta summarising the run (latest
        generated_at, data_until, database fingerprint, run_id).
    TR: root/metrics altındaki bütün metriklerin birleşik görünümü; _meta koşumu özetler (en yeni generated_at,
        data_until, veritabanı parmak izi, run_id).
    """
    docs = load_metrics(Path(root) / "metrics")
    check_consistency(docs)
    view, owners = {s: {} for s in SECTIONS}, {}
    for name, doc in docs.items():
        owners["_current"] = name
        unknown = [k for k in doc if k != "_meta" and k not in SECTIONS]
        if unknown:
            raise SystemExit(f"{name}: unknown section(s) | bilinmeyen bölüm: {unknown}")
        for sec in SECTIONS:
            if sec in doc:
                deep_merge(view[sec], doc[sec], sec, owners)
    metas = [d["_meta"] for d in docs.values()]
    view["_meta"] = {"generated_at": max(m["generated_at"] for m in metas),
                     "data_until": max(m["data_until"] for m in metas), "db": metas[0]["db"],
                     "run_id": next((m["run_id"] for m in metas if m.get("run_id")), None),
                     "scripts": sorted(docs)}
    return view


def has_path(view, dotted):
    """
    EN: True if the dotted path exists in the view and is not None. / TR: Noktalı yol görünümde var ve None değilse True.
    """
    o = view
    for k in dotted.split("."):
        if not isinstance(o, dict) or k not in o:
            return False
        o = o[k]
    return o is not None and not (isinstance(o, float) and math.isnan(o))

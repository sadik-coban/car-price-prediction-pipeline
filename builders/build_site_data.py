"""
build_site_data.py
EN: Assembles data/site_data.json — the site's single data file — from metrics/*.json. No analysis: the
    meta / domain / methodology sections the analysis scripts publish are merged (builders/report_lib/metrics_view.py,
    which also checks that every file comes from the same database and model run), the static column labels
    are added, and the tree is written with the old schema. Also writes data/serving/column_labels.json.
    meta gains generated_at / data_until / run_id so the site can show which run it is.
TR: Sitenin tek veri dosyası data/site_data.json'u metrics/*.json'dan derler. Analiz yok: analiz betiklerinin
    yayımladığı meta / domain / methodology bölümleri birleştirilir (builders/report_lib/metrics_view.py; her dosyanın
    aynı veritabanı ve model koşumundan geldiğini de sınar), durağan kolon etiketleri eklenir ve ağaç eski
    şemayla yazılır. Ayrıca data/serving/column_labels.json'u yazar. meta'ya generated_at / data_until / run_id
    eklenir; site hangi koşumu gösterdiğini söyleyebilir.
Run / Koşum: python builders/build_site_data.py
"""
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from report_lib import metrics_view as MV      # noqa: E402  the single metrics reader | tek metrik okuyucu
from report_lib.column_labels import COLUMN_LABELS  # noqa: E402

# EN: CARDATASYS_OUT redirects the outputs (tests); unset → the repo | TR: CARDATASYS_OUT çıktıları yönlendirir
OUT_ROOT = pathlib.Path(os.environ.get("CARDATASYS_OUT") or ROOT)
SITE_OUT = OUT_ROOT / "data" / "site_data.json"
LABELS_OUT = OUT_ROOT / "data" / "serving" / "column_labels.json"
# EN: the keys the site reads; a missing one stops the build | TR: sitenin okuduğu anahtarlar; eksikse durur
SITE_KEYS = {
    "meta": ["n_dedup", "n_raw", "snapshots", "n_features", "brands", "repro"],
    "domain": ["price_dist", "price_histogram", "segment_ladder", "body_median", "age_depreciation", "km_price_kapsam",
               "km_price", "brand_compare", "series_segment_matrix", "age_km_note", "numeric_correlation", "kmeans",
               "pca_scatter", "pca_scatter_13", "model_compare", "conformal", "quantile_error", "oof_outliers",
               "oof_best", "residual_vs_n", "pred_vs_true", "residual_scatter", "drift", "hedonic",
               "hedonic_reliability", "shap", "dealer_coverage", "model_yil_medyani", "brand_ablation", "final_results"],
    "methodology": ["feature_kept", "feature_drop", "cramers_matrix", "theils_matrix", "cramers_null", "theils_null",
                    "g_mpv", "assoc_model", "lofo", "lofo_agac", "kmeans_selection", "pca_axes", "column_missing",
                    "impute_note", "backtest", "column_missing_all", "sistematik_missing", "kolon_hesabi",
                    "kb_gb_ikiz", "icerik_duplike"],
}


def assemble(view):
    """
    EN: The site tree from the merged metrics view: meta (+ run stamp), domain, methodology, column_labels.
        Stops if a key the site reads is missing.
    TR: Birleşik metrik görünümünden site ağacı: meta (+ koşum damgası), domain, methodology, column_labels.
        Sitenin okuduğu bir anahtar eksikse durur.
    """
    missing = [f"{sec}.{k}" for sec, keys in SITE_KEYS.items() for k in keys if k not in view[sec]]
    if missing:
        raise SystemExit(f"site keys missing | site anahtarı eksik — run | koşun: python analysis/run_all.py · {missing}")
    stamp = {k: view["_meta"][k] for k in ("generated_at", "data_until", "run_id")}
    return {"meta": {**view["meta"], **stamp}, "domain": view["domain"], "methodology": view["methodology"],
            "column_labels": COLUMN_LABELS}


def main():
    """
    EN: Loads the metrics view, assembles the site tree and writes site_data.json + column_labels.json.
    TR: Metrik görünümünü yükler, site ağacını derler, site_data.json + column_labels.json'u yazar.
    """
    site = assemble(MV.load_view(ROOT))
    SITE_OUT.parent.mkdir(parents=True, exist_ok=True)
    LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    SITE_OUT.write_text(json.dumps(site, ensure_ascii=False, indent=2), encoding="utf-8")
    LABELS_OUT.write_text(json.dumps(COLUMN_LABELS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[✓] {SITE_OUT} ({SITE_OUT.stat().st_size / 1e6:.1f} MB) · "
          f"domain {len(site['domain'])} · methodology {len(site['methodology'])} · run {site['meta']['run_id']}")


if __name__ == "__main__":
    main()

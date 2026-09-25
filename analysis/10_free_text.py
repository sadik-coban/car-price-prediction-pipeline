"""
10_free_text.py
EN: Technical report §10 — what the listings' free text adds. The text ablation (R² of structured features
    alone vs structured + text) is a frozen measurement from the archived text-analysis run
    (analysis/frozen/text_ablation.json; no LLM is run here). This script counts how much of the modelled data
    the LangExtract extraction covers (the extraction file is only read and counted) and publishes the frozen
    ablation with its setup.
TR: Teknik rapor §10 — ilanların serbest metni ne katıyor. Metin ablasyonu (yalnız yapısal özniteliklerin R²'si
    ile yapısal + metin) arşivdeki metin analizi koşumundan dondurulmuş bir ölçüm
    (analysis/frozen/text_ablation.json; burada LLM koşulmaz). Bu betik LangExtract çıkarımının modellenen
    verinin ne kadarını kapsadığını sayar (çıkarım dosyası yalnız okunur ve sayılır) ve dondurulmuş ablasyonu
    kurulumuyla yayımlar.
Output / Çıktı: metrics/10_free_text.json
"""

# %% [1] Setup | Kurulum
import json

from lib.common import ROOT, load_clean, save_metrics

EXTRACTION_FILE = ROOT / "data" / "langextract" / "extraction_results.jsonl"
FROZEN_FILE = ROOT / "analysis" / "frozen" / "text_ablation.json"
# EN: the frozen file's (Turkish) LLM class names -> the ids published | TR: dondurulmuş dosyanın LLM sınıf adları
LLM_CLASSES = {"hasar": "damage", "bakim": "maintenance", "modifiye": "modification"}


# %% [2] Analysis functions | Analiz fonksiyonları — pure: no file I/O, they only return values
def extraction_coverage(document_ids, listing_ids):
    """
    EN: How many texts were extracted and how many of them belong to a modelled listing (document_id starts
        with the ad_id). Returns: {"texts", "in_model", "listings"}.
    TR: Kaç metin çıkarıldı ve kaçı modellenen bir ilana ait (document_id ad_id ile başlar).
        Döndürür: {"texts", "in_model", "listings"}.
    """
    ids = {int(str(d).split("_")[0]) for d in document_ids}
    return {"texts": len(document_ids), "in_model": len(ids & set(listing_ids)), "listings": len(listing_ids)}


# %% [3] Metrics assembly | Metrik derleme — naming and rounding only | yalnız adlandırma ve yuvarlama
def to_metrics(res):
    """
    EN: Published in the report inputs (error_drivers.text_source) and report.text_ablation. The frozen evidence
        keeps its original (Turkish) field names; they are translated here, and an unknown field or class stops.
    TR: Rapor girdilerinde (error_drivers.text_source) ve report.text_ablation'da yayımlanır. Dondurulmuş kanıt
        özgün (Türkçe) alan adlarını korur; burada çevrilir, bilinmeyen alan ya da sınıf durdurur.
    """
    cov, fz = res["coverage"], res["frozen"]
    setup, llm = fz["setup"], fz["llm"]
    assert set(llm) == {"kutuphane", "model", "siniflar"}, f"frozen llm fields changed: {sorted(llm)}"
    return {"error_drivers": {"text_source": {
                "llm_texts": cov["texts"], "llm_listings_in_model": cov["in_model"], "listings": cov["listings"],
                "llm_coverage_pct": round(100 * cov["in_model"] / cov["listings"], 1),
                "ablation_trees": setup["n_estimators"], "ablation_categorical": setup["categorical"],
                "ablation_numeric": setup["numeric"],
                "ablation_model_series": any(c in ("model", "series") for c in setup["categorical"] + setup["numeric"])}},
            "report": {"text_ablation": {
                "ablation": fz["ablation"],
                "llm": {"library": llm["kutuphane"], "model": llm["model"],
                        "classes": [LLM_CLASSES[c] for c in llm["siniflar"]]},
                "source": fz["kaynak"]}}}


# %% [4] Load | Yükle — the only cells that read files | dosya okuyan tek hücreler
listings = load_clean()
with open(EXTRACTION_FILE, encoding="utf-8") as fh:
    document_ids = [json.loads(line)["document_id"] for line in fh]
frozen = json.loads(FROZEN_FILE.read_text(encoding="utf-8"))

# %% [5] Compute | Hesapla — look at the results here | sonuçlara burada bak
res = {"coverage": extraction_coverage(document_ids, listings["ad_id"].astype(int).tolist()), "frozen": frozen}
print("extraction texts | çıkarım metni:", res["coverage"]["texts"], "· in model | modelde:", res["coverage"]["in_model"])

# %% [6] Save | Kaydet — the only cell that writes the JSON | JSON'u yazan tek hücre
print("written | yazıldı:", save_metrics("10_free_text", to_metrics(res)))

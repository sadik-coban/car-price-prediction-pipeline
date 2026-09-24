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
    EN: Published in the report inputs (error_drivers.metin_kaynak) and report.text_ablation.
    TR: Rapor girdilerinde (error_drivers.metin_kaynak) ve report.text_ablation'da yayımlanır.
    """
    cov, fz = res["coverage"], res["frozen"]
    setup = fz["setup"]
    return {"error_drivers": {"metin_kaynak": {
                "llm_metin": cov["texts"], "llm_modeldeki_ilan": cov["in_model"], "ilan": cov["listings"],
                "llm_kapsam_pct": round(100 * cov["in_model"] / cov["listings"], 1),
                "ablasyon_agac": setup["n_estimators"], "ablasyon_kategorik": setup["categorical"],
                "ablasyon_sayisal": setup["numeric"],
                "ablasyon_model_seri": any(c in ("model", "series") for c in setup["categorical"] + setup["numeric"])}},
            "report": {"text_ablation": {"ablation": fz["ablation"], "llm": fz["llm"], "kaynak": fz["kaynak"]}}}


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

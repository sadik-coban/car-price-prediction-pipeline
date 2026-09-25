# Yeniden üretilebilirlik

> 🇬🇧 English: [reproducibility.en.md](reproducibility.en.md) · ← [README](../README.md)

Analiz zinciri deterministiktir ve bu **ölçülerek** doğrulandı. 2026-09-23'te hesap tek dosyadan
(`build_site_data.py`, 1.600 satır) soru başına betiklere bölündü; yeni zincir eskisinin çıktısına karşı
sınandı (referans: `archive/backups/referans-2026-09-23/`):
- **metrikler:** her betiğin sayıları eski `site_data.json` / `error_drivers.json` / `oof_shap.json`'daki
  karşılığıyla yaprak yaprak aynı; derlenen `site_data.json` yalnız bootstrap süresinde, kaldırılan
  `noise_floor`'da ve yeni koşum damgasında (`meta.generated_at/data_until/run_id`) ayrılıyor;
- **raporlar:** altı md (teknik, karar notu, SHAP × TR + EN) ve 70 figürün hepsi **bayt bayt aynı** — son tam
  koşum sıfırdan (`metrics/` ve `data/analysis/` silinerek) `run_all.py` ile, 16 dk; yalnız
  `catboost_native` kalıntısı (aşağıda) bir koşumda SHAP §4'ün native sütununu öteki duruma çevirebilir;
- **servis:** üç servis modeli 29.988 ilanın hepsinde eskisiyle aynı tahmini veriyor (fark ₺0);
  `encoders.pkl`, `lightgbm_tfidf_svd.txt` ve servis README'si bayt bayt aynı.

Her metrik dosyası `_meta.generated_at` dışında aynı veriyle aynı çıkar.

Üretilen her JSON ne zaman yazıldığını aynı biçimde söyler: üst düzeyde `_meta.generated_at`, UTC farkıyla
yerel saat, saniye hassasiyetinde (ör. `2026-09-25T04:03:29+03:00`). Kapsam: metrikler,
`data/analysis/oof_info.json`, `data/site_data.json`, `data/serving/column_labels.json` ve S3 manifest'i. Derleyici
çıktılarında damga derlemenin kendi zamanı. `site_data.json`'da ayrıca `meta.metrics_generated_at` var: en yeni
metrik damgası. Yeniden üretim testleri karşılaştırmada yalnız bu alanı dışarıda bırakır. Biçimi
`tests/metrics/test_generated_stamps.py` sabitler. Hangi betiğin en son ne zaman koştuğunu
`python tools/analysis_coverage.py` gösterir.

Bunu sağlayan iki yama:
- dedup sorgusunun dış select'inde `ORDER BY ad_id` — yoksa DuckDB satır sırasını garanti etmez
  ve `KFold(shuffle=True, random_state=42)` pozisyonları karıştırdığı için fold üyeliği koşudan
  koşuya değişir (`SEED` bunu engellemez);
- LightGBM `deterministic=True, force_row_wise=True`.

Koşum parametreleri `site_data.json` → `meta.repro` altında kayıtlı (seed · satır sırası ·
CatBoost cihazı · n_jobs · CV ağaç sayıları · son modelin ağaç sayısı). **CatBoost GPU ile CPU aynı ağacı üretmez**, bu yüzden cihaz
kaydedilir; determinizm için lokal CPU tercih edilir.

**Bilinen kalıntı (ölçüldü 2026-09-23):** `domain.shap.catboost_native` koşumdan koşuma küçük kalemlerde
oynuyor. Sebep SHAP hesabı değil (çok ve tek iş parçacığı bit-birebir aynı), native CatBoost modelinin kendisi:
iki koşumun modeli 29.988 ilanın hepsinde **aynı tahmini** veriyor (fark ₺0, ağaç sayısı aynı), ama iç yapısı
farklı kaydediliyor ve atıf küçük kalemler arasında yer değiştiriyor (kasa tipi 0,0079 ↔ 0,0189, marka
0,0039 ↔ 0,0075; yaş ve km birebir). Tahminler ve metrikler oynamaz; oynayan yalnız SHAP raporu §4'ün native
sütunu (normalize paylar olduğu için tüm sütun; en çok ~1,7 puan). Aynı model dosyasıyla SHAP raporu bayt
bayt aynı üretilir. Yaprak değerleri ve yaprak ağırlıkları iki dosyada da birebir aynı; fark dosyanın
kategori/metin meta verisinde. Atıf iki durum arasında gidip geliyor ve bu yeni zincirde de sürüyor
(2026-09-23: iki koşum bir durumda, sıfırdan tam koşum referansın durumunda çıktı). Teknik rapor yalnız
LightGBM'in SHAP tablosunu kullanır. SHAP ayrı bir raporda (`reports/shap.{tr,en}.md`, 2026-09-21): rapor **veri
ölçeğinde** konuşur — küresel önem, beeswarm, bağımlılık, yaş grupları, km × yaş; §6'da tek bir ilanın
waterfall'ı örnek olarak duruyor (6 figür). Altı vaka dökümlü + üç waterfall'lı eski sürüm,
betiğiyle birlikte `archive/analysis-history/shap-v3-oof-vakalar-2026-09-21/` altında.

**OOF SHAP (2026-09-20, 2026-09-23'ten beri `analysis/shap/02_oof_shap.py`).** Beş fold
`analysis/lib/cv.py` ile, `07_model_comparison`'daki kurulumun aynısıyla yeniden kurulur. Yeniden üretilen
LightGBM OOF tahminleri 07'nin sakladığı OOF ile **kuruşu kuruşuna** eşleşti (max fark ₺0.00). Eşleşme her
koşumda kapı olarak sınanır — tutmazsa dosya yazılmaz. (CatBoost varyantları burada yeniden kurulmaz.)

**Doğrulama kapısı ve metrik referansı (2026-09-24).** Determinizm bir kez ölçülüp bırakılmıyor, her
değişiklikte `tools/verify.py` ile sınanıyor. `tests/baselines/` 24 metrik dosyasının iki anlık görüntüsünü
tutar: `metrics_shape.json` (her anahtar yolu ve JSON tipi) ve `metrics_fingerprint.json` (skalerin değeri;
listelerin uzunluğu ve kanonik JSON'unun sha256'sı). İlk referans `restructure-2026-09` dalındaki `b2395e0`
commit'inin metrikleridir. Kurallar:
- varsayılan **birebir eşitlik**. Oynamasına izin verilen anahtarlar yalnız
  `tests/baselines/exemptions.json`'da, her biri tr/en gerekçesiyle: `_meta.generated_at` (koşum damgası),
  `06_hedonic`'in bootstrap süresi ve `shap/04_variants`'ın `catboost_native*` anahtarları (yukarıdaki
  kalıntı). Artık hiçbir anahtarla eşleşmeyen istisna testi düşürür; bayat istisna kalmaz;
- referans yalnız `python tools/snapshot_metrics.py --accept "<gerekçe>"` ile değişir. Önce `metrics_view`
  tutarlılık kapısı geçmeli, gerekçe boş olamaz; her kabul `tests/baselines/accept_log.jsonl`'a tarih,
  gerekçe, değişen anahtar sayıları ve ilk farklarla yazılır;
- raporlar metriklerin saf fonksiyonudur. Test, derleyicileri `CARDATASYS_OUT` ortam değişkeniyle geçici bir
  klasöre koşar ve çıktıyı depodakiyle karşılaştırır: md'ler satır sonları hariç, figürler bayt bayt,
  `site_data.json` JSON olarak. Referans kabul edildikten sonra raporlar yeniden üretilmezse bu test düşer.

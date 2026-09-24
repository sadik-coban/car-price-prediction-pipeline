# İkinci el araç fiyat analizi — veriden rapora

> 🇬🇧 English: [README.en.md](README.en.md)

Depo, canlı zincirin **tamamını** taşır: ilan toplamadan yayınlanan rapora kadar.
Öncesinde raporların üreticisi yoktu — metinleri başka bir depodaki Next.js sayfalarından
elle transkribe edilmiş, grafikleri o sayfaların ekran görüntüsüydü.

**Altın kural:** hiçbir veri sayısı elle yazılmaz. Her rakam JSON'dan okunur ya da koddan
türetilir. Bilinçli istisna [docs/decisions.md](docs/decisions.md)'de işaretli (elle yazılmış örnek notları). Atılan kolon tablosu
(`feature_drop`) 2026-09-23'e kadar elle yazılıydı ve veriyle uyuşmuyordu; artık her ham kolon koddan tek bir
sınıfa atanıyor ve sayılar hesaplanıyor.

---

## Zincir

```
scraper/ ──► data/raw/{audi,bmw}/<tarih>/details.jsonl
   └─ db/build_duckdb.py (← lib/process_for_db.py) ──► data/cars.duckdb (yarı ham)
        │
        ├─ analysis/NN_*.py        her soru kendi betiğinde ──► metrics/NN_*.json
        │     └─ 07_model_comparison ──► data/analysis/oof.parquet (OOF) ──► 07_final_model · 08_* · shap/*
        ├─ analysis/shap/NN_*.py   SHAP raporunun soruları ──► metrics/shap/*.json + reports/figures/*-sh-*.png
        │
        └─ builders/ (analiz yapmaz, yalnız metrics/*.json okur)
              ├─ build_site_data.py ──► data/site_data.json
              ├─ build_technical_report.py ──► reports/technical ×{tr,en} · reports/figures
              ├─ build_business_report.py ──► reports/business ×{tr,en} · reports/figures
              └─ build_shap_report.py ──► reports/shap ×{tr,en}

   db/build_gold_db.py (← gold_rules.json): data/cars.duckdb ──► data/cars_gold.duckdb (API sözleşmesi)
      └─ db/publish_data_to_s3.py ──► S3 "data/cars.duckdb"  (yayın kolu; rapor zincirinin parçası değil;
                                       yalnız gold sözleşmesini tutan dosyayı yükler — docs/database.md)
```

## Nasıl koşulur

Tüm script'ler yollarını `__file__`'a göre çözer → **nereden koşulursa koşulsun çalışır**,
`cd` şartı yok.

```bash
# 0) ortam (repo kökünde)
python -m venv .venv && .venv/Scripts/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-pipeline.txt

# 1) veri toplama (opsiyonel — yeni snapshot gerekiyorsa)
python scraper/main.py                        # -> data/raw/<marka>/<tarih>/  (kazıma kodu yalnız yerelde)
python db/build_duckdb.py                     # -> data/cars.duckdb (+ yanına duplicate_ad_ids.csv)
python db/build_gold_db.py                    # -> data/cars_gold.duckdb (API'ye giden; yayın yalnız bunu yükler)

# 2) hepsi: analiz betikleri bağımlılık sırasıyla + derleyiciler   (~16 dk)
python analysis/run_all.py
python analysis/run_all.py --from 07          # 07'den itibaren (+ derleyiciler)
python analysis/run_all.py --only 08          # yalnız 08_* betikleri (derleyicisiz)

# tek bir soru: yalnız kendi metrics/<betik>.json'unu yazar
python analysis/08_conformal_coverage.py

# 3) yalnız raporlar (metrikler hazırsa, ~1 dk)
python builders/build_site_data.py
python builders/build_technical_report.py
python builders/build_business_report.py
python builders/build_shap_report.py

# 4) bitti mi: tek karar, 0 ile çıkmalı (aşağıda "Doğrulama kapısı")
python tools/verify.py                        # hızlı kapı (~10 sn)
python tools/verify.py --full                 # önce run_all.py (~16 dk), sonra aynı kapı
```

> **Windows notu:** script'ler Türkçe karakter basar. Çıktı dosyaya yönlendirilirse
> `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` şarttır, yoksa cp1252 `UnicodeEncodeError` verir.

**Veri değişmediyse raporları yeniden üretmek bir dakika sürer** — ağır hesap analiz betiklerinde kalır
(en ağırları `07_lofo` ~6 dk, `07_model_comparison` ~4 dk, `06_hedonic` bootstrap ~1,5 dk, `shap/02` ve
`shap/04` ~1 dk).

### Doğrulama kapısı

`python tools/verify.py` bir değişikliğin bitip bitmediğine karar veren tek komut: 0 ile çıkarsa bitti.
Kendisi yalnız hangi testlerin koşacağını seçer ve özetler; her kontrol `tests/` altında bir pytest testi:
- `tests/db/`, `tests/scraper/`: raw → DB kolu ve toplama ayarı;
- `tests/metrics/`: her `metrics/*.json` referansla (`tests/baselines/`) aynı mı: anahtar yolları, tipler ve
  değerler. Koşumdan koşuma oynaması bilinen anahtarlar gerekçesiyle `tests/baselines/exemptions.json`'da;
- `tests/reports/`: dört derleyici geçici bir klasöre yeniden koşulur; altı rapor, her figür ve
  `site_data.json` depodakiyle aynı olmalı. Elle düzenlenmiş ya da metrikten sonra yeniden üretilmemiş rapor
  burada düşer;
- `tests/repo/`: iki dilli docstring, belgelerdeki göreli bağlantılar, başka bir kaynak dosyayı veri gibi
  okuyan kod yok.

Bir analiz değişikliği sayıları **bilerek** değiştirdiyse kapı düşer ve farkı gösterir. Fark
`python tools/snapshot_metrics.py --diff` ile incelenir, `python tools/snapshot_metrics.py --accept "<gerekçe>"`
ile kabul edilir, sonra raporlar yeniden üretilir. Gerekçesiz kabul reddedilir; her kabul
`tests/baselines/accept_log.jsonl`'a yazılır. Kurallar: [Yeniden üretilebilirlik](docs/reproducibility.md).

## Belgeler

- [Analiz betikleri nasıl okunur](docs/analysis-scripts.md) — üç katman, hücre hücre koşum, ortak kod, tutarlılık kapısı
- [Kararlar ve sınırlar](docs/decisions.md) — iş / teknik ayrımı, metin analizi neden arşivde, gizlilik, dürüst çerçeve
- [Veritabanı notları](docs/database.md) — yarı ham kurallar, güvenli kurulum, **gold adımı (API): DB'yi yeniden kurmadan önce okuyun**
- [Yeniden üretilebilirlik](docs/reproducibility.md) — determinizm, referans karşılaştırması, bilinen kalıntı
- Raporlar: [teknik](reports/technical.tr.md) · [karar notu](reports/business.tr.md) · [SHAP](reports/shap.tr.md) (İngilizceleri `*.en.md`)

---

## İçerik

| yol | ne |
|---|---|
| `scraper/` | ilan toplama → `data/raw/`; ne toplandığı (markalar, fiyat aralıkları, arama sorgusu) `collection_config.json`'da, `collection.py` okur — analiz (`01_dedup_leakage`) de aynı dosyayı okur. Kazıma kodu (`main.py`, `getlistofcars.py`, `getdetails.py`) depoda yok, yalnız yerelde; depoda bu iki ayar dosyası izlenir |
| `db/` | `build_duckdb` (→ yarı ham `cars.duckdb`; güvenli kurulum) + `lib/process_for_db` (ham JSONL → satır ayrıştırıcıları) + `lib/damage_mappings.json` (hasar şemasının 13 parça / 5 durum etiketi; `analysis/01_unspecified_panels` da okur) · `build_gold_db` (→ `cars_gold.duckdb`, API sözleşmesi: bilinmeyen = hayır/0; kurallar ve gerekçeleri `gold_rules.json`'da) · `publish_data_to_s3` (S3 yayını; yalnız gold sözleşmesini tutan dosyayı yükler; `--dry-run` yalnız kontrol eder, S3'e bağlanmaz) + `lib/s3_publish` (S3 bağlantısı) |
| `tests/` | pytest testleri — `tests/db/`: raw → DB kolu (gerçek veride görülen her biçim, yarı ham kurallar, uçtan uca kurulum, güvenli kurulum) ve S3 yayın kolu (doğrulama, sürüm, yükleme sırası, manifest). Kayıtlar sahte ve gerçek biçimli; her şey geçici klasörde; ağ, `.env` ve `data/` yok · `tests/metrics/`, `tests/reports/`, `tests/repo/`: doğrulama kapısı (yukarıda); `tests/repo/test_hooks.py` yerel Claude Code hook'larını sınar, hook'lar yoksa atlanır · `tests/baselines/`: metrik referansı, istisnalar, kabul kaydı. Kurulum `pip install -r requirements-dev.txt`, koşum `python tools/verify.py` (ya da `python -m pytest tests -v`), kapsam `python -m pytest tests --cov=db --cov-branch --cov-report=term-missing` |
| `analysis/` | **bütün hesap**: soru başına bir betik, numara = teknik raporun bölümü (`01_dedup_leakage` · `01_engine_rule` · `01_unspecified_panels` · `01_gold_contract` (API'ye giden gold verisi) · `02_missingness` · `03_association` · `03_segment_quality` · `03_brand_ablation` · `04_target` · `05_segmentation` · `06_hedonic` · `07_model_comparison` · `07_final_model` (servis dosyaları) · `07_lofo` · `07_text_flag` · `08_conformal_coverage` · `08_residuals` · `08_large_errors` · `09_drift` · `09_backtest` · `10_free_text`) + `shap/` (SHAP raporunun bölümleri: `02_oof_shap` · `03_what_sets_price` · `04_variants` · `06_one_listing`) + `lib/` (ortak kod: `common.py` · `cv.py` · `segment_rule.py` · `text_flags.py` · `labels.py`) + `run_all.py` + `frozen/text_ablation.json` (arşivlenen metin analizinden dondurulmuş ablasyon) |
| `builders/` | **analiz yapmayan** derleyiciler, yalnız `metrics/*.json` okur: `build_site_data.py` (→ `site_data.json`, aynı şema) · `build_technical_report.py` · `build_business_report.py` · `build_shap_report.py`; ortak kod `report_lib/` altında: `metrics_view.py` (tek okuyucu + tutarlılık kapısı) · `report_common.py` (ortak sayılar, figürler, biçimleyiciler) · `column_labels.py` |
| `tools/` | `verify.py` (bitti tanımı: hangi testlerin koşacağını seçer ve özetler; `--full` önce zinciri koşar, `--json` tek satır özet basar) · `snapshot_metrics.py` (metrik referansı: `--diff` farkları gösterir, `--accept "<gerekçe>"` yeni referansı yazar) |
| `metrics/` | betik başına bir JSON (`metrics/<betik>.json`, `metrics/shap/<betik>.json`); bölümleri `meta` · `domain` · `methodology` (site ağacı) · `error_drivers` · `oof_shap` · `shap*` · `report` (yalnız raporun kullandığı sayılar), her birinde `_meta` |
| `reports/` | `business.{tr,en}.md` · `technical.{tr,en}.md` · `shap.{tr,en}.md` — hepsi üretilmiş dosya, elle düzenlenmez; hangi figürün hangi rapora girdiği `builders/report_lib/report_common.py`'deki `BUSINESS_FIGS` / `TECHNICAL_FIGS` listelerinde · `figures/` raporların figürleri (`{tr,en}-NN-*.png`, SHAP'inkiler `-sh-` — onları `analysis/shap/` çizer); md dosyaları `figures/...` ile gösterir |
| `docs/` | uzun belgeler, tr + en (aşağıda "Belgeler") |
| **`data/`** | **tüm ağır veri — `.gitignore`'lı**: `raw/` · `cars.duckdb` · `site_data.json` · `serving/` · `analysis/` (OOF ve OOF SHAP artefaktları) · `langextract/` |
| `archive/` | **arşiv, git dışı** (dizini `archive/README.md`): `obsolete/` · `analysis-history/` — metin analizi 2026-09-19'da yayımlanan işten çıkarıldı ([docs/decisions.md](docs/decisions.md)); SHAP raporunun sürümlü arşivi: `shap-v1-deneme-2026-09-20/` (kütüphane varyantlarının tamamı, 46 figür) · `shap-v2-final-model-2026-09-20/` (budanmış, final modelden çizilmiş, 22 figür) · `shap-v3-oof-vakalar-2026-09-21/` (OOF, altı vaka dökümü + üç waterfall, üreteç betiğiyle) · `robustness-2026-09-21/` (model sağlamlığı raporu: betik + 2 md + 10 figür + ölçüm JSON'u) · `backups/` · `experiments/` · `published-report/` |

### `.gitignore` notu

Desen **`/data/`** — baştaki eğik çizgi kasıtlı. Çıplak `data/` yazılsaydı gitignore onu
**her derinlikte** eşleştirirdi ve `metrics` gibi klasörler de sessizce takipsiz
kalırdı. (Bu yüzden o klasörün adı `data` değil `metrics`.)

## Depo kökünün dışında ne kaldı, neden

**Depoya yalnız canlı zincir giriyor.** Aşağıdakiler **diskte duruyor ama git'te izlenmiyor**
(`.gitignore`); hiçbiri silinmedi:

| kök klasörü | ne | git |
|---|---|---|
| `archive/` | arşivin tek çatısı (2026-09-24'e kadar kökte beş ayrı klasör): `obsolete/` (eski `obselete/` — taşındı, silinmedi, gerekçeler kendi `README.md`'sinde; 2026-09-18'e kadarki yöntem ve karar kaydı `archive/obsolete/docs/`'ta, kökteki bugünkü `docs/` ile karıştırılmasın) · `analysis-history/` (eski `_arsiv/`) · `backups/` (eski `backup/`) · `experiments/` (deneyler, büyük hatalar · domain analizleri — kalıcı bulgu çıkarsa üreteci depo köküne taşınır) · `published-report/` (eski `car-price-export/`: yayınlanmış raporun dondurulmuş kopyası — sayı denetiminin referansı). Dizin: `archive/README.md` | izlenmez |
| `.claude/` | yerel Claude Code kurulumu: doğrulama hook'ları (`hooks/`, `settings.local.json`), `methodology-reviewer` agent'ı, skill'ler | izlenmez |

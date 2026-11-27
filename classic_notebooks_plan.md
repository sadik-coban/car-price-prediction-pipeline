> **Durum (2026-11-26): uygulanıyor.** 0., 1. ve 2. aşama commit'lendi (4316eb9, 574a118, 526016d). 1. ve 2. defterin
> bilinen zayıflıkları tek tek gözden geçirildi (`docs/notebook-weaknesses.md`); 3. aşama (`3_price_model`) yazıldı.
> 3. defter ve iki defterin zayıflık düzeltmeleri commit'lenmedi; sırada 4. aşama (`4_errors_and_interval`) var.
> `1_data_and_cleaning`, `2_what_drives_price` ve `3_price_model` okumanı bekliyor. Dal: `classic-notebooks`
> (`simple-notebooks` be8644a'dan açıldı). Dördüncü sürüm: yapı sıfırdan kurulur, fonksiyon klasörünün adı `lib`,
> defterlerin kaynağı jupytext ile düz Python dosyasıdır.

# Klasik defter yapısı, sıfırdan

## Neden

Bugün tek bir analiz altı ayrı yerde yaşıyor: betik (`analysis/NN_*.py`), kart, metrik JSON'u, test, derleyicideki
rapor metni ve üretilen rapor. Üstüne bir de kaydı gösteren defterler var.

| parça | boyut |
|---|---|
| analiz betikleri | 24 dosya, 4.900 satır |
| ortak kütüphane (`analysis/lib`) | 9 dosya, 1.200 satır |
| rapor derleyicileri (`builders/`) | 8 dosya, 4.400 satır |
| testler | 58 dosya, 6.400 satır |
| araçlar ve iç araç | 3.500 satır |
| kart, metrik dosyası | 23 + 23 |

Eski zinciri yanında taşıyarak ilerlemek iki yapıyı aylarca birlikte tutar. Bu dalda eski zincir en başta kalkar;
analiz baştan, defter olarak yazılır.

## Hedef

```
notebooks/                        her defter iki dosya: <ad>.py (kaynak) ve <ad>.ipynb (koşulmuş hali, okunan)
  1_data_and_cleaning             veri, tekilleştirme ve sızıntı, motor kuralı, belirtilmemiş paneller, eksiklik
  2_what_drives_price             bağımlılık, segment kalitesi, hedef, piyasa, hedonik
  3_price_model                   karşılaştırma ve taban, marka ablasyonu, LOFO, ekleme sınaması, son model
  4_errors_and_interval           artıklar, büyük hatalar, aralık ve isabet olasılığı, ileri test, serbest metin
  5_shap
  lib/                            import edilen fonksiyonlar; kurulum yok, import anında dosya okumaz, yazmaz
    data.py                       yükleme ve türetilen kolonlar
    cleaning.py  missing.py  association.py  market.py  hedonic.py
    folds.py  spec_rule.py        ortak beş kat; motor tutarlılık kuralı (2. ve 4. defter)
    model.py                      matrisler, LightGBM, taban
    errors.py  interval.py  shap_tools.py  plots.py
    cache.py                      ağır adımlar için "varsa oku, yoksa koş"
db/                               veritabanı kolu; mantığı değişmez, yalnız db/lib → db/dblib adı değişir
tests/                            bütün kontroller burada; defterlerde kontrol hücresi yok
tools/verify.py                   kapı
tools/build_notebooks.py          kaynağı koşar, defteri yazar, başlık sayılarını karşılaştırır
docs/decisions.md                 karar kaydı
```

### Jupytext

Defterin kaynağı düz bir Python dosyasıdır (`notebooks/<ad>.py`, "percent" biçimi: `# %%` kod hücresi,
`# %% [markdown]` açıklama hücresi). `.ipynb` ondan üretilir ve çıktılarıyla birlikte depoda durur.

- **Sen** `.ipynb` dosyasını okursun; Jupyter ya da VS Code'da her zamanki defter.
- **Ajanlar** `.py` dosyasını okur ve düzenler; JSON'la uğraşmazlar, farklar satır satır okunur.
- `python tools/build_notebooks.py` kaynağı taze bir kernel'de baştan sona koşar ve defteri yazar. Bir hücre düşerse
  ya da başlık sayıları referanstan farklıysa defter yazılmaz.
- Defteri Jupyter'de elle düzenlersen önce `jupytext --sync notebooks/<ad>.ipynb` değişikliği kaynağa aktarır,
  sonra derleme aracı koşulur.
- Kapı, defterin kaynağıyla aynı olduğunu ve kaynak ya da defterin import ettiği `lib` modülleri değiştikten sonra
  yeniden derlendiğini sınar. Defterin kullanmadığı bir modül değişince defter bayatlamaz.

### Sıfırdan ne demek

- **Dal:** `classic-notebooks` (üzerinde henüz iş yok). İlk commit eski zinciri kaldırır. Geçmişsiz (orphan) bir dal
  açmıyorum: ağaç aynı ölçüde temiz olur, ama geçmiş kalır ve iş bitince `main`'e olağan bir birleştirmeyle girer.
- **Kod:** betikler hücre hücre deftere çevrilmez. Yöntem aynı kalır, kod defter için baştan ve sade yazılır.
  Eski betikler yalnız yöntemin kaynağı olarak okunur (`simple-notebooks` dalında duruyorlar).
- **Ağır kayıtlar:** önbelleği yeni kod üretir. Eski zincirin diskteki kayıtları (`data/analysis/oof.parquet` vb.)
  okunmaz; bu "eski koşumdan canlıya bir şey girmez" kuralının gereği. Bedeli: ağır adımlar bir kez yeniden koşar
  (toplam 20–30 dakika).
- **Sayıların referansı:** `simple-notebooks` dalındaki `metrics/` kayıtları. Her defter yazılırken sayıları onlarla
  karşılaştırılır ve fark varsa sana raporlanır. Bu bir kontroldür, girdi değildir; ağaca kopyalanmaz.
  Ağaçta kalan koruma: yeni defterlerin ürettiği küçük bir başlık sayıları dosyası ve onu karşılaştıran test.

### Kalanlar ve gidenler (ilk commit)

| kalır | neden |
|---|---|
| `db/`, `scraper/` (izlenen iki dosya), `tests/db`, `tests/scraper`, `tools/observed_values.py`, `docs/database*.md` | veriyi üreten kol; şikâyetin konusu değil |
| `tools/verify.py`, `pytest.ini`, `requirements-*.txt`, `.gitignore` | kapı ve ortam; `verify.py` içindeki eski zincir koşumu çıkar |
| `docs/decisions*.md`, `backlog/ideas.md` | karar ve fikir kaydı |
| `README*.md` | en sonda yeni yapıya göre yeniden yazılır |

| gider | not |
|---|---|
| `analysis/`, `metrics/`, `builders/`, `plans/` | eski zincir |
| `reports/` | site `main`'den besleniyor; oradaki raporlar yerinde durur |
| `internal_tool/` | rapor ve betik eşlemesine dayanıyordu |
| `notebooks/*.ipynb`, `notebooks/open/` | yerlerine beş yeni defter |
| `tests/analysis`, `tests/metrics`, `tests/baselines`, `tests/reports`, `tests/repo`, `tests/plans`, `tests/internal_tool` | kalkan kodun testleri |
| `tools/` altında kapı ve `observed_values` dışındakiler | kart, plan, referans kabulü, defter koşturucu |
| kökteki eski plan dosyaları, `analysis_status.md`, `docs/analysis-scripts*`, `docs/reproducibility*`, yeniden adlandırma haritaları | eski yapıyı anlatıyor |

Hiçbiri silinmiş olmaz: `main` ve `simple-notebooks` dallarında aynen duruyorlar. Diskteki `data/` (veritabanı,
servis paketi, site verisi) izlenmediği için olduğu gibi kalır.

### `lib` adı

`db/lib` de `lib` adıyla import ediliyor (2 db betiği, 1 araç, `tests/db` altında 6 dosya). İki ayrı `lib` aynı test
oturumunda çakışır; eski yapıdaki "testler dosyayı yolundan yükler" dolambacı buradan çıkmıştı. Çözüm: defterlerin
klasörü `lib` olur, veritabanı kolununki `db/dblib` adını alır. Değişen yalnız import satırları (9 dosya);
`tests/db` aynen geçmeli.

### Defterler neden böyle ayrıldı

| defter | soru | koşum |
|---|---|---|
| 1 | Veri ne, neyi neden temizledik | saniyeler; eğitim yok |
| 2 | Fiyatı ne belirliyor | saniyeler; yalnız doğrusal regresyon |
| 3 | Tahmin modeli nasıl kuruldu, ne kadar iyi | ağır; önbellekten |
| 4 | Tahmine ne kadar güvenilir | ağır; önbellekten |
| 5 | Tahmin modeli bir tahmini nasıl kuruyor | ağır; önbellekten |

Model eğiten her adım 3–5. defterlerde. Onay bölüm bölüm olur: bir defterin içindeki her başlık ayrı okunur.

### Deftere ne girer, modüle ne girer

- **Defterde kalır:** okuyanın görmesi gereken adım. Filtre, gruplama, regresyon formülü, karşılaştırma, tablo, yorum.
- **Modüle gider:** uzun (kabaca 15 satırı aşan), birden çok yerde kullanılan ya da ağır olan fonksiyon. Defter onu
  çağırmadan önce ne yaptığını bir iki cümleyle söyler; fonksiyonun kısa, Türkçe bir açıklaması olur.
- **Ağır adımlar:** `lib/cache.py` üzerinden. Kayıt varsa ve aynı veriden üretilmişse okunur, yoksa defter koşar ve
  kaydeder (`data/cache/`). Defterde her yerde açıkça yazar.
- **Defterde olmayan:** kontrol hücresi, `assert`, kayıtla karşılaştırma, depo kökünü arayan kurulum kodu.

### Import

Paket kurulmaz. `lib/` defterlerin yanında durduğu için ilk hücre doğrudan `from lib import data, cleaning` yazar.
Modüller veri yolunu kendi dosya konumlarından bulur. Testler klasörü `pytest.ini` içindeki tek bir `pythonpath`
satırıyla görür.

### Kontroller

Hepsi `tests/` altında:
- `tests/lib`: modül fonksiyonlarının testleri.
- `tests/notebooks`: defter kaynağıyla aynı ve güncel, baştan sona sırayla ve hatasız koşmuş, içinde `assert` yok,
  kabul edilmiş başlık sayıları var.
- Başlık sayıları: her defterin son hücresi `headline` sözlüğünü tanımlar. Derleme aracı onu
  `tests/baselines/<ad>.json` ile karşılaştırır. Fark ancak `--accept "<gerekçe>"` ile, senin onayınla kabul edilir.
  Bir defterin ilk kaydı, sayıları eski zincirin kabul edilmiş kayıtlarıyla aynıysa bu gerekçeyle açılır.

### Dil

Defterler ve modül açıklamaları yalnız Türkçe; kod adları İngilizce. Dil başına ayrı defter açılmaz. İngilizce
sonradan istenirse aynı deftere, her Türkçe açıklama hücresinin altına eklenir.

### Değişmeyenler

- Veri (`data/`) ve veritabanı kolunun mantığı.
- Yöntem ve sayılar. Servis edilen ana model dosyası bayt bayt aynı çıkmalı.
- Kurallar: `ad_id` ve ilan metni çıktıda görünmez; arşivden ve eski koşumdan canlıya hiçbir şey girmez; kod
  adları İngilizce; gözlenen değerler kuralı; "araç modeli" ve "tahmin modeli" ayrımı.

## Aşamalar

| # | iş | bittiğinde |
|---|---|---|
| 0 | Temiz ağaç: eski zincir kalkar, `db/lib` → `db/dblib`, kapı sadeleşir ve yeşildir. **Yapıldı** | Dalda yalnız veritabanı kolu ve kapı var |
| 1 | Derleme aracı ve defter testleri (**yapıldı**), `notebooks/lib/` (data, cleaning, missing), `1_data_and_cleaning` (**yazıldı**; 34 başlık sayısı eski kayıtlarla aynı, ilk referans onayınla kaydedildi). Önbellek 3. defterle gelir | Üslubu ve kod/modül dengesini onaylarsın |
| 2 | `2_what_drives_price` (**yazıldı**; `lib`: association, market, hedonic, folds, spec_rule, plots; 39 başlık sayısı eski kayıtlarla aynı, ilk referans onayınla kaydedildi; 2026-11-26'da zayıflık maddeleri işlendi: 40 başlık sayısı, silhouette bütün ilanlarda) | Hafif analizlerin hepsi defterde |
| 3 | `3_price_model` (**yazıldı**; `lib`: model, cache, errors; eski kayıtlarla karşılaştırılabilen 318 sayı aynı, son tahmin modeli bugünkü servis dosyasıyla bayt bayt aynı; 37 başlık sayısı; metin bayrağı bölümü 2026-11-28'de senin kararınla kaldırıldı: regex bayrağının isabeti bilinmiyordu, sonuç hiçbir kararı beslemiyordu). Ağır adımlar bir kez koşar (`data/cache/`; kayıtsız yaklaşık 12 dakika, kayıtlıyken saniyeler). Eski zincirde olmayan bir ek (2026-11-25, senin kararın): tahmin modeline girmeyen dört ham kolon (`kb_color`, `kb_seller_type`, `kb_trade_available`, `gb_warranty_status`) LOFO'nun yanında (4.2), aynı katlar ve aynı ölçüyle tek tek eklenip sınanır; sonuç ne çıkarsa çıksın tahmin modeline eklenmezler. Her kolonun karıştırılmış bir kopyası da karşılaştırma olarak eklenir ve fark ilanlar üzerinde eşli bootstrap aralığıyla verilir. Sonuç: satıcı tipi, takas ve garanti kat dışı hatayı ölçülebilir biçimde değiştirmiyor; renk küçük ama ölçülebilir bir katkı veriyor, bilerek eklenmiyor. `1_data_and_cleaning` 4.5 buna göre düzeltildi (sınıf "kurala uymuyor, eklenmedi", metin sınamaya atıf yapıyor) | Modelleme defterde |
| 4 | `4_errors_and_interval` | Hata ve aralık defterde |
| 5 | `5_shap` | Analizin tamamı yeni yapıda |
| 6 | Dışa aktarım: servis paketi ve site verisi `lib` üzerinden, önce ayrı bir yola; bugünkü dosyalarla parite | API ve site eski zincire muhtaç değil |
| 7 | README, kurallar, `main`'e birleştirme kararı | Tek yapı |

## Bilinen bedeller

- Bu dalda kapı eski sayıları artık korumaz (843 testin çoğu kalkar). Koruma, yazım sırasında eski kayıtlarla
  karşılaştırmadır; sonrasında başlık sayıları testi.
- Veri korumalı rapor cümleleri, otomatik iki dilli rapor ve iç araç yok. Site raporlarının nasıl üretileceği 6.
  aşamada kararlaştırılır; o zamana kadar site `main`'deki raporları kullanır.
- `CLAUDE.md` ve `.claude/` kancaları izlenmiyor, yani bütün dallarda ortak; onlara dokunmadım. Bu dalda geçersiz
  kalan satırlar:
  - `tools/snapshot_metrics.py --diff / --accept` → yerini `tools/build_notebooks.py --accept` aldı.
  - "New analyses (harness Faz 1b)" bölümünün tamamı (`new_plan.py`, `new_analysis.py`, `analysis_coverage.py`).
  - "Use the Jupyter MCP for all .ipynb operations": defterler artık `.py` kaynağından derleniyor.
  - `tests/repo/test_no_archive_inputs.py` ve `tests/metrics/test_metric_key_names.py` atıfları (testler kalktı;
    kurallar duruyor).
  - Durma kancası `notebooks/` klasörünü izlemiyor: yalnız defter ya da `lib` değişirse kapıyı kendiliğinden
    yeniden koşmaz. Kapıyı elle koşuyorum; kancaya `notebooks` eklemek senin kararın.
- `README` dosyaları hâlâ eski yapıyı anlatıyor; 7. aşamada yeniden yazılır.

## Kararlar

| # | konu | durum |
|---|---|---|
| 1 | Sıfırdan, yeni dalda | Senin kararın. Dal `classic-notebooks`, geçmiş korunur (önerim) |
| 2 | Defterlerin ayrımı | Bölüm başına değil (senin kararın); beş defter (önerim) |
| 3 | Kontroller | Defterde yok, `tests/` altında (senin kararın) |
| 4 | Dil | Yalnız Türkçe, dil başına ayrı defter yok (senin kararın) |
| 5 | Fonksiyon klasörü | `notebooks/lib/`, kurulumsuz; `db/lib` → `db/dblib` (ad senin kararın, yeniden adlandırma benim çözümüm) |
| 8 | Defter kaynağı | Jupytext: `.py` kaynak, `.ipynb` üretilir (senin kararın) |
| 6 | Veritabanı kolu | Kalır (önerim) |
| 7 | Ağır kayıtlar | Yeni kodla bir kez yeniden üretilir (kuralın gereği) |

## Süre

Kabaca: 0. aşama bir iki saat; 1. aşama yarım gün; 2. aşama bir gün; 3. ve 4. aşama birer gün; 5. aşama yarım gün;
6. ve 7. aşama bir gün. Her defterden sonra senin okuman ve onayın bu sürenin dışında.

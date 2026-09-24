# Kararlar ve sınırlar

> 🇬🇧 English: [decisions.en.md](decisions.en.md) · ← [README](../README.md)

**Yayınlanandan bilinçli sapma — düz LOFO.** Ham `methodology.lofo` hem tekil hem grup
çıkarmalarını taşır; ikisini aynı eksende basmak çift sayımdır (`DAMAGE_COLS` kendi 13 üyesiyle
yarışır). Rapor **çakışmayan** 5 gruba indirger. Ayrıca 6 kategorik öznitelik (`brand`,
`kb_body_type`, `kb_drivetrain`, `segment`, `kb_transmission`, `kb_fuel`) LOFO'da **hiç
ölçülmemiştir** — üretici sayısal öznitelikleri, `model` ile `series`'i, grupları ve tavan/kaput/bagaj
durumlarını tek tek çıkarıyor, bu altısını çıkarmıyor. Rapor bu sınırı söyler.

**Teknik raporların bölüm sırası tek yerde.** Her iki üreteçte (`builders/build_technical_report.py` ve
arşivdeki metin raporu üreteci) bir `SECTIONS` listesi var; her bölüm kendi `sec_*` fonksiyonunda.
Bölüm numaraları `enumerate` ile, metindeki `§` çapraz referansları `secno("anahtar")` ile üretilir —
sırayı değiştirmek için `SECTIONS` satırlarının yerini değiştirmek yeter, numaralar ve referanslar
kendiliğinden doğru kalır.

- **Fiyat raporu:** veri → eksiklik → fazlalık/marka → hedef ve önişleme → piyasa yapısı → hedonik →
  model → kalibrasyon → zaman → serbest metin.
Arşivdeki metin raporunun üreteci de aynı desendedir; o rapor yayımlanan işin parçası değil.

**Bilinçli istisnalar — elle yazılanlar.**

1. ~~`HP_SEARCH_TL = 5000`~~ — 2026-09-20'de kalktı: gürültü tabanı paragrafı raporlardan çıkınca
   bu sayı da hiçbir yerde geçmiyor. Kalan elle yazılı içerik yalnız aşağıdaki örnek notları
   (`feature_drop` 2026-09-23'ten beri hesaplanıyor).
2. `HANDWRITTEN_EXAMPLE_NOTES` (2026-09-17, kullanıcı kararı) — teknik rapordaki "Büyük hatalar nereden
   geliyor" örneklerinin altındaki açıklamalar, ilan metinleri okunarak yazıldı. Üstlerindeki tablo ve
   **otomatik gerekçe** sütunu `analysis/08_large_errors.py`'den gelir. Her not bir örneğe `(model, yıl, fiyat)`
   ile bağlıdır; veri değişip örnek bulunamazsa üreteç **durur** — not sessizce bayatlamaz.

**Türetilen değerler.** Veri gerektiren her sayı bir analiz betiğinde hesaplanır (yalnız raporun kullandıkları
metrik JSON'unun `report` bölümünde: hata bantları, conformal kapsaması, Holm düzeltmesi …). Derleyici yalnız
yayımlanmış sayılar üzerinde biçim düzeyinde aritmetik yapar (ör. modelin tabandan yüzde iyileşmesi,
(taban−model)/taban).
Gürültü tabanından türeyen `1.42×` ve `₺33K` 2026-09-20'de raporlardan çıktı; buradan da kalktılar.

**Veriyle çelişen elle yazılmış notlar düzeltildi (2026-09-16).** Üreticilerde elle yazılmış üç
cümle ölçümle çelişiyordu; rapor artık bu notları değil sayıları okur:

- `kmeans_selection.not` "silhouette k=3'te en yüksek" diyordu; ölçümde k=3 silhouette'in en yüksek
  olduğu yer değil. k=3 silhouette ile değil, yorumlanabilirlik için sabit. Not artık veriden kuruluyor.
- `numeric_correlation.not` "VIF hepsi <3" diyordu. 2026-09-23'ten beri VIF kurulan hedonik modelin kendi
  tasarım matrisinden hesaplanıyor ve teknik rapor ortalanmış ile ortalanmamış hâlini birlikte basıyor.
- `controlled_effects` bayrak etiketi "(gizli hasar)" diyordu; zincirin kendi çerçevesi "gizli
  hasar değil". Etiket: *Çelişkili 'temiz' beyanı (satıcının formu hasar gösteriyor)*.

**Kazanan varyant veriden.** Üreticinin kuralı yalnız MAPE'ye bakar; CPU koşumunda MAPE'de CatBoost
(TF-IDF+SVD) LightGBM'in önünde, yayımlanan GPU koşumunda sıra tersti. Fark gürültü düzeyinde ve
metrikten metriğe yön değiştiriyor; güncel değerler teknik raporun model bölümünde. Rapor ★'ı veriden koyar; "model" dediği yer boyunca
LightGBM'dir — CPU'da deterministik olan o.

**Servis edilen model ölçülen ayarda (2026-09-23).** Raporun metrikleri erken durdurmalı fold
modellerinden gelir; son LightGBM eskiden sabit 900 ağaçla eğitiliyordu. Artık ağaç sayısı, CV'de erken
durdurmanın seçtiği ağaç sayılarının medyanı (`meta.repro.final_lgb_agac`, fold değerleri `cv_agac`).
`data/serving/serve/README.md` girdinin eğitimle birebir nasıl kurulacağını yazar; `encoders.pkl` segment
kuralını (`PERF_RE` dahil) ve %90 aralık için `CONFORMAL_Q`'yu taşır. LOFO'da taban ve çıkarma modelleri
aynı ağaç sınırıyla kurulur, durdukları tur `methodology.lofo_agac`'ta.

## İş / teknik ayrımı

**İş notu** = ne yapmalı, ne kadar para. Yöntem adı geçmez (MAPE, R², conformal, OOF hepsi
teknikte); her şey ₺ ve sade yüzde. **Teknik rapor** = protokol, kontroller, sınırlar.
İkisi **aynı hesaptan** beslenir: ortak değerler `builders/report_lib/report_common.py::derive`'da bir kez türetilir,
iki şablona oradan gider → bir rakam iki raporda farklı çıkamaz. Bazı figürler (çeyrek hatası, lira çeyreği, kapsama, backtest) iki
raporda da geçer: karar notunda karar için, teknik raporda kanıt olarak — bilinçli tekrar.

## Metin analizi neden arşivde

`archive/analysis-history/text_analysis/` — kod, metrikler, figürler ve raporlar diskte duruyor ama
**yayımlanan işin parçası değil** (`.gitignore`'da). İki gerekçe:

1. **Ana bulgusu olumsuzdu.** Yapısal modele metin öznitelikleri eklendiğinde çapraz-doğrulamalı
   doğrulukta ölçülebilir bir katkı bulunamadı (ΔR² ≈ 0.0015). Bu sonuç teknik raporun §10'unda
   ölçüsüyle duruyor (kaynak: `analysis/frozen/text_ablation.json`); ayrı bir rapor taşımasına gerek yok.
2. **Fiyat iddialarını üreten dedektörlerde denetimle bulunmuş kusurlar vardı**
   (`archive/experiments/regex_audit/`). Kusurlar düzeltildi, ama düzeltmeden sonra da geriye fiyat
   iddiası kalmıyordu.

Arşivden bugün hiçbir betik import edilmiyor (2026-09-23): örnek gerekçelerinin ve metin bayrağının
kullandığı dört desen dedektörü aynen `analysis/lib/text_flags.py`'ye taşındı (29.988 ilanın hepsinde arşivle
birebir aynı sonuç), ablasyon ölçüsü ve kurulumu `analysis/frozen/text_ablation.json`'da donduruldu.

## Gizlilik

1. `ad_id` hiçbir rapora ya da site verisine yazılmaz; yalnız veritabanında ve onun yanındaki
   `data/duplicate_ad_ids.csv`'de kalır (ikisi de yerel, `data/` git dışı). İlan metni de hiçbir
   markdown'a girmez.
2. **İlan düzeyindeki satırlar bilinçli olarak yayımlanır** (karar 2026-09-16): en iyi/en kötü
   tahminler — model · fiyat · yaş · km. Nadir model + tam fiyat aramayla bulunabilir; bu kabul
   edilmiş risk.

## Dürüst çerçeve (korunması zorunlu)

- Yapısal veri fiyatı çözer; **metin fiyat doğruluğuna ~0 ekler** (ölçülür, iddia edilmez).
- **Kontrollü ≠ ham.** Her fiyat iddiası araç özellikleri sabitlenerek verilir.
- **Türetilen ≠ beslenen.** `segment` seri ve model adından türetiliyor (ham `gb_segment` bozuk); kural
  `analysis/lib/segment_rule.py`'de (tek kaynak; `07_final_model` onu `encoders.pkl`'ye kopyalar), çözülemeyen
  seri/model kalırsa koşum durur; raporda söylenir.
- **Kapsam toplama filtrelerinden gelir.** Fiyat tavanı (fiyat sağdan kesik), en eski model yılı,
  yalnız `/otomobil/` kategorisi (SUV kategorisi toplanmadı; veride yalnız birkaç SUV var) ve dört yakıt
  türü (elektrikli yok). Teknik rapor §1 bu
  filtreleri scraper'ın ayar dosyasından (`scraper/collection_config.json`) okur ve karşılıklarını veride
  sayar.
- **Motor hacmi = kovanın üst sınırı, güç = alt ve üst sınırın ortalaması** (kullanıcının kuralı). Site
  ikisini ilanların bir kısmında kova olarak veriyor; teknik rapor §1 her adayı aynı modelin kesin değerli
  ilanlarıyla karşılaştıran tabloyu ve figürü basar. Seçilen aday en küçük farkı vermezse
  `analysis/01_engine_rule.py` durur.
- **"Belirtilmemiş" panel orijinal sayılır** (bilinçli karar). Teknik rapor §1 kararın ölçüsünü ve fiyat
  kanıtını basar; bedeli, hasar etkilerinin hafifçe sıfıra çekilmesi. Çeviri analizde yapılır
  (`analysis/lib/common.py`); DB bilgiyi yarı ham tutar (bir sonraki kurulumdan itibaren `NULL`).
- **Hedonik model dönem etkisi içermez** (kullanıcı kararı, 2026-09-23): dönemler havuzlanarak kestirilir;
  piyasa seviyesinin kayması teknik rapor §9'da aynı model ve yılın canlı ilanlarından okunur.
- Kapsam BMW + Audi; başka markalara ne kadar genellenebildiği ölçülmedi.

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
2. `EXAMPLE_NOTES` (2026-09-17, kullanıcı kararı; eski adı `HANDWRITTEN_EXAMPLE_NOTES`) — teknik rapordaki
   "Büyük hatalar nereden geliyor" örneklerinin altındaki açıklamalar, ilan metinleri okunarak yazıldı. Üstlerindeki
   tablo `analysis/08_large_errors.py`'den gelir. **2026-09-27'den beri notlarda elle yazılmış sayı yok:** her notun
   dayandığı karşılaştırma (seri sayısı, aynı adın öteki ilanları, karşılaştırma grubunun n'i ve medyanı) 08'de her
   koşuda hesaplanır (`examples[].compare`) ve nota oradan girer; notun iddiası (ör. "veride tek R8", "ilan aynı
   yılın 730d'leriyle uyumlu") veriyle kapılıdır. Her not bir örneğe `(model, yıl, fiyat)` ile bağlıdır; veri
   değişip örnek bulunamazsa ya da iddia tutmazsa üreteç **durur** — not sessizce bayatlamaz.

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
durdurmanın seçtiği ağaç sayılarının medyanı (`meta.repro.final_lgb_trees`, fold değerleri `cv_trees`).
`data/serving/serve/README.md` girdinin eğitimle birebir nasıl kurulacağını yazar; `encoders.pkl` segment
kuralını (`PERF_RE` dahil) ve %90 aralık için `CONFORMAL_Q`'yu taşır. LOFO'da taban ve çıkarma modelleri
aynı ağaç sınırıyla kurulur, durdukları tur `methodology.lofo_trees`'te.

**Rapor veri katmanlarını değil, ön işlemeyi anlatır (2026-09-26, kullanıcı kararı).** Teknik rapor yarı ham
veritabanından, gold'dan ve API'ye giden veriden bahsetmez; §1 toplanan veriye modelden önce uygulanan adımları
sırasıyla, düz bir listeyle verir (plaka, tekilleştirme, "belirtilmemiş" = yok, motor aralığı, hasar
bayrakları, eksik değer, aykırı değer, hedef). Gold yalnız `db/` tarafında ve `docs/database.md`'de anlatılır;
§1'in eski gold alt bölümünü besleyen `analysis/01_gold_contract.py` arşivde.

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
   doğrulukta ölçülebilir bir katkı bulunamadı. 2026-09-27'den beri bu sonuç arşivden alınmıyor: teknik raporun
   §10'u metnin katkısını `analysis/10_free_text.py` ile her koşuda canlı ölçüyor (ön kayıt
   `plans/10-text-contribution`; modelin OOF'u ve aynı fold'larda + açıklama TF-IDF/SVD).
2. **Fiyat iddialarını üreten dedektörlerde denetimle bulunmuş kusurlar vardı**
   (`archive/experiments/regex_audit/`). Kusurlar düzeltildi, ama düzeltmeden sonra da geriye fiyat
   iddiası kalmıyordu.

Arşivden hiçbir betik import edilmiyor ve 2026-09-27'den beri arşivden hiçbir veri ya da dondurulmuş sayı
da okunmuyor (aşağıdaki "Arşivden canlıya hiçbir şey" kararı). Metin bayrağının iki desen dedektörü
(dönüşüm, modifiye) `analysis/lib/text_flags.py`'de; desenler arşivdeki zincirden taşındı, modifiye kelime
listesi arşivlenmiş bir LLM çıkarım denemesinin sözcük dağarcığından damıtıldı. Her koşuda güncel metne
uygulandıkları ve çıktıları kullanıldığı için kalıyorlar. Hiçbir yeri beslemeyen beygir ve M/RS model dedektörleri
kaldırıldı; `analysis/frozen/` ve arşivdeki LLM çıkarım dosyası artık yok.

## Arşivden canlıya hiçbir şey (2026-09-27, kullanıcı kararı)

"Eski arşivden hiçbir şey alma … tam canlı olmayan ya da etkisiz olan şeyleri kullanma." Canlı zincirin her
girdisi mevcut betiklerin bu koşuda ürettiği şeydir; `archive/`'den, eski koşulardan ya da eski DB'lerden
kopyalanan veri, tablo, dondurulmuş sayı ya da elle yazılmış sayı girmez. Arşiv kökenli kod ve kural (segment
haritası, model ayarları, metin bayrağı kelime listesi) yalnız her koşuda güncel veriye uygulanıyor ve çıktısı
kullanılıyorsa kalır, kökeni yazılır; etkisiz olan tutulmaz. Bekçi: `tests/repo/test_no_archive_inputs.py`.
Bu kararla kalkanlar:
- dondurulmuş metin ablasyonu (`analysis/frozen/text_ablation.json`) ve arşivdeki LLM çıkarım dosyası → §10 canlı
  ölçüm;
- eski DB'den taşınan `dashboard_cache` / `options_cache` → silver ve gold'dan kaldırıldı, gold sözleşmesi fazla
  tabloyu reddediyor;
- §8 notlarındaki elle yazılmış sayılar → her koşuda hesaplanıyor;
- etkisizler: `08_large_errors`'ın eski D serileri (`old_d_group`), kullanılmayan metin gerekçeleri,
  `tools/metric_renames.py`'nin arşivdeki P0 kopyasına karşı kanıt komutları, iki ölü veri dosyası
  (`archive/obsolete/archive-inputs-2026-09-27/`).

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
  yalnız otomobil kategorisi (SUV kategorisi toplanmadı; veride yalnız birkaç SUV var) ve dört yakıt
  türü (elektrikli yok). Teknik rapor §1 bu
  filtreleri scraper'ın ayar dosyasından (`scraper/collection_config.json`) okur ve karşılıklarını veride
  sayar.
- **Kapsam yalnız TR plakalı araçlar** (2026-09-26). Mavi plakalılar (vergi rejimi farklı) ve plakası boş
  ilanlar (rejimi bilinmiyor) modele girmez. Mavi plakalılar veritabanında durur, gold'a gitmez
  (`db/gold_rules.json`). Sayılar teknik rapor §1'de.
- **Kasa tipi verilmeyen ilan doldurulmaz** (2026-09-26). Model adı kasayı belirlemiyor: aynı ad birden çok
  tiple satılıyor, site aynı adı bile tutarsız etiketliyor. Model bunları ayrı bir kategori olarak görür;
  kasa tipine göre medyan fiyat grafiği onları çizmez, başlıkta sayar.
- **Motor hacmi = kovanın üst sınırı, güç = alt ve üst sınırın ortalaması** (kullanıcının kuralı). Site
  ikisini ilanların bir kısmında kova olarak veriyor; teknik rapor §1 her adayı aynı modelin kesin değerli
  ilanlarıyla karşılaştıran tabloyu ve figürü basar. Seçilen aday en küçük farkı vermezse
  `analysis/01_engine_rule.py` durur.
- **"Belirtilmemiş" panel orijinal sayılır** (bilinçli karar); ağır hasar kaydında da aynı kural. Teknik
  rapor §1 kararın ölçüsünü basar. Çeviri analizde yapılır (`analysis/lib/common.py`); veritabanı bilgiyi
  yarı ham tutar (`NULL`).
- **Hedonik model dönem etkisi içermez** (kullanıcı kararı, 2026-09-23): dönemler havuzlanarak kestirilir;
  piyasa seviyesinin kayması teknik rapor §9'da aynı model ve yılın canlı ilanlarından okunur.
- **Yeniden eğitim sabit bir PSI eşiğine bağlanmaz** (2026-09-27). Kayma izlenir, model yeni taramalarla
  yeniden eğitilir. Gerekçe veride: PSI eşiğin çok altındayken bile aynı taramada eğitilen modelin hatası
  test ufku uzadıkça artıyor (teknik rapor §9).
- Kapsam BMW + Audi; başka markalara ne kadar genellenebildiği ölçülmedi.

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
  model → kalibrasyon → serbest metin (zaman bölümü 2026-11-22'de çıktı).
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
metrik JSON'unun `report` bölümünde: hata bantları, conformal kapsaması, Mondrian bantları …). Derleyici yalnız
yayımlanmış sayılar üzerinde biçim düzeyinde aritmetik yapar (ör. modelin tabandan yüzde iyileşmesi,
(taban−model)/taban).
Gürültü tabanından türeyen `1.42×` ve `₺33K` 2026-09-20'de raporlardan çıktı; buradan da kalktılar.

**Veriyle çelişen elle yazılmış notlar düzeltildi (2026-09-16).** Üreticilerde elle yazılmış üç
cümle ölçümle çelişiyordu; rapor artık bu notları değil sayıları okur:

- `kmeans_selection.not` "silhouette k=3'te en yüksek" diyordu; ölçümde k=3 silhouette'in en yüksek
  olduğu yer değil. k=3 silhouette ile değil, yorumlanabilirlik için sabit. Not artık veriden kuruluyor.
  2026-09-29'dan beri kümeler hiç kurulmuyor; §5 yalnız silhouette taramasını tek paragrafla veriyor.
- `numeric_correlation.not` "VIF hepsi <3" diyordu. 2026-09-23'ten beri VIF kurulan hedonik modelin kendi
  tasarım matrisinden hesaplanıyor; 2026-09-28'den beri teknik rapor en yüksek değerini (ortalanmış ve ortalanmamış)
  tek cümleyle veriyor.
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

## Sadeleştirme (2026-09-28, kullanıcı onayı)

Dışarıda hazırlanan bir sadeleştirme listesi madde madde değerlendirildi. Liste sitenin eski kopyasına bakıyordu
(ΔR² 0,0015 ve hedonik dönem etkisi +%5,3 oradan); tek kaynak üreteçtir, site raporun markdown'ını kopyalar.

- **Rapordan çıkanlar:**
  - Cramér's V, permütasyon tabanları ve Theil's U matrisi (asimetri tablosu kalır);
  - Pearson haritası ve |r|>0,5 tablosu (Spearman çiftleri tek cümle);
  - VIF tablosu, yakıta göre cc–hp tablosu, varsayım testleri paragrafı (her biri tek cümle);
  - en iyi 5 ve iki en kötü 6 tablosu (bulgu cümleleri kaldı);
  - teknik rapordaki lira paneli (karar notunda kalır);
  - kayma p-değerleri, ayrık tablo ve Holm (taramalar ilan paylaşıyor, test bağımsızlık varsayar);
  - LOFO kapsam tablosu (tek cümle).
- **Yöntemi değişenler:**
  - Hedonik güven aralıkları modele göre kümelenmiş standart hatalardan. Satır bootstrap'i ilanları bağımsız
    sayıyordu; 22 seri kümelemek için az ve dengesiz.
  - Hedonik iki sütun: segment kontrolü ve model kontrolü. Karar notu model sütununu basar (kullanıcı kararı).
  - Fiyat düzeyine göre her gruplama tahmin edilen fiyattan; gerçek fiyata göre gruplama ortalamaya dönüş üretir.
  - Kapsama çapraz kalibreli. Banda göre hata payı (Mondrian) ön kayıtla ölçüldü (`plans/08-mondrian-coverage`,
    H1 doğrulandı); servis edilen tek q değişmedi.
  - Backtest manşet kurulumla koşar (son insample bloğu manşet OOF'a birebir eşit); model kümeli bootstrap %95
    aralıklar ve tek dönem/kümülatif eşli karşılaştırma. "Saf zaman etkisi" ve "daha çok veri, daha az hata"
    iddiaları aralıkların taşıdığı kadara indi.
- **Alınmayanlar:**
  - hedonik dönem kuklası (aşağıda);
  - log ölçekte EMD ve hücre değişiminin log ortalaması;
  - LOFO'ya altı kategorik öznitelik (2026-09-25 kararı);
  - veriden gelmeyen ±%1 yeniden eğitim eşiği.
- `tools/snapshot_metrics.py --drop-exemption`, bilinçli kaldırılan bir anahtarın muafiyetini siler (bootstrap
  süresi böyle kalktı).
- **Bağımsız denetim (aynı gün) ve düzeltmeleri:**
  - Model sütunu tekil tasarımda pseudo-inverse ile çözülüyordu; kümeli SH'ler küçük değişikliklerde patlıyordu. Artık
    model içi dönüşüm (sabit etkiler) kullanılıyor; tahminler aynı, tasarım tam ranklı, testli.
  - Sütun farkı cümlesi artık terim başına kümeli fark testine dayanıyor.
  - Duyarlılıklar eklendi: seriye göre kümeleme ve motor değeri tutarsız ilanlar olmadan +100 hp.
  - "Lira ölçeğinde en büyük hatalar düşük tahmin" bir bulgu değil: simetrik log hatası da yaklaşık 68/100 düşük
    tahmin verir; rapor bunu söylüyor.
  - Ufuk artık sabit bir test kümesinde de ölçülüyor (aynı ilanlar, eğitim taraması eskidikçe hata artıyor).
  - **Ön kayıttan sapma:** `plans/08-mondrian-coverage`'ın ölçütü rastgele katlarda neredeyse tanım gereği tutuyordu;
    ölçüt kodu sınadı, yöntemi değil. Asıl soru yeni bir ön kayıtla soruldu (`plans/09-forward-coverage`): servis
    tarifi ileriye uygulanınca banda göre pay en ucuz bandı her kurulumda düzeltiyor (H1 doğrulandı), ama her bantta
    %88'i tutmuyor (H2 çürüdü) ve emsalsiz ilanlarda kapsamayı düşürüyor. Karar notunun önerisi bu sonuca göre yazıldı.
- **İkinci liste (2026-09-29, kullanıcı onayı):**
  - §5 tek paragraf: veride belirgin küme yok (silhouette hep 0,25 altında); sabit k'li kümeler, PCA ve üç figür
    kalktı, numaralar kaymadı.
  - §3'teki Theil's U tablosu iki cümleye indi.
  - §7'deki "en iyi durum" örnek tahmin tablosu kalktı; yerine servis edilen modelin ağaç kuralı.
  - İki raporda da yeniden eğitim önce hataya bakar (sabit test kümesindeki fark), kayma tanı içindir.
  - §1 motor alt bölümü olduğu gibi kaldı (2026-09-26 kararı yinelendi); LOFO'daki altı öznitelik açılmadı.
- **Site karşılaştırması (2026-10-12 / 10-19, kullanıcı onayı):**
  - §6'daki motor tutarlılık sayısı §8'in iki sayısını da veriyor: §8 bütün ilanlarda 15 ilan işaretliyor, hedonik
    örneklemde 14'ü var (birinin geçerli motor hacmi yok).
  - §9 girişi artık sabit test kümesindeki en eski eğitim farkını veriyor; yeniden eğitimi yalnız ölçülen hataya
    bağlıyor, kayma tanı içindir. Eskiden farklı test kümelerindeki %6.29 → %7.34'ü manşet yapıyordu.
  - §8'deki süreç dili (ön kayıt dosya yolu, "kodu sınadı" cümlesi) okuyucu diline çevrildi.
  - Figür 15 karar notundan çıktı: "Eski dönemleri atma" maddesinin kanıtı gibi okunuyordu ama zamansal olmayan düz
    OOF'u çiziyor; maddenin dayanağı eşli karşılaştırma. Teknik §9'da "zamansal değil" açıklamasıyla kaldı.

## İş / teknik ayrımı

**İş notu** = ne yapmalı, ne kadar para. Yöntem adı geçmez (MAPE, R², conformal, OOF hepsi
teknikte); her şey ₺ ve sade yüzde. **Teknik rapor** = protokol, kontroller, sınırlar.
İkisi **aynı hesaptan** beslenir: ortak değerler `builders/report_lib/report_common.py::derive`'da bir kez türetilir,
iki şablona oradan gider → bir rakam iki raporda farklı çıkamaz. Bazı figürler (çeyrek hatası, kapsama) iki
raporda da geçer (lira çeyreği 2026-09-28'den beri yalnız karar notunda): karar notunda karar için, teknik raporda kanıt olarak — bilinçli tekrar.

## Metin analizi neden arşivde

`archive/analysis-history/text_analysis/` — kod, metrikler, figürler ve raporlar diskte duruyor ama
**yayımlanan işin parçası değil** (`.gitignore`'da). İki gerekçe:

1. **Ana bulgusu olumsuzdu.** Yapısal modele metin öznitelikleri eklendiğinde çapraz-doğrulamalı
   doğrulukta ölçülebilir bir katkı bulunamadı. 2026-09-27'den beri bu sonuç arşivden alınmıyor: teknik raporun
   serbest metin bölümü metnin katkısını `analysis/10_free_text.py` ile her koşuda canlı ölçüyor (ön kayıt
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
- dondurulmuş metin ablasyonu (`analysis/frozen/text_ablation.json`) ve arşivdeki LLM çıkarım dosyası → serbest metin
  bölümünde canlı ölçüm;
- eski DB'den taşınan `dashboard_cache` / `options_cache` → silver ve gold'dan kaldırıldı, gold sözleşmesi fazla
  tabloyu reddediyor;
- §8 notlarındaki elle yazılmış sayılar → her koşuda hesaplanıyor;
- etkisizler: `08_large_errors`'ın eski D serileri (`old_d_group`), kullanılmayan metin gerekçeleri,
  `tools/metric_renames.py`'nin arşivdeki P0 kopyasına karşı kanıt komutları, iki ölü veri dosyası
  (`archive/obsolete/archive-inputs-2026-09-27/`).

## Kayma çıktı, tek ileri test kaldı (2026-11-22, kullanıcı kararı)

Kullanıcı zaman ve kaymayı analizden çıkarmak istedi ("çok karışıyor"). Karar, verinin nasıl toplandığına
bakılarak verildi.

**Araştırma bulguları:**
- **Taramalar stok dökümü.** Her tarama o gün sitede yayında olan bütün BMW/Audi ilanlarını alıyor: 10.901 · 11.254 ·
  11.478 · 11.526 ilan.
- **Aralar düzensiz.** Taramalar arasında 9, 53 ve 98 gün var. Önceki taramalarda görülmemiş ilan payı %26, %63 ve %77.
- **`listing_date` yayın tarihi değil.** Birden çok taramada görülen 10.831 ilanın 6.057'sinde (%56) değişmiş; son
  güncelleme günü. Taramadaki medyan "ilan yaşı" bu yüzden 10–13 gün. Tek güvenilir zaman ekseni dört tarama.
- **İlan yaşının etkisi küçük.** OOF artığı ilan yaşına göre neredeyse düz: medyan −%0,46 ile +%0,12 arası.

**Neden kayma çıktı.** Stok dökümleri akışı görmüyor. Uzun süre satılmayan ilan birçok taramaya giriyor, iki tarama
arasında açılıp kapanan hızlı ilan hiçbirine girmiyor. İki tarama arasındaki fiyat dağılımı farkı bu yüzden üç şeyin
karışımı: piyasanın gerçekten kayması, hangi ilanın stokta kaldığı ve bileşim. PSI, KS, EMD ve yoğunluk eğrileri
bunları ayıramıyor; "son görüldüğü taramaya göre artık" da hayatta kalmayla karışık.

**Neden tek ileri test kaldı.** "Eskiden öğren, sonra gelen yeni ilanı fiyatla" sorusu iyi tanımlı ve modelin
kullanımına en yakın ölçüm. Aralığın emsalsiz araçlarda tutmadığının tek kanıtı da bu test.
- **Kurulum:** son taramadan önce görülen bütün ilanlarla eğit (ilan başına son fiyat), son taramada ilk kez görülen
  ilanlarda ölç.
- **Ölçülenler:** MAPE ve modele göre yeniden örneklemeli %95 aralık; servis tarifinin ileri kapsaması (genel, bant
  başına, emsalsiz).
- **Yer:** `analysis/08_forward_test.py`; raporda §8'de bir paragraf.

**Çıkanlar:**
- `analysis/09_drift.py` ve `09_backtest`'in tek, biriken, eşli, ufuk, insample ve tarama başına kolları;
- figür 13, 14, 15; teknik raporun "Zaman" bölümü (serbest metin §10 → §9);
- `08_large_errors`'ın son görüldüğü taramaya göre artığı ve canlı/kalkan karşılaştırması; `08_residuals`'ın dönem
  cümlesi;
- iş notunun yeniden eğitim, PSI ve rejim maddeleri.

Modelin zamanı görmediği raporda tek bir sınır cümlesiyle kalır. Hedonik modelde dönem etkisi zaten yoktu
(2026-09-23).

**Geri çekilen plan.** `plans/09-forward-coverage` (kilitli, 2026-09-28) dokuz ileri kurulumu ön kayda almıştı:
- H1 doğrulandı: banda göre pay en ucuz bandı her kurulumda düzeltiyor.
- H2 çürütüldü: her bantta %88 tutmuyor.
- Hipotezler "her kurulumda" diye yazıldığı için tek kurulum onları taşımaz. Plan geri çekildi; dizin yerelde
  `archive/plans-withdrawn/`'da. Dokuz kurulumun sayıları ve kanıtı commit `fd0376a`'da (`metrics/09_backtest.json`).
- Yeni plan açılmadı: tek testin sonucu dokuz kurulumdan biri olarak zaten görülmüştü; şimdi "ön kayıt" sonradan
  yapılmış olurdu.

## Servis edilen aralık: ölçek modeli ve isabet olasılığı (2026-11-23, kullanıcı onayı)

**Karar.** Servis edilen %90 aralık artık araca göre. Küçük bir ikinci model (ölçek modeli, s) bu tür araçta beklenen
hata büyüklüğünü tahmin eder; aralık `expm1(log tahmin ± k·s)`. Yanında bir isabet olasılığı verilir: ilan fiyatının
tahminin ±%10 yakınında olma olasılığı. Fiyat tahmini değişmedi: ana model dosyası birebir aynı. Tarif ve kod
`analysis/lib/conformal.py`'de; plan `served_interval_plan.md` (repo kökü).

**Önceki durum.** Herkese tek pay (q = 0,1412, yani −%13,2 / +%15,2). Genel kapsama %90, ama gruplarda tutmuyordu:
0–7 yaşta %97, 19+ yaşta %75, emsalsiz araçta %71, F segmentinde %66.

**Denenen kollar** (OOF, çapraz kalibreli, 29.988 ilan; ileri test 8.889 yeni ilan). IS log uzayında aralık skoru,
küçük daha iyi:

| kol | OOF kapsama | OOF IS | OOF emsalsiz | ileri kapsama | ileri IS | ileri emsalsiz | sonuç |
|---|---|---|---|---|---|---|---|
| tek q | 89,99 | 0,4211 | 70,7 | 87,40 | 0,4438 | 68,35 | raporda kalır (eski servis) |
| bant başına q (plan 08) | 90,01 | 0,4059 | 66,8 | 87,99 | 0,4296 | 65,32 | raporda kalır (ön kayıtlı) |
| CQR | 90,00 | – | 73,7 | 88,68 | – | 70,03 | koda girmedi |
| bant × emsal | 90,06 | 0,3929 | 90,8 | 87,68 | 0,4178 | 90,91 | koda girmedi |
| ölçek modeli | 90,09 | 0,3657 | 90,3 | 86,83 | 0,3977 | 88,55 | **servis edilir** |
| üç girdili ölçek (tahmin, yaş, emsal) | – | 0,3780 | – | – | fark gürültü | – | koda girmedi |

- **Ara yol** (k'yı emsal kovası başına hesaplamak) ileri testte hiçbir şey değiştirmedi; koda girmedi.
- CQR, bant × emsal ve üç girdili kolun sayıları salt okunur denetim betiklerinden; canlı zincirde yoklar.

**Neden ölçek modeli.** Bant × emsal yalnız gördüğümüz iki hatayı (ucuz bant, emsalsiz araç) düzeltiyordu. Ölçek
modeli öteki eksenleri de düzeltiyor: girdisi olan eksenlerin 42 grubunda tek q'da 18 grup hedeften 5 puandan fazla
sapıyor, ölçek modelinde hiçbiri. OOF'ta en iyi aralık skoru onda. 30 girdiyi üç girdiye tercih etme gerekçesi:
kullanıcının girdiği hasar bilgisi payı etkilemeli (ağır hasarlıda kapsama 80,8'e karşı 89,7). İleri testte iki
ölçek modeli arasındaki fark gürültü içinde.

**Vazgeçilen şey.** Bant × emsal'in 16 hücresinin her birinde ayrı bir kalibrasyonu vardı. Ölçek modelinde gruplardaki
iyi kapsama ampirik; daha keskin aralık ve öteki eksenler için o hücre kalibrasyonundan vazgeçildi.

**Dürüstlük notları.**
- **Seçim sonuçtan sonra.** Yöntem, s'nin ayarları (300 ağaç, 15 yaprak, yaprakta en az 100 ilan, alt sınır 0,02),
  monotonluk kısıtı ve ±%10 sonuçlar görüldükten sonra kondu. Hiçbirinde arama yapılmadı. Bağımsız test kalmadı.
- **%90 yaklaşık tutar, aynı dönem için.** Servis edilen merkez ve s bütün veriyle, skorlar kat modelleriyle kuruluyor;
  bu kurulumun sonlu örneklem güvencesi yok. Zaman içinde tutmaz: sonradan gelen ilanlarda 86,8.
- **Sızıntı payı ölçüldü.** Testler yalnız s'nin puanlanan katı görmediğini kanıtlar; hatalar ana modelin çapraz
  uydurmasından gelir. Her dış kat için her şey (ana model, iç OOF hataları, s, k) öteki dört kattan yeniden kurulunca
  kapsama 90,11, emsalsiz 90,03, IS 0,3669 çıktı. Fark yok sayılır.
- **Kararlılık.** 10 tohum ve 7 ayarda OOF kapsama 90,00–90,25, k 2,075–2,105, ileri kapsama 86,69–87,06. Ama aynı
  aracın payı tohumdan tohuma medyan %2,5, %95'lik %9,5 oynuyor. Tohum ortalaması alınmadı: servis tek modelle
  deterministik, beş model servis etmek bu kazanca değmez.
- **Monotonluk bir tasarım kuralı.** "Daha çok emsal payı hiç genişletmez." Kapsamaya ölçülebilir katkısı yok
  (kısıtlı ve kısıtsız ileri testte aynı).
- **Dar aralık önce eskir.** Fiyat seviyesi kayarsa en dar aralıklar kapsamayı ilk kaybeder; rapor bunu bir varsayım
  hesabıyla gösterir (`report.scale.level_shift`).
- **s'nin görmediği bilgi.** Satıcı, il, renk ve ilan metnindeki bir kusur s'de yok; bu eksenlerde ölçek modeli tek
  q'dan ne iyi ne kötü.
- **Hedef ilan fiyatı**, satış fiyatı değil. **Model adı serbest metin:** emsal eğitimdeki dizeyle birebir aranır;
  yazım hatası emsali 0 yapar ve aralığı genişletir (ihtiyatlı yönde).

**Ön kayıt.** Kilitli `plans/08-mondrian-coverage` tek q ve bant kollarını kapsıyor; ölçek modeli onun dışında ve o plan
olduğu gibi duruyor. Yeni ön kayıt açılmadı: sonuçlar görülmüştü, sonradan "ön kayıt" yanıltıcı olurdu
(`plans/09-forward-coverage` geri çekilirken izlenen yol).

**Sitede sunum.** İsabet "10'da N" olarak gösterilir, aşağı yuvarlanır ve en çok 9 olur (yuvarlamayla araçların %11'i
"10'da 10" görürdü, gerçekleşen %94–96,5). Cümleler dönemi taşır: "Ocak–Haziran 2026 ilan fiyatlarına göre".
Yüzde, s, k ya da "yüksek / orta / düşük" etiketi gösterilmez.

**Zaman denetimi (aynı gün).** Kullanıcı "ileri testteki +2,2 log puan fiyat kayması mı, proje kayma analizine uygun mu"
diye sordu; üç bağımsız denetim yapıldı.
- **+2,2 fiyat artışı değil.** İleri testte işaretli hata ortalama +2,46, medyan +2,02. Bunun +1,16'sı eğitim
  kümesinin kuruluşundan: eğitim fiyatlarının %46'sı ocak tarihli ve marttan önce çıkmış ilanlara ait; çıkan ilanlar
  aynı gün kalanlardan ~2 log puan ucuz. Fiyat seviyesi 98 günde yaklaşık +1 ile +1,5 (yalnız mart stokuyla eğitim:
  +1,30 [1,0–1,6]). Nominal ilan fiyatı; enflasyondan arındırılmadı.
- **PSI, KS, EMD geri gelmez.** Ardışık taramalarda fiyat PSI'ı ortak ilanları sabit tutan gürültü tabanının altında;
  0,10 eşiği yalnız `model` ve konumda aşılıyor, orada değer gürültüye eşit. "Kayma çıktı" kararı doğrulandı.
- **Tarih modele girmez, sabit ofset uygulanmaz (kullanıcı kararı).** Tarama girdisi çıkan ucuz ilanları tarih diye
  öğreniyor (9 gün için +1,28; stokun hareketi 0,42). Hedefi kaydırıp yeniden eğitmek sabit ofsetle aynı sonucu veriyor.
  Sabit ofset (+1,73) Haziran ilanlarında yanlılığı sıfırlar ve kapsamayı 88,7'den 89,3'e çıkarır, ama ileri tarihte
  bir test yapılmayacağı için doğrulanamaz; uygulanmadı. Model son taramanın ilanlarının ortalama %1,7 altında kalır;
  bu bir sınırlama olarak yazılır.

**Repo dışı.** API ve site (`sadik-portfolio`) bu repodan değiştirilmez; şartname planın 7–9. adımlarında, parite
örnekleri `data/serving/serve/parity_examples.json`'da.

## Hedonik bölüm yeniden kuruldu (2026-11-23, kullanıcı isteği)

**Neden.** İki sütunlu katsayı tablosu ("segment kontrolü", "model kontrolü") soruyu tersinden soruyordu. Kullanıcının
görmek istediği: aracın ne olduğu bilinmeden fiyatın ne kadarı açıklanıyor, yalnız kimlikle ne kadarı, ve kimlik
sabitlenince her etki nasıl değişiyor. Hedonik regresyon çıkarılmadı: km, ağır hasar ve değişen parça etkilerini bu
kadar temiz veren başka bir bölüm yok (SHAP'ta yaş ile km'nin payı tahmin modeline göre değişiyor).

**Eklenenler** (`analysis/06_hedonic.py`, `report.hedonic`):
- **Merdiven:** altı regresyonun örneklem içi ve 5 katlı kat dışı R²'si. Kat dışı: yalnız segment, marka, yakıt, vites
  0,38; yalnız durum (yaş, km, hasar, boya, değişen) 0,73; yalnız araç modelinin adı 0,82; durum + motor 0,87; segment
  kontrolü 0,93; araç modeli kontrolü 0,96.
- **Kimlik kontrolsüz üçüncü sütun.** Km kimlik olmadan −%10, kimlik sabitken −%15; +100 hp kimlik olmadan +%50,
  segment girince +%20. Yani kontrolsüz km yanıltıyor, gücün "etkisi"nin çoğu aracın sınıfı.
- **Yıllık basamaklarla yaş** (araç modeli içinde yaş kuklaları ve km spline'ı): yıl başına −%4,7 ile −%6,8, yaşa göre.
  İkinci dereceden terimin medyan araçtaki tek sayısı (−%5,8) fonksiyon biçimine bağlıydı; denetimde esnek biçimler
  −%4,1…−%4,4 verdi. Veri tek dönem: ölçülen bir model yılının farkı, bir aracın yaşlanması değil.
- **Sayıya göre parça:** boyalı parça doğrusal değil (1: −%2,3; 2: −%3,6; 5 ve fazlası: −%7,0; doğrusal katsayı parça
  başına −%1,0 diyordu). Değişen parça doğrusala yakın (1: −%3,7; 2: −%6,4; 3: −%9,3).

**İş notundan çıkan.** "+100 hp" satırı: aynı araç modeli içinde güç neredeyse değişmiyor (735 araç modelinin 354'ünde
değişiyor, araç modeli içi yayılım 9 hp), sayı tek bir araç modeline duyarlı. Teknik raporda uyarılarıyla kalır.
Başlıktaki "fiyatı ne kadar oynattığı" ifadesi nedensel olduğu için değişti.

**Ön kayıt yok.** Hepsi sonuçlar görüldükten sonra eklendi; sonradan ön kayıt yanıltıcı olurdu. Yayımlanan iki sütunun
sayıları değişmedi.

**Kelime kuralı (aynı gün).** Raporlarda "model" iki şeyi anlatıyordu. Verideki kolon için "araç modeli", LightGBM için
"tahmin modeli" yazılır; ikisinin karışabileceği her yerde.

## Gizlilik

1. `ad_id` hiçbir rapora ya da site verisine yazılmaz; yalnız veritabanında ve onun yanındaki
   `data/duplicate_ad_ids.csv`'de kalır (ikisi de yerel, `data/` git dışı). İlan metni de hiçbir
   markdown'a girmez.
2. **İlan düzeyindeki satırlar bilinçli olarak yayımlanır** (karar 2026-09-16): §7'deki örnek
   tahminler ve §8'deki örnekler — model · fiyat · yaş · km (en iyi 5 ve iki en kötü 6 tablosu 2026-09-28'de
   çıktı). Nadir model + tam fiyat aramayla bulunabilir; bu kabul
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
  piyasa seviyesinin kayması 2026-11-22'den beri ölçülmüyor (aşağıda "Kayma çıktı, tek ileri test kaldı"). 2026-09-28'de
  yeniden önerildi (+%5,3) ve alınmadı: sayı sitenin eski Temmuz verisinden, veride "dönem" ilanın son görüldüğü
  tarama olduğu için hayatta kalmayla karışık.
- **Model zamanı görmez** (2026-11-22). Taramaların fiyatları birlikte öğrenilir; kayma, dönem etkisi ve yeniden
  eğitim önerileri analizden çıktı. Raporlar bunu tek bir sınır cümlesiyle söyler; son taramanın yeni ilanlarındaki
  hata teknik rapor §8'de (`analysis/08_forward_test.py`). Önceki kural ("yeniden eğitim sabit bir PSI eşiğine
  bağlanmaz", 2026-09-27/28) onunla birlikte kalktı.
- Kapsam BMW + Audi; başka markalara ne kadar genellenebildiği ölçülmedi.

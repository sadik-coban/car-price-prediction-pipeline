# İkinci El Araç Piyasası Analizi — Teknik Rapor

Bu rapor iki soruya yanıt arar: İkinci el araç fiyatını ne belirler ve model bunu ne kadar isabetle öngörebilir? Analiz; **29.988** TR plakalı BMW/Audi ilanında veri temizliği ve sızıntı kontrolünden geçerek kontrollü fiyat etkileri, piyasa yapısı, model karşılaştırması ve zamansal testleri ortaya koyar. LightGBM ortalama **%6.49** yüzde hatayla (MAE: **₺110K**, R²: **0.9745**) çalışarak emsal medyanına (aynı model ve yıl; emsal yoksa daha geniş medyan) göre ortalama mutlak hatada (MAE) **%43** daha iyi sonuç verir. Paylaşılan tüm metrikler, modelin daha önce görmediği veriler üzerinden **5-fold out-of-fold** kurgusuyla hesaplanmıştır. Hedef ilan fiyatıdır, satış fiyatı değil: satış fiyatı pazarlıkla bundan ayrılır.

## 1. Veri temizleme ve sızıntı tespiti

**TR plakalı 45.159 tarama kaydı → 29.988 ilan.** Aradaki 15.171 satır aynı ilanın sonraki taramalarda yeniden görülmesi; `ad_id` başına en son kayıt alındı. Tekrarlar bilgi taşıyor ama model bunu kullanmıyor: birden çok taramada görülen 10.831 ilandan 4.703 tanesinin fiyatı değişmiş (2.801 indirim, 1.854 zam; 48 ilan eski fiyatına döndü).

Medyan ilan fiyatı ₺1.55M, ₺0.85M–₺3.43M arası (P10–P90).

**Kapsam: yalnız TR plakalı araçlar.** 22 mavi plakalı ilan (Türkiye'de oturan yabancıların aracı) modele ve analize alınmadı: vergilendirme rejimleri farklı, modeli ve analizi yanıltır. Plaka bilgisi boş olan 54 ilan da, hangi rejime girdiği bilinmediği için alınmadı.

**Kapsam: toplama filtreleri.** Veri Audi ve BMW ilanlarından, sitenin otomobil kategorisinden şu filtrelerle toplandı: fiyat ₺300K–₺6.50M, en fazla 700.000 km, 2005 ve sonrası model yılı, yakıt Benzin, Dizel, Hibrit, LPG. Üç sonucu var:

- **Fiyat sağdan kesik.** En pahalı ilan tam tavanda (₺6.50M); tavanda 10 ilan var, üstünde hiç yok. Tavanın üstündeki araçlar veride değil; en pahalı uçtaki tahminler bu sınırla birlikte okunmalı.
- **Yaş en fazla 21.** 2005 model yılında 541 ilan var; daha eski araçlar toplanmadı, yani en yaşlı kova toplama sınırına dayanıyor.
- **Gövde ve yakıt.** Yalnız otomobil kategorisi toplandı, öteki kategoriler toplanmadı: veride 6 SUV var. Yakıt filtresi elektrikliyi dışarıda bırakıyor: 0 elektrikli ilan.

### Ön işleme ve filtreler

Toplanan veriye modelden önce sırasıyla şunlar uygulandı:

1. **Plaka:** yalnız TR plakalı satırlar alındı: 45.315 tarama satırından 45.159 satır. 22 mavi plakalı ve plaka bilgisi boş 54 ilan tümden dışarıda kaldı (gerekçesi yukarıda).
2. **Tekilleştirme:** `ad_id` başına en son tarama: 45.159 satır → 29.988 ilan.
3. **"Belirtilmemiş" = yok:** belirtilmemiş panel orijinal, belirtilmemiş ağır hasar kaydı ağır hasarsız sayıldı (aşağıda).
4. **Motor aralığı → tek sayı:** hacim aralığın üst sınırı, güç alt ve üst sınırın ortalaması (aşağıda).
5. **Hasar:** 39 ham panel bayrağı 12 özniteliğe indirildi (aşağıda).
6. **Eksik değer:** sayısal özniteliklerde doldurma yok, boş değer modele boş gidiyor; kategorik boşluk ayrı bir `missing` kategorisi (aşağıda).
7. **Aykırı değer atılmadı:** fiyat, km ve yaş kırpılmadı; TR plakalı her ilan modelde.
8. **Hedef:** `log1p(price)`.

### Ölçek

| kalem | değer |
|---|---:|
| TR plakalı tarama kaydı (tüm taramalar) | 45.159 |
| tekil ilan (`ad_id` dedup) | 29.988 |
| tarama dönemi | 4 (2026-01-18 – 2026-06-27) |
| ham kolon (besleme) | 117 |
| modele giren öznitelik | 25 |
| BMW / Audi | 17.896 / 12.092 |
| hedef | `log1p(price)` |

### Üç veri katmanı

1. **Yapısal** — yaş · km · motor gücü/hacmi · kasa · yakıt · vites · çekiş · segment.
2. **Hasar / ekspertiz** — 13 kaporta paneli × {değişen, boyalı, lokal boya} + ağır hasar kaydı. Bu 39 ham bayrak modele 12 öznitelik olarak giriyor: tavan · kaput · bagaj tek panel olduğu için **durum** (orijinal/lokal/boyalı/değişen), kapı · çamurluk · tampon ise grup içi **sayı** (kapı 0–4, çamurluk 0–4, tampon 0–2).
3. **Serbest metin** — satıcı açıklaması; modelde **kullanılmıyor**. Ölçüldü: log R²'ye katkısı 0.0008; ayrıntısı §10'da.

**"Belirtilmemiş" panel orijinal sayıldı.** Site her panel için beş cevaptan birini veriyor: orijinal, belirtilmemiş, boyalı, lokal boyalı, değişmiş. Modele giren ilanlarda 59.651 panel (%15.3) belirtilmemiş; 1.235 ilanda 13 panelin hiçbiri belirtilmemiş. Bu cevap bilinçli bir kararla orijinal gibi kodlandı; gerekçe, satıcının hasarı yazmayı unutmuş olabileceği ama hasar olmamasının daha olası sayılması.

**Ağır hasar kaydında da aynı kural.** Modele giren ilanlarda sayfanın ağır hasar bilgisi vermediği ("Belirtilmemiş" ya da alan yok) ilanların payı %68.2; bunlar ağır hasarsız sayıldı. Ağır hasarlı olup bunu belirtmeyen bir ilan modelde ağır hasarsız görünür.

### Motor gücü ve hacmi: aralıktan tek sayıya

Site motor hacmini ve gücünü ilanların bir kısmında kesin değer, bir kısmında **aralık** olarak veriyor: modele giren ilanlardan 7.865 tanesinde hacim, 7.797 tanesinde güç aralıklı (en sık aralıklar 1401–1600 cc ve 151–175 hp). Model tek sayı kullanıyor. Kural: **hacim = aralığın üst sınırı, güç = alt ve üst sınırın ortalaması** (açık uçlu aralıkta bilinen sınır). Kural veriyle sınandı: aralıklı ilanın üç adayı, aynı modelin kesin değerli ilanlarının medyanıyla karşılaştırıldı — hacimde 7.176, güçte 7.148 ilan (kesin değerli ilanı olan modellerde).

![Sitenin verdiği değerler: alt × üst sınır](figures/tr-30-engine-bounds.png)

Örnek: 1401–1600 cc aralığı verilen bir ilanın aynı modeli, hacmi tek sayı verilen ilanlarında en çok 1598 cc yazıyor; bu ilan için adaylar alt sınır 1401 (197 cc uzak), orta nokta 1500.5 (97.5 cc), üst sınır 1600 (2 cc). Aşağıdaki tablodaki sayı, bu uzaklıkların bütün aralıklı ilanlar üzerindeki medyanı; her modelin referansı, kendi tek sayılı ilanlarının medyanı.

| aday | hacim: medyan mutlak fark | güç: medyan mutlak fark |
|---|---:|---:|
| alt sınır | 194 cc | 17 hp |
| orta nokta | 95.5 cc | **7 hp** |
| üst sınır | **5 cc** | 14 hp |

Hacimde kesin değer aralığın üst sınırına çok yakın (aralık içindeki medyan konumu %97.5, alt çeyrek %83.9; en sık aralığı paylaşan modellerin en sık kesin değeri 1598 cc), bu yüzden üst sınır neredeyse tam isabet ediyor; en az 100 ilanlı aralıkların hepsinde (5/5) en yakın aday üst sınır. Güçte en yakın aday güç düzeyine göre değişiyor (en az 100 ilanlı aralıklar): 101–125 hp orta nokta; 126–175 hp üst sınır; 176–225 hp alt sınır; 226–250 hp alt sınır ile orta nokta berabere; 251–275 hp alt sınır. Kesin değerin aralık içindeki konumu da bu yüzden dağınık (çeyrekler %29.2–%79.2). Tek bir kural olarak orta nokta, bütün aralıklı ilanlarda medyan farkı en küçük aday. Aynı modelin kesin değer medyanı aralığın içine düşüyor: hacimde %97.9, güçte %84.2 ilanda; güçte dışarıda kalanlarda birden çok kesin güç değeri olan modellerin payı %90.3 — aynı model adı farklı motor seçenekleri taşıyor, yani bu oran sitenin aralığının değil referansın kabalığını gösteriyor. Kural veriyle çelişirse (seçilen aday en küçük medyan farkı vermezse) üreteç durur.

Sebep motorların litre etiketi: "1.6" diye satılan bir motorun gerçek hacmi 1598 cc gibi, etiketin birkaç cc altında. Sitenin aralıkları da bu etiket değerlerinde bitiyor (1401–1600 cc), bu yüzden gerçek hacim neredeyse her zaman aralığın üst sınırına çok yakın. Güçte böyle bir etiket yok; gerçek değerler aralığın içine dağılıyor, orada ortalama daha iyi tutuyor.

![Aralıktan tek sayıya: adayın aynı modelin kesin değerinden uzaklığı](figures/tr-29-hp-cc-rule.png)

### Tutulan öznitelikler (25)

Model (`model`) · Seri (`series`) · Marka (`brand`) · Kasa Tipi (`kb_body_type`) · Çekiş (`kb_drivetrain`) · Segment (`segment`) — seri ve model adından türetildi · Vites Tipi (`kb_transmission`) · Yakıt Tipi (`kb_fuel`) · Tavan Durumu (`roof_state`) · Kaput Durumu (`hood_state`) · Bagaj Durumu (`trunk_state`) · Yaş (yıl) (`vehicle_age`) · Kilometre (`gb_mileage`) · Motor Gücü (hp) (`power_hp_val`) · Motor Hacmi (cc) (`engine_cc_val`) · Kapı Değişen (`door_changed`) · Kapı Boyalı (`door_painted`) · Kapı Lokal Boya (`door_local`) · Çamurluk Değişen (`fender_changed`) · Çamurluk Boyalı (`fender_painted`) · Çamurluk Lokal Boya (`fender_local`) · Tampon Değişen (`bumper_changed`) · Tampon Boyalı (`bumper_painted`) · Tampon Lokal Boya (`bumper_local`) · Ağır Hasarlı (`is_heavy_damaged`)

### Atılan öznitelik grupları

117 ham kolonun 53 tanesi modele doğrudan ya da türetilerek giriyor (bunların 39 tanesi hasar bayrağı), biri hedef (fiyat), 63 tanesi atıldı. Her ham kolon aşağıdaki sınıflardan tam birine atanıyor; tablo koddan hesaplanıyor.

| grup | gerekçe | kolon | kolonlar |
|---|---|---:|---|
| C | Kimlik / metin / zaman | 8 | `ad_id`, `listing_date`, `ad_title`, `location`, `eids_model`, `url` … |
| F | Türetilmiş tekrar | 9 | `engine_cc_low`, `engine_cc_val`, `engine_cc_is_range`, `power_hp_val`, `power_hp_is_range`, `count_changed` … |
| B | kb/gb ikizi | 11 | `kb_year`, `kb_mileage`, `gb_transmission`, `gb_fuel`, `gb_body_type`, `gb_color` … |
| A | Yarı-sabit (en sık değer ≥ %99) | 4 | `kb_condition`, `gb_usage_type`, `gb_is_first_owner`, `gb_plate_origin` |
| D | Eksik > %40 | 5 | `gb_mtv_yearly`, `tramer_fee`, `transmission_brand`, `gb_kasko_avg`, `gb_traffic_insurance_avg` |
| E | Katalog bloğu (birlikte eksik) | 22 | `kb_fuel_cons_avg`, `kb_fuel_tank`, `gb_segment`, `torque_nm`, `cylinder_count`, `max_speed_kmh` … |
| G | Modele alınmadı, gerekçe kayıtlı değil | 4 | `kb_color`, `kb_trade_available`, `kb_seller_type`, `gb_warranty_status` |

*`engine_cc_val`, `power_hp_val`: veritabanındaki kolonlar (aralığın orta noktası). Model ikisini de alt ve üst sınırlardan yeniden türetiyor (yukarıdaki motor kuralı): güçte alt–üst ortalaması, yani veritabanı kolonuyla aynı değer; hacimde üst sınır, yani aralıklı ilanlarda veritabanındakinden farklı.*

Sayısal özniteliklerde doldurma yapılmadı: eksik değerler LightGBM ve CatBoost'a boş (NaN) olarak girer ve kütüphanenin kendi eksik-değer yönlendirmesi kullanılır; kategorik boşluklar ayrı bir `missing` kategorisi olur. Yalnız KMeans/PCA için genel medyanla dolduruldu; %27.6 eksik olan `torque_nm` ise analiz dışı bırakıldı.

**Sızıntı kontrolü.** Dedup (ilanların tekilleştirilmesi) `ad_id` üzerinden ve CV'den ÖNCE yapıldı. Değerlendirme 5-fold out-of-fold: her ilan tam olarak bir kez, kendisini görmemiş bir modelle tahmin edildi.

### İçerik bazlı tekrar

| tanım | fazla satır | pay |
|---|---:|---:|
| katı — tüm ayırt edici alanlar aynı | 137 | %0.46 |
| gevşek | 209 | %0.70 |
| fiyat hariç katı | 1.278 | %4.26 |

`ad_id`'nin göremediği risk: `ad_id` farklı ama ilan aynı. Katı tanım 127 tekrar grubu buluyor; kolonları: `price`, `gb_mileage`, `gb_year`, `brand`, `series`, `model`, `kb_fuel`, `is_heavy_damaged`, `count_painted`, `count_changed`, `power_hp_up`, `engine_cc_up`. Bir kısmı gerçek tekrar ilan, bir kısmı yaygın modellerde tesadüfi çakışma. İlk iki tanım fiyat eşitliği istiyor, yani fiyatı değiştirilip yeniden yayımlanan ilanı göremez. Fiyatı dışarıda bırakan katı tanım 1.278 fazla satır (%4.26) buluyor; bunun ne kadarı fiyatı değiştirilmiş yeniden ilan, ne kadarı yaygın bir modelde tesadüfi çakışma, veriden ayrılamıyor. Fold'lar arasına sızabilecek pay bu iki tanıma göre %0.46 (katı) ile %4.26 (fiyat hariç) arasında; kilometresi de değiştirilmiş bir yeniden ilanı ikisi de göremez.

En çok tekrar eden ilanlar:

| model | yıl | fiyat | tekrar |
|---|---:|---:|---:|
| 320i Sport Line | 2025 | ₺4.900.000 | 4 |
| 318i Standart | 2005 | ₺717.000 | 3 |
| A3 Sportback 1.6 Ambition | 2010 | ₺890.000 | 3 |
| 116d Joy Plus | 2015 | ₺1.044.950 | 3 |
| 320i First Edition Sport Line | 2019 | ₺2.530.000 | 3 |

## 2. Eksiklik rastgele değil

32 kolon %2'nin üzerinde eksik. 22 tanesi birlikte düşen dört blokta: bunlar "eksik veri" değil, katalog eşleşmesinin çöktüğü ilanlar — standart modeller eşleşir, niş varyantlar eşleşmez, tüm özellik listesi birden boşalır. Sistematik olduğu için güvenilir imputasyon yok → bu kolonlar çıkarıldı. Kalan 10 kolonun eksikliği başka kaynaktan; her biri kendi sınıfında atıldı (§1 tablosu) — eksik > %40: `tramer_fee`, `transmission_brand`, `gb_traffic_insurance_avg`, `gb_kasko_avg`, `gb_mtv_yearly` · kb/gb ikizi: `gb_drivetrain`, `gb_trade_available` · modele alınmadı, gerekçe kayıtlı değil: `gb_warranty_status`, `kb_trade_available` · kimlik / metin / zaman: `eids_model`.

![Eksiklik oranı (%) — renk = birlikte eksik blok (aşağıdaki tablo) · gri = blok dışı
(etiketsiz olanlar ham kolon adı)](figures/tr-16-missing.png)

Geriye kalan 25 öznitelikte eksiklik sorun değil: en yükseği `kb_drivetrain` ile %1.5, 21'inde hiç eksik yok. Yukarıdaki grafik yalnız atılan kolonları gösteriyor.

**"Belirtilmemiş", eksik veri değil.** Sayfanın söylemediği bilgi veride boş duruyor. 45 kolonda bu boşluk eksik veri değil, satıcının "belirtilmemiş" cevabı; bu kolonlar yukarıdaki listeye ve bloklara girmiyor. Belirtilmemiş payı ağır hasar kaydında %68.2, 39 panel bayrağının her birinde %12.2–%19.1. Model bu bilinmeyenleri bilinçli bir kararla "yok" okuyor: belirtilmemiş panel orijinal, belirtilmemiş ağır hasar kaydı ağır hasarsız sayılıyor (§1).

### Birlikte eksik bloklar

| kolon | ort. eksik | birlikte eksik | örnek kolonlar |
|---:|---:|---:|---|
| 16 | %27.6 | %98.8 | `weight_kg`, `kb_fuel_cons_avg`, `kb_fuel_tank`, `gb_segment` … |
| 2 | %31.4 | %100.0 | `city_fuel_cons`, `highway_fuel_cons` |
| 2 | %29.9 | %100.0 | `production_year_start`, `production_year_end` |
| 2 | %29.3 | %100.0 | `rpm_max`, `rpm_min` |

*Birlikte eksik: blok kolonlarının hepsinin eksik olduğu ilanların, en az birinin eksik olduğu ilanlara oranı.*

### gb_ / kb_ çift kaynak

| alan | Genel Bakış (gb) boş | KısaBilgi (kb) ikizi |
|---|---:|---|
| Çekiş (`gb_drivetrain`) | %74.0 | `kb_drivetrain` · eksik %1.5 |
| Ort. Trafik Sigortası (`gb_traffic_insurance_avg`) | %54.3 | yok |
| Ortalama Kasko (`gb_kasko_avg`) | %51.1 | yok |
| Yıllık MTV (`gb_mtv_yearly`) | %40.6 | yok |

**kb/gb.** İlan sayfasında aynı bilgi iki sekmede yer alabiliyor: `kb` kısa bilgi, `gb` genel bakış. 10 çiftin 8 tanesinde iki sekme birebir aynı. 6 çiftte bir taraf modelde, öteki atıldı; 4 çiftte (`color`, `condition`, `trade_available`, `seller_type`) hiçbir taraf modelde değil. §1 tablosundaki ikiz sınıfında bir kolon daha var: `kb_is_heavy_damaged`, `is_heavy_damaged` ile birebir aynı. Çekişte `gb` ilanların %74.0 kadarında boş, `kb` %1.5. **Kasa tipinde `kb`, daha genel olduğu için seçildi:** 9 kategori; `gb` aynı bilgiyi koltuk sayısıyla birleştirip 23 değere bölüyor. Karşılığı olmayan ve %40'tan fazlası boş 3 alan (Ort. Trafik Sigortası, Ortalama Kasko, Yıllık MTV) ise elendi.

## 3. Fazlalık, bağıntı ve marka

Theil's U(a | b), b bilinince a'nın belirsizliğinin ne kadarının gittiğini verir (0–1) ve yönlüdür. Asimetri bulgunun kendisi: `model` `brand`, `segment`, `series` değerini neredeyse tam belirliyor (U ≥ 0.99) ama tersi değil — yani `seri`, `model`in kabalaştırılmış hâli, bağımsız bilgi değil.

![Seri × segment — medyan fiyat (₺M); 3 seri birden fazla segmente düşüyor · • = tek ilan](figures/tr-19-series-segment.png)

### Theil's U asimetrisi

| yön | okunuşu | U |
|---|---|---:|
| U(seri \| model) | model bilinince seri ne kadar belli | 0.999 |
| U(model \| seri) | seri bilinince model ne kadar belli | 0.387 |
| U(marka \| model) | model bilinince marka | 1.000 |
| U(marka \| seri) | seri bilinince marka | 1.000 |

Model seriyi 1.00 belirliyor, seri modeli yalnız 0.39. Marka hem modelden hem seriden tamamen okunuyor → marka ayrı bilgi taşımaz (aşağıdaki marka ablasyonu aynı sonucu ölçer).

Sayısal öznitelikler arasında Spearman korelasyonu (sıra ilişkisi; aşağıdaki harita). |ρ| > 0.5 olan çiftler: Yaş (yıl)–Kilometre 0.74, Motor Gücü (hp)–Motor Hacmi (cc) 0.61, Kapı Boyalı–Çamurluk Boyalı 0.60. Hedonik modeldeki çoklu bağlantı §6'da VIF ile ölçülüyor; orada kapı ve çamurluk boyaları ayrı değil, toplam boyalı parça sayısı olarak giriyor.

![Spearman korelasyonu](figures/tr-21-spearman.png)

### Segment beslemeden gelmiyor, türetiliyor

Ham `gb_segment` kullanılmıyor ve gerekçesi yalnız eksiklik değil: beslemenin "G" segmenti gerçek bir segment değil. O etiketi taşıyan 215 ilandan 214 tanesinin gövdesi MPV ve hepsi tek seriden geliyor — bozuk kaynak. Segment bu yüzden türetiliyor; MPV bilgisi kasa tipinde duruyor.

İlanların 29.832 tanesi segmentini doğrudan serisinden alıyor. Bazı ailelerde (`M Serisi`, `RS`, `S`, `i Serisi`) segment seriden değil model adından çözülüyor — örneğin M3 → 3 Serisi, S3 → A3 — toplam 156 ilan. Bu yüzden segment yalnız serinin değil (seri, model) çiftinin fonksiyonu: U(segment | seri) = 0.997, tam 1 değil. Çözülemeyen seri ya da model kalırsa üreteç durur; sessiz bir varsayılan segment yok.

Türetilen etiket, ham segmenti dolu 21.723 ilanın 404 tanesinde (%1.9) beslemeden ayrılıyor; 215 tanesi bilinçli G düzeltmesi, en büyük ikinci kaynak beslemenin E dediği 95 ilanın burada D olması.

### Marka ablasyonu

| kimlik kolonları | MAPE | MAE | R² |
|---|---:|---:|---:|
| yalnız marka | %8.80 | ₺163K | 0.9413 |
| seri + model | %6.49 | ₺110K | 0.9745 |
| marka + seri + model (rapordaki model) | %6.49 | ₺110K | 0.9745 |

Tam modelde yalnız kimlik kolonları değişiyor, diğer öznitelikler sabit; aynı 5-fold OOF. "Yalnız marka" kolu segmenti de dışarıda bırakıyor, çünkü segment seriden türetiliyor. Seri+model yerine yalnız marka verilince MAE ₺53K kötüleşiyor. Seri+modelin üzerine marka eklemek MAE'yi ₺1 değiştiriyor (MAPE farkı 0.00 puan) — manşet modelin 5 fold'unda marka 26 bölmede kullanılıyor ama hatayı değiştirmiyor. Bu bir ölçümden çok verinin tanımı: bu korpusta her seri tek bir markaya ait (U(marka | seri) = 1.00), marka seriden okunabiliyor.

## 4. Hedef ve önişleme

Ham fiyat sağa çarpık (çarpıklık 1.62); log dönüşümü simetriğe yaklaştırıyor (0.28). Model `log1p(price)` üzerinde eğitildi: log ölçekte fark göreli (yüzde) farka karşılık gelir, yani ucuz ve pahalı araçta aynı yüzde hata aynı ağırlığı taşır. Bu bir modelleme kararı, piyasa bulgusu değil.

![Fiyat histogramı — tüm veri (kesikli çizgi = medyan)](figures/tr-25-price-hist.png)

![Kasa tipine göre medyan fiyat (en az 80 ilanlı tipler; kasa tipi verilmeyen 291 ve daha az ilanlı tiplerdeki 101 ilan dışarıda)](figures/tr-01-body-median.png)

## 5. Piyasa yapısı — segmentasyon (KMeans + PCA)

**k=3 silhouette ile seçilmedi.** k=3 için silhouette 0.188 — denenen 7 değer içinde 7. sırada; en yüksek k=2 (0.242). Hepsi 0.25'in altında: veride belirgin doğal küme yok. k=3 yorumlanabilirlik için sabit seçildi; kümeler aşağıdaki eksenleriyle okunmalı, "piyasanın doğal yapısı" olarak değil.

![k seçimi — dirsek + siluet](figures/tr-24-k-selection.png)

![PCA — PC1 %19.7 × PC2 %12.4](figures/tr-22-pca-scatter.png)

![PCA — PC1 %19.7 × PC3 %11.0](figures/tr-23-pca-scatter-13.png)

### Kümeleri ayıran eksenler

| küme | ilan | ortalamadan en çok ayrıldığı 3 eksen |
|---|---:|---|
| Küme 1 · ağır hasar %5 | 9.046 | Kilometre ↑ · Çamurluk Lokal Boya ↑ · Motor Hacmi (cc) ↑ |
| Küme 2 · ağır hasar %2 | 15.976 | Kilometre ↓ · Yaş (yıl) ↓ · Motor Hacmi (cc) ↓ |
| Küme 3 · ağır hasar %13 | 4.966 | Kapı Boyalı ↑ · Çamurluk Boyalı ↑ · Çamurluk Değişen ↑ |

↑/↓ = kümenin ortalaması genelin üstünde/altında (z-skoru büyüklüğüne göre ilk 3). Kümelere ad verilmedi: k yorumlanabilirlik için sabitlendi, ayrım bu sütunda okunur.

### PCA yükleri

| PC | varyans | en büyük 4 yük |
|---|---:|---|
| PC1 | %19.7 | Kilometre (+0.46) · Yaş (yıl) (+0.45) · Çamurluk Boyalı (+0.41) · Kapı Boyalı (+0.41) |
| PC2 | %12.4 | Motor Gücü (hp) (+0.65) · Motor Hacmi (cc) (+0.61) · Çamurluk Boyalı (-0.24) · Kapı Boyalı (-0.24) |
| PC3 | %11.0 | Çamurluk Lokal Boya (+0.58) · Kapı Lokal Boya (+0.57) · Motor Gücü (hp) (-0.28) · Tampon Lokal Boya (+0.22) |

İlk 3 bileşenin açıkladığı varyans: %43.1. PC1 ≈ Kilometre + Yaş (yıl) + Çamurluk Boyalı + Kapı Boyalı · PC2 ≈ Motor Gücü (hp) + Motor Hacmi (cc) · PC3 ≈ Çamurluk Lokal Boya + Kapı Lokal Boya.

## 6. Hedonik model — kontrollü etkiler

Hedonik regresyon her sürücünün *kontrollü* (diğer her şey sabitken) fiyat etkisini verir; hedef log fiyat, n **29.554**. İki sütun yan yana. **Segment kontrolü** segment, marka, yakıt ve vites kuklalarıyla kurulur (R² **0.9312**). **Model kontrolü** bunlara `C(model)` ekler (735 model adı), yani her etki aynı model adı içinde ölçülür (R² **0.9648**). Model kimliği, hedonik R² ile modelin aynı ölçekteki OOF R²'si (0.9699, log fiyat) arasındaki farkın yaklaşık %87 kadarını kapatıyor — bir üst tahmin: C(model)'li R² örneklem içi, modelinki OOF. Modelin başlıktaki R²'si (0.9745) ham ₺ ölçeğinde; hedonikle o karşılaştırılmamalı.

**Güven aralıkları.** Aynı modelin ilanları birbirinden bağımsız değil ve hata varyansı eşit değil (Breusch-Pagan p <0.001); bu yüzden %95 güven aralıkları modele göre kümelenmiş standart hatalardan (735 küme). Segment sütununda 10 terimin hepsinin aralığı sıfırı dışlıyor; model sütununda yaş², +1 litre terimlerinin aralığı sıfırı içeriyor. Duyarlılık: seriye göre kümelenince (22 küme; az ve dengesiz, bu yüzden yalnız karşılaştırma için) aralığı sıfırı içeren terimler segment sütununda +1 litre, model sütununda yaş², yaş×km, +1 litre. Aynı serinin modelleri de birbirinden bağımsız değil; bu terimlerin anlamlılığı kümelemenin seçimine bağlı.

**Not:** Hedonik model bir OLS modelidir ve eksik değerlerle çalışamaz; bu yüzden eksik motor gücü (426) ve eksik motor hacmi (356) bulunan ilanlar analiz öncesinde elenmiştir. Her iki alanın da ortak eksik olduğu satırlar düşüldüğünde veri setinden toplam 434 satır çıkarılmıştır.

![Hedonik etkiler (nokta + modele göre kümeli %95 GA)](figures/tr-03-hedonic-ci.png)

### Kontrollü etkiler

| terim | segment kontrolü [%95 GA] | model kontrolü [%95 GA] |
|---|---:|---:|
| yaş | -%6.64 [-%6.92, -%6.37] | -%5.80 [-%6.33, -%5.27] |
| yaş² | +%0.08 [+%0.04, +%0.13] | +%0.03 [-%0.03, +%0.08] |
| km (100 bin) | -%15.11 [-%15.81, -%14.40] | -%14.83 [-%15.31, -%14.34] |
| km² | +%1.56 [+%1.08, +%2.05] | +%1.67 [+%1.34, +%2.01] |
| yaş×km | -%0.62 [-%0.89, -%0.35] | -%0.33 [-%0.56, -%0.10] |
| ağır hasar | -%11.60 [-%12.51, -%10.68] | -%12.46 [-%13.25, -%11.67] |
| boyalı | -%1.05 [-%1.17, -%0.92] | -%0.98 [-%1.09, -%0.88] |
| değişen | -%3.06 [-%3.33, -%2.79] | -%3.12 [-%3.35, -%2.88] |
| +100 hp | +%19.86 [+%13.89, +%26.15] | +%19.30 [+%6.90, +%33.12] |
| +1 litre | +%7.49 [+%0.71, +%14.73] | +%5.97 [-%0.92, +%13.35] |

Etki = exp(β)−1. Yaş ve km **medyan araca** (11 yaş, 181.000 km) ortalandı: yaş ve km satırları o araçtaki marjinal etki. Kare ve etkileşim terimleri (yaş², km², yaş×km) tek başına okunmaz; eğrinin bükülmesini taşır. İki sütunun farkı her terim için modele göre kümeli bir testle sınandı: fark yaş, yaş×km, ağır hasar, boyalı terimlerinde örnekleme hatasından büyük (%5 düzeyinde); öteki terimlerde değil. Karar notundaki etkiler model sütunundan: aynı model adı içinde.

**Katsayılar nedensel etki değil, kontrollü ilişkidir.** Ör. boyalı panelin katsayısı boyamanın fiyatı düşürdüğünü değil, boyalı panelli ilanların benzerlerinden o kadar ucuz ilan edildiğini söyler.

**Çoklu bağlantı.** Segment sütununun tasarımında en yüksek VIF yaş×km 7.25 (10'un altında); ortalanmamış tasarımda en yüksek yaş×km 96.28. Yaş, yaş², km, km² ve yaş×km aynı iki değişkenden türediği için yapısal olarak bağlı; medyan araca ortalamak bunu giderir, tahminler ve R² değişmez.

### Motor etkisi

+100 hp segment kontrolünde **+%19.9**, model kontrolünde **+%19.3**; +1 litre **+%7.5** ile **+%6.0** (aynı regresyonda, diğeri sabitken). Model adı motoru büyük ölçüde belirlediği için model sütununda motor terimleri yalnız aynı model adı içindeki güç ve hacim farkından ölçülür; aralıkları bu yüzden daha geniş. Bu fark ince: model adlarının yalnız 354/735 tanesinde güç ilanlar arasında değişiyor (model içi standart sapma 9.0 hp, genelde 43.1 hp) ve bir kısmı katalog hatası: §8'in motor değeri tutarsız 14 ilanı çıkarılınca model sütununda +100 hp +%19.3 → +%26.0. Hacim ve güç birbirine bağlı (Pearson korelasyonu 0.73): hacmin etkisi güç sabitken kalan kısım, iki katsayı birlikte okunmalı; birimler farklı olduğu için doğrudan kıyaslanmaz.

### LOFO — çıkarma testi

LOFO ikinci ve bağımsız bir yöntem: her özniteliği çıkarıp CV hatasının ne kadar büyüdüğüne bakar. SHAP'tan farklı bir şeyi ölçer — öznitelik yokken geri kalanların telafi edemediği kısmı. Sıralama SHAP'la örtüşmüyor: LOFO'da Kilometre > Yaş (yıl) > hasar grubu > model/seri adı > motor (hp + cc); SHAP'ta Yaş (yıl) > motor (hp + cc) > Kilometre > model/seri adı > hasar grubu. En keskin fark motor (hp + cc): SHAP'ta 2. sırada, LOFO'da 5. (ΔRMSE ₺524) — çıkarılınca model onu büyük ölçüde başka özniteliklerden telafi ediyor.

![LOFO — öznitelik çıkınca ΔRMSE (çakışmayan gruplar)](figures/tr-04-lofo-flat.png)

Grafik 5 çubuk gösteriyor, model 25 öznitelik kullanıyor: 19 öznitelik ölçüldü (2'si kendi çubuğunda, 17'si 3 grubun içinde: `DAMAGE_COLS` · `MODEL_SERIES` · `ENGINE`); **6 öznitelik hiç ölçülmedi**: `brand` · `kb_body_type` · `kb_drivetrain` · `segment` · `kb_transmission` · `kb_fuel`.

## 7. Model karşılaştırma ve kısıtlar

Rapordaki model: **LightGBM (model/seri adı TF-IDF+SVD)** — MAPE **%6.49**, R² **0.9745**, MAE **₺110K**. Hedef `log1p(price)`, 25 öznitelik. Emsal medyanı tabanına (aynı model ve yıl; emsal yoksa daha geniş medyan) göre ortalama mutlak hatada (MAE) %43 daha iyi.

**TF-IDF+SVD neye uygulanıyor.** Serbest ilan metnine değil, yalnız `model` ve `series` ad dizgilerine ("A4 Sedan 2.0 TDI" gibi). Amaç, nadir ad kombinasyonlarının isim benzerliği üzerinden komşularından bilgi ödünç almasıdır; target encoding'in seyrek hücrelerde zayıfladığı yeri kapatır. Satıcı açıklaması modele hiçbir biçimde girmez (bkz. §1, üçüncü katman).

### Model varyantları

| varyant | MAPE | R² | MAE | MedAE | RMSE |
|---|---:|---:|---:|---:|---:|
| LightGBM (model/seri adı TF-IDF+SVD) | %6.49 | 0.9745 | ₺109.776 | ₺75.320 | ₺176.225 |
| CatBoost (model/seri adı TF-IDF+SVD) ★ | %6.44 | 0.9745 | ₺110.085 | ₺74.927 | ₺176.130 |
| CatBoost (model/seri adı native text) | %6.58 | 0.9739 | ₺112.925 | ₺77.660 | ₺178.312 |
| emsal medyanı (taban, merdivenli) | %11.20 | 0.9235 | ₺191.224 | ₺130.000 | ₺305.050 |

★ = yalnız MAPE'ye bakan kuralın kazananı: **CatBoost (model/seri adı TF-IDF+SVD)**. Ama iki TF-IDF+SVD varyantı arasındaki fark 0.05 MAPE puanı ve ₺309 MAE; LightGBM şu metriklerde önde: MAE; CatBoost şunlarda: MAPE, MedAE, RMSE → pratikte **eşitler**. Rapor boyunca "model" LightGBM'dir: CPU'da deterministik, CatBoost'un ağaçları ise cihaza (GPU/CPU) göre değişir — önceki bir GPU koşumunda MAPE sırası tersti. Conformal aralık, marka ablasyonu ve örnek tahminler LightGBM'den.

### Taban basamak kırılımı

| basamak | ilan | pay | MAPE | MAE | R² |
|---|---:|---:|---:|---:|---:|
| model+yıl | 29.236 | %97.49 | %10.69 | ₺179K | 0.9450 |
| model | 596 | %1.99 | %26.70 | ₺559K | 0.6082 |
| global | 156 | %0.52 | %47.38 | ₺1.10M | -0.2359 |

Merdiven: (model, yıl) medyanı → (model) medyanı — tüm yıllar → global medyan. Test'teki (model, yıl) hücresi eğitim fold'unda yoksa taban bir alt basamağa iner; her inişte hata belirgin büyür — emsalsiz araçta taban zaten zayıf. Medyanlar her fold'da yalnız eğitim kısmından hesaplanır (sızıntısız, modelle aynı 5-fold).

**Model Kısıtları ve Gözlemler.** Modifiye, özel donanım veya ÖTV muafiyeti gibi form alanlarında yer almayıp serbest metne gizlenen bilgiler modele girmiyor; metninde dönüşüm ya da modifiye ifadesi geçen ilanlarda büyük hata oranı ham olarak %4.9, diğerlerinde %3.9; yaş, km, fiyat, performans ailesi, emsal sayısı ve marka sabitken olasılık oranı 1.14 (%95 GA 0.94–1.37): anlamlı bir fark ölçülmedi. Bu ifade performans ailelerindeki ilanların %43.6 kadarında geçiyor. Emsali olmayan ilanlarda hata belirgin şekilde büyüyor: aynı model ve yıldan başka ilan yoksa büyük hata oranı %19.0, 100+ emsal varsa %2.9; lira ölçeğindeki en büyük hatalar da bu uçta (§8). Kapsamlı bir hiperparametre optimizasyonuna bilinçli olarak gidilmedi; getirisi bu raporda ölçülmedi.

### Örnek tahminler

| bant | araç | yaş | km | gerçek | LightGBM | sapma | OOF artık | CatBoost (model/seri adı SVD) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ekonomik | 116i Comfort | 17 | 174.000 | ₺718.000 | ₺714.062 | %0.5 | %0.0 | ₺709.186 |
| orta | 520d Premium | 14 | 300.000 | ₺1.480.000 | ₺1.490.900 | %0.7 | %0.0 | ₺1.437.492 |
| premium | 520i Luxury Line | 4 | 96.000 | ₺3.680.000 | ₺3.640.718 | %1.1 | %0.0 | ₺3.615.892 |

> **Bunlar tipik değil, en iyi durum örnekleri.** Her fiyat diliminde ağır hasarsız ve |OOF artık|'ı en küçük ilanı seçer. "sapma" tüm veriyle eğitilmiş final modelin tahminidir (ilanı eğitimde görmüştür); sızıntısız ölçü "OOF artık". Tipik hata için MAPE'ye bakın.

## 8. Kalibrasyon, artıklar ve zayıflık

OOF (sızıntısız) tahminler gerçek fiyata karşı — R² **0.9745**. Artık% ortalamada sıfıra yakın (ort. -%0.47, std %9.27). Tahmin edilen fiyatın her düzeyinde de: gerçek fiyatın tahmine göre eğimi **1.003**, tahmin çeyreklerinde ortalama sapma −₺12K ile −₺3K arasında.

![Tahmin vs Gerçek (R² 0.975)](figures/tr-08-pred-vs-true.png)

### Hata dağılımı

| \|hata\| bandı | ilan | pay |
|---|---:|---:|
| ≤ %5 | 15.775 | %52.6 |
| %5 – %10 | 8.221 | %27.4 |
| %10 – %20 | 4.784 | %16.0 |
| > %20 | 1.208 | %4.0 |

![OOF hata dağılımı — artık % = (gerçek − tahmin) / gerçek](figures/tr-26-error-hist.png)

Tüm 29.988 ilanın OOF hatası. Dağılım sıfırda tepe yapıyor (medyan artık -%0.27); ±%10 içinde kalan ilan payı **%80.0**. Ortalama |hata| %6.5 — MAPE'nin kendisi; medyan |hata| %4.7. Yüzde artıkta kuyruk asimetrik görünüyor: model gerçeğin %20'den fazla **üstünü** 768 ilanda, **altını** 440 ilanda söylüyor (en uçlar -%178.1 ve +%56.5). Bunun tamamı tanımdan: artık gerçek fiyata bölündüğü için düşük tahmin en fazla %100 olabilir, fazla tahminin sınırı yoktur. Simetrik ölçekte (bir taraf ötekinin 1.2 katından büyük) fazla tahmin 768, düşük tahmin 812 ilan — asimetri tersine dönüyor. Std (%9.27) bu kuyruk yüzünden şişik; tipik hatayı medyan |hata| daha iyi anlatır.

### Büyük hatalar nereden geliyor

Hatası ±%20 sınırını aşan 1.208 ilan (768 fazla, 440 düşük tahmin). Aşağıdaki kırılımlar yalnız yapısal alanlardan sayılır — metin dedektörüne dayanmaz.

| aynı model+yılda ilan | ilan | büyük hata | fazla tahmin | düşük tahmin |
|---|---:|---:|---:|---:|
| 1 | 610 | %19.0 | %10.7 | %8.4 |
| 2–4 | 1.584 | %10.5 | %6.2 | %4.3 |
| 5–19 | 5.732 | %4.7 | %3.0 | %1.7 |
| 20–99 | 16.505 | %3.0 | %1.9 | %1.1 |
| 100+ | 5.557 | %2.9 | %2.0 | %0.9 |

1. **Emsal yok.** Aynı model ve yıldan başka ilan yoksa büyük hata oranı %19.0, 100+ emsal varsa %2.9.
2. **Uç ya da yaşlı araç.** F/S segmentte %19.1 — ama o iki segmentte toplam 444 ilan var (F 325 · S 119), yani en riskli işaretlenen yer aynı zamanda en ince olanı. Diğerleri %3.8; yaş ≥ 18'de %12.0 (daha gençlerde %3.2).
3. **Zaman ve hayatta kalma.** Medyan artık ilanın son görüldüğü taramaya göre değişiyor: 2026-01-18 -%3.27 → 2026-06-27 +%1.38. Bunun bir kısmı dönem: model zamanı görmüyor ve piyasa bu aralıkta +%1.98 kaydı (§9). Bir kısmı hayatta kalma: son taramada hâlâ yayında olan 11.526 ilanın medyan artığı +%1.38, daha önce kalkan 18.462 ilanınki -%1.24; piyasanın yerinde saydığı ilk iki tarama arasında (%0.00) bile medyan artık -%3.27 → -%1.75 geçiyor — erken kalkan ilanlar modelin söylediğinden ucuza fiyatlanmıştı.

**18 yaş verinin seçtiği bir eşik değil.** Büyük hata oranı yaşla artıyor: 5 yaş %0.35 · 10 yaş %1.56 · 15 yaş %7.86 · 18 yaş %11.69 · 21 yaş %15.34. En büyük bir yıllık sıçrama 16→17 yaş arasında (%7.79 → %12.72). Kesim 8 ile 21 arasında kaydırılınca yaşlı/genç oranı 3.5–6.7 kat arasında kalıyor; 18'de 3.7 kat. Ayrıca en yaşlı kova toplama sınırına dayanıyor: veri 2005 model yılıyla başlıyor, yani 18 ve üstü kovada yalnız 4 model yılı var.

İlişki; ilan ilan doğrulanmadı. Formda olmayan bilgi (hasar geçmişi, donanım, modifiye) olası katkı, burada ölçülmedi.

#### Örnekler

| araç | yıl | km | fiyat | model tahmini | artık |
|---|---:|---:|---:|---:|---:|
| BMW 640i | 2011 | 160.000 | ₺5.600.000 | ₺3.236.953 | +%42.2 |
| Audi 4.2 FSI Quattro R-tronic (R8) | 2008 | 112.550 | ₺4.690.000 | ₺2.744.815 | +%41.5 |
| BMW 750i Long | 2007 | 271.000 | ₺1.190.000 | ₺2.831.842 | -%138.0 |

- **BMW 640i · 2011:** İlan metnine göre araç komple M6 dönüşümü: M6 motoru ve M6 kasa parçaları takılmış. Form hâlâ 640i dediği için model onu sıradan bir 640i gibi fiyatlıyor; alıcı ise bir M6'ya bakıyor.
- **Audi 4.2 FSI Quattro R-tronic (R8) · 2008:** Veride tek R8. Formdaki model adı yalnız "4.2 FSI Quattro R-tronic"; aynı motor adını taşıyan S5 4.2 FSI Quattro'ların medyanı ₺2.62M (4 ilan) ve model tahmini buna yakın. Emsali olmayan bir süper otomobili model, adı benzeyen S5 gibi fiyatlamış.
- **BMW 750i Long · 2007:** Veride bu addan 2 ilan var; diğeri ₺5.30M'lik dönüşümlü bir 2009 araç. Bu ilan ise aynı yılın 730d'leriyle (15 ilan, medyan ₺1.18M) uyumlu ve metni bakımlı, masrafsız diyor. İlan piyasaya uygun, yanılan model: emsali olmadığı için muhtemelen adın diğer, pahalı ilanından etkileniyor.

Tahminler OOF; ilan kimliği (`ad_id`) bilerek yazılmadı.

![Artık% vs Tahmin](figures/tr-09-residual.png)

![Emsali az olan modelde hata büyük — model başına medyan hata](figures/tr-11-n-vs-error.png)

Her nokta bir model; y ekseni o modelin ilanlarındaki medyan hata. Kova medyanı tek ilanlı modellerde %8.9, 100+ ilanlıda %4.6. Yukarıdaki tablo iki yönden farklı ölçer: büyük hata **oranını** sayar ve ilanları model+**yıl** bazında gruplar. İkisi aynı yönü gösteriyor — emsal azaldıkça hata büyüyor.

### Conformal aralık ve kapsama

**Conformal aralık**, modelin tek bir fiyatın yanında veriye dayalı bir fiyat bandı da (örneğin ₺1.34M – ₺1.78M) sunmasıdır.

- **Dağılım varsayımı yapmaz:** Hataların bir formüle (çan eğrisi vb.) uyduğu varsayılmaz. Modelin daha önce hiç görmediği araçlardaki gerçek hataları sıralanır, en kötü %10'u dışarıda bırakılır ve pay doğrudan veriden okunur. Tek varsayım, yeni ilanların eskilere benzemesidir — piyasa kaydıkça (§9) bu varsayım zayıflar.
- **Oransaldır:** Hata payı lira değil yüzde olarak uygulanır (tahminin yaklaşık %13 altı ile %15 üstü). Bu yüzden pahalı araçta lira bandı geniş, ucuz araçta dar çıkar.
- **Kalibrasyonu başka ilanlardan:** Kapsama çapraz ölçülür: her katın aralığı yalnız öteki katların hatalarından kurulur, hiçbir ilanın kendi hatası kendi aralığını ayarlamaz. Katlar rastgele olduğu için genel kapsama (%89.99) yine de hedefe neredeyse kendiliğinden oturur; asıl sınav, sonraki taramanın yeni ilanlarındaki ileri kapsama (aşağıda).

**Zayıflık fiyata bağlı:** medyan hata tahmini en ucuz çeyrekte %6.95, en pahalıda %3.55. Tüm piyasaya tek bir yüzde uygulanınca, modelin oransal olarak daha çok yanıldığı ucuz bantta aralık dar kalıyor: Q1 kapsaması %81.0.

![Tahmin edilen fiyat çeyreğine göre medyan hata (%)](figures/tr-10-quartile-error.png)

Liraya çevrilince tablo değişiyor: toplam lira hatasının %38.9 kadarı tahmini en pahalı çeyrekte, %17.6 kadarı en ucuzda; ortalama mutlak hata ₺171K ile ₺77K.

**Hata payı banda göre (Mondrian).** Hata payı tahmin edilen fiyatın dört çeyreğinde ayrı ayrı hesaplanınca kapsama her bantta %89.96–%90.04. Rastgele katlarda bu neredeyse tanım gereği: ilk ön kayıttaki ölçüt (her bantta %88–%92) kodu sınadı, yöntemi değil; asıl sınav aşağıdaki ileri kapsama. Bedeli genişlik: en ucuz bantta aralık tahminin %28.3 kadarından %37.1 kadarına genişliyor, en pahalıda %28.3 kadarından %21.4 kadarına daralıyor.

![%90 aralık kaç ilanda tuttu (hedef %90)](figures/tr-12-coverage.png)

| bant | tahmin edilen fiyat | kapsama: tek oran | kapsama: banda göre | genişlik: banda göre |
|---|---|---:|---:|---:|
| Q1 | ₺1.15M altı | %81.0 | %90.0 | tahminin %37.1 kadarı |
| Q2 | ₺1.15M – ₺1.53M | %91.0 | %90.0 | tahminin %27.2 kadarı |
| Q3 | ₺1.53M – ₺2.26M | %93.2 | %90.0 | tahminin %24.4 kadarı |
| Q4 | ₺2.26M üstü | %94.7 | %90.0 | tahminin %21.4 kadarı |

**Not:** Bantlar tahmin edilen fiyattan kesildi: bir fiyatlama aracının bildiği tek şey o. Gerçek fiyata göre gruplamak ortalamaya dönüş üretir. Genişlik, aralığın alt ve üst ucu arasındaki farkın o banttaki medyanı.

**İleri kapsama (ön kayıt `plans/09-forward-coverage`).** Servis edilen tarif ileriye uygulandı: q eğitim taramasının kendi OOF hatalarından, test sonraki taramanın yeni ilanlarında (§9'daki dokuz ileri kurulum). Tek payla genel kapsama %87.3–%92.4, en ucuz bantta %77.1–%85.3. Banda göre pay en ucuz bandı her kurulumda düzeltiyor (%88.8–%92.5) ama her bantta %88'i tutmuyor: en düşük %84.9. Aynı model ve yılı eğitimde hiç olmayan ilanlarda kapsama iki kolda da düşük (tek pay %64.3–%76.2, banda göre %63.0–%73.0); banda göre pay orada her kurulumda kapsamayı daha da düşürüyor.

| bant | tek pay | banda göre |
|---|---:|---:|
| Q1 | %77.1–%85.3 | %88.8–%92.5 |
| Q2 | %88.6–%92.1 | %87.3–%91.0 |
| Q3 | %89.8–%94.5 | %84.9–%91.6 |
| Q4 | %92.9–%96.2 | %88.0–%92.5 |
| emsalsiz (model+yıl eğitimde yok) | %64.3–%76.2 | %63.0–%73.0 |

Hücreler dokuz ileri kurulumdaki kapsamanın en düşüğü–en yükseği.

### En büyük hatalar

**Motor değeri tutarsız ilanlar.** Motor gücü ya da hacmi kendi modelinin medyanından 1.5 kattan fazla sapan 15 ilan var (%0.05): katalog eşleşmesi çökmüş, model aracın sahip olmadığı bir motoru fiyatlıyor; medyan hataları %12.8, geri kalanınki %4.7. Bu kontrol yalnız en az 5 ilanı olan modellerde çalışıyor: daha az ilanlı 307 modelin 632 ilanı (%2.11) onun kör noktası — emsalsizliğin en yoğun olduğu yer.

**Lira ölçeğinde en büyük hatalar.** İlk 100 lira hatasından 73 tanesi düşük, 27 tanesi fazla tahmin. Bu yön bir model eğilimi sayılmaz: gerçek fiyat tahminin etrafında log ölçekte simetrik dağılsaydı da ilk 100'ün ortalama 67.8'i (%95: 60–77) düşük tahmin olurdu; aynı yüzde hata, fiyat tahminin üstündeyken liraca daha büyük. 84 tanesi tahmini en pahalı çeyrekte. Segmentini model adından alan seriler (M Serisi, RS, S, i Serisi) verinin %0.52 kadarı ama ilk 100 içinde 23 ilan — verideki paylarının 44 katı. Fiyat tavanı (₺6.50M) bu uçta modelin öğrendiği aralığı da kesiyor; en pahalı ilanlardaki düşük tahmin bu sınırla birlikte okunmalı.

![Lira ölçeğinde hata — tahmin − gerçek](figures/tr-28-residual-lira.png)

## 9. Zaman — dönem etkisi, dağılım kayması ve backtest

İki ölçü var. **Dağılım kayması:** dönemler arası fiyat dağılımı az kayıyor (en yüksek PSI 0.005, "kayma yok" eşiği 0.10). **Zamansal backtest:** eski dönemde eğitip sonraki dönemin yalnızca yeni ilanlarında test edince hata ufuk uzadıkça büyüyor (%6.29 → %7.34). Aynı model ve yılın ilanlarında piyasa seviyesi +%2.0 kaydı ve model zamanı görmüyor → yeniden eğitim takvime değil **ölçülen hataya ve kaymaya** bağlanmalı (bölümün sonu).

### Dönem etkisi

| dönem | canlı piyasa (aynı model+yıl) | dağılım mesafesi (EMD) |
|---|---:|---:|
| 01-18 (taban) | %0.00 | — |
| 01-27 | %0.00 (776) | ₺10.109 |
| 03-21 | +%0.85 (728) | ₺20.560 |
| 06-27 | +%1.98 (701) | ₺48.059 |

İki sütun iki ayrı soruya cevap veriyor. **Canlı piyasa**: aynı model ve yılın ilanlarında medyan fiyat ne kadar değişti (parantez içinde karşılaştırılan hücre sayısı) — ilan bileşiminden arınmış, model varsayımı yok. **EMD**: iki dönemin fiyat dağılımını üst üste getirmek için gereken ortalama kaydırma; bileşim dahil. Hedonik model dönem etkisi içermiyor, dönemler havuzlanarak kestirildi. Rapordaki model (LightGBM) de zamansızdır: dönem özniteliği almaz.

### Zamansal backtest

| tek dönem: eğitim → test | MAPE [%95 GA] | n | kümülatif: eğitim → test | MAPE [%95 GA] | n |
|---|---:|---:|---|---:|---:|
| 01-18 → 01-27 | %6.29 [5.90, 6.73] | 2.960 | ≤01-18 → 01-27 | = tek dönem | = |
| 01-18 → 03-21 | %6.59 [6.24, 6.96] | 8.182 | ≤01-18 → 03-21 | = tek dönem | = |
| 01-18 → 06-27 | %7.34 [6.95, 7.73] | 10.529 | ≤01-18 → 06-27 | = tek dönem | = |
| 01-27 → 03-21 | %6.54 [6.18, 6.91] | 7.413 | ≤01-27 → 03-21 | %6.47 [6.09, 6.83] | 7.238 |
| 01-27 → 06-27 | %7.18 [6.82, 7.60] | 10.313 | ≤01-27 → 06-27 | %7.23 [6.85, 7.64] | 10.257 |
| 03-21 → 06-27 | %6.90 [6.49, 7.33] | 9.099 | ≤03-21 → 06-27 | %6.78 [6.42, 7.17] | 8.889 |

Tek dönem = yalnız bir taramada eğit, sonrakini tahmin et. Kümülatif = t'ye kadarki tüm taramalarda eğit. Test kümesi yalnız eğitimde hiç görülmemiş `ad_id`'ler (sızıntısız); **n** bu ilanların sayısı ve MAPE bu ilanlarda. Ör. 01-27 taramasındaki 11.254 ilanın 8.294 tanesi 01-18 taramasında da yayındaydı; test edilen kalan 2.960 ilan. Kümülatifte t'ye kadarki her taramada görülen ilan çıktığı için n tek dönemden küçük ya da eşit. "=" işaretli kümülatif hücreler tek dönem koluyla aynı deneydir (ilk taramaya kadar birikim tek bir taramadır).

Kurulum manşet modelinki: model ve seri adı TF-IDF+SVD, aynı LightGBM ayarları. Ağaç sayısını, servis edilen modeldeki gibi, eğitim kümesinin kendi içindeki 5 katlı erken durdurma seçiyor (161–211 ağaç); test taraması durdurmak için hiç kullanılmıyor. Köşeli parantez %95 güven aralığı: test ilanları modele göre yeniden örneklenerek (bir modelin ilanları birbirinden bağımsız değil) 1.000 kez hesaplandı.

Aynı eğitim taramasından (01-18) test ufku uzadıkça MAPE %6.29 → %7.34; güven aralıkları örtüşmüyor. Ama her ufkun test ilanları farklı: başka ilanlar, başka bileşim, başka test taraması. Bu fark yalnız zamana bağlanamaz.

**Sabit test kümesinde ufuk.** Bu çekince aynı ilanlarda kalkıyor: 06-27 taramasına yeni gelen 8.889 ilan (önceki hiçbir taramada görülmemiş) önceki her tek taramayla eğitilen modelle fiyatlandı. MAPE eğitim taramasına göre 01-18 %7.29, 01-27 %7.13, 03-21 %6.89; en yeni eğitime göre fark 01-18 +0.40 [+0.27, +0.52], 01-27 +0.24 [+0.13, +0.34] (eşli, modele göre kümeli). Aynı ilanlarda eğitim taraması eskidikçe hata artıyor.

| test | tek dönem eğitimi | kümülatif eğitimi | ortak ilan | tek dönem | kümülatif | fark [%95 GA] |
|---|---|---|---:|---:|---:|---:|
| 03-21 | 01-27 | ≤01-27 | 7.238 | %6.52 | %6.47 | -0.04 [-0.12, +0.03] |
| 06-27 | 01-27 | ≤01-27 | 10.257 | %7.17 | %7.23 | +0.06 [-0.01, +0.12] |
| 06-27 | 03-21 | ≤03-21 | 8.889 | %6.89 | %6.78 | -0.11 [-0.20, -0.02] |

Tek dönem ile kümülatif, aynı test taramasında ikisinin de test ettiği aynı ilanlarda eşli karşılaştırıldı (aynı yeniden örneklemeler). Üç karşılaştırmanın bir tanesinde birikim hatayı güven aralığı sıfırın altında kalacak kadar düşürüyor; kalan iki tanesinde fark örnekleme hatası içinde.

### Dönem başına OOF

| dönem (bağımsız) | MAPE | n | kümülatif | MAPE | n |
|---|---:|---:|---|---:|---:|
| 01-18 | %6.91 | 10.901 | ≤01-18 | %6.91 | 10.901 |
| 01-27 | %6.92 | 11.254 | ≤01-27 | %6.67 | 13.861 |
| 03-21 | %6.93 | 11.478 | ≤03-21 | %6.45 | 21.099 |
| 06-27 | %7.09 | 11.526 | ≤06-27 | %6.49 | 29.988 |

Bu tablo zamansal değil: her satır düz 5-fold OOF, yeni ilan kuralı yok. Son kümülatif satır manşet modelin OOF'unun kendisi (%6.49, 29.988 ilan). İlk ileri test (01-18 → 01-27, %6.29) aynı taramanın kendi içindeki OOF'undan (%6.91) düşük. Bu bir çelişki değil, iki ayrı ölçüm: ileri model taramanın tamamıyla eğitiliyor (OOF'ta her kat beşte dördüyle), test ise yalnız sonraki taramaya yeni gelen ilanlar — başka bir ilan kümesi.

![Tek dönem ve biriken dönemler — ortalama yüzde hata](figures/tr-15-backtest.png)

### Dağılım kayması

| dönem çifti | ortak ilan (ilk taramanın payı) | KS | PSI | EMD (₺) |
|---|---:|---:|---:|---:|
| 01-18→01-27 | %76.1 | 0.0055 | 0.0004 | ₺10.109 |
| 01-18→03-21 | %30.2 | 0.0173 | 0.0015 | ₺20.560 |
| 01-18→06-27 | %9.1 | 0.0309 | 0.0049 | ₺48.059 |
| 01-27→03-21 | %36.1 | 0.0161 | 0.0011 | ₺15.717 |
| 01-27→06-27 | %10.8 | 0.0301 | 0.0038 | ₺39.115 |
| 03-21→06-27 | %21.1 | 0.0157 | 0.0017 | ₺28.620 |

**Sütunlar ne ölçüyor.** Üç ölçü de iki dönemin **ilan fiyatı dağılımını** karşılaştırır (ham fiyat, ₺; her dönemin o gün ilanda olan tüm ilanları, dönem başına ~11 bin).

| ölçü | ne ölçer, nasıl okunur |
|---|---|
| **KS** | İki dağılımın en çok ayrıldığı nokta; 0–1 arası. "Şu fiyatın altında kalan ilan payı" iki dönemde en fazla ne kadar farklı? 0.031 = en ayrık noktada 3.1 puan fark. |
| **PSI** | Fark pratikte büyük mü? İlk dönemin fiyatları 10 dilime bölünür; ikinci dönemde bu dilimlerin payı ne kadar kaymış? < 0.10 kayma yok · 0.10–0.25 orta · > 0.25 büyük. |
| **EMD (₺)** | Fark kaç lira? Bir dönemin fiyat dağılımını ötekine çevirmek için fiyatların ortalama kaç lira kaydırılması gerektiği. Lira cinsinden tek ölçü olduğu için en doğrudan okunanı bu. |

**Taramalar bağımsız örneklem değil.** Aynı ilan birkaç taramada birden görülüyor: ilk taramadaki ilanların %76.1 kadarı (01-18→01-27) ikinci taramada da var. Anlamlılık testleri (ör. KS p-değeri) iki örneklemin bağımsız olduğunu varsayar; bu yüzden p-değeri verilmiyor, tablo farkın büyüklüğünü gösteriyor.

**Kayma tablosunun söylediği.** Tam taramalar arasındaki fark küçük: en yüksek PSI 0.0049, "kayma yok" eşiğinin (0.10) yirmide biri. EMD bunu liraya çeviriyor: dokuz günde ~₺10 bin, beş ayda ~₺48 bin — medyan ilan fiyatının (₺1.55M) yaklaşık %3 kadarı. En yakın iki taramada ilkindeki ilanların %76.1 kadarı ikincisinde de var, bu yüzden aralarındaki mesafe küçük çıkıyor.

![Fiyat dağılımı — dönemlere göre](figures/tr-13-drift-hist.png)

![Log-fiyat yoğunluğu — dönemlere göre](figures/tr-14-drift-kde.png)

### Yeniden eğitim ne zaman

- **Kaymayı izle, modeli yeniden eğit.** Canlıda bir **kayma servisi** PSI · KS · EMD'yi izlesin ve model yeni taramalarla yeniden eğitilsin. Sabit bir PSI eşiği yetmez: bugünkü en yüksek PSI 0.0049, ama aynı yeni ilanlarda en eski taramayla (01-18) eğitilen model en yenisinden +0.40 puan [+0.27, +0.52] daha çok yanılıyor: dağılım neredeyse kıpırdamazken model eskiyor.
- **Hatayı doğrudan izle.** Fiyat her taramada geldiği için modelin yeni ilanlardaki hatası doğrudan ölçülebilir; yukarıdaki backtest tam bunu yapıyor. Kayma ölçüleri (PSI · KS · EMD) tanı için kalır.
- **Fiyat rejimini değiştiren gelişmeler.** Vergi/ÖTV düzenlemesi, teşvik, ithalat kuralı, kur hareketi ya da ani piyasa anomalisi gibi dışsal olaylar kaymayı bir ölçüm penceresi dolmadan yaratabilir; bunlar ayrıca **tetikleyici** sayılmalı ve eğitim planı bunlara göre yapılmalı.
- **Eski dönemleri atma.** Aynı test ilanlarındaki eşli karşılaştırmada birikimli eğitim üç karşılaştırmanın bir tanesinde hatayı anlamlı düşürüyor, hiçbirinde anlamlı artırmıyor (aralıklar test ilanlarının örneklemesini taşır, eğitimin değişkenliğini değil). Yeniden eğitim eski dönemleri atarak değil, **üstüne ekleyerek** yapılmalı.

## 10. Serbest metin: ölçüldü, dahil edilmedi

Satıcı açıklaması modele **girmiyor**. Bu bir ihmal değil, ölçüm sonucu: §7'deki model log fiyatta R² **0.9699**; aynı fold'larda açıklama metni eklenince **0.9707** — ΔR² **0.0008**, modelin açıklayamadığı log varyansın %2.6 kadarı. MAPE %6.49 → %6.41, ortalama mutlak hata ₺109.776 → ₺108.770.

Metin kolu modelle aynı öznitelikleri, fold'ları ve LightGBM ayarlarını kullanıyor; açıklama kelime TF-IDF'i (1–2 gram) ve 50 SVD bileşeni olarak ekleniyor, ikisi de her fold'un yalnız eğitim kısmında kuruluyor. Fark ön kayıtlı eşiğin altında (ΔR² < 0.005 ve MAPE iyileşmesi < 0,2 puan), bu yüzden metin modele eklenmedi. Ölçüm her koşuda yeniden yapılıyor; eşik aşılırsa bu bölüm üretilmez.

Metinde dönüşüm, motor değişimi ya da modifiye ifadesi geçen ilanlar ayrıca işaretleniyor: 3.107 ilan (%10.4), §7'deki metin bayrağı. Bayrak düz bir kelime kuralı; kelime listesi arşivlenmiş bir LLM çıkarım denemesinin sözcük dağarcığından damıtıldı ve her koşuda bugünkü metne uygulanıyor.

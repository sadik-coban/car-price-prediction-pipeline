# Defterlerin bilinen zayıflıkları ve açık noktaları

Bu liste `notebooks/` altındaki defterler yazılırken görülen zayıflıkları toplar: yöntemin sınırları, otomatik
korunmayan yerler, sınanmadan söylenen cümleler ve gösterim eksikleri. Hiçbiri sayıları yanlış yapmaz; her biri bir
sonucun ne kadar güvenle okunabileceğini ya da bir değişikliğin fark edilmeden geçip geçemeyeceğini söyler.

- **Kapsam:** `1_data_and_cleaning`, `2_what_drives_price`, `3_price_model` ve `4_errors_and_interval`. Sonraki defterler yazıldıkça eklenir.
- **Tarih:** 2026-11-24. 1. defterin maddeleri 2026-11-25'te tek tek gözden geçirildi; 3. defterin maddeleri 2026-11-25'te eklendi; 2. defterin maddeleri 2026-11-26'da gerçek veride ölçülüp gözden geçirildi; 4. defterin maddeleri 2026-11-28'de eklendi.
- **Durum:** "açık" olanlara dokunulmadı; "bilerek" bir karardır, "bilinen sınır" veriden ya da koddan çözülemeyen yerdir; yöntem eski zincirdekiyle aynı tutuldu. Yöntemi değiştiren bir düzeltme
  sayıları oynatır ve başlık sayılarının (`tests/baselines/`) yeniden kabul edilmesini gerektirir.

## 1. defter: veri ve temizlik

### Otomatik korunmayan yerler

| konu | ne | durum |
|---|---|---|
| Etiket olarak yazılan değerler | Defter kaynaklarını tarayan test (aşağıda, "Sonradan giderilenler") yalnız bir kolonla karşılaştırılan değerleri görüyor. Satır etiketi olarak yazılan değerler (`.loc["bmw"]`, `reindex(["bmw", "audi"])`) taranmıyor, çünkü hangi kolona ait oldukları koddan okunmuyor. Bugün yalnız 2. defterin marka tablosunda var; yanlış yazılırsa satır boş kalır ve defter tam sayıya çevirirken durur. | bilinen sınır |
| Kategori filtresi (1.4) | Toplama filtresinin "otomobil" kategorisi sınanamıyor: veride kategorinin kolonu yok, kasa tipi kolonu birkaç SUV gösteriyor. Defter bunu tabloda gösteriyor. | bilinen sınır |
| 4.5'in cümlesi 3. defterin sonucuna dayanıyor | 4.5 dört ham kolon için "üçü kat dışı hatayı ölçülebilir biçimde değiştirmiyor, renk küçük ama ölçülebilir bir katkı veriyor" diyor. Bu sonuç 3. defterde (4.2) hesaplanıyor; 1. defter onu hesaplamıyor, yalnız söylüyor. 3. defterin sayıları değişirse başlık sayıları (`add_*` anahtarları) durdurur, ama 1. defterin cümlesini elle güncellemek gerekir. | bilinen sınır |

### Eksik ya da açık kalanlar

| konu | ne | durum |
|---|---|---|
| "Hibrit" yakıt etiketi tutarsız (1.4) | Dönüşümde hata yok: veritabanındaki yakıt bütün satırlarda ham sayfadakiyle aynı. Tutarsızlık sitedeki etikette. 900 "Hibrit" ilanın 849'u, aynı adla "Benzin" ya da "Dizel" olarak da geçen araç modellerinde (ör. A3 35 TFSI, 520i, A6 40 TDI): aynı araç modeli, aynı model yılı, aynı güç, satıcıya göre farklı etiket. Bunlar hafif hibrit sürümler; etiket 2019 model yılından önce hiç görülmüyor, sonra payı artıyor. Yalnız 51 ilan hep "Hibrit" geçen araç modellerinde (şarj edilebilir hibritler: i8, 230e, 740e). `kb_fuel` tahmin modelinin bir özniteliği ve hedonik regresyonun bir kontrolü; "Hibrit" değerinin etkisi hibrit teknolojisinin etkisi diye okunmamalı. Düzeltilmedi: hangi ilanın gerçekten hibrit olduğu veriden çıkarılamaz, grup da küçük (ilanların %3'ü). Kalıcı çözüm veride değil: aracın kesin özelliklerini veren sabit bir araç kataloğu (`backlog/ideas.md`). | bilerek |
| Belirtilmemiş paneller ham sayfalardan sayılıyor (3) | Sayım yalnız veritabanından da yapılabilir. Ölçüldü (2026-11-25): bütün ilanlar ham sayfasıyla eşleşiyor ve veritabanından sayım birebir aynı sayıları veriyor (parça cevabı, belirtilmemiş cevap, en az bir ve 13 parçası belirtilmemiş ilan). Ham sayfalar yine de okunuyor: defter "veritabanındaki boş bayrak = sitenin belirtilmemiş cevabı" cümlesini veride gösteriyor ve 4.1 bunun üstüne kuruluyor; maliyeti birkaç saniye. Aynı eşitliği artık bir test de sınıyor ("Sonradan giderilenler"). | bilerek |

### Sonradan giderilenler

| konu | ne yapıldı |
|---|---|
| Gerekçesi olmayan dört ham kolon (4.5) | `kb_color`, `kb_seller_type`, `kb_trade_available` ve `gb_warranty_status` tahmin modeline girmiyordu ve neden girmediği kayıtlı değildi: öznitelik listesi kurulurken hiç denenmemişlerdi. Karar (2026-11-25): katkıları tahmin modelinin kendisiyle ölçülür, sonuç ne çıkarsa çıksın eklenmezler. 3. defter 4.2 bunu yapıyor: her kolon tek tek eklenir, karıştırılmış kopyası karşılaştırma olur, fark eşli bootstrap aralığıyla verilir. Sonuç: satıcı tipi, takas ve garanti kat dışı hatayı ölçülebilir biçimde değiştirmiyor; renk MAPE'yi 6,490'dan 6,449'a indiriyor (beş katın beşinde), RMSE'yi değiştirmiyor. 4.5'te sınıfın adı "gerekçe kayıtlı değil" yerine "kurala uymuyor, eklenmedi" oldu, metin sınamaya atıf yapıyor; başlık anahtarı `raw_columns_no_reason` → `raw_columns_not_added` (değer aynı, 4). Renk bilerek dışarıda: kazanç küçük, karşılığında yayındaki tahmin modelinden ayrılmak gerekir. |
| Boş bayrak = belirtilmemiş (3.3) | Defter "bir parçanın üç bayrağı birden boşsa cevap belirtilmemiş" cümlesini sayıyla gösteriyordu ama cümle bozulsa hiçbir şey durmuyordu; test yalnız ham cevaptan kurulan bayrakların veritabanıyla aynı olduğunu sınıyordu ve boş bayrağı 0 sayıyordu. Aynı teste eklendi (`tests/lib/test_real_data.py`): her parçanın üç bayrağı ya birlikte boş ya birlikte dolu, ve birlikte boş olması tam olarak "belirtilmemiş" cevabı. |
| Kolon sınıfları tablosu (4.5) | "Tahmin modelinde" satırı 53 kolonu tek tek yazıyordu. Parça bayrakları artık tek tek yazılmıyor, sayıları yazılıyor (39); satırda geri kalan 14 kolon adıyla duruyor. Sayılar aynı. |
| Başlık sayıları | 34 satır "ana sonuçlar" için fazlaydı. Satır çıkarılmadı: tablo aynı zamanda defter yeniden kurulduğunda sayıların değişmediğini denetleyen kayıt (`tests/baselines/`), çıkarılan her satır korumasız kalır. Bunun yerine her satıra hesaplandığı bölüm eklendi ve metin tablonun iki işini söylüyor. 2. defterin tablosu (39 satır) da aynı biçime geçti. Sayılar aynı. |
| Toplama filtresi sınırları (1.4) | Defter filtre ile veriyi yan yana gösteriyordu ama veri filtrenin dışına çıksa hiçbir şey durmuyordu. Gerçek veride sınandı, bütün satırlar filtrenin içinde. Test eklendi (`tests/lib/test_real_data.py`): markalar seçili markalarla aynı, fiyat her markanın kendi aralığında, kilometre ve model yılı sınırın içinde, yakıt değerleri filtredekilerle aynı (filtredeki "LPG" ilanda "LPG & Benzin"). Yakıt denetimi "elektrikli yok" cümlesini de koruyor. |
| Defterdeki değer sabitleri | Defterin bir kolonla karşılaştırdığı değerlerin ("SUV", "bmw", "audi") gözlenen değerler kaydında olduğu elle kontrol edilmişti. Artık bir test defter kaynaklarını tarıyor (`tests/notebooks/test_notebooks.py`): bir veri kolonu yalnız o kolonda görülmüş bir değerle ya da boş değerin dolgusuyla karşılaştırılabilir. "Belirtilmemiş" hasar eşlemesinin bir anahtarı; eşlemenin kayıtla iki yönlü uyumu zaten sınanıyordu (`tests/lib/test_cleaning.py`). |
| Elektrikli araç (1.4) | Defter "elektrikli ilan = 0" sayısını hesaplamıyor, çünkü "Elektrik" değeri veride hiç görülmedi ve görülmemiş bir değer koda yazılmıyor. Araştırıldı: elektrikli araçlar toplamada hiç istenmedi. Arama adresi ilk commit'ten beri yalnız dört yakıtı gönderiyor (Benzin, Dizel, Hibrit, LPG; `scraper/collection.py`, adresi `tests/scraper/test_collection.py` sınıyor). Veride sitenin iki yakıt alanında da bütün satırlarda yalnız dört değer var ve adı tam elektrikli bir araca ait seri ya da araç modeli yok. "Hibrit" şarj edilebilir hibritleri de içeriyor (i8, 230e, 740e). Toplama filtresi testi yakıt değerlerinin filtredekilerle aynı kaldığını sınıyor; elektrikli bir ilan gelirse durur. |
| Motor değeri kolonları (2.1) | "Veritabanının `*_val` kolonları aralığın orta noktası" cümlesi sınanmadan yazılmıştı. Gerçek veride sınandı ve bütün satırlarda tutuyor: iki sınır varsa orta nokta, tek sınır varsa o sınır, sınır yoksa boş. Test eklendi (`tests/lib/test_real_data.py`). Aynı yerde ikinci bir test, kuralın bir değeri yalnız sitenin hiç sınır vermediği ilanlarda boş bıraktığını sınıyor: hacim kuralı üst sınırı aldığı için yalnız alt sınırı olan bir hacim ("… cm3 ve üzeri") boş kalırdı. Site hacimde bu biçimi hiç göstermedi; gösterirse önce veritabanı kurulumu görülmemiş değer diye durur, test ondan sonraki adımı (biçim tanıtıldı ama kural uyarlanmadı) yakalar. |

## 2. defter: fiyatı ne belirliyor

İlk listedeki dokuz madde 2026-11-26'da gerçek veride ölçüldü. Yedisi defterde giderildi, kalanlar aşağıda.

### Kalan sınırlar

| konu | ne | durum |
|---|---|---|
| Hedonik katlar (5.3) | Katlar regresyona giren 29.554 ilan üzerinde kuruluyor; tahmin modelinin katları aynı fonksiyonla (`folds.make_folds`) ama 29.988 ilan üzerinde. Ölçüldü: tahmin modelinin bölmesi regresyon ilanlarına indirilip kullanılınca altı regresyonun kat dışı R²'si en çok 0,0011 değişiyor (yalnız araç modelinin adı: 0,8220 → 0,8231); üç parça 0,139 / 0,230 / 0,592 yerine 0,138 / 0,230 / 0,593. İki defter bu sayıları birbiriyle karşılaştırmıyor; bölme değiştirilmedi, defter farkı söylüyor. | bilerek |
| KMeans tohumu (4.4) | Silhouette artık bütün ilanlarda hesaplanıyor, ama KMeans tek tohumla (42) kuruluyor. Beş tohumla ölçüldü: k = 4'te skor 0,200 ile 0,263, k = 5'te 0,193 ile 0,259 arasında oynuyor; öteki k'lerde oynama 0,04'ün altında. Hepsi düşük kalıyor; vargı (belirgin küme yok) etkilenmiyor. "En iyi k" okunmamalı; defter bunu söylüyor. | bilinen sınır |
| Motor kuralının kör noktası (5.9) | Kural en az 5 ilanı olan araç modellerinde çalışıyor; daha az ilanlı araç modellerindeki 632 ilan sınanamıyor. En az ilan eşiği 3, 10 ya da 20 alınınca işaretlenen ilan sayısı (14–18) ve +100 hp'nin etkisi fiilen değişmiyor. | bilinen sınır |
| Theil's U tabanı yalnız araç modeli için (1.1) | Karıştırılmış kolonla ölçülen taban defterde yalnız bilinen kolon araç modeli iken gösteriliyor. Öteki bilinen kolonlarda da ölçüldü: seride en çok 0,023, kalanlarda en çok 0,011; okumayı değiştirmiyor. | bilinen sınır |
| Belirtilmemiş ağır hasar (5.10) | Araç modeli kontrolünde belirtilmemiş ilanlar kaydı olmayanlardan %0,82 ucuz ve aralığı sıfırı dışlıyor: fark küçük ama sıfır değil. Terim 0/1 olarak bırakıldı; tahmin modelindeki kodlama da aynı (1. defter). | bilinen sınır |

### Sonradan giderilenler

| konu | ne yapıldı |
|---|---|
| Theil's U yukarı sapıyor (1.1) | Sapma ölçüldü: araç modeli kolonu karıştırılıp (10 karıştırmanın ortalaması) U yeniden hesaplandı. Kasa tipi, çekiş, vites, yakıt, segment, marka ve seride taban 0,02–0,06; düzeltilmiş değerler ham değerlerden en çok 0,01 ayrılıyor (ör. vites 0,720 → 0,710). İlk listedeki "0,72–0,93 arası değerlerin bir kısmı bu sapma" kaygısı bu yüzden önemsiz. Tavan, kaput ve bagaj durumunda ise ham değerin büyük kısmı taban: 0,232 / 0,104 / 0,122'ye karşı 0,156 / 0,042 / 0,061. Defter tabanı üçüncü kolon olarak gösteriyor ve bu üç satırın bağ olarak okunmaması gerektiğini söylüyor. |
| Silhouette tek örneklemde (4.4) | 30 farklı 3.000 ilanlık örneklemde skor k başına ±0,01 oynuyordu. Bütün ilanlarda hesaplamak k başına 5 saniye sürüyor ("çok yavaş" doğru değildi); skor artık bütün ilanlarda hesaplanıyor, örneklem kalktı. `best_silhouette` 0,242 → 0,245 (onayınla). "0,25'in altında" eşiği metinden çıktı. |
| Ağır hasar terimi (5.1) | Yeni 5.10 bölümü terimi üç durumla (var, yok, belirtilmemiş) kuruyor. Belirtilmemiş ilanlar kaydı olmayanlardan segment kontrolünde %0,54 [−0,11, 1,20], araç modeli kontrolünde −%0,82 [−1,22, −0,42] ayrılıyor; "var"ın etkisi iki kodlamada yakın (araç modeli kontrolünde −%12,46'ya karşı −%12,98). Öteki dokuz terim en çok 0,27 puan değişiyor. Yeni başlık anahtarı `damage_unspecified_pct`. |
| Motor tutarlılık kuralı (5.9) | Oran 1,25, 1,5 ve 2 ile yinelendi (defterde tablo): çıkarılan ilan 609, 14 ve 7; +100 hp'nin etkisi üçünde de %25–26 (bütün ilanlarda %19,3). Kayma oranın seçimine bağlı değil; en uçtaki birkaç ilan yapıyor (oran 3'te tek ilan çıkınca bile %21,9). |
| Kimliksiz sütunun karşılaştırması (5.4) | Aralıkların örtüşmemesine bakan hücre kalktı. 5.5'teki fark testi artık iki adımı da sınıyor (kimlik yok − segment kontrolü, segment kontrolü − araç modeli kontrolü); iki ucun farkı 5.4'teki kimlik bileşeni tablosunda. Kimlik yok − segment kontrolü: km, ağır hasar, boyalı, değişen parça ve +100 hp anlamlı (km t = 6,56, +100 hp t = 6,64). |
| Kilometre açıklaması (5.4) | Sınandı. Araç modeli kontrolü regresyonunun kimlik bileşeni (kontrol kuklaları ve araç modelinin sabit etkisi) on terime regreslendi; km'nin katsayısı +0,053 [0,035, 0,071]: öteki terimler sabitken km'si yüksek ilanlar daha pahalı araç modellerinden. Bu katsayı iki sütunun km katsayısı farkına cebirsel olarak eşit. Defterde tablo olarak duruyor. Depo dışında ayrıca bakıldı: km artığı beşte birlik gruplarda kimlik bileşeni düzenli artıyor (−0,037'den +0,036'ya); E segmenti aynı yaş ve motorda daha çok km yapmış ve daha pahalı, C segmenti tersine. |
| Breusch–Pagan p değeri (5.5) | Hücre artık LM istatistiğini ve serbestlik derecesini yazıyor (2558, 21); p sıfıra taşınca `0` yerine "gösterilebilen en küçük sayının altında" yazıyor. |
| Log fiyat histogramı (3) | Fiyat histogramının yanına log fiyat histogramı eklendi (60 kova, eksende karşılık gelen fiyat). |

### Yazım sırasında giderilenler

| konu | ne yapıldı |
|---|---|
| Sıfır ya da eksi güç ve hacim | Eski kod boş değerlerin yanında sıfır ve eksi değerleri de regresyondan atıyordu; veride böyle bir değer yok, o dal kaldırıldı. Böyle bir değer gelirse regresyona sessizce girmesin diye gerçek veride güç ve hacmin pozitif olduğunu sınayan test eklendi (`tests/lib/test_real_data.py`). |
| Sitenin segment değeri | Metin kırpma yerine açık bir eşleme (`cleaning.raw_segment`); görülmemiş bir değerde duruyor. |
| `patsy` | Defter doğrudan import ediyor; `requirements-pipeline.txt` dosyasına eklendi. |

## 3. defter: tahmin modeli

Yöntem eski zincirdekiyle aynı; ilk yazımda eski kayıtlarla karşılaştırılabilen 318 sayının hepsi aynı çıktı. İki fark:
4.2'deki ekleme sınaması eklendi, metin bayrağı bölümü 2026-11-28'de kaldırıldı.

### Yöntem zayıflıkları

| konu | ne | durum |
|---|---|---|
| Erken durdurma puanlanan katta (1.3) | Ağaç sayısını, hatası ölçülen katın kendisi seçiyor; kat dışı hata bu yüzden biraz iyimser. Ne kadar iyimser olduğu ölçülmedi. LightGBM ile CatBoost karşılaştırması da eşit koşullarda değil: CatBoost sabit 500 turla eğitiliyor. Defter ikisini de söylüyor. | açık |
| Ekleme sınaması tek koşum (4.2) | Tek tohum, tek kat bölmesi, kolon başına tek plasebo çekilişi. Bootstrap aralığı yalnız ilanların örneklenmesini kapsıyor, tahmin modelini yeniden kurmanın oynamasını kapsamıyor. Bir kolon ile kendi plasebosu arasındaki fark ayrıca sınanmıyor. RMSE farkını birkaç pahalı araç belirliyor, aralıkları geniş. | açık |
| Plasebo hatayı büyütüyor (4.2) | Bilgisi olmayan bir kategorik kolon eklemek kat dışı hatayı biraz büyütüyor. Bu yüzden "eklenince değişmiyor" ile "plasebodan iyi" aynı şey değil: satıcı tipi, takas ve garanti hatayı düşürmüyor, ama plasebolarından kötü de değil. Defter kararı ilk ölçüye göre söylüyor. | açık |
| LOFO tek koşum (4.1) | Tek tohum. Tablonun alt yarısındaki küçük farklar plasebo farklarıyla aynı büyüklükte; sıralama olarak okunmamalı. Altı kategorik öznitelik tek tek çıkarılmıyor (2026-09-25 kararı). | açık |
| Son tahmin modelinin ağaç sayısı (5) | Katlarda seçilen ağaç sayılarının medyanı, her biri ilanların beşte dördüyle eğitilen koşumlardan geliyor; son tahmin modeli bütün ilanlarla eğitiliyor. Aynı sayının daha büyük eğitim kümesine uyup uymadığına bakılmadı. | açık |

### Otomatik korunmayan yerler

| konu | ne | durum |
|---|---|---|
| Servis dosyasıyla eşitlik | Son tahmin modelinin bugünkü servis dosyasıyla (`data/serving/serve/lightgbm_tfidf_svd.txt`) bayt bayt aynı olduğu depo dışında sınandı (2026-11-25). Depoda bunu koruyan test yok; dışa aktarım aşamasında (6. aşama) parite denetimi olarak gelecek. | açık |
| Başka makine | Sonuçlar bu makinede koşumdan koşuma birebir aynı (LightGBM 1, 4, 8 ve 16 iş parçacığında; CatBoost üç koşumda). Başka bir makinede (başka BLAS, başka işlemci) SVD kolonlarının ve dolayısıyla başlık sayılarının aynı çıkacağı sınanmadı. | açık |
| İş parçacığı cümlesi (1.3) | "İş parçacığı sayısı LightGBM'in sonucunu değiştirmiyor" cümlesini depodaki test uydurma 400 ilanda sınıyor; gerçek veride depo dışında sınandı. | bilinen sınır |
| Önbellek anahtarı | Anahtar veritabanını, lib modüllerinin metnini, paket sürümlerini ve argümanların içeriğini kapsıyor; işlemciyi ve BLAS'ı kapsamıyor. `model.py`, `cache.py`, `data.py` ya da `folds.py`'deki her değişiklik, yorum satırı bile, bütün ağır adımları yeniden koşturur (yaklaşık 12 dakika). Eski kayıtlar silinmiyor; `data/cache` elle temizlenir. | bilerek |
| Emsal tablosu (2.3) | Hücre büyüklükleri bütün ilanlarda, ilanın kendisi dahil sayılıyor; merdiven ise katların içinde çalışıyor. İki sayım aynı şeyi ölçmüyor; defter farkı söylüyor. | bilerek |

### Yazım sırasında giderilenler

| konu | ne yapıldı |
|---|---|
| "Tesisat" denetimi | Eski kod, bir öznitelik çıkarıldığında matrisin gerçekten değiştiğini koşum sırasında denetliyordu. Artık `model.fold_matrices` bilinmeyen bir öznitelik ya da kolon adında duruyor ve bir test, kullanılan bir kolon çıkınca hatanın büyüdüğünü sınıyor (`tests/lib/test_model.py`). |
| Aynı ilanlarda karşılaştırma (2.4) | Eski kod tahmini 0 ile 15 milyon ₺ arasına kırpıyordu; kat dışı tahminlerin hiçbiri bu aralığın dışında değil, kırpma kaldırıldı. Artık tanımı (`errors.residual_pct`) kırpmayı koruyor. |
| Plasebo permütasyonu (4.2) | İlk yazımda dört kolon aynı permütasyonla karıştırılıyordu; dört plasebo birbirinden bağımsız değildi. Her kolon artık kendi permütasyonuyla karıştırılıyor. |
| Önbelleğin veritabanı yolu | `cache.py` veritabanı yolunu kendisi tanımlıyor. Parmak izi alınan dosyanın defterlerin okuduğu dosya olduğunu bir test sınıyor (`tests/lib/test_cache.py`). |
| Metin bayrağı bölümü kaldırıldı | 3. defterin eski 5. bölümü ilanın metninde dönüşüm ya da modifiye ifadesi arayan bir regex bayrağı kuruyor ve bayraklı ilanlarda büyük hatanın daha sık olup olmadığına bakıyordu. Karar (2026-11-28, senin): bölüm tamamen kalktı. Gerekçe: bayrak sabit bir kelime listesi, isabeti hiç ölçülmedi ve kaçırdığı biçimler var ("yazılım" geçen 454 ilanın 156'sı bayraksızdı; içinde hem "yazılım vardır" hem "yazılım yok" var); sonuç ("ayrı bir ilişki ölçülmedi") bu yüzden "etki yok" ile "bayrak kötü"yü ayıramıyordu ve hiçbir kararı beslemiyordu. Kalkanlar: bölüm, `lib/text_flags.py`, testi ve altı başlık sayısı; eski 6. bölüm (son tahmin modeli) 5. bölüm oldu. Kalktığı andaki sayılar: 3.228 bayraklı ilan (başlık dahil), olasılık oranı 1,155 [0,962, 1,386]. `lib/errors.py` ve `model.base_oof` şu an hiçbir defterde kullanılmıyor; ikisi de 4. defter için duruyor (`model.py`'ye dokunmak önbelleği baştan koşturur). |

## 4. defter: hatalar ve aralık

Yöntem eski zincirdekiyle aynı (`08_residuals`, `08_large_errors`, `08_conformal_coverage`, `08_forward_test`,
`10_free_text` ve `07_final_model`'in servis edilen aralık kısmı); kod `lib/interval.py`, `lib/forward.py`,
`lib/free_text.py` ve `lib/errors.py`'ye eklenen iki fonksiyonla yeniden yazıldı. İlk yazımda eski kayıtlarla
karşılaştırılan 665 sayının 623'ü aynı, 39'u bilerek hesaplanmıyor (aşağıda "çıkarılanlar"), 3'ü farklı (boş değer
grupları, aşağıda). Yeni ağır adımlar (çapraz kalibreli ölçek kolu, servis edilen ölçek modeli, ileri test, açıklamalı
koşum) önbellekten okunuyor; 3. defterin hiçbir kaydı yeniden üretilmiyor.

### Yöntem zayıflıkları

| konu | ne | durum |
|---|---|---|
| Aralık kat dışı hatalarla kalibre ediliyor (3) | k ve tek pay, temel koşumun kat dışı hatalarından. Bu hatalar erken durdurmanın iyimserliğini taşıyor (3. defter, 1.3): durma turunu puanlanan kat seçiyor. Aralık bu yüzden biraz dar kalibre edilmiş olabilir; ne kadar olduğu ölçülmedi. | açık |
| Çapraz kalibrasyon yalnız s düzeyinde (3.2) | Puanlanan katı ne s ne k görüyor; ama hataların kendisi tahmin modelinin çapraz uydurmasından geliyor ve emsal sayımı öteki dört katın tamamını kullanıyor (iç katlara göre ayrılmıyor). Her şeyin öteki dört kattan yeniden kurulduğu sızıntısız kopya eski zincirde depo dışında ölçülmüştü (kapsama 90,11; karar kaydı, 2026-11-23); bu defter onu yeniden üretmiyor. | bilinen sınır |
| Servis edilen aralığın güvencesi yok (5) | Servis edilen merkez (son tahmin modeli) ve s bütün ilanlarla, k ve skorlar kat modellerinin hatalarıyla kuruluyor. Bu karışık kurulumun sonlu örneklem güvencesi yok; %90 "yaklaşık". Defter bunu söylüyor. | bilerek |
| Seçim sonuçtan sonra (3.2) | Ölçek modelinin yöntemi, ayarları, monotonluk kısıtı, alt sınırı ve ±%10 toleransı sonuçlar görüldükten sonra kondu; arama yapılmadı ama bağımsız test de kalmadı. Tek tohum: aynı aracın payının tohumdan tohuma ne kadar oynadığı bu defterde ölçülmedi (karar kaydındaki ölçüm: medyan %2,5). | bilerek (2026-11-23 kararı) |
| Gruplarda kapsama, çoklu karşılaştırma (3.4) | 42 girdi ve 34 görülmeyen grup; her grubun kapsaması tek bir sayı, binom aralığı yok. "En kötü grup" 34 grup arasından seçiliyor; küçük gruplarda (ör. 302 ilanlı "ilk sahip: evet") 3 puanlık fark rastlantı olabilir. Defter bunu söylüyor ama sayısal bir eşik vermiyor. | açık |
| Girdi eksenlerinde düzelme kısmen tasarım gereği (3.4) | s bu eksenlerin bilgisini görüyor; bu eksenlerde hedefe yakın kapsama, ölçek modelinin işe yaradığının bağımsız kanıtı değil. Bağımsız eksenler görülmeyen eksenler; orada iki kol arasında fark yok. | bilinen sınır |
| Seviye kayması varsayımı (3.7) | Bütün fiyatları aynı oranda kaydırıyor; gerçek bir kayma araç türüne göre farklı olabilir. Tablo yalnız "dar aralık önce eskir" cümlesinin dayanağı; kaymanın ne kadar olacağını söylemiyor. | bilinen sınır |
| Büyük hata kırılımları birbirinden ayrılmıyor (2.2, 2.3) | Emsal, yaş, segment ve fiyat düzeyi birlikte değişiyor. 2.3'teki "emsal kovası içinde" tablosu kaba (beş kova) ve F/S grupları küçük (64, 148, 232 ilan). Eski zincirin denetiminde emsal kontrollü olasılık oranları F/S için 1,19 [0,87–1,63], yaş için 1,30 çıkmıştı (`backlog/ideas.md`, 2026-11-23); bu defter çok değişkenli bir model kurmuyor, yalnız "ayrılmadı" diyor. | açık |
| İleri test tek kurulum (6) | Tek bir zaman farkı, tek eğitim kümesi. MAPE aralığı (araç modeline göre küme bootstrap'i) yalnız test ilanlarının örneklenmesini kapsıyor, tahmin modelini yeniden kurmanın oynamasını kapsamıyor. İleri kuruluma özgü fark (aynı ilanlarda +0,18 puan [0,10, 0,26]) iki şeyi birlikte taşıyor: son taramanın ilanlarının eğitimde olmaması (daha az, daha eski veri) ve taramalar arası fiyat değişimi; tek test bunları ayıramıyor. Farklı `ad_id` ile yeniden yayımlanan aynı araç eğitimde eski fiyatıyla bulunabilir (1. defter, 1.5); etkisi ölçülmedi. | bilinen sınır |
| İleri testte isabet olasılığı iyimser (6.3) | Ortalama öngörü gerçekleşenden yaklaşık 4 puan yüksek; düzeltilmedi (test ilanlarına bakıp düzeltmek ayar olurdu). | bilerek (2026-11-23 kararı) |
| Açıklamalı koşum (7) | Tek koşum, tek tohum; kazanç küçük ve beş katta aynı yönde, ama bir aralığı yok (anlamlılık testi kurulmadı). Aynı açıklamayla yeniden yayımlanan ilanlar katlar arasına düşerse metin kolu ikizini tanıyabilir; eski denetimde kazancın yaklaşık %11'i buradan geliyordu (`backlog/ideas.md`, 2026-11-23); bu defterde ölçülmedi, yalnız söyleniyor. Açıklama fiyatı da yazabilir; ölçülen fark bunu da içerir. | açık |
| İşlenmiş örnekler (5) | Dört örnek tek bir seçim kuralına bağlı (en az 100 ilanlı hücrelerde tahmini en yüksek ilan). "Görülmemiş kategori" örneği yakıt kolonuna kasa tipinin adını yazıyor; yapay bir girdi, yalnız −1 kodunun yolunu gösteriyor. | bilerek |

### Otomatik korunmayan yerler

| konu | ne | durum |
|---|---|---|
| Nitel cümleler | "Ölçek modeli her çeyrekte hedefe yakın", "F/S'nin farkı kovalar içinde daralıyor ama kapanmıyor", "18'de sıçrama yok", "gözlenen sayı simetrik karşılaştırmanın aralığında", "ileri testte en dar beşte bir en geniş beşte birden düşük", "10'da N'de gerçekleşen her N'de N/10'un üstünde" gibi cümleler yalnız tablolara dayanıyor. Başlık sayıları (45) bunların bir kısmını dolaylı koruyor; kova içi tablolar, beşte bir tabloları ve "10'da N" tabloları başlıkta yok. Veri değişirse bu cümleler sessizce bayatlayabilir. | açık |
| Eski kayıtlarla karşılaştırma | 665 sayının karşılaştırması depo dışında yapıldı; depoda eski sayıları koruyan test yok (plan gereği). Koruma bundan sonra başlık sayıları. | bilerek |
| Başka makine | Yeni ağır adımlar bu makinede iki ayrı süreçte bit bit aynı çıktı ve önbellek kaydıyla aynı. Başka işlemcide ya da BLAS'ta aynı çıkacakları sınanmadı (3. defterle aynı durum). | açık |
| Ölçek modelinin iş parçacığı | Ölçek modeli her ağaçta ilanların %80'ini çekiyor; iş parçacığı sayısının sonucu değiştirmediği uydurma ilanlarda sınanıyor (`tests/lib/test_interval.py`), gerçek veride sınanmadı. | bilinen sınır |
| İl gruplaması (3.4) | `location` kolonunun "ilçe, il" biçimine dayanıyor. Biçimi gerçek veride bir test sınıyor (`tests/lib/test_real_data_errors.py`); kolon gözlenen değerler kaydında yok. | bilinen sınır |
| Emsal tablosu: yaş ve model yılı (5) | Servis edilen emsal (araç modeli, yaş) ile aranıyor, defter (araç modeli, model yılı) ile sayıyor. Bugün bütün ilanlar 2026 taramalarından geldiği için aynı hücreler; bir test sınıyor (`tests/lib/test_interval.py`). Başka bir yılın taraması gelirse test durur. | bilinen sınır |

### Yazım sırasında giderilenler ve çıkarılanlar

| konu | ne yapıldı |
|---|---|
| Boş değerler grup dışı kalıyordu (3.4) | Eski kodda görülmeyen eksenler `astype(str)` ile gruplanıyordu; bu pandas sürümünde boş değer NaN kalıyor ve hiçbir gruba girmiyor, yani garantisi boş 6.618 ve takası boş 5.730 ilan sessizce dışarıdaydı. Artık boş değer kendi grubu (`missing`). Görülmeyen grup sayısı 32 → 34; tek payda kendi kapsamasının 3 puan altında kalan grup 4 → 6 (yeni ikisi garanti ve takas boş), ölçek modelinde 4 → 4. Eski yolla yeniden sayınca 32 / 4 / 4 birebir çıkıyor. |
| %20 tanımının asimetrisi (1.2, 2.1) | "768 fazla / 440 eksik" tek başına tahmin modelinin fazla tahmine eğilimli olduğu gibi okunuyordu (`backlog/ideas.md`, 2026-11-23). Defter tanımın asimetrisini açıklıyor ve yönü eşit tutan 1,2 kat sayımını yanına koyuyor (768 / 812). |
| ₺ sapmanın işareti (1.3) | Eski kod sapmayı tahmin − gerçek, artığı gerçek − tahmin diye tanımlıyordu; "ortalama yanlılık −₺12K…−₺3K" cümlesinde işaret karışıyordu. Defterde tek işaret var: ₺ artık = gerçek − tahmin, yüzde artıkla aynı yönde. Sayılar aynı, işaret ters. |
| İleri test "zaman etkisi" okuması (6.1) | 6,78'e karşı 6,49 zaman etkisi diye okunuyordu. Defter aynı test ilanlarının kat dışı MAPE'sini (6,60) ve ilan başına farkın küme bootstrap aralığını (+0,18 [0,10, 0,26]) yan yana veriyor; aralığın neyi kapsadığını yazıyor. |
| F/S ve yaş "ayrı etken" okuması (2.3) | Defter emsal kovası içinde oranları gösteriyor ve "F/S ayrı etken", "yaş tek başına büyütüyor" okumalarını açıkça dışlıyor. |
| "Payının 44 katı" (2.4) | Kat sayısı verilmiyor; segmenti adından çözülen ilanların payı hem bütün ilanlarda hem en pahalı çeyrekte yazdırılıyor ve metin oranın tabana göre değiştiğini söylüyor. |
| Serbest metin ifadeleri (7) | "Ön kayıtlı" ve "yeniden konan ilanlar iki kolu eşit etkiler" cümleleri yok (`backlog/ideas.md`, 2026-11-23); ikiz ilan etkisi ölçülmemiş bir sınır olarak yazılıyor. Kat kat fark tablosu eklendi. |
| % hata için taban ve tavan | Eski `08_residuals` yüzde hatanın paydasını 10 bin ₺'de, sonucu %500'de kesiyordu. En düşük fiyat 345 bin ₺, en büyük mutlak artık %178; ikisi de hiç bağlamıyordu, kaldırıldı. Tanım `errors.residual_pct`. |
| Kısmi bağımlılık (3.5) | Eski kod s'yi alt sınırsız okuyordu; defter servis edilen s gibi alt sınırda kırpılmış hâlini kullanıyor. Sınır bağlamıyor, sayılar aynı. |
| Çalışma sırasında durma denetimleri | Eski kodun `assert` ve "örnekler değişti" denetimleri: `interval.conformal_quantile` artık `ValueError` veriyor; (araç modeli, yaş) = (araç modeli, model yılı) denetimi ve "test ilanları ilan tablosunda aynı fiyatla" denetimi test oldu (`tests/lib/test_interval.py`, `tests/lib/test_forward.py`). Defterde tek `raise` kaldı: "görülmemiş kategori" örneğinin yakıt değeri veride görülmüşse. |
| İleri testte çeyrek başına ölçek kolu (6.2) | Eski kayıt çeyrek başına yalnız tek pay ve bant kollarını veriyordu; defter ölçek kolunu da ekliyor (yeni sayı, eski kayıtta karşılığı yok). |
| Çıkarılanlar | Bant başına q (Mondrian) kolu ve kabul aralığı (sahibin kararı); son taramanın ilanlarında kapsama (tarama, yani dönem gruplaması); `08_large_errors`'ın elle seçilmiş üç örnek ilanı (anahtarları gözlenen değerler kaydında olmayan araç modeli adları, notları ilan metni okunarak yazılmıştı); model başına medyan fiyat (yalnız site figürünü besliyordu). Karşılaştırmadaki 39 "hesaplanmıyor" bunlar. |

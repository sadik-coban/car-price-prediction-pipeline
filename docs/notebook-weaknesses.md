# Defterlerin bilinen zayıflıkları ve açık noktaları

Bu liste `notebooks/` altındaki defterler yazılırken görülen zayıflıkları toplar: yöntemin sınırları, otomatik
korunmayan yerler, sınanmadan söylenen cümleler ve gösterim eksikleri. Hiçbiri sayıları yanlış yapmaz; her biri bir
sonucun ne kadar güvenle okunabileceğini ya da bir değişikliğin fark edilmeden geçip geçemeyeceğini söyler.

- **Kapsam:** `1_data_and_cleaning` ve `2_what_drives_price`. Sonraki defterler yazıldıkça eklenir.
- **Tarih:** 2026-11-24.
- **Durum:** "açık" olanlara dokunulmadı; yöntem eski zincirdekiyle aynı tutuldu. Yöntemi değiştiren bir düzeltme
  sayıları oynatır ve başlık sayılarının (`tests/baselines/`) yeniden kabul edilmesini gerektirir.

## 1. defter: veri ve temizlik

### Otomatik korunmayan yerler

| konu | ne | durum |
|---|---|---|
| Toplama filtresi sınırları (1.4) | Verinin toplama filtresinin içinde kaldığı yalnız bir tabloyla gösteriliyor. Fiyat tavanı başlık sayılarında var (`price_at_cap`); kilometre ve yıl sınırının başlık anahtarı yok, değişirse hiçbir test yakalamaz. | açık |
| Defterdeki değer sabitleri | Defterin kendi yazdığı değerler ("SUV", "bmw", "audi", "Belirtilmemiş") gözlenen değerler kaydında (`db/observed_values.json`) var, ama bu elle kontrol edildi. İki yönlü test yalnız `notebooks/lib` içindeki sabitleri tarıyor, defter kaynaklarını taramıyor. | açık |

### Eksik ya da açık kalanlar

| konu | ne | durum |
|---|---|---|
| Gerekçesi olmayan dört ham kolon (4.5) | `kb_color`, `kb_trade_available`, `kb_seller_type` ve `gb_warranty_status` tahmin modeline girmiyor ve neden girmediği kayıtlı değil. | açık |
| Elektrikli araç sayısı (1.4) | "Elektrikli ilan = 0" sayısı hesaplanmıyor, çünkü "Elektrik" değeri veride hiç görülmedi ve görülmemiş bir değer koda yazılmıyor. Yerine yakıt tablosu elektrikli değer olmadığını gösteriyor. | bilerek |
| Belirtilmemiş paneller ham sayfalardan sayılıyor (3) | Veritabanındaki boş değerin "belirtilmemiş" cevabına tam karşılık geldiği gösterildiği için sayım yalnız veritabanından da yapılabilirdi. Yöntem aynı kalsın diye ham sayfalar okunuyor. | bilerek |
| Kolon sınıfları tablosu (4.5) | "Tahmin modelinde" satırı 53 kolonu tek tek sayıyor; 39 hasar bayrağı özetlenebilir. | açık |
| Başlık sayıları | 34 satır; "küçük tablo" için fazla olabilir. | açık |

### Sonradan giderilenler

| konu | ne yapıldı |
|---|---|
| Motor değeri kolonları (2.1) | "Veritabanının `*_val` kolonları aralığın orta noktası" cümlesi sınanmadan yazılmıştı. Gerçek veride sınandı ve bütün satırlarda tutuyor: iki sınır varsa orta nokta, tek sınır varsa o sınır, sınır yoksa boş. Test eklendi (`tests/lib/test_real_data.py`). Aynı yerde ikinci bir test, kuralın bir değeri yalnız sitenin hiç sınır vermediği ilanlarda boş bıraktığını sınıyor: hacim kuralı üst sınırı aldığı için yalnız alt sınırı olan bir hacim ("… cm3 ve üzeri") boş kalırdı. Site hacimde bu biçimi hiç göstermedi; gösterirse önce veritabanı kurulumu görülmemiş değer diye durur, test ondan sonraki adımı (biçim tanıtıldı ama kural uyarlanmadı) yakalar. |

## 2. defter: fiyatı ne belirliyor

### Yöntem zayıflıkları

| konu | ne | durum |
|---|---|---|
| Theil's U yukarı sapıyor (1.1) | Bilinen kolonun çok değeri olduğunda ölçü şişiyor; araç modelinin 745 değeri var. Örnek: U(tavan durumu \| araç modeli) 0.23 çıkıyor, oysa ikisi arasında gerçek bir bağ beklenmez. Marka, seri ve segment için 1'e yakın değerler yapısal ve güvenli. Kasa tipi, çekiş, vites ve yakıt için 0.72–0.93 arası değerlerin bir kısmı bu sapma. | açık |
| Silhouette tek örneklemde (4.4) | Skor 3.000 ilanlık tek bir örneklemde, tek bir tohumla hesaplanıyor. k = 2 (0.242) ile k = 6 (0.239) fiilen eşit; "en iyi k" okunmamalı. Vargı (belirgin küme yok) bundan etkilenmiyor. | açık |
| Ağır hasar terimi (5.1) | "Belirtilmemiş" 0 sayılıyor; ilanların %68'inde bu bilgi belirtilmemiş (1. defter, 4.1). Katsayı "kayıt yok" ile "bilinmiyor"u aynı grupta tutuyor. | açık |
| Motor tutarlılık kuralı (5.9) | Eşikler (medyanın 1.5 katı, en az 5 ilan) keyfi ve yalnız bir duyarlılık sınaması var. Az ilanlı araç modelleri kuralla sınanamıyor. | açık |
| Hedonik katlar (5.3) | Katlar regresyona giren 29.554 ilan üzerinde kuruluyor. Tahmin modelinin 29.988 ilanlık katlarıyla aynı fonksiyon (`folds.make_folds`), ama aynı katlar değil; iki defterin kat dışı sayıları aynı bölmeden gelmiyor. | açık |
| Kimliksiz sütunun karşılaştırması (5.4) | Kimliksiz sütun ile kontrollü iki sütun arasında terim başına fark testi yok; "km ve güç belirgin değişiyor" gözlemi aralıkların örtüşmemesine dayanıyor. Fark testi yalnız segment kontrolü ile araç modeli kontrolü arasında yapılıyor (5.5). | açık |

### Sınanmadan söylenen

| konu | ne | durum |
|---|---|---|
| Kilometre açıklaması (5.4) | Kimlik olmadan kilometrenin zayıf görünmesi "yaşı, hasarı ve motoru aynı ilanlar arasında kilometresi yüksek olanlar daha pahalı araç modellerinden" diye açıklanıyor. Bu, dışarıda kalan değişken mantığına dayanan bir yorum; defterde ayrıca sınanmıyor. | açık |

### Gösterim eksikleri

| konu | ne | durum |
|---|---|---|
| Breusch–Pagan p değeri (5.5) | Sayı alttan taşıyor ve `0` yazıyor; doğrusu "ölçülemeyecek kadar küçük". | açık |
| Log fiyat histogramı (3) | Çarpıklık iddiasını yalnız iki çarpıklık sayısı destekliyor; log fiyatın histogramı yok. | açık |

### Yazım sırasında giderilenler

| konu | ne yapıldı |
|---|---|
| Sıfır ya da eksi güç ve hacim | Eski kod boş değerlerin yanında sıfır ve eksi değerleri de regresyondan atıyordu; veride böyle bir değer yok, o dal kaldırıldı. Böyle bir değer gelirse regresyona sessizce girmesin diye gerçek veride güç ve hacmin pozitif olduğunu sınayan test eklendi (`tests/lib/test_real_data.py`). |
| Sitenin segment değeri | Metin kırpma yerine açık bir eşleme (`cleaning.raw_segment`); görülmemiş bir değerde duruyor. |
| `patsy` | Defter doğrudan import ediyor; `requirements-pipeline.txt` dosyasına eklendi. |

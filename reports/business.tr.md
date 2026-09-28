# İkinci El Araç Piyasası Analizi — Karar Notu

**Kime:** fiyatlama ekibi ve galeri. **Karar:** model, fiyat önerisi aracında birincil referans olarak kullanılabilir; ucuz (₺1.15M altı), emsalsiz ve yaşlı (kabaca 18 yaş ve üstü) araçlarda tek başına kullanılmamalı. **Kazanç:** araç başına ~₺81K daha az fiyatlama hatası. **Sınır:** ilan fiyatını tahmin eder, satış fiyatını değil.

## Ne kadar değerinde?

**Emsal medyanı** — *aynı model ve yılın ortadaki fiyatı; o yıl için emsal yoksa modelin tüm yıllarının, o da yoksa bütün piyasanın medyanı* — ortalama **₺191K** yanılıyor; model **₺110K** — **%43 daha iyi**, araç başına **₺81K**. 100 araçlık bir stokta bu, yaklaşık **₺8M**'lik fiyatlama hatası farkı demek. Yalnız emsali olan ilanlarda (%97.5) karşılaştırınca taban ₺179K, model ₺106K: araç başına ₺73K, %41.

Farkı kapatan, model ve yılın ötesi en çok kilometre ve hasar: modelden çıkarılınca hata (karesel ortalama) sırasıyla ₺55K ve ₺25K büyüyor. Motor bilgisi ise büyük ölçüde model adında zaten var: ayrıca çıkarılınca hata yalnız ₺524 artıyor.

![Ortalama hata: emsal medyanı vs model](figures/tr-00-base-vs-model.png)

**Emsal yoksa taban çöküyor** — en alt basamakta ortalama hata, model+yıl basamağının **6.1 katı**. Model de emsalsiz araçta zorlanıyor — nerede, ileride.

| taban basamağı | ilan | pay | ortalama hata |
|---|---:|---:|---:|
| model+yıl | 29.236 | %97.49 | ₺179K |
| model | 596 | %1.99 | ₺559K |
| global | 156 | %0.52 | ₺1.10M |

## Piyasa fiyatı nasıl kuruyor

Her kalemin fiyatı ne kadar oynattığı — **aynı model adı içinde, diğer her şey sabitken**:

| kalem | fiyat |
|---|---:|
| yaş (yıl başına, tipik araçta) | -%5.8 |
| kilometre (100 bin km başına, tipik araçta) | -%14.8 |
| ağır hasar kaydı | -%12.5 |
| değişen panel (her biri) | -%3.1 |
| boyalı panel (her biri) | -%1.0 |
| +100 hp motor gücü | +%19.3 |

En belirsiz kalem +100 hp motor gücü: %95 güven aralığı +%6.8 ile +%33.3 arasında.

Yaş ve kilometre birbirine bağlı iki eksen; tablodaki yaş ve km satırları biri sabitken ötekinin etkisi, tipik araçta (11 yaş, 181.000 km) ölçüldü.

![Yaşa göre ham fiyat — düzeltilmemiş (medyan + ort.)](figures/tr-05-age-price.png)

![Kilometreye göre ham fiyat — düzeltilmemiş (medyan + ort.; 400 bin km altı, 653 ilan dışarıda)](figures/tr-06-km-price.png)

**Markadan hareket çıkmaz.** Seri+modelin üzerine markayı eklemek ortalama hatayı hiç oynatmıyor — marka zaten modelin içinde.

![Ham medyan fiyat: BMW vs Audi — model karmasını yansıtır](figures/tr-07-brand.png)

## Sayıya nerede güvenme

Model ucuz araçlarda **yüzde olarak** zorlanıyor — hata fiyat çeyreğine göre belirgin değişiyor.

![Fiyat çeyreğine göre medyan hata (%)](figures/tr-10-quartile-error.png)

**Liraya çevrilince tablo tersine dönüyor.** Toplam lira hatasının en büyük payı (%39.7) Q4'te; ortalama mutlak hata en pahalı çeyrekte ₺176K, en ucuzda ₺77K. Tahmin edilen fiyata göre bakınca (fiyatlama aracının bildiği tek şey) model hiçbir çeyrekte belirgin yanlı değil: gerçek fiyat ile tahmin arasındaki eğim 1.003.

![Fiyat çeyreğine göre lira hatası](figures/tr-27-quartile-lira.png)

**Emsal azaldıkça büyük sapma (±%20 üstü) oranı artıyor.** Aynı model ve yıldan başka ilan yoksa bu oran %19.0, 100+ emsal varsa %2.9. Üst/spor segment (%19.1) ve 18 yaş ve üstü araçlar (%12.0) da riskli; genel oran %4.0.

### Neden tek sayı değil aralık

İlan fiyatında iki yönlü hata da para kaybettirir: **fazla tahmin alıcıya patlar** — pahalıya alınmış araç; **düşük tahmin satıcıya** — ucuza gitmiş araç. Tek sayı ne kadar emin olunduğunu saklar; aralık bunu söyler ve kullanıcıyı belirsizliğin büyük olduğu yerde uyarır.

Bu yüzden çıktı tek sayı değil, **%90 aralık**. Ama aralık ucuz araçlarda tutmuyor: en ucuz çeyrekte gerçek kapsama **%81.6**, hedefin altında.

![%90 aralık kaç ilanda tuttu (hedef %90)](figures/tr-12-coverage.png)

**Ne yapmalı**

- Ucuz araçlarda aralığı genişlet — tek sayıya güvenme.
- Nadir ve uç araçları elle fiyatla; model orada saçılıyor.
- Metninde dönüşüm, motor değişimi ya da modifiye geçen ilanı yayına almadan önce gözden geçir: bu bilgi formda yok. Araç özellikleri sabitken bu ilanlarda hata oranında anlamlı bir fark ölçülmedi; gözden geçirme model hatasına değil, formun göremediği bilgiye karşı.
- **Kaymayı izle ve modeli yeniden eğit:** canlıda fiyat dağılımını izleyen bir servis kur, model yeni verilerle yeniden eğitilsin. Fiyat dağılımı bugün az kayıyor (en yüksek PSI 0.005), ama piyasa seviyesi 4 dönemde +%2.0 kaydı ve model zamanı görmüyor.
- **Fiyat rejimini değiştiren gelişmeleri takip et** (vergi/ÖTV düzenlemesi, teşvik, ani piyasa hareketi gibi) — eğitim planı bunlara göre yapılmalı. Eski dönemleri atma: veri biriktikçe hata düşüyor.

![Daha çok veri, daha az hata — tek dönem vs biriken dönemler](figures/tr-15-backtest.png)

## Bu model neyi vermez

- **Satış fiyatını.** İlan fiyatını tahmin eder; satış fiyatı pazarlıkla bunun altına iner.
- **Nedenselliği.** Bunlar kontrollü ilişkiler; "boya yaptır, fiyat düşer" demez.
- **Belirli bir hasarlının değerini.** Model hasarı panel ve panel grubu düzeyinde görüyor (boyalı · değişen · ağır hasar kaydı) ama **şiddetini** görmüyor: hafif bir çizik de derin bir göçük de aynı "boyalı" bayrağına düşüyor.
- **Paketin ötesindeki donanımı ve modifiyeyi.** Model adındaki donanım paketi (M Sport, S Line gibi) okunuyor; paketin dışındaki opsiyonlar ve sonradan yapılan değişiklikler görünmüyor.
- **BMW ve Audi dışını.** Kapsam bu iki marka; başka markalara ne kadar genellenebildiği ölçülmedi.

**Ölçek:** 29.988 ilan, 4 tarama dönemi. **Veri:** 2026-01-18 – 2026-06-27. Medyan ilan fiyatı ₺1.55M.

# İkinci El Araç Piyasası Analizi — SHAP Raporu

Model fiyatı **neye bakarak** kuruyor? Burada açıklanan model LightGBM — teknik raporun kısaca "model" dediği. CatBoost'un iki varyantı yalnız §4'teki karşılaştırmada geçiyor. Sorular veri ölçeğinde: hangi öznitelik ne kadar pay alıyor, etki değerler değiştikçe nasıl değişiyor. Tek ilan ölçeğinde yalnız §6'da bir örnek var.

## 1. SHAP nedir, LOFO'dan farkı ne

**SHAP** tek bir tahmini parçalarına ayırır: bu ilanda yaş fiyatı şu kadar yukarı itti, kilometre şu kadar aşağı çekti; parçaların toplamı tahmini tam verir. Teknik rapordaki **ablasyon / LOFO** (Leave-One-Feature-Out) ise bir öznitelik çıkarılınca *hatanın* ne kadar büyüdüğüne bakar.

## 2. Nasıl hesaplandı

- **Veri:** 29.988 ilanın **tamamı** — örnekleme yok.
- **Model: OOF.** Her ilan, onu eğitimde hiç görmemiş fold modeliyle açıklanıyor — zincirin kendi 5-fold kurulumu. Yeniden kurulan fold'ların tahminleri zincirin yazdığı OOF tahminlerle eşleşti (en büyük fark ₺0.00).
- **Yöntem:** `shap.TreeExplainer` (exact). Toplamsallık hatası 5.1e-13 (final model üzerinde ölçüldü), yani parçalar tahmini tam veriyor.
- **Ölçek:** hedef `log1p(fiyat)`, yani katkılar log'da toplanıyor ve fiyatta **çarpan** oluyor (aşağıdaki tablo).
- **Gruplama:** model/seri adının 170 SVD boyutu tek tek anlamsız, `MODEL_SERIES` altında toplandı. §3'teki tabloda `ENGINE` = hp + cc, `DAMAGE` = 13 panelden türeyen 12 öznitelik + ağır hasar kaydı; beeswarm ve grup grafiklerinde bunlar ayrı satır. §6'nın dökümü yalnız SVD boyutlarını (`MODEL_SERIES`) toplar, öteki öznitelikler ayrı — orada motor gücü kendi satırında; hacmin bu ilandaki payı küçük olduğu için "diğer öznitelik" çubuğunda kalıyor. Grubun değeri satır başına üyelerinin **işaretli toplamı** — grubun o ilandaki net etkisi; tablodaki sayı onun ortalama mutlak değeri (§7).

### Hangi grafikte hangi ölçek

Her grafikte okura en anlamlı ölçek seçildi:

| ölçek | nerede | neden |
|---|---|---|
| **log** | beeswarm, grup, etkileşim | İlanlar arası karşılaştırılabilir ve **toplanabilir** tek ölçek: `+0.62` ucuz araçta da pahalı araçta da aynı anlama gelir. |
| **%** | bağımlılık eğrileri, §3 tablosundaki **fiyatta tipik etki** sütunu | Katkıyı çarpana çevirir: `e^s − 1`. +0.10 → ×1.105, yani %10.5 daha pahalı; −0.10 → ×0.905, %9.5 daha ucuz. |

**Örnek hesap.** Yaş (yıl) kaleminin ortalama |SHAP|'i 0.2870 → `e^0.2870 = 1.332`: bu büyüklükte bir katkı fiyatı %33.2 oynatır (ortalamanın üsteli tipik etkiyi büyütür; tipik değer tabloda). Aynı ilanda Motor (hp + cc) de 0.1347 eklemişse yüzdeler toplanmaz, çarpanlar çarpılır: `1.332 × 1.144 = 1.525` → %52.5, %33.2 + %14.4 = %47.7 değil.

## 3. Fiyatı ne belirliyor

![Fiyatı en çok ne belirliyor](figures/tr-sh-01-importance.png)

| öznitelik | ortalama \|SHAP\| | pay | fiyatta tipik etki |
|---|---:|---:|---:|
| Yaş (yıl) | 0.2870 | %43.9 | %23.5 |
| Motor (hp + cc) | 0.1347 | %20.6 | %12.3 |
| Kilometre | 0.0842 | %12.9 | %7.8 |
| Model/seri adı | 0.0666 | %10.2 | %5.5 |
| Hasar (panel + ağır hasar) | 0.0412 | %6.3 | %3.9 |
| Segment | 0.0282 | %4.3 | %2.4 |
| Kasa tipi | 0.0070 | %1.1 | %0.4 |
| Vites | 0.0043 | %0.7 | %0.2 |
| Yakıt | 0.0009 | %0.1 | %0.1 |
| Çekiş | 0.0001 | %0.0 | %0.0 |
| Marka | 0.0001 | %0.0 | %0.0 |

*Paylar LightGBM'e özgü (§4). **Fiyatta tipik etki** katkının fiyattaki karşılığının (`|e^s − 1|`) ilanlar üzerindeki medyanı.*

Atfın %43.9 kadarını tek başına **Yaş (yıl)** alıyor, ilk üç öznitelik %77.4 kadarını.

Sıralama fold'dan fold'a tam sabit değil: 3., 4. sıradaki kalem en az bir fold'da değişiyor; payları yakın kalemlerin sırası tek başına okunmamalı.

### Yön kontrolü

![Her nokta bir ilan](figures/tr-sh-02-beeswarm.png)

Etki yönü korelasyonu (Spearman): yaş **-0.99**, kilometre **-0.96**, motor gücü **+0.89**. Yaş ve km arttıkça fiyat aşağı, güç arttıkça yukarı gidiyor. Bu bir tutarlılık kontrolü ve üreteçte kapı: işaretlerden biri ters dönerse rapor üretilmez.

![Öznitelik değeri ve fiyata etkisi](figures/tr-sh-03-dependence.png)

Üç eğri:

- **Yaş:** medyan katkı 0–2 yaşta +%90.4, 18+ yaşta -%47.0. Bu, modelin yaşa verdiği payda uçtan uca **3.6 kat** fark; aynı iki grubun gerçek medyan fiyat oranı 5.7 kat. Aradaki fark yaşlı araçların başka özelliklerde de farklı olmasından; SHAP o kısmı o özniteliklere yazıyor — en çok Kilometre, Motor (hp + cc), Hasar (panel + ağır hasar).
- **Kilometrenin bedeli** (ardışık pencerelerin medyan katkı farkı; pencere merkezi = penceredeki medyan kilometre; pencereler eşit aralıklı olmadığı için 100 bin km'ye oranlandı): 28→127 bin km arasında %8.7, 127→223 binde %11.4, 223→385 binde %7.2. Son pencerede oran belirgin düşüyor (bir öncekinin 0.64 katı): çok yüksek kilometrede ek kilometrenin bedeli azalıyor ama sürüyor. 350 bin km ve üstünde 1.723 ilan var.
- **Motor gücü:** medyan katkı güç bantlarında <150 hp -%14.9 → 150–200 hp +%7.7 → 200–250 hp +%15.5 → 250+ hp +%43.2; 150 hp altından 250 hp üstüne atıf farkı **1.7 kat**.

### shap'in kendi bölmesi: iki grup

![İki grup](figures/tr-sh-04-cohorts.png)

Bölmeyi biz vermedik: shap kendi karar ağacıyla veriyi **Yaş (yıl) = 9.5** eşiğinden ikiye ayırdı. *Yaş (yıl)* için ortalama |SHAP| `Yaş (yıl) < 9.5` grubunda 0.396, `Yaş (yıl) >= 9.5` grubunda 0.230. Bunu "model yeni araçta yaşa daha çok bakıyor" diye okumak yanıltır: |SHAP| ortalamadan sapmayı ölçtüğü için tipik yaştaki araçta yaşın katkısı küçük. Yaş bantlarına göre ortalama |SHAP|: 0–4 0.577 · 5–8 0.333 · 9–12 0.084 · 13–16 0.271 · 17+ 0.611 — V şeklinde; yaşın ağırlığı hem yeni hem çok yaşlı araçta büyük.

### Yaş ve kilometre birlikte çalışıyor

![Aynı kilometre, farklı yaş](figures/tr-sh-05-km-age.png)

Renk aracın yaşı. Aynı **150–250 bin km** bandında kilometre katkısının medyanı 10 yaş altında -0.030 (2.232 ilan), 10 yaş ve üstünde -0.006 (9.000 ilan): model yaşlı araçta kilometreye daha az ceza kesiyor. İki öznitelik bağımsız değil (§7).

## 4. Üç varyant aynı fiyatı farklı gerekçeyle kuruyor

| öznitelik | LightGBM | CatBoost (SVD) | CatBoost (native) |
|---|---:|---:|---:|
| Yaş (yıl) | %43.9 | %30.9 | %32.1 |
| Motor (hp + cc) | %20.8 | %17.2 | %17.1 |
| Kilometre | %12.9 | %22.1 | %22.3 |
| Model/seri adı | %8.8 | %13.7 | %16.7 |
| Hasar (panel + ağır hasar) | %6.5 | %7.6 | %5.5 |
| Segment | %5.2 | %7.1 | %0.1 |
| Kasa tipi | %1.0 | %0.4 | %2.9 |
| Vites | %0.7 | %0.9 | %1.4 |
| Yakıt | %0.1 | %0.1 | %0.3 |
| Çekiş | %0.0 | %0.1 | %0.5 |
| Marka | %0.0 | %0.0 | %1.1 |

Doğrulukta üç varyant birbirine yakın (MAPE %6.49 · %6.44 · %6.58). Gerekçede ayrılıyorlar: LightGBM yaşa %43.9 pay veriyor, CatBoost (SVD) %30.9 (13.0 puan fark); kilometrede durum tersine dönüyor (%12.9 · %22.1). Yaş ile kilometre birlikte hareket ettiği için payın hangisine yazılacağı modelin tercihi. **Sonuç:** ilk üç kalem üç modelde de aynı (Yaş (yıl), Motor (hp + cc), Kilometre), ama sıraları ve payları modele bağlı; CatBoost (native) modelinde 3. ile 4. kalem arasındaki fark yalnız 0.4 puan.

*Tablo üç modeli **final** hâlleriyle ve §3'le aynı kuralla karşılaştırır: grup değeri satır başına işaretli toplam. Native varyantta model ve seri adı ayrı iki öznitelik; burada onlar da satır başına toplandı. §3'ün payları OOF modellerinden, buradaki LightGBM sütunu final modelden; aynı kalem iki tabloda en fazla 1.4 puan ayrılıyor (Model/seri adı).*

## 5. Marka neden sıfıra yakın

`brand`'in ortalama |SHAP|'i **0.0001**, atfın %0.0 kadarı. Model adı markayı zaten belirliyor (teknik rapor §3: U(marka | model) = 1.00); seri+modelin üzerine marka eklemek ortalama hatayı ₺1 değiştiriyor.

Üç yöntem — bağımlılık ölçüsü, ablasyon, SHAP — aynı yere çıkıyor: **marka ayrı bilgi taşımıyor.** Bu "marka fiyatı etkilemez" demek değil; etkisi model adının içinde.

## 6. Tek bir ilanda karar

![118i Standart](figures/tr-sh-06-waterfall.png)

Alttaki `E[f(X)]` modelin hiçbir öznitelik bilmeden verdiği değer, üstteki `f(x)` bu ilana verdiği tahmin; aradaki her ok bir özniteliğin katkısı, "15 diğer öznitelik" oku ise kalanların toplamı (log ölçek, §2).

- **İlan:** 118i Standart · 2007 · 330.000 km
- **Gerçek fiyat:** ₺595.000
- **İlanı görmemiş modelin tahmini:** ₺519.680 (-%12.7)

Tahmini kuran en büyük üç kalem:

1. Yaş (yıl) = 19 yıl → ×0.54
2. Model/seri adı = 118i Standart → ×0.87
3. Motor gücü (hp) = 138 hp → ×0.89

Listede olmayan 21 kalem birlikte fiyatı ×0.76 yapıyor (-%23.5) — toplamda küçük değil.

Gerçek fiyatla aradaki %12.7 fark hiçbir çubukta görünmüyor: SHAP **tahmini** parçalara ayırır, gerçek fiyatı değil. Modelin neyi bilmediği dökümde yazmaz.

## 7. Kısıtlar

- **Atıf, nedensellik değil.** SHAP modelin neyi kullandığını söyler, piyasanın nasıl çalıştığını değil.
- **Birlikte hareket eden öznitelikler payı bölüşür.** Yaş ile kilometrenin payı modele göre değişiyor (§4). TreeSHAP burada `tree_path_dependent` modda, yani gözlemsel; pay dağılımı bu seçime de bağlı.
- **Katkılar log uzayında.** Kalemleri karşılaştırırken **çarpanlara** bakın: her ilanda aynı anlama gelirler. Liraya çevirmek ilana özgü ve sıraya bağlı olduğu için bu raporda yapılmadı.
- **Gruplama bir karardır.** `MODEL_SERIES` 170 SVD boyutunun satır başına işaretli toplamı, yani adın o ilandaki net etkisi; boyutlara tek tek bakılsa her biri küçük görünür. Üyelerin |SHAP|'ini ayrı ayrı toplamak kullanılmadı: o toplam girdinin kaç parçaya bölündüğüne göre büyür — aynı ad 170 boyutta ayrı toplanınca 2.5 kat büyük çıkar.

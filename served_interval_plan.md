> **Durum (2026-11-23): onaylandı; 0–6. adımlar (bu repo) uygulandı.** Kalan: 7. adım (API), 8. adım (canlıya alma) ve
> 9. adım (site); üçü de repo dışı, kullanıcıda. Bu plan `mondrian_band_comparables_plan.md`'nin (bant × emsal taslağı ve
> ölçek modeli eki) yerini aldı; o dosya `archive/`'e taşındı.
>
> **Değerlendirme (2026-11-23):** iki bağımsız denetim yapıldı. Karar: yöntem aynı kaldı, plan sondaki
> "Değerlendirme kararları" bölümündeki değişikliklerle uygulandı. O bölüm, yukarıdaki metinle çeliştiği yerde geçerlidir.
> API paritesi için işlenmiş örnekler: `data/serving/serve/parity_examples.json` (07_final_model yazar).

# Servis edilen aralık: ölçek modeli ve isabet olasılığı

## Özet

Bugün her tahmine aynı pay veriliyor: `CONFORMAL_Q` = 0,1412, yani −%13,2 / +%15,2. Genel kapsama %90, ama gruplarda
tutmuyor. Yeni araçta pay gereğinden geniş (%97), eski, emsalsiz ve hasarlı araçta dar (%66–81).

Plan iki şeyi servise alıyor:
1. **Araca göre aralık.** Küçük bir ikinci model (ölçek modeli, s) bu tür araçta beklenen hata büyüklüğünü tahmin eder.
   Aralık bu büyüklükle ölçeklenir; %90'ı conformal yöntem sağlar.
2. **İsabet olasılığı.** Bu tür araçta ilan fiyatının tahminin ±%10 yakınında olma olasılığı. Aralıkla aynı hesaptan
   çıkar; yeni bir model gerekmez.

Fiyat tahmini değişmez: ana model, MAPE ve öteki bölümler aynı kalır.

**Süre:** ~1,5 iş günü.

| bölüm | adımlar | süre |
|---|---|---|
| cardatasys: lib, 08, 07, raporlar, defter, kapı | 0–6 | 6–7 sa |
| API (`sadik-portfolio/api`, repo dışı) | 7 | ~1,5 sa |
| canlıya alma | 8 | ~20 dk + kullanıcı |
| site (`sadik-portfolio`, repo dışı) | 9 | ~1,5 sa |

## Kararlar

| konu | karar | gerekçe |
|---|---|---|
| yöntem | ölçek modeli (normalized conformal) | Bant × emsal yalnız gördüğümüz iki hatayı düzeltiyor, ölçek modeli öteki eksenleri de. En dar aralığı ve en iyi aralık skorunu veriyor (bkz. Kanıt). |
| s'nin girdisi | ana modelin kodlu matrisi, metin sütunları hariç (`cv.fold_matrices(…, use_text=False)`) + log tahmin + emsal sayısı | API aynı kodlamayı zaten kuruyor (`cat_maps`, bilinmeyen → −1): servis paritesi kendiliğinden. Pandas kategorileriyle aynı sonucu veriyor. |
| emsal | eğitimde aynı (model, yaş) hücresindeki ilan sayısı. OOF'ta öteki katlardan, serviste bütün eğitimden sayılır. | API yılı değil yaşı alıyor; verinin hepsinde yaş = 2026 − yıl (ölçüldü). Bir test korur. |
| monotonluk | s, emsal sayısına göre artmayan olacak biçimde kısıtlı (`monotone_constraints` = −1, yalnız emsal) | Baştan kondu: "daha çok emsal payı hiç genişletmez". Kısıtsız eğri zaten neredeyse monotondu. Kısıtla emsalsiz OOF kapsaması 88,8'den 90,3'e çıktı. |
| s'nin ayarları | 300 ağaç, öğrenme oranı 0,05, 15 yaprak, yaprakta en az 100 ilan, subsample ve colsample 0,8, seed 42, `deterministic`, `force_row_wise`, alt sınır 0,02 | Bir kez kondu, aranmadı. Alt sınır bugün hiç devreye girmiyor (en küçük s 0,021). |
| isabet olasılığı | ±%10, yani \|log fiyat − log tahmin\| ≤ log 1,1 | Bir kez kondu, aranmadı. |
| sitede sunum | "10'da N" sıklığı ve nokta göstergesi; "yüksek / orta / düşük" etiketi yok | Sıklık yüzdeden doğru okunur; etiket keyfi eşik ister. |
| adım 0 | yeni ön kayıt yok, karar notu var | Sonuçlar görüldü. `plans/09-forward-coverage` geri çekilirken de aynı yol izlendi. |
| bant × emsal | koda girmez, yalnız karar notunda sayılarıyla anılır | Etkisiz kod tutulmaz (CLAUDE.md). |
| `08_conformal_coverage` | `records.HEAVY`'ye girer | Artık model eğitiyor (25 küçük LightGBM). Defter onu kayıttan okur. |
| repo dışı (API, site) | Bu plan şartnameyi verir. Kod kullanıcı yapar ya da açıkça izin verir. | Repo dışı yalnız okunur. |
| model adı öneri listesi | bu planın dışında, `backlog/ideas.md`'ye | Emsal 0 uyarısı yazım hatasını görünür kılıyor; liste ayrı iş. |

## Tarif

Bir aracın tahmini `p`, ölçeği `s(x)`:

1. **Hata:** her eğitim ilanının OOF hatası `e = |log1p(fiyat) − log1p(tahmin)|`.
2. **Ölçek modeli:** `s(x)` = e'yi tahmin eden küçük LightGBM, alttan 0,02'de kırpılır.
3. **İç içe skorlar:** her katın ilanlarına, o katı görmemiş bir s ile `e / s(x)` hesaplanır. 29.988 skor.
4. **Katsayı:** `k` = skorların sonlu örneklem %90 yüzdeliği (`conformal_quantile`). Bugünkü veride k = 2,0874.
5. **Aralık:** `expm1(clip(log1p(p) ± k·s(x)))`. Kırpma `lib.conformal.interval` ile aynı.
6. **İsabet olasılığı:** skorlardan `log(1,1) / s(x)`'in altında ya da ona eşit olanların payı.

Servis edilen s bütün OOF hatalarıyla eğitilir; k ve skorlar 3. adımdaki iç içe s'lerden gelir. İleri test bu ayrımı
birebir kopyalıyor.

**Kapsama güvencesi:** yalnız genel %90 güvenceli, s ne öğrenirse öğrensin. Gruplardaki kapsama ampirik. s yanlış
öğrenirse kapsama yine %90 olur, ama pay yanlış araçlara gider ve aralık tek q'dan da kötü olabilir. Yani kapsama s'den
bağımsız, genişliğin kalitesi değil.

## Kanıt

**Kaynak:** salt okunur kontrol betikleri, oturumun geçici dizininde.
- `plan_numbers.py`: bu plandaki tarif, ölçek modeli ve isabet olasılığının bütün sayıları.
- `interval_audit2.py`, `interval_audit3.py`: öteki kollar.
- `confidence_score.py`: kısıtsız isabet kontrolü.

Betikler cardatasys'in kendi kodunu import ediyor (`load_clean`, `load_oof`, `make_folds`, `fold_matrices`,
`lib.conformal`, 08_forward_test işlevleri). Tek q'nun ve ileri testin yayımlanan değerleri assert'le eşleniyor. Sayılar
buraya kopyalandı. Uygulama bunları kendi koşusunda yeniden üretir; sayılar canlı zincire buradan girmez.

**Terimler:**
- **Genişlik:** tahminin yüzdesi.
- **IS:** log uzayında aralık skoru (Winkler). Genişlik ile dışarıda kalanların cezasının toplamı; küçük daha iyi.
- **Grup:** en az 200 ilanlı grup.

### Yöntemlerin karşılaştırması

**OOF, çapraz kalibreli (29.988 ilan):**

| yöntem | kapsama | genişlik medyan / ort. | IS | emsalsiz (752) |
|---|---|---|---|---|
| tek q (bugün) | 89,99 | 28,3 / 28,3 | 0,4211 | 70,7 |
| bant başına q (plan 08, yayımlı) | 90,01 | 25,9 / 27,5 | 0,4059 | 66,8 |
| CQR | 90,00 | 27,3 / – | – | 73,7 |
| bant × emsal | 90,06 | 25,9 / 27,6 | 0,3929 | 90,8 |
| **ölçek modeli** | **90,09** | **24,7 / 27,2** | **0,3657** | **90,3** |

**İleri test (son taramadan önce görülen 21.099 ilanla eğitilir, son taramada yeni 8.889 ilanda ölçülür):**

| yöntem | kapsama | genişlik medyan / ort. | IS | emsalsiz (297) |
|---|---|---|---|---|
| tek q | 87,40 | 28,2 / 28,2 | 0,4438 | 68,35 |
| bant başına q | 87,99 | 27,1 / 27,4 | 0,4296 | 65,32 |
| CQR | 88,68 | 27,6 / – | – | 70,03 |
| bant × emsal | 87,68 | 25,8 / 27,4 | 0,4178 | 90,91 |
| **ölçek modeli** | **86,83** | **24,0 / 26,8** | **0,3977** | **88,55** |

**Gruplarda kapsama** (tek q ve ölçek modeli, aynı betik):
- **s'nin girdisi olan eksenler** (yaş, km, beygir, segment, yakıt, vites, kasa, ağır hasar, değişen parça, seri sıklığı;
  OOF'ta 42 grup):
  - tek q'da 18 grup 90'dan 5 puandan fazla sapıyor, en büyük sapma 25,2;
  - ölçek modelinde hiçbiri, en büyük sapma 3,5;
  - bant × emsal 44 grupluk aynı kontrolde 8 grupta sapıyordu, en büyük sapma 20,9.
  - Bu düzelme kısmen tasarım gereği, çünkü bu eksenler s'nin girdisi. Yine de dışarıda ölçüldü, ve kullanıcının
    girdiği bilgiye göre kapsama tam da istenen şey.
- **s'nin görmediği eksenler** (satıcı, il, renk, garanti, ilk sahip, takas, tramer): yöntemler arasında fark yok.
  - Her yöntemde 3–4 grup kendi genel kapsamasının 3 puan altında.
  - En kötüsü ileri testte kırmızı araçlar (n 201): ölçek modelinde −10,7, tek q'da −5,8. Küçük bir grup ve 26 grup
    arasından seçildi, gürültü olabilir; raporda yazılır.

**Seçilmeyen ara yol:** k'yı emsal kovası başına hesaplamak. İleri testte emsalsiz kapsama (88,55) ve IS (0,3976 /
0,3974) değişmedi. s emsal etkisini zaten öğreniyor.

### Araç türüne göre (OOF)

Aralık, grubun medyan tahmini etrafında milyon ₺. İsabet: ±%10 içinde olma olasılığı; öngörülen → gerçekleşen.

| araç türü | ilan | kapsama bugün → yeni | pay yeni | aralık bugün | aralık yeni | isabet |
|---|---|---|---|---|---|---|
| hepsi | 29.988 | 90,0 → 90,1 | −12 / +13 | 1,33–1,76 | 1,36–1,73 | 78 → 78 |
| 0–3 yaş | 2.649 | 97,4 → 90,8 | −8 / +8 | 3,41–4,53 | 3,64–4,25 | 92 → 92 |
| 4–7 yaş | 4.420 | 97,5 → 90,5 | −7 / +8 | 2,36–3,13 | 2,52–2,93 | 92 → 92 |
| 8–12 yaş | 12.264 | 92,7 → 89,9 | −10 / +12 | 1,38–1,83 | 1,42–1,77 | 82 → 82 |
| 13–18 yaş | 8.853 | 83,3 → 90,3 | −15 / +18 | 0,96–1,27 | 0,94–1,30 | 67 → 67 |
| 19+ yaş | 1.802 | 75,2 → 88,5 | −18 / +22 | 0,60–0,80 | 0,57–0,85 | 59 → 59 |
| emsalsiz | 752 | 70,7 → 90,3 | −20 / +25 | 1,49–1,98 | 1,38–2,14 | 53 → 56 |
| 20+ emsal | 20.081 | 91,9 → 90,3 | −11 / +12 | 1,30–1,73 | 1,34–1,68 | 81 → 81 |
| en ucuz çeyrek | 7.497 | 81,0 → 89,9 | −16 / +19 | 0,79–1,04 | 0,76–1,08 | 65 → 64 |
| en pahalı çeyrek | 7.497 | 94,7 → 90,7 | −8 / +9 | 2,71–3,59 | 2,87–3,39 | 87 → 88 |
| km'si en yüksek %20 | 5.998 | 79,9 → 89,9 | −16 / +20 | 0,91–1,20 | 0,87–1,25 | 64 → 63 |
| ağır hasarlı | 1.407 | 80,5 → 89,7 | −16 / +19 | 1,04–1,38 | 1,01–1,42 | 64 → 63 |
| F segmenti | 325 | 65,8 → 86,5 | −22 / +28 | 2,30–3,05 | 2,07–3,39 | 50 → 52 |

**Payın dağılımı:**
- İlanların %62'sinde pay daralıyor, %38'inde genişliyor.
- Yüzdelikler: %5'te −7 / +7, %25'te −9 / +10, %50'de −12 / +13, %75'te −15 / +18, %95'te −21 / +27.
- En dar aralık %8,9, en geniş %185 (OOF). İleri testte en dar %8,4, en geniş %128.
- s'nin emsale göre kısmi bağımlılığı monoton azalıyor: emsal 0'da 0,093, 5'te 0,069, 20'de 0,063, 200'de 0,060.

### İsabet olasılığı güvenilir mi?

Olasılığın onda birlik dilimlerinde öngörülen → gerçekleşen (%):

| dilim | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| OOF | 49 → 51 | 62 → 61 | 68 → 68 | 74 → 72 | 78 → 79 | 83 → 82 | 88 → 87 | 91 → 91 | 94 → 94 | 96 → 97 |
| ileri test | 50 → 51 | 62 → 57 | 70 → 64 | 76 → 69 | 80 → 74 | 84 → 81 | 88 → 83 | 91 → 88 | 94 → 91 | 97 → 94 |

- **OOF'ta kalibre:** ortalama öngörü 78,3, gerçekleşen 78,3. Brier skoru, herkese aynı olasılığı veren skordan %12,2 iyi.
- **Ayırt ediyor:** araçların %10'unda olasılık %58'in altında, %10'unda %95'in üstünde.
- **Yeni ilanlarda iyimser:** ortalama öngörü 79,1, gerçekleşen 75,1. Orta dilimlerde fark 4–6 puan; Brier skoru yine
  sabit skordan %9,7 iyi. Aralığın yeni ilanlarda %87'de kalmasıyla aynı zaman etkisi. Test kümesine bakıp düzeltmek ayar
  olur: yapılmaz, yazılır.

## Dürüstlük notları (karar notuna ve rapora girer)

- **Seçim sonuçtan sonra.** Yöntem, s'nin ayarları, monotonluk kısıtı ve ±%10 sonuçlar görüldükten sonra kondu.
  Hiçbirinde arama yapılmadı.
- **Bağımsız test yok.** Veri dondu, ileri test sayıları görüldü.
- **Yalnız genel %90 güvenceli.** Gruplardaki iyi kapsama ampirik. Bant × emsal'in 16 hücresinin her birinde ayrı bir
  güvencesi vardı (zaman değişmediği sürece). Daha keskin aralık ve öteki eksenler için bundan vazgeçildi; karar
  notu bunu açıkça yazar.
- **Zamanı düzeltmez.** Yeni ilanlarda kapsama %86,8, isabet olasılığı ~4 puan iyimser. Raporda yeni ilanlar için öne
  çıkan sayı bu olur, %90 değil.
- **s'nin görmediği bilgiyi bilemez.** Satıcı, il, renk, ilan metnindeki bir kusur. Bu eksenlerde ölçek modeli tek q'dan
  ne iyi ne kötü.
- **Hedef ilan fiyatı.** Kapsama ve isabet ilan fiyatı içindir: satış fiyatı ya da "gerçek değer" değil. Veri Haziran
  2026'da kaldı.
- **Model adı serbest metin.** Emsal araması eğitimdeki dizeyle birebir. Yazım hatası emsali 0 yapar ve aralığı
  genişletir: ihtiyatlı yönde bir hata, ama kullanıcı sebebini görmeli (adım 9).
- **Çok geniş aralıklar var.** %185'e kadar; site metni bunu karşılar.

## Adımlar

### 0. Karar notu (~20 dk)

`docs/decisions.md` ve `docs/decisions.en.md`'ye "Servis edilen aralık: ölçek modeli ve isabet olasılığı (2026-11-23)"
girdisi:
- bugünkü durumun sayıları;
- denenen kollar ve sayıları (tek q, bant başına q, CQR, bant × emsal, ara yol, ölçek modeli);
- seçim gerekçesi ve takas (hücre güvencesi ↔ keskinlik ve öteki eksenler);
- dürüstlük notları;
- kilitli `plans/08-mondrian-coverage`'ın bu işi kapsam dışı saydığı ve neden yeni ön kayıt açılmadığı.

08 planına dokunulmaz.

### 1. `analysis/lib/conformal.py` — ortak parçalar (~1 sa)

`lib/cv.py`'ye ve `lib/__init__.py`'ye dokunulmaz, yoksa bütün metrikler bayatlar. Tek kaynak olarak yeni parçalar:
- **Sabitler:** `SCALE_PARAMS`, `S_FLOOR = 0.02`, `WITHIN = 0.10`.
- **`other_fold_counts(cells, fold)`:** her ilan için öteki katlarda aynı (model, yıl) hücresindeki ilan sayısı.
- **`scale_matrix(X, tr, va, log_pred, comparables)`:** `fold_matrices(…, use_text=False)` + log tahmin + emsal sayısı.
  Döndürür: (eğitim, değerlendirme, kategorik sütunlar).
- **`fit_scale` / `predict_scale`:** monotonluk kısıtı yalnız emsal sütununda; tahmin `S_FLOOR`'da kırpılır.
- **`nested_scores(...)`:** her ilanın skoru, o ilanın katını görmemiş bir s'den.
- **`p_within(sorted_scores, s, within=WITHIN)`:** isabet olasılığı.

Testler (`tests/analysis/test_08_conformal_coverage.py` içinde, mevcut düzen):
- iç içe skorlarda s puanlanan katı hiç görmüyor: o katın fiyatları değişince skoru üreten s aynı kalıyor;
- emsal sayımına ilanın kendi katı girmiyor;
- s emsal sayısına göre artmıyor (kısıtın tuttuğu);
- alt sınır uygulanıyor;
- iki koşu aynı s'yi veriyor;
- `p_within` s büyüdükçe artmıyor;
- k, skorların sonlu örneklem %90 yüzdeliğine eşit;
- veride yaş = 2026 − yıl, yani (model, yaş) bölümlemesi (model, yıl) ile aynı.

### 2. `analysis/08_conformal_coverage.py` — ölçek kolu ve isabet (~1,5 sa, koşu ~30 sn)

- **Kollar:** tek q ve bant başına q kalır (biri eski servis, öteki plan 08'in ön kaydı). Ölçek kolu eklenir: her kat s'yi
  ve k'yı yalnız öteki katlardan alır.
- **Bütün kollara IS ve genişlik dağılımı** eklenir.
- **`report.scale`:**
  - `k`, `floor_hits`;
  - genişlik: medyan, ortalama, en dar, en geniş, yüzdelikler;
  - kapsama bant, emsal kovası, girdi eksenleri ve görmediği eksenler başına;
  - s'nin emsale göre kısmi bağımlılık eğrisi;
  - araç türü tablosu (yukarıdaki satırlar).
- **`report.confidence`:** onda birlik güvenilirlik tablosu, Brier skoru (sabit skorla karşılaştırmalı), grup başına
  öngörülen ve gerçekleşen.
- **Site ağacı:** yalnız bir site sayfasının ya da rapor figürünün kullandığı alanlar girer, ör.
  `domain.conformal.scale_by_quantile`. Rapor figürünün verisi dışa aktarılmazsa figür testi düşer.
- **Kart** (`analysis/cards/08_conformal_coverage.json`): yöntem, iç içe şema, kanıt.
- **`records.HEAVY`'ye girer.** Gerekçe: "25 küçük LightGBM eğitir".
- **Beklenen sayılar** (kontrol betiğindekiyle birebir, kod deterministik):
  - genel 90,09; emsal 0 kovası 90,29; IS 0,3657; k 2,0874;
  - isabet ortalaması 78,3 öngörülen / 78,3 gerçekleşen.
  - Tek q ve bant kollarının sayıları değişmez. Tutmazsa durulur, sebebi bulunur.

### 3. `analysis/08_forward_test.py` — ileri kol (~45 dk, koşu ~45 sn)

- **`forward_coverage`'a ölçek kolu:**
  - s'ler ve k eğitim kümesinin iç içe OOF hatalarından gelir; eğitim ilanının emsali kendi iç CV katına göre sayılır;
  - servis edilen s bütün eğitim kümesiyle eğitilir;
  - test ilanının emsali bütün eğitim kümesindeki sayıdır.
- **İsabet olasılığı:** güvenilirlik tablosu ve ortalama öngörülen / gerçekleşen.
- **Test:** `test_forward_coverage_calibrates_on_training_only` genişler: test fiyatları değişince k ve s yerinde
  kalmalı.
- **Beklenen sayılar:**
  - ölçek kolu: genel 86,83, emsalsiz 88,55, IS 0,3977, k 2,0997;
  - isabet: 79,1 öngörülen / 75,1 gerçekleşen;
  - mevcutlar değişmez: MAPE 6,78, kapsama 87,4 / 87,99, emsalsiz 68,35 / 65,32.

### 4. `analysis/07_final_model.py` → `encoders.pkl` (~1 sa, koşu birkaç dakika)

Yeni anahtarlar (hepsi `lib.conformal`'ın aynı işlevleriyle):

| anahtar | içerik |
|---|---|
| `CONFORMAL_SCALE_MODEL` | servis edilen s, LightGBM metni (`model_to_string`) |
| `CONFORMAL_SCALE_COLS` | s'nin sütun sırası ve kategorik sütunlar |
| `CONFORMAL_K` | k (bugün 2,0874) |
| `CONFORMAL_S_FLOOR` | 0,02 |
| `CONFORMAL_SCORES` | iç içe skorların sıralı listesi (29.988 sayı, ~240 KB), isabet olasılığı için |
| `CONFIDENCE_WITHIN` | 0,10 |
| `COMPARABLE_COUNTS` | {(model, yaş): ilan sayısı} |
| `DATA_UNTIL` | verideki son tarama tarihi |

Ayrıca:
- `CONFORMAL_Q` kalır, çünkü eski backend onu okuyor. `CONFORMAL_TANIM` ve `serving_readme` yeni tarifi anlatır. Girdi
  bölümüne "emsal = (model, yaş) sayısı, model dizesi birebir" eklenir.
- **Servis s'si ana modelin `cat_maps` kodlarıyla eğitilir** (`full_matrix`'in metin dışı sütunları + log tahmin +
  emsal). API bu satırı zaten kuruyor.
- **Ana model değişmemeli.** `lightgbm_tfidf_svd.txt`'nin sha256'sı bugünkü serve dosyasıyla aynı çıkmalı; o zaman S3'e
  yalnız `encoders.pkl` gider. Farklıysa durulur.
- **`to_metrics`** k'yı, emsal sözlüğünün boyutunu ve toplamını, örnek satırlarda s'yi yazar.
- **`test_07_final_model`:**
  - `COMPARABLE_COUNTS` toplamı 29.988;
  - 07'nin k'sı 08'deki `report.scale.k` ile aynı;
  - örnek satırlarda servis s'si lib'in hesabıyla aynı.

### 5. Raporlar (~1,5 sa)

**Teknik rapor §8** (`builders/build_technical_report.py`):
- aralık paragrafı üç kolla: tek q, bant başına q, ölçek modeli; IS sütunu;
- ölçek modeli paragrafı, takas ve dürüstlük notları;
- araç türü tablosu;
- isabet olasılığı paragrafı ve tek bir yeni figür: güvenilirlik eğrisi, OOF ve ileri test;
- figür 12'ye ölçek kolu;
- ileri test paragrafına ölçek kolu ve isabet satırı.
- 99fe61e'de eklenen "emsalsiz kapsama hedefin altında" assert'i tek q ve bant kolları için kalır. Ölçek kolu için cümle
  ayrı yazılır, yeni veri kapısıyla: "ölçek modeli emsalsizde hedefe yakın, öteki iki kol değil".

**İş notu** (`build_business_report.py`, "Neden tek sayı değil aralık"):
- aralık artık araca göre; isabet olasılığı tek cümleyle;
- "emsalsiz araçta aralık tutmuyor" cümlesi değişir;
- yeni ilanlarda ~%87.

**Ortak:** `report_common` (REQUIRED, `derive`, figür listesi), `build_site_data` (SITE_KEYS), iç aracın
`section_map.json` + `catalog.py --stamp`. Raporlar yeniden üretilir; `reports/` elle değiştirilmez.

### 6. Defter, kapı ve belgeler (~45 dk)

- **Defter:** `notebooks/3_errors_and_text`'te 08_conformal_coverage hücresi `run()` yerine kayıttan okur.
  `python tools/run_notebooks.py`.
- **Kapı:**
  - `python tools/verify.py` 0 ile çıkmalı;
  - `tools/analysis_coverage.py` kartları ve testleri dolu göstermeli.
- **Referans:** `python tools/snapshot_metrics.py --diff` kullanıcıya gösterilir; `--accept` yalnız onayla.
- **Belgeler:**
  - README.md ve README.en.md'de aralık cümlesi;
  - `docs/analysis-scripts`'te ağır betik sayısı (8 → 9);
  - `backlog/ideas.md`'ye model adı öneri listesi.

### 7. API — `sadik-portfolio/api` (repo dışı; kullanıcı yapar ya da açık izin verir, ~1,5 sa)

`app/services/lgb_service.py`:
- **Yükleme:** `_assert_bundle_has_interval` yeni anahtarları da ister. Eski paket sessizce tek q'ya düşmez,
  yüklenmeyi reddeder.
- **Tahmin:**
  1. Mevcut satır kurulur (`cat_maps`, bilinmeyen → −1) ve fiyat tahmin edilir; bu kısım değişmez.
  2. s girdisi: `CONFORMAL_SCALE_COLS` sütunları + log tahmin + `COMPARABLE_COUNTS.get((model.strip(), yaş), 0)`.
  3. `s = max(booster.predict(...), S_FLOOR)`, `r = K · s`.
  4. Aralık `expm1(clip(log tahmin ± r))`, kırpma analizdekiyle aynı.
  5. İsabet `searchsorted(SCORES, log(1 + WITHIN) / s, side="right") / len(SCORES)`.
- **Yanıt (yalnız eklenir):**
  - `price_range`'in mevcut alanları aynı kalır, değerleri araca göre değişir;
  - yeni: `price_range.comparables`, `confidence: {within_percent: 10, probability}`, `data_until`.
- Docstring'deki "One q for the whole market" yeniden yazılır.
- **Parite:** analizdeki örnek satırlarda API'nin s'si ve aralığı lib'inkiyle aynı. Yerel başlatıcı
  `cardatasys/data/serving/serve`'den okur. Elle karşılaştırılacak örnekler:
  - bol emsalli pahalı bir araç (~−8 / +8);
  - emsalsiz bir araç (~−20 / +25);
  - yazım hatalı bir model adı (emsal 0).

### 8. Canlıya alma (~20 dk + kullanıcı)

1. Kullanıcı yeni `encoders.pkl`'yi S3'e yükler (`SERVING_ENCODERS_KEY`). Eski backend yeni anahtarları görmez,
   `CONFORMAL_Q` ile çalışmaya devam eder, yani güvenli.
2. API main'e push edilir; Railway yeniden başlar ve yeni paketi yükler.
3. Canlıya üç örnek istek atılır, sonuçlar adım 7'deki değerlerle karşılaştırılır.

Frontend bu sırada değişmek zorunda değil: `lower_percent` / `upper_percent` zaten okunuyor.

### 9. Site — `sadik-portfolio` (repo dışı; kullanıcı yapar ya da açık izin verir, ~1,5 sa)

**Tahmin kartı** (`app/_site/predict/FinalPredict.tsx`):
- nokta fiyat yuvarlanır ("1,53 milyon ₺");
- aralık sabit ölçekli bir çubukla gösterilir;
- emsal sayısıyla tek bir sebep cümlesi;
- emsal 0 ise uyarı ve model adını kontrol etme önerisi;
- isabet satırı ("10'da N" ve nokta göstergesi);
- veri dönemi ("Ocak–Haziran 2026 ilan fiyatlarına göre");
- "Nasıl hesaplandı" bağlantısı rapordaki §8'e.

"%90 conformal aralık" satırı teknik ayrıntıya iner.

```
Tahmini piyasa değeri
3,93 milyon ₺

Olası aralık   3,64 ────●──── 4,25 M₺        −%8 / +%8
İsabet         ●●●●●●●●●○

Eğitim verisinde bu model ve yıldan 143 ilan var. Bu tür araçlarda
ilan fiyatı 10 durumdan 9'unda tahminin ±%10 yakınında.
Ocak–Haziran 2026 ilan fiyatlarına göre · Nasıl hesaplandı →
```

```
Tahmini piyasa değeri
1,72 milyon ₺

Olası aralık   1,38 ──────────●────────── 2,14 M₺   −%20 / +%25
İsabet         ●●●●●○○○○○

⚠ Eğitim verisinde bu model adı ve yıl için ilan yok. Fiyat benzer
araçlardan tahmin edildi; aralık bu yüzden geniş. Model adını
kontrol edin (ör. "520i Executive M Sport").
```

Örneklerdeki emsal sayısı (143) uydurma; gerçek sayı API'den gelir.

**Sunum kuralları:**
- olasılık "10'da N" olarak aşağı yuvarlanır, en çok 9 (N = min(floor(10·p), 9)); "10'da 10" hiç gösterilmez;
  yüzde, s, k ya da "güven puanı" gösterilmez;
- "emsal" kelimesi yerine "eğitim verisinde bu model ve yıldan N ilan";
- "Nasıl hesaplandı"da: kapsama ve isabet ilan fiyatı içindir, veri Ocak–Haziran 2026, yeni ilanlarda gerçekleşen
  birkaç puan daha düşük; dar aralıklar önce eskir.

**Metinler:**
- `app/_site/overview/FinalOverview.tsx` ve `content/building-car-price-predictor.mdx` (81. ve 87. satırlar): "one
  margin for the whole market" yerine "pay, bu tür araçta beklenen hata kadar";
- i18n anahtarları (TR/EN).

**Rapor senkronu:** `content/reports/car-price/*`, `public/report-data.json`, `public/site_data.json`.
- **Ön koşul:** portfolyonun drift sayfası `methodology.backtest`'i okuyor ve FinalDrift.tsx §9'a bağlantı veriyor. İkisi
  de 99fe61e'de kalktı. Senkrondan önce bu sayfanın ne olacağına karar verilmeli.

## Bitti tanımı

- **Beklenen sayılar:** adım 2 ve 3'teki sayılar birebir üretildi; mevcut kolların sayıları değişmedi.
- **Ana model:** sha256 aynı.
- **Kapı:**
  - `python tools/verify.py` PASS; sonuç satırı aktarılır;
  - iç aracın testleri `.venv-tool` ile geçiyor;
  - `analysis_coverage` dolu;
  - `snapshot_metrics --diff` kullanıcıya gösterildi ve onayla kabul edildi.
- **Çıktılar:** raporlar ve defterler yeniden üretildi; raporlarda tek q'nun "herkese aynı pay" anlatımı kalmadı.
- **Canlı:** API paritesi yerelde, ardından canlıda üç istekle doğrulandı.

## Kapsam dışı

- s'nin ayarlarını, girdilerini, alt sınırını, ±%10 toleransını ya da monotonluk kısıtını aramak.
- s'ye model/seri metnini ya da ilan metnini eklemek.
- CQR'ı ya da bant × emsali servis etmek; emsal kovası başına k (ara yol).
- Zaman etkisini düzeltmek (ileri kalibrasyon), yeni tarama, yeniden eğitim.
- %90 dışında bir hedef; "yüksek / orta / düşük" etiketleri.
- Model adı öneri listesi (`backlog/ideas.md`).

## Kullanıcıyı bekleyenler

1. Planın onayı ("Kararlar" tablosundaki varsayılanlar dahil).
2. Referans kabulü (`snapshot_metrics --accept`).
3. S3'e `encoders.pkl` yüklemesi.
4. API ve site: kodu kullanıcı yazar ya da bu repo dışı iş için açık izin verir; push onayı.
5. Drift sayfasının akıbeti (site senkronundan önce).

## Değerlendirme kararları (2026-11-23)

İki bağımsız denetim planı çürütmeye çalıştı: biri geçerlilik ve kararlılık, öteki ileri test. Betikleri salt okunur,
oturumun geçici dizininde (`scale_validity/`, `scale_forward/`). Buradaki sayılar oradan kopyalandı; uygulama kendi
koşusunda yeniden üretir.

### Doğrulananlar

- **Sayılar:** `plan_numbers.py` değiştirilmeden koşuldu, plandaki bütün sayılar birebir çıktı.
- **Sızıntı:** her dış kat için her şey (ana model, iç OOF hataları, s, k) öteki dört kattan yeniden kuruldu.
  Kapsama 90,11, emsalsiz 90,03, IS 0,3669, isabet 78,3 → 78,2. Plandaki çapraz uydurmanın payı yok sayılabilir.
- **Kararlılık:** 10 tohum ve 7 ayar (ağaç 150/600, yaprak 7/31, yaprakta en az 50/200, kısıtsız).
  - OOF kapsama 90,00–90,25, IS 0,3654–0,3672, k 2,075–2,105;
  - ileri kapsama 86,69–87,06.
  - Seçilen ayar tipik, şanslı bir çekiliş değil.
- **Uçlar:** en geniş %1'de ortalama s 0,203, gerçekleşen hata 0,196, kapsama 90,3. En dar %1'de s 0,027, hata 0,027.
  Genişlik hak edilmiş.
- **İsabet varsayımı:** e / s dağılımı yaş gruplarında ve s dilimlerinde aynı (KS en çok 0,029). Grup içinde öngörülen ile
  gerçekleşen arasındaki fark en çok ~2 puan.
- **İleri testte gruplar:** tek q'da kapsama 67–98 arasında, ölçek modelinde 84–91. 19+ yaş 70 → 86, emsalsiz 68 → 89.

### Kararlar

| # | konu | karar | gerekçe |
|---|---|---|---|
| 1 | yöntem | 30 girdili s kalır. "En iyi IS" gerekçesi "OOF'ta" diye daraltılır. | Üç girdili s (log tahmin, yaş, emsal) OOF'ta geride: IS 0,3780'e karşı 0,3657, fark gerçek (SE 0,0012) ve hasar eksenlerinden geliyor (ağır hasarlı 80,8 → 89,7). İleri testte IS farkı gürültü (−0,0033, SE 0,0025), kapsama 0,76 puan düşük (SE 0,39). Seçim gerekçesi: kullanıcının girdiği hasar bilgisi payı etkilemeli. Üç girdili kol koda girmez, karar notunda sayılarıyla anılır. |
| 2 | güvence cümlesi | "Yalnız genel %90 güvenceli" çıkar. Yerine: "%90 tasarım gereği yaklaşık tutar; sızıntısız ölçüm 90,11. Yalnız aynı dönemin ilanları için geçerli, zaman içinde değil (ileri 86,8)." | Servis edilen merkez ve s bütün veriyle, skorlar kat modelleriyle kuruluyor. Bu kurulumun sonlu örneklem güvencesi yok. |
| 3 | kayma duyarlılığı | `08_conformal_coverage` tek bir alan daha üretir: bugünkü veride fiyatlar %2 / %5 / %10 yüksek olsaydı kapsama (genel ve en dar beşte bir; tek q ve ölçek). Rapora tek cümle girer. | Dar aralık önce eskir. Ölçülen: genel 89 / 81 / 57, en dar beşte bir 88 / 70 / 27; tek q 89 / 86 / 71. İsabet %5'te 78'den 69'a iner. Zaman analizi değil: veriye dokunmayan bir varsayım hesabı; zaman tek cümle olarak kalır. Tablo "yalnız seviye" diye etiketlenir: %2 satırı kabaca Haziran'ın bugünkü durumu, ve seviye tek başına ileri kaybın yarısını açıklıyor (aşağıda). |
| 4 | ileri test tablosu | `08_forward_test` genişlik beşte birleri başına kapsamayı (iki kol) ve kaçırmanın yönünü de yazar. Raporda "%87"nin yanında verilir. | En dar beşte birde 86,8 (aynı araçlar tek q'da 98,0). Kaçırmaların %80'i aralığın üstünde (10,5'e 2,7): yeni ilanlar tahminin medyanda ~2 log puan üstünde. Bu sayı 98 günlük fiyat artışı değil; aşağıdaki "Zaman denetimi"ne bakın. |
| 5 | "10'da N" | Aşağı yuvarlanır, en çok 9. "10'da 10" hiç gösterilmez. | Yuvarlamayla araçların %11'i "10'da 10" görür; gerçekleşen 94–96,5. İleri testte "8" görenlerde gerçekleşen 7,4. |
| 6 | sitede tarih | İsabet ve aralık cümleleri dönemi taşır: "Ocak–Haziran 2026 ilan fiyatlarına göre" ("Haziran 2026" değil). "Nasıl hesaplandı"da: dar aralıklar önce eskir; Haziran ilanlarında gerçekleşen kapsama ayrıca verilir. | Model dört taramanın fiyatlarını birlikte öğreniyor. Bugünkü tek q ile Haziran ilanları OOF tahminin ortalama +1,73 log puan üstünde, kapsama 88,7 (n 11.526); önceki taramaların ilanlarında 90,6–91,2. Ölçek modelinde aynı sayı uygulamada ölçülür. |
| 7 | monotonluk | Kısıt tasarım kuralı olarak kalır. "88,8 → 90,3" gerekçesi çıkar. | Fark gürültü içinde (n 752, SE 1,1). İleri testte kısıtlı ve kısıtsız aynı (88,55). |
| 8 | en geniş aralık | "%185" öne çıkarılmaz; %99'luk yüzdelik raporlanır. | En geniş değer ayara göre 141–233 arasında oynuyor. |
| 9 | alt sınır | "Hiç devreye girmiyor" → "OOF'ta girmiyor; ileri testte 8.889 araçtan 2'sinde giriyor". | Ölçüldü (ileri en küçük s 0,0193). |
| 10 | tohum gürültüsü | Tohum ortalaması alınmaz. Dürüstlük notuna yazılır. Adım 1'deki "iki koşu aynı s" testi determinizm testi diye adlandırılır. | Aynı aracın payı tohumdan tohuma medyan %2,5, %95'lik %9,5 değişiyor; toplam sayılar değişmiyor. Servis tek modelle deterministik. Beş model servis etmek bu kazanca değmez. |
| 11 | bilinmeyen kategori | Adım 7'nin parite örneklerine sözlükte olmayan bir kategori değeri (kod −1) taşıyan bir araç eklenir. | Kanıtta −1 kodu hiç geçmiyor; API yolu sınanmamış. |
| 12 | adım 1 test notu | "s puanlanan katı görmüyor" testinin yalnız s düzeyini kanıtladığı yazılır; sızıntısız kopyanın sayısı karar notuna girer. | Ana model düzeyi çapraz uydurma. |
| 13 | rapor metni (adım 5) | İş notundaki "bant başına payla emsalsizde kapsama daha da düşüyor" yan cümlesi çıkar. §8'de "zayıflık fiyata bağlı" yerine "yaşa ve emsale bağlı" yazılır. | Birincisi gürültü (n 297, eşli fark −6,8…+0,7). İkincisi ölçüldü: yaş kontrol edilince çeyrekler arası fark −3,43'ten +0,10 puana iniyor. Ölçek modelinin gerekçesi de bu. |

**Süreye etkisi:** yaklaşık +1,5 saat (3, 4 ve 13. kararlar).

### Zaman denetimi (2026-11-23, ek)

Kullanıcı "+2,2 log puan güvenilir mi, proje kayma analizine uygun mu" diye sordu. Üç bağımsız denetim yapıldı
(geçici dizinde `drift_level/`, `drift_measures/`, `drift_advocate/`). 4. ve 6. kararlar buna göre düzeltildi.

- **+2,2 fiyat artışı değil, modelin geride kalma payı.** İleri testte işaretli hata ortalama +2,46 [2,18–2,75],
  medyan +2,02. İki parçası var:
  - **eğitim kümesinin kuruluşu, +1,16 [1,03–1,29]:** eğitim fiyatlarının %46'sı ocak tarihli ve marttan önce çıkmış
    ilanlara ait; çıkan ilanlar aynı gün kalanlardan ~2 log puan ucuz;
  - **fiyat seviyesi, 98 günde yaklaşık +1 ile +1,5:** yalnız mart stokuyla eğitilen kol +1,30 [1,0–1,6] (iki denetim
    ayrı ayrı buldu), hedonik +1,45…+1,52, (model, yıl) hücre medyanları +0,9…+1,0.
  - Aynı gün yeni ilan ile stokta kalan ilan arasındaki fark yok sayılır (−0,5…0).
- **Sayı nominal ilan fiyatıdır;** enflasyondan arındırılmadı, satış fiyatı değil.
- **Seviye, ileri kaybın yaklaşık yarısı.** Yalnız +%2,2 seviye kayması kapsamayı 89,3'e indirir; gözlenen 87,4. Kalanı
  hataların yayılmasının genişlemesi (SS 9,04 → 9,34). Tek denetimin sayısı.
- **Kayma ölçüleri (PSI, KS, EMD) geri gelmez.** Ardışık taramalarda fiyat PSI'ı gürültü tabanının altında
  (0,0004–0,0017; taban 0,0009–0,0024). 0,10 eşiği yalnız `model` ve konumda aşılıyor, orada değer gürültüye eşit.
  Eğitim ile sonraki ilanları ayıran sınıflayıcının AUC'si 0,527; ileri hata bu skora göre düz.

### Tarih etkisini modelden çıkarma araştırması (2026-11-23, ek)

Kullanıcı tarih etkisinin modelden doğrudan çıkarılmasını araştırmamı istedi. Üç denetim yapıldı (`date_methods/`,
`date_identify/`, etki haritası). Hiçbir şey uygulanmadı.

**Kullanıcı kararı (2026-11-23):** sabit ofset uygulanmayacak, tarih modele girmeyecek. Gerekçe: ileri tarihte bir test
yapılmayacak, ofset doğrulanamaz. Fiyat tahmini olduğu gibi kalır; zaman tek cümlelik sınırlama olarak durur ve 6.
karardaki "Ocak–Haziran 2026" ifadesi geçerlidir. Aşağıdaki sayılar kayıt içindir.

- **Çıkarılabilen şey küçük.** En iyi yöntemler ileri testte MAPE'yi 6,78'den 6,64–6,67'ye, kapsamayı 87,4'ten
  88,3–88,5'e getiriyor. Sonraki 98 günün hareketi (+1,4) hiçbir yöntemle çıkmıyor.
- **Tarama girdisi sağlam değil.** 9 gün için +1,28 öğreniyor; stokun hareketi 0,42. Çıkan ucuz ilanları tarih diye
  etiketliyor. OOF'taki 6,49 → 6,35 kazancı serviste yok (Haziran ilanlarında fark −0,005). Canlı API yeni sütunu
  hata vermeden 0 ile doldurur.
- **Hedefi kaydırıp yeniden eğitmek** sabit ofsetle aynı sonucu veriyor (fark gürültü içinde), bedeli ise 14 betik,
  referans kabulü ve bu plandaki bütün sayıların yeniden üretilmesi.
- **Sabit ofset (yeniden eğitim yok):** son taramanın ilanlarındaki ortalama OOF log artığı (+1,73) servis edilen
  tahmine eklenir. Haziran ilanlarında yanlılık +1,73 → 0, kapsama 88,7 → 89,3 (q aynı), MAPE aynı (6,69 → 6,70).
  İleri testte MAPE 6,66, kapsama 88,4.
- **Ofsetin adı "tarih düzeltmesi" olamaz.** Tarama grupları arasındaki 4,9 puanlık farkın yaklaşık %60'ı tarih
  (%40–70), kalanı seçim. Doğru adı: tahmini son taramadaki stokun seviyesine hizalamak.
- **Kabul sınaması (hangi yöntem seçilirse):** ileri kurulumda, düzeltme yalnız eğitim taramalarından kestirilince son
  eğitim gününün yeni ilanlarında yanlılık ~0 olmalı, ileri yanlılık "yalnız son stok" kolunun (+1,30) altına inmemeli.

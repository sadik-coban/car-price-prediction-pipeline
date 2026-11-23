> **Durum (2026-11-23, akşam):** anlık döküm; aralık planının repo adımları uygulandıktan sonra güncellendi. Kaynakları: `tools/analysis_coverage.py`, `plans/`, `backlog/ideas.md`,
> `internal_tool/section_map.json`. Tablolar elle yazıldı. Güncellemek için bu kaynaklara yeniden bakılır.

# Analizler: biten ve kalan

## Özet

- **Biten:**
  - analiz zincirinde 19 betik, SHAP tarafında 4 betik;
  - hepsinin kartı, testi, referans değeri ve kaynak hücresi dolu (`analysis_coverage`);
  - bayat metrik yok.
- **Servis edilen aralık:** onaylandı, repo tarafı (0–6. adımlar) uygulandı
  ([served_interval_plan.md](served_interval_plan.md)). Kalan: API, canlıya alma ve site; üçü de repo dışı.
- **Bekleyen fikirler:** 6 analiz fikri ve 3 temizlik işi var. İçlerinde gerçek değer taşıyan tek analiz fikri, silinip
  yeniden konan ilanlar.
- **Sonraki tur:** rapor düzeltme turu. 2026-11-23 değerlendirmesinin bulguları `backlog/ideas.md`'de 11 satır.
- **Repo dışı:** site senkronu ve push.

## Biten analizler

### Teknik rapor

| bölüm | betik | cevapladığı soru |
|---|---|---|
| §1 Veri temizliği | `01_dedup_leakage` | Tarama satırlarından tekil ilana nasıl gelindi; fold'lar arasına ne sızabilir? |
| | `01_engine_rule` | Site hacim ve gücü aralık verdiğinde model hangi sayıyı kullanmalı? |
| | `01_unspecified_panels` | Kaç panel ve kaç ilan "Belirtilmemiş" hasar cevabı taşıyor? |
| §2 Eksiklik | `02_missingness` | Hangi kolonlar eksik ve birlikte eksik; her kolon modele neden girdi ya da atıldı? |
| §3 Bağımlılık ve marka | `03_association` | Kategorik ve sayısal öznitelikler birbirine nasıl bağlı? |
| | `03_segment_quality` | Segment beslemeden mi alınmalı, türetilmeli mi? |
| | `03_brand_ablation` | Marka; seri ve modelin yanında ayrı bilgi taşıyor mu? |
| §4 Hedef | `04_target` | Fiyat nasıl dağılıyor; model neden log fiyatı hedef alıyor? |
| §5 Piyasa yapısı | `05_segmentation` | Segment, kasa, yaş ve km'ye göre fiyat; belirgin kümeler var mı? |
| §6 Hedonik | `06_hedonic` | Yaş, km, motor ve hasarın fiyata kontrollü etkisi ne kadar? |
| §7 Model | `07_model_comparison` | Hangi model en iyi tahmin ediyor, tabanı ne kadar geçiyor? |
| | `07_final_model` | Serviste kullanılan modeller hangileri, nasıl eğitiliyor? |
| | `07_lofo` | Hangi öznitelik ya da grup olmadan model ne kadar kötüleşir? |
| | `07_text_flag` | Dönüşüm veya modifiye geçen ilanlarda model daha çok mu yanılıyor? |
| §8 Hata ve aralık | `08_residuals` | Hatalar nasıl dağılıyor; fiyat düzeyine göre yanlılık var mı? |
| | `08_large_errors` | %20'yi aşan hatalar nereden geliyor? |
| | `08_conformal_coverage` | %90 aralık her fiyat bandında ve araç türünde %90 kapsıyor mu; Mondrian ve ölçek modeli ne değiştiriyor; isabet olasılığı güvenilir mi? |
| | `08_forward_test` | Eskiden öğrenen model son taramanın yeni ilanlarında ne kadar doğru, aralık orada ne kadar kapsıyor? |
| §9 Serbest metin | `10_free_text` | Satıcı açıklaması eklenince doğruluk ne kadar değişir? |

### SHAP raporu

| betik | konu |
|---|---|
| `shap/02_oof_shap` | Sızıntısız SHAP: her ilanı onu görmemiş fold modeli açıklıyor. |
| `shap/03_what_sets_price` | Fiyatı ne belirliyor: motor ve hasar grupları, tipik etki. |
| `shap/04_variants` | Üç model aynı fiyatı farklı gerekçelerle kuruyor. |
| `shap/06_one_listing` | Tek bir tahmin nasıl kuruluyor. Örnek, hatası medyana en yakın ilan. |

### Planlar ve denemeler

- **Ön kayıtlı planlar:**
  - `plans/08-mondrian-coverage`: kilitli, H1 doğrulandı;
  - `plans/10-text-contribution`: kilitli, H1 doğrulandı;
  - `plans/09-forward-coverage`: geri çekildi (`archive/plans-withdrawn/`). Gerekçe: zaman analizi çıktı, tek ileri test
    kaldı (2026-11-22).
- **Denenip kazanç getirmeyenler** (`archive/experiments/`):
  - büyük hatalı ilanları eğitimden atmak: hiçbir eşikte tabanı geçmiyor;
  - derinliği 1 olan LightGBM: MAPE 0,75 puan kötü.

## Kalan işler

### 1. Servis edilen aralık: repo tarafı bitti, repo dışı kaldı

[served_interval_plan.md](served_interval_plan.md) iki şey getirdi:
- **araca göre aralık:** her araca beklenen hata büyüklüğüyle ölçeklenen bir aralık (ölçek modeli);
- **isabet olasılığı:** fiyatın tahminin ±%10 yakınında olma olasılığı.

Fiyat tahmini değişmedi; ana model dosyası birebir aynı. 0–6. adımlar uygulandı: `lib/conformal.py`,
`08_conformal_coverage`, `08_forward_test`, `07_final_model`, raporlar, defter, karar notu, referans kabulü.

Kalan (repo dışı, kullanıcıda):
- **API (7. adım):** paketten ölçek modelini okuyup aralığı ve isabeti hesaplar. Parite örnekleri
  `data/serving/serve/parity_examples.json`'da.
- **Canlıya alma (8. adım):** yeni `encoders.pkl` S3'e; ana model dosyası yüklenmez.
- **Site (9. adım):** tahmin kartı ve rapor senkronu.

Aynı gün alınan kararlar (`docs/decisions.md`): kayma ölçüleri geri gelmez; tarih modele girmez; sabit ofset uygulanmaz.

### 2. Bekleyen analiz fikirleri (`backlog/ideas.md`)

| fikir | durum | öneri |
|---|---|---|
| CQR (aralığı ilan ilan ayarlama) | Aralık çalışmasında ölçüldü. OOF kapsaması 90,00, emsalsizde 73,7. İleri testte 88,68, emsalsizde 70,03. Ölçek modeli geçti. | **Kapanabilir.** Sayılar aralık planının 0. adımındaki karar notuna yazılır, backlog satırı silinir. |
| Silinip yeniden konan ilanlar | Keşif yapıldı, rapora girmedi. Aynı `ad_id`'nin arada kaybolması 385 ilan. İleri testte "yeni" sayılan 8.889 ilanın ~131'i (%1,5) daha önce görülmüş araç. | **Değerli olan tek fikir.** Yayımlanmış bir sonucu (ileri test) ilgilendiriyor. Etkisi küçük beklenir. Rapora girecekse önce plan açılır (`tools/new_plan.py`). |
| SHAP etkileşimleri / ALE | Etkileşimin toplam payı ölçüldü (derinlik 1 denemesi: log R² 0,9639'a karşı 0,9699). Hangi çiftlerden geldiği ölçülmedi. | İsteğe bağlı. |
| φK ya da tek değişkenli OOF R² tablosu | Cramér's V'nin yerine kategorik ve sayısal ilişkiyi tek ölçüde verir. | İsteğe bağlı; §3'e yeni bir şey söylemez. |
| Hacmi tazelikten ayırmak (kayan ve genişleyen pencere) | §9 "Zaman" bölümüne bağlıydı; o bölüm çıktı. | Gelecek fikri olarak kalsın. Dört düzensiz taramayla (9, 53, 98 gün ara) zayıf kalır. |
| Endeksle düzeltilmiş hedef | Model zamanı görmüyor; dönem farkı hedefte düzeltilebilir. | Gelecek fikri olarak kalsın. Zaman analizini çıkarma kararıyla çelişiyor. |

### 3. Analiz dışı temizlik (`backlog/ideas.md`, ertelendi)

- **`03_segment_quality`:** boş kasa `"(bos)"` diye yayımlanıyor; İngilizce `"(empty)"` ya da null olmalı. Metriği
  değiştirir; referans kabulü gerekir.
- **Not metinleri:** değerlerin içindeki düzyazı eski anahtar adlarını anıyor (`01_engine_rule`, `03_brand_ablation`,
  `02_missingness`).
- **Uzun tireler:** raporlarda 193 uzun tire (—). Biri hariç hepsi derleyicilerde; biri `07_model_comparison`
  etiketinde, o da referans kabulü ister.

### 4. Repo dışı ve teslim

- **Site senkronu:** drift sayfası artık olmayan `methodology.backtest`'i okuyor. Site bir sonraki senkronda
  `methodology.forward_test`'e geçmeli. `sadik-portfolio` bu repodan yalnız okunur.
- **Push:** `simple-notebooks` dalındaki 39c420f ve 99fe61e henüz gönderilmedi.

## Önerilen sıra

1. **Aralık planının repo dışı adımları:** API, canlıya alma, site.
2. **Rapor düzeltme turu:** `backlog/ideas.md`'deki 2026-11-23 satırları.
3. **Yeniden konan ilanlar:** istenirse plan olarak açılır.
4. **Temizlik işleri:** tek seferde yapılır, ardından referans kabulü.
5. **Site senkronu ve push.**

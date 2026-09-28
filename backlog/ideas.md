# Plan dışı fikirler · Ideas outside the plan

Bir analiz sürerken akla gelen ama kilitli planda olmayan fikir buraya yazılır, o turda **uygulanmaz**.
Değerli görülürse ayrı bir plan olur (`python tools/new_plan.py new ...`, bkz. `plans/README.md`).

An idea that comes up during an analysis but is not in the locked plan is written here and **not done** in that
turn. If it is worth it, it becomes its own plan (`python tools/new_plan.py new ...`, see `plans/README.md`).

Biçim · Format: `- YYYY-AA-GG · <plan id ya da -> · <fikir> · <neden ilginç>`

<!-- fikirler aşağıya · ideas below -->
- 2026-09-25 · - · 03_segment_quality'nin G segmenti kasa sayımında boş kasa `"(bos)"` diye yayımlanıyor (Türkçe veri etiketi); İngilizce `"(empty)"` ya da null olsun · ad değişikliği saf kalsın diye değer değiştirilmedi, metriği değiştirir
- 2026-09-25 · - · Değerlerin içindeki düzyazı eski anahtar adlarını anıyor (01_engine_rule `definition`: "ciftler = [alt, ust, …]", 03_brand_ablation `note`: "dogrulama: …", 02_missingness `note`: "belirtilmemis'te") · saf ad değişikliği metin değerini değiştirmediği için eski adlar metinde kaldı
- 2026-09-25 · - · 09_drift PSI eşikleri (0.10 / 0.25) yalnız `domain.drift.note` düzyazısında; derleyiciler bunları regex'le okuyor (`build_technical_report.py` ve `report_common.py`, `re.search(r"PSI<…>…", …["note"])`). `psi_safe` / `psi_retrain` gibi yapılandırılmış alanlara taşınsın · metin değişirse rapor sessizce bozulur; metriği değiştirir, saf ad değişikliği değil
- 2026-09-25 · - · 09_drift `overlap_note` sütun adlarını Türkçe ASCII yazıyor ("cift, ortak ilan %, KS_ayrik…"); satırlar anahtarlı değil konumlu · anahtarlı satırlar ya da İngilizce sütun listesi kodla metin arasındaki bağı açık yapar; metriği değiştirir
- 2026-09-26 · - · Raporlardaki em dash'ler (—) değiştirilecek: 193 adet (teknik 66/68, iş 17/17, SHAP 12/13 TR/EN); biri hariç hepsi builders/ üreteçlerinde, biri 07_model_comparison etiketi ("(model) medyanı — tüm yıllar") · kullanıcı "sonra" dedi; yalnız dört üreteci koşmak yeter, o tek etiket için 07_model_comparison + referans kabulü
- 2026-09-28 · - · Hacim ile tazeliği ayır: eşit n'de kayan ve genişleyen pencere backtest'i ve öğrenme eğrisi · §9'un eşli karşılaştırması birikimin etkisini ölçüyor ama veri miktarı ile yakın dönemin katkısını ayırmıyor (sadeleştirme listesi, 4. bölüm)
- 2026-09-28 · - · Endeksle düzeltilmiş hedef: fiyatı aynı model+yıl endeksiyle bugüne çek, modelle, tahmini geri şişir · model zamanı görmüyor; dönem kayması hedefte düzeltilebilir (sadeleştirme listesi, 4. bölüm)
- 2026-09-28 · - · SHAP etkileşimleri / ALE / etkileşim payı taraması · etkileşimin toplam payını 2026-09-28'deki stump deneyi ölçtü (archive/experiments/lgb-depth1-2026-09-28, log R² 0,9639'a karşı 0,9699); hangi çiftlerden geldiği ölçülmedi (sadeleştirme listesi, 4. bölüm)
- 2026-09-28 · - · Conformalized quantile regression (CQR) · Mondrian bant başına tek q veriyor; CQR aralığı ilan ilan ayarlar (sadeleştirme listesi, 4. bölüm)
- 2026-09-28 · - · φK ya da tek değişkenli OOF R² tablosu · Cramér's V'nin yerine kategorik ve sayısal ilişkiyi tek ölçüde verir (sadeleştirme listesi, 4. bölüm)

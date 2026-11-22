> **Durum (2026-11-22): tamamlandı.** Bütün adımlar yapıldı; 08_forward_test eski kurulumla birebir aynı (n 8.889, MAPE 6,78 [6,42, 7,17]); referans kabul edildi; verify PASS. Kullanıcı gözden geçirince `archive/`'e taşınır.

# Kaymayı çıkar, tek ileri test kalsın (2026-11-22)

## Context
Kullanıcı zaman ve kaymayı analizden çıkarmak istedi ("çok karışıyor"). Araştırma şunu gösterdi:
- **Taramalar stok dökümü.** Her tarama o gün yayındaki ~11 bin ilanın dökümü. Aralar düzensiz: 9, 53 ve 98 gün. Yeni
  ilan payı %26, %63 ve %77.
- **`listing_date` yayın tarihi değil.** Birden çok taramada görülen ilanların %56'sında değişmiş; son güncelleme günü.
  Tek güvenilir zaman ekseni dört tarama.
- **Kayma ölçüleri ayrıştıramıyor.** PSI, KS, EMD ve yoğunluk eğrileri piyasa değişimini, stokta kalma etkisini ve
  bileşim farkını birbirinden ayıramıyor. Karışıklığın kaynağı bu.
- **Backtest iyi tanımlı.** "Eskiden öğren, sonra gelen yeni ilanı fiyatla" sorusunu cevaplıyor; modelin kullanımına
  en yakın ölçüm o.

Kullanıcı kararı: **kayma tamamen çıksın, tek bir ileri test kalsın.**

## Ne kalıyor, ne gidiyor
**Kalan ileri test (`analysis/08_forward_test.py`, 09_backtest'in yerine):**
- **Eğitim:** son taramadan önce görülen bütün ilanlar, ilan başına son fiyat (`latest_per_ad`).
- **Test:** son taramada ilk kez görülen ilanlar (`test_rows`).
- **Ağaç sayısı:** eğitim kümesinde iç 5-fold CV'den (`fit_forward`).
- **Ölçülenler:**
  - MAPE ve modele göre yeniden örneklemeli %95 aralık (`cluster_boot` + `interval`);
  - servis tarifinin ileri kapsaması (`forward_coverage`): genel, bant başına, emsalsiz (model+yıl eğitimde yok).
- **Kod:** bu fonksiyonlar 09_backtest'ten aynen taşınır; yeni hesap yazılmıyor.
- **Değerler:** eski kayıttaki "→03-21 → 06-27" satırıyla aynı çıkmalı: n 8.889, MAPE 6,78, kapsama 87,4 / 87,99,
  emsalsiz 68,35 / 65,32.

**Gidenler:**
- `09_drift` tamamen; `09_backtest`'in tek, biriken, eşli, ufuk, insample ve tarama başına kolları.
- Figür 13, 14, 15.
- 08_large_errors'ta `by_snapshot` ve `live_vs_gone`; 08_residuals'ta dönem cümlesi (`snap_median`).
- Teknik rapor §9 ("Zaman").
- İş notundaki üç madde: yeniden eğitim, PSI, rejim.
- `report_common`'da PSI eşiğini nottan regex'le okuyan kod.

## Verdiğim kararlar
- **Ad ve yer:** 08_forward_test §8'i besliyor, numarası bölümle tutsun diye 08. "Serbest metin" §10 → §9 olur.
  `10_free_text` adı kalır, çünkü kilitli `plans/10-text-contribution` o adı anıyor; README'de bu istisna yazılır.
- **Kilitli `plans/09-forward-coverage`:**
  - H1 ve H2 "her kurulumda" diye yazılmış ve dokuz kurulumda sonuçlanmış (H1 doğrulandı, H2 çürütüldü). Tek kurulum bu
    iddiaları taşımaz.
  - Plan geri çekilir: dizin yerel `archive/plans-withdrawn/`'a taşınır, içeriği düzenlenmez.
  - `docs/decisions` (TR/EN) sonuçları ve onları taşıyan son commit'i kaydeder.
  - Yeni plan açılmıyor: tek testin sonucu zaten biliniyor, şimdi "ön kayıt" yapmak sonradan olur. Karar notunda açıkça
    yazılır.
- **Rapor:**
  - §8'in "İleri kapsama" paragrafı tek teste iner: yeni ilanlarda MAPE ve aralığı (manşet OOF'un yanında), bant
    başına kapsama tablosu, emsalsiz satırı.
  - "Ölçüt sonuçtan önce yazıldı" başlığı kalkar.
  - "(§9)" atıfları ve §8'deki "Zaman ve hayatta kalma" maddesi kalkar.
  - Giriş "zamansal testler" yerine "yeni ilanlarda ileri test" der.
  - §7'ye bir sınır cümlesi: "Model zamanı görmez; manşet hata taramaları karıştıran 5-fold OOF'tan, yeni ilanlardaki
    hata §8'de." Sayılar derleyiciden gelir.
- **İş notu:**
  - Aralık paragrafı tek testin sayılarıyla kalır: en ucuz bantta tek pay → banda göre pay, emsalsizde iki kol.
  - Üç zaman maddesi kalkar.
  - "Bu model neyi vermez"e **Zamanı** maddesi eklenir: dönem farkını görmediği, yeni ilanlardaki hatası ve manşet
    hatası (ikisi de metrikten).
- **Site:** `sadik-portfolio`'ya dokunulmuyor.
  - `site_data.json`'dan `domain.drift` ve `methodology.backtest` düşer, `methodology.forward_test` gelir.
  - Site bu sayfaları bir sonraki senkronda güncellemeli.
  - Canlı API drift sekmesi analizden bağımsız.
- **Branch ve commit:** `simple-notebooks`. Önce bekleyen notebook işi kendi commit'ine girer, sonra bu değişiklik ayrı
  commit olur. Atıf satırı yok, push yok.

## Değişiklikler (kritik dosyalar)
0. **Önce karar kaydı (başka hiçbir şeye dokunmadan):**
   - **`docs/decisions.md` / `.en.md`:** "2026-11-22 · Kayma çıktı, tek ileri test kaldı" girdisi. İçeriği:
     - araştırma bulguları: taramalar stok dökümü; aralar 9/53/98 gün; yeni ilan payı %26/%63/%77; `listing_date` son
       güncelleme günü (%56 değişmiş); ilan yaşına göre artık düz;
     - kayma ölçülerinin neden ayrıştıramadığı;
     - neden tek ileri testin kaldığı;
     - neyin çıktığı;
     - `plans/09-forward-coverage`'ın geri çekilişi ve sonuçları (H1 doğrulandı, H2 çürütüldü);
     - yeni plan açılmama gerekçesi.
   - **Plan dosyası:** repo köküne `forward_test_plan.md` olarak kaydedilir; her adımda durum satırı güncellenir, iş
     bitince `archive/`'e taşınır.
   - **Hafıza:** bir kayıt eklenir; yeniden önerilmesin diye: kayma çıktı, `listing_date` zaman ekseni değil, tek ileri
     test kaldı.
1. **`analysis/08_forward_test.py`:**
   - `tools/new_analysis.py` iskeletinden açılır, fonksiyonlar 09_backtest'ten taşınır;
   - metrik `methodology.forward_test`;
   - kart `analysis/cards/08_forward_test.json`; ayrılma: temporal, eğitim testten önce;
   - test `tests/analysis/test_08_forward_test.py`: `test_09_backtest.py`'deki geçerli invariantlar taşınır (ör.
     `test_forward_coverage_calibrates_on_training_only`, kapsama aralıkları);
   - `run_all` ORDER'a girer; `records.HEAVY`'de `09_backtest` yerine geçer.
2. **Silinenler:**
   - `analysis/09_drift.py`, `analysis/09_backtest.py`, kartları, `tests/analysis/test_09_*.py`, `metrics/09_*.json`;
   - `analysis/lib/conformal.py` docstring'inde 08_forward_test anılır.
3. **`analysis/08_large_errors.py`, `analysis/08_residuals.py`:** zaman parçaları çıkar; kartlar ve
   `tests/analysis/test_08_large_errors.py` güncellenir.
4. **Kartlar:** `temporal` notları 09'u anmaz; "KFold dönemleri karıştırır; model zamanı görmez; yeni ilanlardaki hata
   08_forward_test'te". Örnek: `analysis/cards/07_model_comparison.json`.
5. **Derleyiciler:**
   - `builders/build_technical_report.py`: `section_time` ve SECTIONS satırı kalkar; §8 ileri paragrafı yeniden yazılır;
     §7 sınır cümlesi eklenir.
   - `builders/build_business_report.py`: aralık paragrafı ve maddeler.
   - `builders/report_lib/report_common.py`: REQUIRED, `derive` (`psi_*`, `bt_*` yerine `ft`), figür 13–15 ve
     TECHNICAL_FIGS.
   - `builders/build_site_data.py`: SITE_KEYS.
6. **İç araç:** `internal_tool/section_map.json` (9. bölüm ve figür 13–15 kalkar, 08_forward_test §8'e girer);
   `python internal_tool/catalog.py --stamp`.
7. **Notebook'lar:** `3_errors_and_time` → `3_errors_and_text`; üreteçle yeniden üretilir;
   `python tools/run_notebooks.py`.
8. **Belgeler:**
   - README (TR/EN): betik listesi, süreler, numara istisnası;
   - `docs/analysis-scripts` (TR/EN);
   - `docs/reproducibility` (TR/EN);
   - `docs/decisions` (TR/EN);
   - `restructure_plan.md`;
   - `backlog/ideas.md`: 09_drift'e özgü iki fikir silinir, zaman fikirleri gelecek fikri olarak kalır.
9. **Koşum:** `08_large_errors`, `08_residuals`, `08_forward_test` (~1–2 dk), sonra dört derleyici ve notebook'lar.
   OOF değişmiyor; tam zincir gerekmiyor.
10. **Referans:**
    - `snapshot_metrics --diff` yalnız beklenen farkları göstermeli: 09 dosyaları gider, 08_forward_test gelir,
      `by_snapshot` ve `live_vs_gone` gider, `residual_scatter.note` değişir;
    - sonra `--accept "kayma çıktı, tek ileri test kaldı (kullanıcı kararı 2026-11-22)"`; gerekirse `--drop-exemption`;
    - **plan onayı bu kabulü kapsar**; fark beklenenden başkaysa durup sorarım.

## Doğrulama
- 08_forward_test'in değerleri eski kayıttaki "→03-21 → 06-27" satırıyla birebir aynı: n, MAPE, aralık, ağaç,
  kapsama.
- `snapshot_metrics --diff`: yalnız beklenen farklar.
- Raporlar: `grep -n "PSI\|kayma\|drift\|§10\|hayatta kalma" reports/*.md` boş.
- Canlı kodda `git grep -n "09_drift\|09_backtest\|period_shift\|live_vs_gone"` yalnız `docs/decisions`, rename
  haritaları (tarihçe) ve backlog'da.
- `python tools/run_notebooks.py`: hafif betiklerin hepsi "Kayıtla aynı".
- `python tools/verify.py` → PASS (sonuç satırı aktarılır); iç araç testleri `.venv-tool` ile geçer.

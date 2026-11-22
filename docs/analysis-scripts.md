# Analiz betikleri nasıl okunur

> 🇬🇧 English: [analysis-scripts.en.md](analysis-scripts.en.md) · ← [README](../README.md)

Her betik bir soruyu yanıtlar ve **üç katmanlıdır**: (1) saf analiz fonksiyonları — dosya okumaz/yazmaz,
yalnız değer döndürür; (2) `to_metrics` — yalnız adlandırma ve yuvarlama; (3) çalıştırma hücreleri —
`[4] Yükle · [5] Hesapla · [6] Kaydet`. `save_metrics` her betikte **bir kez**, en sondaki hücrede çağrılır;
bir fonksiyon hata verirse JSON yazılmaz, eski JSON olduğu gibi kalır. Dosya adları İngilizce, her dosyanın
ve her fonksiyonun açıklaması iki dilli (`EN:` / `TR:`).

**VS Code'da hücre hücre:** betikler `# %%` hücreleriyle yazıldı; Python + Jupyter eklentileriyle
**Shift+Enter** hücreyi Interactive Window'da koşar. `[5]`'e kadar koşup sonuca bakılabilir; `[6]`
koşulmazsa JSON değişmez.

**Defterler (`notebooks/`):** sonuçları okumak ve hafif analizleri denemek için dört defter. Sıraları teknik raporun
bölümleriyle aynı:
- [1_data](../notebooks/1_data.ipynb): 1–4. bölümler;
- [2_market_and_model](../notebooks/2_market_and_model.ipynb): 5–7;
- [3_errors_and_text](../notebooks/3_errors_and_text.ipynb): 8–9;
- [4_shap](../notebooks/4_shap.ipynb): SHAP raporu.

Her betik için defter kartını, sonuçlarını tablo olarak ve raporun figürlerini gösterir. Sonuç iki yoldan gelir:
- **Hafif betikler defterde koşar.** `script = run("08_residuals")` betiğin [1]–[5] hücrelerini çalıştırır: veriyi ve
  kayıtlı model çıktılarını (OOF tahminleri, `oof_shap.npz`) okur, hesabı yapar. Kaydet hücresi [6]
  `if __name__ == "__main__":` altında olduğu için çalışmaz. Analiz kodu betiğin kendisidir; defterde kopyası yok.
  `check` sonucun `metrics/` kaydıyla aynı olup olmadığını gösterir. `script.res`, yüklenen tablolar ve betiğin
  fonksiyonları sonraki hücrelerde denenebilir.
- **Ağır betikler kayıttan okunur.** Model eğiten ya da çapraz doğrulama koşan sekiz betik
  (`analysis/lib/records.py`'deki `HEAVY`) `metrics/`'ten gelir; `run` onları reddeder.

Kayıtlı metrikler derleyicilerin okuyucusuyla (`builders/report_lib/metrics_view.py`) yüklenir; bu yüzden karışık ya
da bayat bir kayıtta defter de raporlar gibi durur. Defter hiçbir dosya yazmaz. Bir sonucu kalıcı olarak değiştirmek
için betik değişir ve zincir koşulur. Ardından `python tools/run_notebooks.py` defterleri taze kernel'de baştan sona
koşar ve kaydeder (~30 sn). Yeni bir betik iskeleti kaydet hücresini korumalı açar; betik ağırsa `HEAVY`'ye eklenir.
`tests/repo/test_notebooks.py` şu durumlarda düşer:
- HEAVY dışında kaydet hücresi korumasız bir betik;
- defterde dosya yazan ya da model uyduran bir hücre;
- arşiv yolu, kırık bağlantı ya da hata çıktısı.

**Ortak kod:** `analysis/lib/common.py` dışarıya iki şey açar — `load_clean()` (TR plakalı, fiyatı > 0,
`ad_id` başına son kayıt, `ORDER BY ad_id`, türetilen öznitelikler) ve `save_metrics()` (JSON + `_meta`:
betik adı, koşum zamanı, verideki son tarama tarihi, DB parmak izi, `run_id`). `analysis/lib/cv.py` 5-fold
OOF'un tek kopyası (aynı fold'lar, fold içi TF-IDF+SVD, aynı LightGBM ayarı) ve OOF artefaktının
okuyucusu/yazıcısı; `analysis/lib/segment_rule.py` segment kuralının tek kaynağı; `analysis/lib/text_flags.py`
ilan metninden dört desen bayrağı (LLM yok).

**Tutarlılık kapısı:** derleyiciler bütün metrik dosyalarının aynı DB parmak izini, modelden türeyenlerin
aynı `run_id`'yi taşıdığını sınar; karışık ya da bayat çıktıyla rapor yazılmaz, hangi betiğin yeniden
koşulacağı söylenir. 07'nin artefaktına bağlı bir betik, artefakt yoksa ya da başka DB'den üretilmişse durur.

## Yeni analiz nasıl eklenir

Kapı yeni bir analizi kartsız ve testsiz "bitti" saymaz (`tests/analysis/test_coverage.py`). Sıra:

1. **Plan (yeni bir soru için):** `python tools/new_plan.py new <id> "soru" "question"`. Doldur, kullanıcıya göster,
   yalnız açık onaydan sonra `python tools/new_plan.py lock <id> --by "<onaylayan>"`. Kilitli planda yalnız
   hipotez durumu ve kanıt değişir (`plans/README.md`).
2. **İskelet:** `python tools/new_analysis.py NN_ad "soru" "question"`. Üç katmanlı betiği, bilerek düşen taslak
   testi ve durumu `todo` kartı yazar.
3. **`ORDER`:** betiği `analysis/run_all.py`'deki listeye numara sırasıyla ekle.
4. **Analiz:** betiği yaz ve koş (`python analysis/NN_ad.py`). `save_metrics` metriğe kaynak izini (betiğin ve
   yüklediği `analysis/lib` dosyalarının sha256'sı) yazar.
5. **Test:** taslağı en az bir gerçek invariantla değiştir. Örnekler: aralık, toplam, başka bir betikle tutması
   gereken sayım, raporun bir iddiasının yanındaki sayılardan çıkması. Yalnız metrik JSON'unu okur.
6. **Kart:** `analysis/cards/NN_ad.json`:
   - soru, veri, yöntem, bölme;
   - dört sızıntı tipi (`present` / `unknown` için not zorunlu);
   - kanıt: toplanan test kimlikleri + metrikteki rapor anahtarları;
   - `status: complete`.
7. **Referans:** `python tools/snapshot_metrics.py --diff`, sonra kullanıcı onayıyla `--accept "<gerekçe>"`.
8. **Bitti:** `python tools/verify.py` → PASS. Ne eksik kaldığını `python tools/analysis_coverage.py` gösterir.

- **Bayat metrik:** bir betik ya da import ettiği `analysis/lib` dosyası değişince metrik bayat sayılır; kapı
  hangi betiğin yeniden koşulacağını söyler (`tests/metrics/test_provenance.py`). `lib/common.py`'ye dokunmak
  bütün betikleri bayat yapar (tam koşum ~16 dk).
- **Adlar:** kod düzeyinde her şey İngilizce snake_case: metrik anahtarları, kodun karşılaştırdığı değerler
  (`"series_map"`, `"model_year"`, `"economy"`…), değişken ve fonksiyon adları. Türkçe yalnız raporda görünen
  metinde kalır: `L("tr", "en")` yazıları ve etiket sözlüklerinin `(tr, en)` değerleri (`TIER_LABEL`,
  `BAND_LABEL`). Kod görünen bir terimle indekslemez; terimi `HED_TERM_ID` gibi bir sözlükle kimliğe çevirir ve
  bilinmeyen terimde durur. Eski→yeni anahtar eşlemesi `docs/metric-key-renames.json`'da, site için yol ve değer
  eşlemesi `docs/site-data-renames.json`'da. `tests/metrics/test_metric_key_names.py` metriklerde emekli bir
  anahtar ya da değer, kodda ya da belgede eski adlı bir noktalı yol görürse kırmızı olur; hiçbir gruba ait
  olmayan yeni bir betiğin metriği de sınanır.
- **Eski betikler:** kart ve testten önce yazılan 18 betiğin hepsi artık kartlı ve testli;
  `tests/analysis/legacy.json` boş. Liste yalnız kısalabilir, yeni bir ad eklenemez.
- **Ağır metodoloji testleri** (hızlı kapıda değil): `python -m pytest -m full tests/analysis`. Denetimler:
  fold'lar arası `ad_id` örtüşmesi yok, karıştırılmış hedefte R² ≈ 0, bit-birebir yeniden koşum.

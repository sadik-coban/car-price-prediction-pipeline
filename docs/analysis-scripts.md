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
- **Eski betikler:** kart ve testten önce yazılan 18 betik `tests/analysis/legacy.json`'da, kaynaklarının
  sha256'sıyla. Değişen eski betik kart ve test ister; liste yalnız kısalır.
- **Ağır metodoloji testleri** (hızlı kapıda değil): `python -m pytest -m full tests/analysis`. Denetimler:
  fold'lar arası `ad_id` örtüşmesi yok, karıştırılmış hedefte R² ≈ 0, bit-birebir yeniden koşum.

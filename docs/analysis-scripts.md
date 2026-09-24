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

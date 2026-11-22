# Kök dizini sadeleştirme ve okunur analiz defterleri: plan

> **Durum (2026-11-22): kararlar verildi, hiçbir şey uygulanmadı.** Bu dosya yalnız plandır. Uygulama başlayınca
> adımların durumu buraya yazılır; iş bitince dosya `db_plan.md` gibi `archive/`'e taşınır.
>
> **Güncelleme (2026-11-22, kullanıcı kararı):**
> - `simple-notebooks` denemesi (01'in `.py` ↔ `.ipynb` ikizi, `sync_notebooks`) geri alındı.
> - Yerine kökte `notebooks/` altında dört defter kuruldu: `1_data`, `2_market_and_model`, `3_errors_and_text`,
>   `4_shap`.
> - Hafif betikler defterde kaydetmeden koşar (`records.run`; kaydet hücresi `if __name__ == "__main__":` altında)
>   ve sonuç kayıtla karşılaştırılır. Ağır sekiz betik (`records.HEAVY`) `metrics/`'ten okunur. Defter tarafının
>   kodu `analysis/lib/records.py`'de. Çıktıları `tools/run_notebooks.py` yeniler.
> - Kullanıcı bu defterleri düzenleyecek; yapı oradan organize edilecek. Aşağıdaki "Defterler" bölümü (24 eşli
>   defter) bu yüzden askıda.

## Neden

- **Kök kalabalık.** Kökte git'in izlediği 20, git dışı 11 öğe var (`.git` hariç). Neyin nerede olduğu ilk bakışta
  anlaşılmıyor.
- **Analizler okunamıyor.** Bugün bir analize bakmanın yolu `.py` betiği ile `metrics/*.json`'dur. Defter ikizi
  denemesi var ama okunmuyor:
  - deneme `simple-notebooks` dalında, commit'lenmemiş: `analysis/01_dedup_leakage.ipynb`, `analysis/jupytext.toml`,
    `tools/sync_notebooks.py`, `tests/repo/test_notebook_pairs.py`;
  - yalnız 01'de var;
  - defterin büyük kısmı fonksiyon tanımı; tek çıktısı hesap hücresinin bastığı iki satırlık sözlük;
  - diğer 23 betik de hesap hücresinde 1–3 satır basıyor.

## Kararlar (2026-11-22, kullanıcı)

| # | konu | karar |
|---|---|---|
| 1 | defterlerin yeri | kökte `notebooks/`. Kaynak yine betik (`.py`); jupytext ikisini eşler |
| 2 | defter biçimi | okuma biçimi: soru → tablo ya da figür → karar, adım adım. Fonksiyonlar en üstte kapalı hücrede |
| 3 | kök | ağır gruplama: zincir `pipeline/`, altyapı `dev/` altında toplanır |
| 4 | defter dili | açıklama metinleri ve hücre başlıkları yalnız Türkçe; raporlar TR + EN kalır |

## Hedef yapı

```
cardatasys/
├─ README.md · README.en.md     "nereye bakmalı": notebooks/ · reports/ · docs/
├─ notebooks/                   analizler, son temiz koşumun çıktılarıyla (.ipynb) — okumak için buraya
│  └─ shap/
├─ reports/                     üretilmiş raporlar + figures/ (değişmez)
├─ docs/                        belgeler, TR + EN (değişmez)
├─ pipeline/                    veriden rapora zincir
│  ├─ scraper/                  ilan toplama (izlenen iki ayar dosyası + yerel kazıma kodu)
│  ├─ db/                       DuckDB kurulumu, gözlenen değerler, gold, S3 yayını
│  ├─ analysis/                 soru başına betik (kaynak), lib/, cards/, shap/, run_all.py
│  ├─ builders/                 rapor derleyicileri (yalnız metrics/ okur)
│  └─ metrics/                  betik başına JSON (üretilir)
├─ dev/                         altyapı
│  ├─ tests/                    kapı testleri, baselines/
│  ├─ tools/                    verify, sync_notebooks, snapshot_metrics, analysis_coverage, new_analysis, new_plan, observed_values
│  ├─ plans/                    kilitli ön kayıt planları
│  ├─ backlog/                  plan dışı fikirler
│  └─ internal_tool/            yerel Streamlit aracı
├─ .gitignore · pytest.ini · requirements-dev.txt · requirements-pipeline.txt · requirements-pipeline.lock.txt
└─ git dışı: data/ · archive/ · .venv/ · .venv-tool/ · .claude/ · .env · CLAUDE.md · notes.txt
```

**Kökte izlenen öğeler:** 20'den 12'ye iner (bu plan dosyası iş bitene kadar 13.). `requirements*.txt` ve
`pytest.ini` kökte kalır: kurulum komutları ve pytest kökten çalışır.

## Taşıma tablosu

| şimdi | sonra | not |
|---|---|---|
| `scraper/` | `pipeline/scraper/` | klasör olarak yeniden adlandırılır. İçindeki izlenmeyen yerel kazıma kodu (`main.py`, `getlistofcars.py`, `getdetails.py`) da onunla gider; tek tek kopyalanmaz |
| `db/` | `pipeline/db/` | |
| `analysis/` | `pipeline/analysis/` | `analysis/jupytext.toml` kalkar; yerine kökte tek `jupytext.toml` gelir |
| `builders/` | `pipeline/builders/` | |
| `metrics/` | `pipeline/metrics/` | |
| `tests/` | `dev/tests/` | `pytest.ini`: `testpaths = dev/tests` |
| `tools/` | `dev/tools/` | |
| `plans/` | `dev/plans/` | |
| `backlog/` | `dev/backlog/` | |
| `internal_tool/` | `dev/internal_tool/` | kendi ortamı `/.venv-tool/` kökte kalır, yalnız yollar güncellenir |
| — | `notebooks/` | yeni. `analysis/01_dedup_leakage.ipynb` buraya taşınır |
| `catboost_info/` (git dışı) | `archive/obsolete/` | CatBoost'un varsayılan eğitim klasörü. Yeniden oluşmasın diye `train_dir` `data/` altına alınır |
| `db_plan.md` | `archive/` | bitmiş plan ("üç parça da yapıldı") |
| `harness-research/` (git dışı) | `archive/` | araştırma notları |
| `.coverage` (git dışı) | `data/` altına | coverage ayarıyla (`data_file`) |
| `notes.txt` (git dışı) | yerinde | kişisel not; kullanıcı karar verir |

**Taşıma kuralları:**
- hiçbir şey silinmez;
- taşımalar yeniden adlandırmadır;
- arşive giden her öğe için `archive/README.md`'ye bir satır eklenir.

## Defterler

### Eşleme

- Kökte tek bir `jupytext.toml`: `formats = "notebooks///ipynb,pipeline/analysis///py:percent"`.
  - `notebooks/shap/` ile `pipeline/analysis/shap/` gibi alt klasörler birbirini eşler.
  - Filtreler bugünkü gibi kalır (`notebook_metadata_filter = "-all"`). Kapalı hücre bilgisi
    (`jupyter.source_hidden`) taşınsın diye `cell_metadata_filter`'a `jupyter` eklenir.
- Defteri yalnız soru betikleri alır: `pipeline/analysis/*.py` (20) ve `shap/*.py` (4). `lib/` ve `run_all.py`
  defter almaz.
- `sync_notebooks.py`:
  - yeni yolları kullanır;
  - kernel'i bugünkü gibi taze açar; çalışma klasörünü kurulum hücresinin beklediği yere ayarlar;
  - `--check` ve `test_notebook_pairs.py` aynen kalır.
- `run_all.py`: ikizi olan betik bugünkü gibi defter üzerinden koşar. Sonunda her betiğin ikizi olur.
- Klasörler arası eşleme pilotta sınanır: biri Jupyter'de kaydedilince öteki doğru yerde güncellenmeli.

### Okuma biçimi: hücre sırası

1. **[md] Giriş:** başlık, soru, veri ve bu defterin beslediği kararlar (rapordaki bölüm).
2. **[kod, kapalı] Fonksiyonlar:** bugünkü `[2]` katmanı (saf, dosya okumaz ve yazmaz) ve `[3] to_metrics`. Tek
   hücrededir, varsayılan olarak kapalıdır, tıklayınca açılır.
3. **[kod] Kurulum ve yükleme:** import'lar, lib yolu ve dosya okuyan tek hücre (bugünkü `[4]`).
4. **Her alt soru için üç hücre:**
   - [md] `## n. <soru>`;
   - [kod] tek fonksiyon çağrısı; son ifadesi bir tablo (DataFrame) ya da figür;
   - [md] `Okuma:` / `Karar:`.
5. **[kod] Kaydet:** JSON'u yazan tek hücre (bugünkü `[6]`).

**Kurallar:**
- **Md hücrelerine sayı yazılmaz.** Sayılar yalnız çıktılarda görünür; altın kural defter için de geçerli.
- **Hesap sonucu ve metrikler değişmez.** Sonuçlar bugünkü gibi `res` sözlüğünde toplanır; `to_metrics` ve kaydet
  hücresi aynen kalır. Böylece metrik değerleri değişmez, yalnız `_meta` değişir.
- **Tablo önce gelir.** Figür yalnız raporda o adım için bir figür varsa çizilir; `builders/report_lib`'deki aynı
  çizim fonksiyonuyla çizilir, ikinci bir figür kodu yazılmaz. Ayrıntısı pilotta netleşir.
- **Tablo başlıkları:** Türkçe etiket gerekirse `lib/labels.py` kullanılır; yeni bir etiket haritası açılmaz.

### Dil

- **Yalnız Türkçe:** defterdeki md hücreleri ve hücre başlıkları (`# %% [4] Yükle — …` gibi).
- **Değişmeyenler:**
  - değişken, fonksiyon ve metrik anahtarları İngilizce;
  - docstring'ler bugünkü gibi EN + TR (`test_bilingual_docstrings` aynen kalır);
  - loglar.
- CLAUDE.md'nin "Names" bölümüne kural olarak yazılır: defter açıklaması Türkçe.

### lib yolu

- Jupyter kernel'i defterin klasöründe (`notebooks/`) açar; lib `pipeline/analysis/lib/`'dedir.
- Kurulum hücresi üç durumda da çalışmalı:
  - `python pipeline/analysis/<ad>.py`;
  - Jupyter ya da VS Code'da `notebooks/<ad>.ipynb`;
  - `sync_notebooks`'un taze kernel'i.
- Yöntem: bugün `shap/` betiklerindeki `sys.path.insert(0, … globals().get("__file__", Path.cwd() / "_") …)` kalıbı
  genelleştirilir ve her betikte aynı satır olur.
- Alternatif: projeyi paket olarak kurmak (`pip install -e .`). Paket kurulumu yalnız kullanıcı onayıyla yapılır.
- lib değişince açık kernel eski kodu tutar. VS Code'da `jupyter.runStartupCommands` ile `%load_ext autoreload` ve
  `%autoreload 2` açılır. Bu satırlar betiğe değil, editör ayarına konur.

### Temiz koşum koruması (önerildi, onay bekliyor)

- **Sorun:** `save_metrics`, `_meta.source`'a diskteki dosyaların sha'sını yazıyor (`lib/common.py`,
  `_source_hashes`). Etkileşimli kernel'de bellekteki kod diskteki koddan farklı olabilir:
  - lib yeniden yüklenmemiştir;
  - bir hücre değişmiş ama kaydedilmemiştir;
  - hücreler sırasız koşulmuştur.

  Bu durumda kaydet hücresi taze görünen ama başka koddan gelen bir metrik yazar. Kapı bunu yakalamaz.
  `notebooks/` kökte olunca etkileşimli koşu artacak, bu yüzden koruma bu işle birlikte ele alınır.
- **Öneri:**
  - `sync_notebooks` kernel'i bir ortam değişkeniyle başlatır (ör. `CDS_CLEAN_RUN=1`);
  - `save_metrics` IPython içinde bu değişkeni görmezse yazmaz ve şunu basar: "taslak — `python dev/tools/sync_notebooks.py <ad>` koş";
  - düz betik koşusu etkilenmez.

## Ağır gruplama: neler kırılır (envanter, 2026-11-22)

| bağ | bugün nerede | ne yapılır |
|---|---|---|
| kök bulma (`Path(__file__).parents[N]`, `.parent.parent`) | 67 izlenen `.py`, çoğu `tests/`'te; ayrıca `internal_tool/`, `tools/`, `analysis/`, `db/`, `builders/` | taşınan her dosyada bir seviye artar. Her modülün `ROOT`'unun depo köküne çözüldüğünü sınayan bir test eklenir |
| Python'daki yol dizgileri (`"analysis/"`, `"metrics"` …) | ~190 | güncellenir. Kabul: canlı kodda eski kök yollarına `git grep` sıfır |
| md belgeleri | ~220 kod span + 6 göreli bağlantı | güncellenir. Bağlantıları `test_relative_links` denetler; kod span'ları grep ile |
| `pytest.ini` | `testpaths = tests` | `dev/tests` |
| `.gitignore` | `/scraper/*` ve iki istisnası; Jupyter yorumu (`analysis/` altında ikizler) | `/pipeline/scraper/*` ve istisnaları; yorum `notebooks/` olur |
| yerel hook'lar (`.claude/hooks/`, git dışı) | `guard_paths`: `metrics/*`, `reports/*`, `data/*`, `analysis/frozen/*`, `tests/baselines/*`. `guard_plan`: `plans/*/analysis_plan.json`. `stop_gate`: `analysis`, `builders`, `tools`, `metrics`, `reports`, `plans` ve `tools/verify.py` | yeni yollara geçer. `analysis/frozen/` bugün yok; deseni kalkar ya da gerekçesi yazılır. `tests/repo/test_hooks.py` sınar |
| `.claude/agents/methodology-reviewer.md`, `.claude/skills/statistical-analysis/SKILL.md` | eski yolları anıyor | güncellenir |
| README "Nasıl koşulur" ve "İçerik", CLAUDE.md komutları, `docs/analysis-scripts.{md,en.md}` | `python analysis/run_all.py`, `python tools/verify.py` … | yeni komutlara geçer; "nereye bakmalı" bölümü eklenir |
| metriklerin kaynak izi (`_meta.source`) | depoya göreli yollar (`analysis/…`) | yollar değişince bütün metrikler bayat görünür. Taşımadan sonra zincir bir kez yeniden koşulur (`verify --full`); değerler aynı kalmalı |

## Adımlar

0. **Hazırlık:**
   - `simple-notebooks` dalındaki commit'lenmemiş denemenin akıbeti kullanıcıya sorulur (aşağıda açık kararlar);
   - taban kaydedilir: `verify` ve `verify --full` PASS, `snapshot_metrics.py --diff` boş.
1. **Artıklar arşive:**
   - `catboost_info/`, `db_plan.md`, `harness-research/` arşive taşınır;
   - `.coverage` `data/` altına alınır;
   - `train_dir` ayarı yapılır.

   Bu adımda yol değişikliği yok.
2. **Kökü grupla:** taşıma tablosu ve envanterdeki her bağ güncellenir. Yalnız taşıma ve yol değişikliği yapılır;
   davranış değişmez. Bu adımın sonunda zincir yeniden koşulur.
3. **Defter altyapısı:**
   - kökte `jupytext.toml`;
   - `sync_notebooks` ve `run_all` yeni yollarla;
   - `test_notebook_pairs`;
   - lib yolu kurulum satırı;
   - VS Code ayarı;
   - temiz koşum koruması (onaylanırsa).
4. **Pilot:** iki defter okuma biçimine çevrilir: `01_dedup_leakage` (veri tarafı) ve model tarafından biri (ör.
   `08_large_errors`). Kullanıcı ikisini açıp okur. Biçim onaylanınca kalıp `docs/analysis-scripts.{md,en.md}`'ye ve
   `tools/new_analysis.py` iskeletine yazılır.
5. **Yayılım:** kalan 22 betik aynı kalıba çevrilir. Defter üzerinden koşmak süreyi değiştirmez; zincir bir kez
   koşulur.
6. **Belgeler:**
   - README (TR + EN): "nereye bakmalı", "İçerik", "Nasıl koşulur";
   - CLAUDE.md: komutlar ve defter dili kuralı;
   - `docs/`'taki yollar;
   - `archive/README.md` satırları.

## Kabul (bitti tanımı)

- `python dev/tools/verify.py` ve `--full` 0 ile çıkar.
- `python dev/tools/snapshot_metrics.py --diff`: hiçbir metrik değeri değişmemiş; yalnız `_meta` değişmiş.
- `notebooks/`'ta 24 defter var. Hepsi `sync_notebooks --check` ile hizada: tek temiz koşum, hata çıktısı yok.
- Her defter okuma biçiminde:
  - fonksiyonlar tek kapalı hücrede;
  - her alt sorunun çıktısı bir tablo ya da figür;
  - md hücreleri Türkçe ve el yazısı sayı yok.
- Kök "Hedef yapı"daki gibi.
- Canlı kodda ve belgelerde eski kök yolları (`analysis/`, `tools/` …) geçmiyor; arşiv hariç.
- Hiçbir şey silinmedi: artıklar `archive/`'de ve `archive/README.md`'de satırları var.

## Riskler

- **Taşımadan sonra kapı kırmızı:** kaynak izindeki yollar değiştiği için zincir yeniden koşulana kadar kapı
  kırmızı kalır. Bu beklenen bir durum; 2. adımın sonunda `--full` koşulur.
- **Git'te büyüme:** defter çıktıları (HTML tablolar, PNG figürler) depoyu büyütür. Figür yalnız gerekiyorsa konur.
- **jupytext klasörler arası eşleme:** yanlış yere yazarsa iki ayrı kopya oluşur. Pilotta sınanır; `--check` yakalar.
- **Hook güncellemeleri commit'e girmez:** `.claude/` git dışı. Güncellemeler önce yedeklenir.
- **Temiz koşum koruması olmazsa** etkileşimli oturumdan metrik yazılabilir.

## Açık kararlar (uygulamadan önce)

- ~~`simple-notebooks` dalındaki commit'lenmemiş deneme önce commit'lensin mi, yoksa bu işin dalına mı taşınsın?~~
  Karar verildi (2026-11-22): deneme geri alındı; yerini kayıttan okuyan `notebooks/` aldı (en üstteki güncelleme).
- Temiz koşum koruması kurulsun mu?
- Docstring'ler tek dile indirilsin mi? Bu ayrı bir konu; karar verilene kadar EN + TR kalır.
- `notes.txt` yerinde mi kalsın?

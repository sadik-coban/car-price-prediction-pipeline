# İç araç · Internal tool

Yalnız yerelde (127.0.0.1) koşan Streamlit aracı. Kazınan veri hiçbir yerde barındırılamaz; araç ağa açılmaz.
A Streamlit tool that runs on 127.0.0.1 only. The scraped data must never be hosted; the tool never opens to a network.

## Kurulum · Setup
Aracın kendi ortamı `/.venv-tool/` (git dışı). Pipeline'ın `.venv`'ine ve lock dosyasına dokunmaz; pipeline lock'u kısıt
olarak kullanıldığı için pandas / numpy / duckdb / pyarrow / plotly sürümleri kapıyla aynıdır.

```
python -m venv .venv-tool
.venv-tool\Scripts\python -m pip install -r internal_tool/requirements.txt -c requirements-pipeline.lock.txt -c requirements-dev.txt
```
`internal_tool/requirements.lock.txt`: kurulan tam sürümler (`pip freeze`).

## Koşum · Run
```
.venv-tool\Scripts\python internal_tool\launch.py            # tarayıcı açılır · opens the browser
.venv-tool\Scripts\python internal_tool\launch.py --headless # tarayıcısız · no browser
```
Adres: http://127.0.0.1:8501. Başlatıcı loopback olmayan adresi reddeder; sayfalar da sunucu loopback'e bağlı değilse
durur.

## Sayfalar · Pages
- **Başlangıç** (ilk açılan): üç sayfanın ne yaptığı, bağlantıları ve verinin durumu (analizlerin son
  koşumu; ham / silver / gold dosyalarının tarihi ve boyutu).
- **Raporlar:** Rapor (teknik · karar notu · SHAP), dil ve bölüm seçilir. Bölüm figürleriyle birlikte gösterilir.
  Altında o bölümü besleyen betikler listelenir; her birinin en son ne zaman koştuğu ve kodunun o zamandan beri
  değişip değişmediği görünür.
  - Bölüm → betik eşlemesi `section_map.json`'da, elle yazılmıştır. Testler eşlemeyi raporlara göre sınar
    (`tests/internal_tool/test_tool_section_map.py`).
  - Bir derleyici değişince sayfa uyarır. Eşlemeyi gözden geçirdikten sonra `python internal_tool/catalog.py
    --stamp` çalıştırılır.
- **Betikler:** 25 analiz betiğinin tablosu. Sütunlar: son koşum, kod değişti mi, `run_id`, güncel silver'ı okumuş mu,
  hangi bölümlerde kullanıldığı. Bir betik seçilince kartı, metrikleri, testleri ve kaynak kodu görünür. **Salt
  okunur**: buradan hiçbir betik koşturulmaz.
- **Veri gezgini:** Kaynak seçilir:
  - **ham**: `data/raw/*/*/details.jsonl`, bütün kayıtlar ve alanlar olduğu gibi;
  - **silver**: `data/cars.duckdb`;
  - **gold**: `data/cars_gold.duckdb`.

  Sonra satır kümesi seçilir (bütün satırlar · ilan başına son satır · analiz kümesi · tek tarama) ve **her kolona**
  koşul eklenir. Koşullar VE ile birleşir. Örnek: plaka = Mavi plakalı VE `KısaBilgi - Yıl (sayı)` 2024–2024 VE
  Kimden = Galeriden.
  - **Ham alanlar:**
    - Ham alanlar yazıldığı gibi kalır.
    - Sayıya benzeyen alanların yanında ayrıştırılmış bir `(sayı)` kolonu vardır, aralıkla filtrelemek için.
    - `Hasar_Listesi`, `Hasar - <Parça>` kolonlarına açılır.
  - **İlan metni araması:** Ham kaynakta `Aciklama_HTML`, DB'de `description_text` aranabilir. Arama büyük/küçük harfe
    ve Türkçe harflere duyarsızdır.
  - **Silver ↔ gold:** Silver ile gold aynı koşulları paylaşır. Sayfa aynı koşulların öbür DB'deki sonucunu da
    gösterir.
  - **Tablo:** `ad_id` ve tıklanır `url` içerir.
  - **İlan penceresi:** Bir ilanın **herhangi bir hücresine** tıklayınca ilan bir pencerede bütünüyle açılır:
    - üst bilgi ve "İlanı sitede aç";
    - gruplanmış bütün alanlar;
    - 13 parçalık hasar tablosu;
    - aynı ilanın bütün taramaları ve fiyat farkı;
    - ilan metni;
    - ham kaynakta kaydın JSON hâli.

## Testler · Tests
- `python tools/verify.py`: saf modüllerin testleri. AppTest burada atlanır, çünkü pipeline ortamında streamlit yok.
- `.venv-tool\Scripts\python -m pytest tests/internal_tool`: AppTest dahil hepsi.
- `python tools/verify.py --data`: gerçek veri sayımları. Ham kayıt 45.335; mavi plaka 38; plaka alanı yok 138;
  analiz kümesi = `n_dedup`.

## Kurallar · Rules
- **Salt okunur:** Veri yalnız okunur. DuckDB `read_only` açılır ve hemen kapanır; böylece araç açıkken DB yeniden
  kurulabilir.
- **`ad_id` ve `url` yalnız bu yerel araçta görünür.** Hiçbir log'a, rapora ya da site verisine girmezler.
- **Kaynak okuma istisnası:** `catalog.read_source`, betik kaynağını yalnız göstermek için okur. Bu,
  `tests/repo/test_hygiene.py`'deki tek istisnalardan biridir.

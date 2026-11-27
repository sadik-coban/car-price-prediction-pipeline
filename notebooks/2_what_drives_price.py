# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2. Fiyatı ne belirliyor
#
# **Bu defter.** İlan fiyatı aracın hangi özellikleriyle birlikte değişiyor? Önce tahmin modelinin özniteliklerinin
# birbirine ne kadar bağlı olduğuna ve segmentin nereden geldiğine bakıyoruz. Sonra hedefin, yani fiyatın dağılımına
# ve ham fiyatın segmente, kasa tipine, yaşa, kilometreye ve markaya göre nasıl değiştiğine. Son bölüm hedonik
# regresyonla, aracın kimliği sabitken durumunun ve motorunun fiyatla ilişkisini ölçüyor. Bu defterde tahmin modeli
# eğitilmez, yalnız doğrusal regresyonlar kurulur; her sayı her koşumda yeniden hesaplanır.
#
# | bölüm | soru |
# |---|---|
# | Kurulum ve veri | Hangi ilanlarla çalışıyoruz? |
# | 1. Öznitelikler birbirine nasıl bağlı | Hangi kolon ötekini belirliyor, hangi sayısal kolonlar birlikte değişiyor? |
# | 2. Segment nereden geliyor | Segment nasıl türetiliyor, sitenin verdiği segment neden kullanılmıyor? |
# | 3. Hedef: fiyat | Fiyat nasıl dağılıyor, tahmin modeli neden log fiyatı hedef alıyor? |
# | 4. Piyasa | Ham fiyat segmente, kasa tipine, yaşa, km'ye ve markaya göre nasıl değişiyor; doğal kümeler var mı? |
# | 5. Hedonik regresyon | Fiyatı ne kadar kimlik, ne kadar durum açıklıyor; kimlik sabitken durum ve motor ne diyor? |
# | Özet | Defterin gösterdikleri |
# | Başlık sayıları | Defterin sayıları, hesaplandıkları bölümle, tek tabloda |
#
# Defter hiçbir dosya yazmaz; veritabanını yalnız okur. İlan numaraları (`ad_id`) ve satıcıların yazdığı metinler
# hiçbir çıktıda gösterilmez. Sonuçların sayıları metinde yazılmaz, hücre çıktılarında görünür; metin neye
# bakılacağını ve ne görüldüğünü söyler.

# %% [markdown]
# > **Terimler** (araç modeli, tahmin modeli, ilan, öznitelik ve model yılı 1. defterdeki anlamıyla)
# >
# > - **log fiyat**: log(1 + fiyat). Log ölçekte bir fark, fiyatta yüzde bir farka karşılık gelir.
# > - **kimlik**: aracın ne olduğu. Kaba hâli segment, marka, yakıt ve vites; ince hâli araç modelinin adı.
# > - **durum**: aracın hâli: yaş, kilometre, ağır hasar kaydı, boyalı ve değişen parça sayısı.
# > - **hedonik regresyon**: log fiyatı aracın özelliklerine bağlayan doğrusal regresyon. Burada fiyat tahmini için
# >   değil, öteki özellikler sabitken bir özelliğin fiyatla ilişkisini ölçmek için kurulur.
# > - **kat dışı**: ilanlar beş kata bölünür; her katın ilanları, öteki dört katta kurulan regresyonla tahmin edilir.
# >   Kat dışı bir sayı, regresyonun görmediği ilanlarda ölçülmüştür.

# %% [markdown]
# ## Kurulum ve veri
#
# İlk hücre kütüphaneleri ve defterin yanındaki `lib` klasöründen bu defterin modüllerini yükler: `data` (veriyi okuma
# ve türetilen kolonlar; 1. defterde anlatıldı), `cleaning` (sitenin segment değeri), `association` (Theil's U),
# `market` (Cliff's delta), `hedonic` (kümeli regresyon ve kat dışı tahmin), `folds` (ortak beş kat), `spec_rule`
# (motor tutarlılık kuralı) ve `plots` (ısı haritası). Her modülün ne yaptığı kullanıldığı yerde anlatılıyor.

# %%
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from lib import association, cleaning, data, folds, hedonic, market, plots, spec_rule

pd.set_option("display.max_colwidth", None)   # uzun etiketler tablolarda kesilmesin

# %% [markdown]
# Defter 1. defterin kurduğu ilan tablosuyla çalışır: her ilanın en son taramadaki satırı (TR plakalı, fiyatı 0'dan
# büyük), tahmin modelinin kolonları `data.add_features` ile eklenmiş. Satırların nasıl seçildiği ve kolonların nasıl
# türetildiği 1. defterde.

# %%
SEED = 42   # defterin rastgele adımlarının (karıştırma, KMeans) tohumu
rows_raw = data.load_rows()
listings = data.add_features(rows_raw)
print("İlanlar (satır, kolon):", listings.shape)

# %% [markdown]
# ## 1. Öznitelikler birbirine nasıl bağlı
#
# **Soru.** Tahmin modelinin öznitelikleri birbirinden ayrı bilgi mi taşıyor, yoksa biri ötekini mi belirliyor?
# Kategorik ve metin kolonları için Theil's U, sayısal kolonlar için Spearman korelasyonu kullanıyoruz.
#
# ### 1.1 Kategorik kolonlar: Theil's U
#
# Theil's U(a | b), b bilindiğinde a'nın belirsizliğinin ne kadarının gittiğini söyler: 0 hiç, 1 tamamı. Belirsizlik
# entropiyle ölçülür: U(a | b) = (H(a) − H(a | b)) / H(a). Ölçü yönlüdür: araç modeli seriyi belirleyebilir ama seri
# araç modelini belirlemez. `association.theils_u` bu payı bir kolon çifti için hesaplar. Tahmin modelinin 9 kategorik
# kolonu ve 2 metin kolonu (araç modeli `model` ve seri `series`) için bütün çiftler: satırda a, kolonda b.

# %%
categorical = data.CAT + data.TEXT
theils = pd.DataFrame([[association.theils_u(listings[a], listings[b]) for b in categorical] for a in categorical],
                      index=categorical, columns=categorical)

theils.round(3).rename_axis(index="a (belirlenen)", columns="b (bilinen)")

# %% [markdown]
# Matrisin araç modeline ait kolonu ve satırı yan yana: araç modeli bilinince her kolonun belirsizliğinin ne kadarı
# gidiyor, ve tersi.
#
# Üçüncü kolon ölçünün tabanı. Bilinen kolonun çok değeri varsa U yukarı sapar: az ilanlı bir araç modelinin ilanları
# rastlantıyla da birbirine benzer, ölçü bunu bilgi sayar. Araç modeli kolonu ilanlar arasında rastgele karıştırılınca
# hiçbir bilgi taşımaz ama farklı değer sayısı aynı kalır; U'nun o durumda verdiği değer sapmanın büyüklüğüdür. 10
# karıştırmanın ortalaması.

# %%
SHUFFLES = 10
rng = np.random.default_rng(SEED)
shuffled_models = [pd.Series(listings["model"].to_numpy()[rng.permutation(len(listings))], index=listings.index)
                   for _ in range(SHUFFLES)]
floor = [np.mean([association.theils_u(listings[a], shuffled) for shuffled in shuffled_models]) for a in categorical]
print("Araç modelinin farklı değeri:", listings["model"].nunique())

asymmetry = pd.DataFrame({"given_model": theils["model"], "model_given": theils.loc["model"],
                          "given_shuffled_model": floor}).drop("model")

asymmetry.round(3).rename_axis("kolon").rename(columns={"given_model": "U(kolon | araç modeli)",
                                                        "model_given": "U(araç modeli | kolon)",
                                                        "given_shuffled_model": "U(kolon | karıştırılmış araç modeli)"})

# %% [markdown]
# **Ne görüyoruz.** Asimetri bulgunun kendisi. Araç modeli seriyi, markayı ve segmenti neredeyse tamamen belirliyor;
# ters yönde seri araç modelini ancak kısmen belirliyor. Yani seri, araç modelinin kabalaştırılmış hâli; ondan ayrı
# bir bilgi değil. Marka hem araç modelinden hem seriden tamamen okunuyor (matriste `brand` satırı): bu veride her
# seri tek bir markaya ait, marka ayrı bilgi taşımıyor. Segment de seriden neredeyse tamamen okunuyor ama tam değil
# (2. bölüm). Kasa tipi, çekiş, yakıt ve vitesin de büyük kısmı araç modelinden okunuyor; bu satırlarda taban küçük,
# okuma değişmiyor. Tavan, kaput ve bagaj durumları ise segment, marka ve kasa tipi gibi kolonlarla neredeyse hiç
# bağlı değil. Araç modeliyle bağlı gibi görünmelerinin önemli bir kısmı taban: karıştırılmış araç modeli de bu üç
# kolonun belirsizliğinin bir kısmını "açıklıyor". Bu üç satır bir bağ olarak okunmamalı.
#
# ### 1.2 Sayısal kolonlar: Spearman korelasyonu
#
# Spearman korelasyonu iki kolonun sıralarının korelasyonu: biri büyürken öteki de büyüyor (ya da küçülüyor) mu,
# ilişki doğrusal olmasa bile? −1 ile 1 arasında. Tahmin modelinin 14 sayısal kolonu için matris; boş değerler çift
# çift dışarıda kalır.

# %%
NUM_TR = {"vehicle_age": "yaş", "gb_mileage": "km", "power_hp_val": "güç", "engine_cc_val": "hacim",
          "door_changed": "kapı değişen", "door_painted": "kapı boyalı", "door_local": "kapı lokal",
          "fender_changed": "çamurluk değişen", "fender_painted": "çamurluk boyalı", "fender_local": "çamurluk lokal",
          "bumper_changed": "tampon değişen", "bumper_painted": "tampon boyalı", "bumper_local": "tampon lokal",
          "is_heavy_damaged": "ağır hasar"}   # kolon -> gösterilen ad

spearman = listings[data.NUM].corr(method="spearman")

# %%
fig, ax = plt.subplots(figsize=(8.4, 6.8))
image = plots.heatmap(ax, spearman.rename(index=NUM_TR, columns=NUM_TR), cmap="RdYlGn", vmin=-1, vmax=1)
fig.colorbar(image, ax=ax, shrink=0.8)
ax.set_title("Sayısal öznitelikler: Spearman korelasyonu", fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# Mutlak değeri 0.5'i aşan çiftler, en güçlüden:

# %%
HIGH_CORR = 0.5
high_pairs = pd.DataFrame([{"a": a, "b": b, "rho": spearman.loc[a, b]}
                           for a, b in itertools.combinations(data.NUM, 2) if abs(spearman.loc[a, b]) > HIGH_CORR])
high_pairs = high_pairs.sort_values("rho", key=abs, ascending=False, ignore_index=True)

high_pairs.replace({"a": NUM_TR, "b": NUM_TR}).round(3).rename(
    columns={"a": "birinci kolon", "b": "ikinci kolon", "rho": "Spearman"})

# %% [markdown]
# **Ne görüyoruz.** 0.5'i aşan çiftler beklenen bağlar: yaşlı araç çok yol yapmış, büyük motor güçlü, kapıları boyanmış
# aracın çamurlukları da çoğunlukla boyalı. Yaş ve km boyalı ve değişen parça sayılarıyla da zayıf ama pozitif
# ilişkili: eski araçta işlem görmüş parça daha çok. Bu bağların hedonik regresyondaki karşılığı 5.6'da VIF ile
# ölçülüyor; orada kapı ve çamurluk boyası ayrı değil, toplam boyalı parça sayısı olarak giriyor.

# %% [markdown]
# ## 2. Segment nereden geliyor
#
# **Soru.** Site her ilanda bir segment de veriyor (`gb_segment`), ama tahmin modeli onu kullanmıyor; segmenti seriden
# türetiyor (1. defter, Kurulum ve veri). Kural ilanları nasıl çözüyor ve sitenin segmenti neden alınmadı?
#
# ### 2.1 Kural ve çözüm yolları
#
# Segment kuralı `data` modülünde üç tablodan oluşur:
#
# - **seri eşlemesi** (`data.SEGMENT_MAP`): her seri bir segmente, ör. 3 Serisi ve A4 D.
# - **performans ailesi** (`data.PERF_BASE`): M, S ve RS aileleri tek bir segmente düşmez; araç modeli adındaki rakam
#   temel seriyi verir: M3 → 3 Serisi → D, RS 6 → A6 → E.
# - **araç modeli adı tablosu** (`data.MODEL_SEG`): rakam kuralının yanlış sonuç verdiği iki ad, önce bunlara bakılır:
#   i8 ve Z4 M spor araç (S).
#
# `data.resolve_segment` bir ilanın segmentini ve onu çözen yolu verir; `data.add_features` segment kolonunu aynı
# kuralla kurar ve çözülemeyen ilan kalırsa durur. Her yoldan kaç ilan çözülüyor?

# %%
PATHS = {"series_map": "seri eşlemesi", "base_class": "performans ailesi: temel seri",
         "model_table": "araç modeli adı tablosu"}   # yol kimliği -> gösterilen ad

segment_path = pd.Series([data.resolve_segment(s, m)[1] for s, m in zip(listings["series"], listings["model"])],
                         index=listings.index)
print("Seri eşlemesindeki seri:", len(data.SEGMENT_MAP))

segment_path.value_counts().rename(index=PATHS).rename_axis("çözüm yolu").rename("ilan").to_frame()

# %% [markdown]
# Seri eşlemesi dışında çözülen ilanlar hangi serilerden ve hangi segmentlere düşüyor?

# %%
from_model_name = listings[segment_path != "series_map"]

from_model_name.groupby("series")["segment"].value_counts().unstack(fill_value=0).rename_axis(
    index="seri", columns="segment")

# %% [markdown]
# Birden çok segmente yayılan seriler ve 1. bölümdeki U(segment | seri):

# %%
segments_per_series = listings.groupby("series")["segment"].nunique()
print("Birden çok segmente düşen seri:", ", ".join(segments_per_series.index[segments_per_series > 1]))
print("U(segment | seri):", round(theils.loc["segment", "series"], 3))

# %% [markdown]
# **Ne görüyoruz.** İlanların neredeyse hepsi segmentini doğrudan serisinden alıyor. Performans aileleri birden çok
# segmente yayılıyor: M Serisi'nin bir araç modeli C, bir başkası E olabiliyor. Bu yüzden segment yalnız serinin değil,
# seriyle araç modeli adının birlikte fonksiyonu; U(segment | seri) tam 1 değil. Araç modeli adı tablosu i8 ve Z4 M
# ilanlarını spor araç yapıyor.
#
# ### 2.2 Sitenin segmenti
#
# Sitenin segmenti `gb_segment` kolonunda, "D Segment" gibi; ilanların bir kısmında boş. `cleaning.raw_segment` her
# değeri kısa bir kimliğe (B … G) çevirir; veride hiç görülmemiş bir değer gelirse durur. Önce sitenin "G" segmenti:
# bu ilanların serisi ve kasa tipi ne? Kasa tipi kısa bilgi sekmesindeki `kb_body_type`: genel bakış sekmesindeki ikizi
# kasa tipini koltuk sayısıyla birleştirdiği için daha genel olan bu taraf kullanılıyor (1. defter, 4.4).

# %%
raw_segment = cleaning.raw_segment(listings["gb_segment"])
g_listings = listings[raw_segment == "G"]
print("Sitenin G dediği ilan:", len(g_listings))
print("Serileri:", g_listings["series"].value_counts().to_dict())

g_listings["kb_body_type"].value_counts().rename_axis("kasa tipi").rename("ilan").to_frame()

# %% [markdown]
# **Ne görüyoruz.** "G" gerçek bir segment değil: bu ilanların neredeyse hepsi MPV kasalı ve hepsi tek bir seriden.
# Sitenin segment kolonu burada kasa tipini segment gibi taşıyor. MPV bilgisi kasa tipi kolonunda zaten var; segment bu
# yüzden seriden türetiliyor ve bu ilanlar serilerinin segmentini alıyor.
#
# Sitenin segmenti dolu olan ilanlarda türetilen segment siteninkinden ne sıklıkla ayrılıyor, hangi yönde?

# %%
known = raw_segment.notna()
differs = known & (raw_segment != listings["segment"])
print(f"Sitenin segmenti dolu ilan: {known.sum()}; türetilenden farklı: {differs.sum()} "
      f"(%{100 * differs.sum() / known.sum():.2f})")

mismatch = pd.DataFrame({"raw": raw_segment[differs], "derived": listings.loc[differs, "segment"]}).value_counts()
mismatch.rename_axis(["sitenin segmenti", "türetilen segment"]).rename("ilan").to_frame()

# %% [markdown]
# **Ne görüyoruz.** Ayrılan ilanların yarısından fazlası bilinçli G düzeltmesi: sitenin G dediği ilanlar türetilen
# segmentte serilerinin segmentini alıyor. Geri kalanlarda sitenin E ya da F dediği bazı ilanlar kuralda bir alt
# segmente ya da spor sınıfına düşüyor. Ayrışma ilanların küçük bir kısmında: seriden türetilen segment sitenin
# bilgisiyle büyük ölçüde örtüşüyor ve bozuk G etiketini düzeltiyor.
#
# ### 2.3 Seri ve segmente göre fiyat
#
# Her (seri, segment) hücresinde medyan fiyat (milyon ₺). Boş hücre o seride o segmentten ilan olmadığını, nokta
# hücrenin tek bir ilandan oluştuğunu gösterir.

# %%
cells = listings.groupby(["series", "segment"])["price"].agg(["median", "size"])
median_table = cells["median"].unstack() / 1e6
single = (cells["size"].unstack() == 1).to_numpy()

fig, ax = plt.subplots(figsize=(5.8, 9))
image = plots.heatmap(ax, median_table, cmap="YlGnBu")
rows, columns = np.nonzero(single)
ax.plot(columns + 0.32, rows - 0.3, "o", ms=3.5, color="black")
fig.colorbar(image, ax=ax, shrink=0.6, label="medyan fiyat (milyon ₺)")
ax.set_title("Seri ve segmente göre medyan fiyat (milyon ₺)\n• = tek ilan", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Ne görüyoruz.** Seriler çoğunlukla tek bir segmentte; yalnız performans aileleri birden çok hücreye yayılıyor ve
# B dışında her segmentte en pahalı hücre bir performans ailesinin. Aynı segmentin içinde seriler arasında belirgin
# fiyat farkı var: segment fiyatın yalnız kaba bir göstergesi. Performans ailelerinin bazı hücreleri tek bir ilan; o
# hücrenin medyanı bir ilanın fiyatı.

# %% [markdown]
# ## 3. Hedef: fiyat
#
# **Soru.** İlan fiyatı nasıl dağılıyor ve tahmin modeli neden fiyatın kendisini değil log fiyatı hedef alıyor? Önce
# fiyatın yüzdelikleri:

# %%
price = listings["price"]
price.quantile([0.1, 0.5, 0.9]).rename({0.1: "%10", 0.5: "%50 (medyan)", 0.9: "%90"}).astype(int).rename_axis(
    "yüzdelik").rename("fiyat (₺)").to_frame().T

# %% [markdown]
# Çarpıklık dağılımın ne kadar tek yana uzandığını ölçer: 0 simetrik, pozitif değer sağa (pahalı uca) uzanan bir
# kuyruk demek. Fiyatın ve log fiyatın çarpıklığı:

# %%
skewness = pd.Series({"price": stats.skew(price), "log_price": stats.skew(np.log1p(price))})
skewness.round(2).rename({"price": "fiyat", "log_price": "log fiyat"}).rename_axis("hedef").rename(
    "çarpıklık").to_frame()

# %% [markdown]
# İki histogram yan yana. Solda fiyat: sıfırdan en yüksek fiyata kadar 250 bin ₺ genişliğinde kovalar. Sağda log
# fiyat: en düşük değerden en yükseğe eşit genişlikte 60 kova. Log ölçekte eşit aralık eşit oran demek; eksendeki
# işaretler karşılık gelen fiyatı (milyon ₺) gösteriyor. Kesikli çizgi iki grafikte de medyan.

# %%
BIN, LOG_BINS = 250_000, 60
LOG_TICKS = [0.5, 1, 2, 4]   # log fiyat ekseninde gösterilen fiyatlar (milyon ₺)
edges = np.arange(0, price.max() + BIN, BIN)
counts, _ = np.histogram(price, bins=edges)
log_counts, log_edges = np.histogram(np.log1p(price), bins=LOG_BINS)
print("Kovalara düşen ilan: fiyat", counts.sum(), "| log fiyat", log_counts.sum(), "/", len(price))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
axes[0].bar(edges[:-1] / 1e6, counts, width=0.9 * BIN / 1e6, align="edge")
axes[0].axvline(price.median() / 1e6, color="black", ls="--", lw=1)
axes[0].set(xlabel="fiyat (milyon ₺)", ylabel="ilan")
axes[0].set_title("Fiyat", fontsize=10)
axes[1].bar(log_edges[:-1], log_counts, width=0.9 * np.diff(log_edges), align="edge")
axes[1].axvline(np.log1p(price.median()), color="black", ls="--", lw=1)
axes[1].set_xticks(np.log1p(np.array(LOG_TICKS) * 1e6), [f"{tick:g}" for tick in LOG_TICKS])
axes[1].set(xlabel="fiyat (milyon ₺, log ölçek)", ylabel="ilan")
axes[1].set_title("Log fiyat", fontsize=10)
fig.suptitle("Fiyat ve log fiyat histogramı (kesikli çizgi: medyan)", fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Ne görüyoruz.** Ham fiyat sağa çarpık: ilanların çoğu ucuz ve orta kovalarda, pahalı uç uzun ve ince bir kuyruk.
# Son kova toplama tavanında bitiyor (1. defter, 1.4); tavanın üstünde ilan yok. Log dönüşümü dağılımı simetriğe çok
# yaklaştırıyor: sağdaki histogramda medyanın iki yanı birbirine yakın, uzun kuyruk yok. Tahmin modeli log fiyat
# üzerinde eğitilir (3. defter): log ölçekte fark göreli (yüzde) farka karşılık gelir, yani ucuz ve pahalı araçta aynı
# yüzde hata aynı ağırlığı taşır. Bu bir modelleme kararı, piyasa bulgusu değil. Bu defterdeki regresyonlar da log
# fiyat üzerinde.

# %% [markdown]
# ## 4. Piyasa
#
# **Soru.** Ham fiyat, yani başka hiçbir şey sabitlenmeden, segmente, kasa tipine, yaşa, kilometreye ve markaya göre
# nasıl değişiyor? Bu tablolar düzeltilmemiş: bir grubun fiyatı o gruptaki araçların karışımını da taşır (ör. yaşlı
# araçlar daha çok kilometre yapmış). Öteki özellikler sabitken karşılaştırma 5. bölümde.
#
# ### 4.1 Segment ve kasa tipi
#
# Segment başına medyan fiyat ve ilan sayısı, en ucuzdan:

# %%
by_segment = listings.groupby("segment")["price"].agg(median="median", n="size").sort_values("median")

by_segment.round().astype(int).rename_axis("segment").rename(columns={"median": "medyan fiyat (₺)", "n": "ilan"})

# %% [markdown]
# Kasa tipi başına aynı tablo, en az 80 ilanı olan kasa tipleri. Kasa tipi kısa bilgi sekmesindeki daha genel kolon
# (`kb_body_type`, 2.2). Kasa tipi verilmemiş ilanlarda kasa tipi araç modeli adından çıkarılmıyor, çünkü aynı ad
# farklı kasa tiplerinde satılıyor; bu ilanlar ayrı bir grup (`missing`).

# %%
MIN_BODY_N = 80
by_body = listings.groupby("kb_body_type")["price"].agg(median="median", n="size")
by_body = by_body[by_body["n"] >= MIN_BODY_N].sort_values("median")

by_body.round().astype(int).rename(index={"missing": "(verilmemiş)"}).rename_axis("kasa tipi").rename(
    columns={"median": "medyan fiyat (₺)", "n": "ilan"})

# %% [markdown]
# Grafik kasa tipi verilmemiş grubu çizmiyor; o grubu ve daha az ilanlı kasa tiplerini başlıkta sayıyor.

# %%
shown_bodies = by_body.drop("missing")
n_unknown_body = (listings["kb_body_type"] == "missing").sum()
n_small_bodies = len(listings) - n_unknown_body - shown_bodies["n"].sum()

fig, ax = plt.subplots(figsize=(7, 3))
ax.barh([f"{body} · {n}" for body, n in shown_bodies["n"].items()], shown_bodies["median"] / 1e6)
ax.invert_yaxis()
ax.set_xlabel("medyan fiyat (milyon ₺)")
ax.set_title(f"Kasa tipine göre medyan fiyat (en az {MIN_BODY_N} ilanlı tipler)\nkasa tipi verilmemiş "
             f"{n_unknown_body} ve daha az ilanlı tiplerdeki {n_small_bodies} ilan dışarıda", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Ne görüyoruz.** Segment sırası beklenen sıra: B en ucuz, F en pahalı, spor araçlar (S) F'nin hemen altında. Kasa
# tipinde 3 kapılı hatchback en ucuz, coupe en pahalı. Kasa tipi verilmemiş ilanların medyanı yüksek, ama bu bir kasa
# tipi değil, farklı araç modellerinden gelen karışık bir grup; grafikte bu yüzden çizilmiyor.
#
# ### 4.2 Yaş ve kilometre
#
# Yaş başına (en çok 25 yaş, en az 10 ilanlı yaşlar) ve 25 bin km'lik kilometre kovası başına (400 bin km'nin altı)
# medyan ve ortalama fiyat. Ortalama pahalı ilanlardan etkilenir, medyan tipik ilanı gösterir; aradaki açıklık
# fiyatın çarpıklığından.

# %%
MAX_AGE, MIN_AGE_N = 25, 10
by_age = listings[listings["vehicle_age"] <= MAX_AGE].groupby("vehicle_age")["price"].agg(
    median="median", n="size", mean="mean")
by_age = by_age[by_age["n"] >= MIN_AGE_N]

KM_BIN, KM_LIMIT = 25_000, 400_000
below_limit = listings[listings["gb_mileage"] < KM_LIMIT]
by_km = below_limit.groupby(below_limit["gb_mileage"] // KM_BIN * KM_BIN)["price"].agg(
    median="median", n="size", mean="mean")

print(f"Gösterilen yaşlar: {by_age.index.min()}-{by_age.index.max()} ({by_age['n'].sum()} ilan)")
print(f"{KM_LIMIT // 1000} bin km ve üstü, grafiğin dışında:", (listings["gb_mileage"] >= KM_LIMIT).sum(), "ilan")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, table, x, label in [(axes[0], by_age, by_age.index, "yaş (yıl)"),
                            (axes[1], by_km, (by_km.index + KM_BIN / 2) / 1000, "kilometre (bin km, kova ortası)")]:
    ax.plot(x, table["median"] / 1e6, "o-", ms=3, label="medyan")
    ax.plot(x, table["mean"] / 1e6, "s--", ms=3, label="ortalama")
    ax.set(xlabel=label, ylabel="fiyat (milyon ₺)")
    ax.legend(frameon=False)
axes[0].set_xticks(range(0, int(by_age.index.max()) + 1, 5))
axes[0].set_title("Yaşa göre", fontsize=10)
axes[1].set_title(f"Kilometreye göre ({KM_LIMIT // 1000} bin km altı)", fontsize=10)
fig.suptitle("Ham fiyat: düzeltilmemiş, öteki özellikler sabit değil", fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Ne görüyoruz.** Ham fiyat yaşla ve kilometreyle düzenli düşüyor. Ortalama çoğu noktada medyanın üstünde: her
# grubun içinde pahalı bir kuyruk var. Bu eğriler düzeltilmemiş: yaşlı araç hem daha çok kilometre yapmış hem başka
# araç modellerinden. Yaşın ve kilometrenin, öteki özellikler sabitken fiyatla ilişkisi 5. bölümde.
#
# ### 4.3 BMW ve Audi
#
# İki markanın ilan sayısı, medyan ve ortalama fiyatı. Farkın büyüklüğü için Cliff's delta: rastgele bir BMW ilanının
# rastgele bir Audi ilanından pahalı olma olasılığı eksi ucuz olma olasılığı. −1 ile 1 arasında; 0 iki dağılımın iç
# içe olduğunu, ±1 tamamen ayrıldığını söyler. `market.cliffs_delta` bunu bütün ilan çiftleri üzerinden, örnekleme
# yapmadan hesaplar. Anlamlılık testi (ör. Mann–Whitney) yapılmadı: bu büyüklükte iki grupta çok küçük bir fark da
# "anlamlı" çıkar ve farkın büyüklüğü hakkında bir şey söylemez.

# %%
by_brand = listings.groupby("brand")["price"].agg(n="size", median="median", mean="mean").reindex(["bmw", "audi"])
delta = market.cliffs_delta(listings.loc[listings["brand"] == "bmw", "price"],
                            listings.loc[listings["brand"] == "audi", "price"])
print("Cliff's delta (BMW ile Audi):", round(delta, 3))

by_brand.round().astype(int).rename(index={"bmw": "BMW", "audi": "Audi"}).rename_axis("marka").rename(
    columns={"n": "ilan", "median": "medyan fiyat (₺)", "mean": "ortalama fiyat (₺)"})

# %%
fig, ax = plt.subplots(figsize=(4.4, 3))
ax.bar(["BMW", "Audi"], by_brand["median"] / 1e6)
ax.set_ylabel("medyan fiyat (milyon ₺)")
ax.set_title("Ham medyan fiyat: BMW ve Audi\n(araç modeli karmasını da yansıtır)", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Ne görüyoruz.** BMW'nin ham medyanı Audi'ninkinden biraz yüksek, ama delta sıfıra çok yakın: iki markanın fiyat
# dağılımları neredeyse tamamen iç içe. Ham fark iki markanın araç modeli karmasını da yansıtıyor; 1. bölümde marka
# araç modelinden ve seriden tamamen okunuyordu. Markanın tahmin modeline katkısı 3. defterde ölçülüyor.
#
# ### 4.4 Doğal kümeler var mı
#
# Piyasayı segment ve kasa tipi gibi açık kolonlarla okuyoruz. Bunların yerine geçebilecek doğal kümeler var mı? 14
# sayısal öznitelikte boş değerler kolon medyanıyla dolduruluyor, her kolon standartlaştırılıyor (ortalama 0,
# standart sapma 1) ve KMeans k = 2 … 8 küme için kuruluyor. Her k için silhouette skoru: −1 ile 1 arasında; 1'e yakın
# değer kümelerin iyi ayrıldığını, 0'a yakın değer iç içe olduğunu söyler. Kümeler de skor da bütün ilanlarda
# hesaplanıyor; bu hücre defterin en uzun süren hücresi (yarım dakika kadar).

# %%
K_RANGE = range(2, 9)
numeric = listings[data.NUM].fillna(listings[data.NUM].median())
scaled = StandardScaler().fit_transform(numeric)

silhouette = {}
for k in K_RANGE:
    labels = KMeans(k, random_state=SEED, n_init=10).fit(scaled).labels_
    silhouette[k] = silhouette_score(scaled, labels)

pd.Series(silhouette).round(3).rename_axis("k").rename("silhouette").to_frame().T

# %% [markdown]
# **Ne görüyoruz.** Silhouette denenen bütün k değerlerinde düşük: bu ölçekte belirgin bir küme yapısı yok. k
# değerleri arasındaki farklar da küçük; en yüksek skoru veren k "doğru küme sayısı" diye okunmamalı. Kümeler bu
# yüzden analizde kullanılmıyor; piyasa segment ve kasa tipi gibi açık özniteliklerle okunuyor.

# %% [markdown]
# ## 5. Hedonik regresyon
#
# **Soru.** İki soru. Birincisi: log fiyatın ne kadarını yalnız aracın kimliği, ne kadarını yalnız durumu, ne kadarını
# ikisi birlikte açıklıyor? İkincisi: aracın kimliği sabitken durumu ve motoru fiyatla nasıl ilişkili, ve her ilişki
# kimlik sabitlenince nasıl değişiyor?
#
# Yöntem hedonik regresyon: log fiyatı aracın özelliklerine bağlayan doğrusal regresyon (en küçük kareler, OLS).
# Burada fiyat tahmini için değil, ikinci soru için kuruluyor; fiyatı tahmin etmek tahmin modelinin işi (3. defter).
# Katsayılar nedensel etki değil, kontrollü ilişkidir: ör. boyalı parçanın katsayısı boyamanın fiyatı düşürdüğünü
# değil, boyalı parçalı ilanların benzerlerinden o kadar ucuz ilan edildiğini söyler. Regresyonda tarama (dönem) etkisi
# yok: dört taramanın ilanları birlikte kestiriliyor.
#
# ### 5.1 Regresyon tablosu
#
# Regresyonun kolonları:
#
# - `log_price`: log(1 + fiyat), hedef.
# - `age`: yaş (yıl); `km100k`: kilometre, 100 bin km cinsinden.
# - `heavy_damage`: ağır hasar kaydı (1 var, 0 yok ya da belirtilmemiş).
# - `painted`, `changed`: boyalı ve değişen parça sayısı, 13 parçanın tamamında: veritabanının `count_painted` ve
#   `count_changed` sayaçları. Tahmin modelinin kapı, çamurluk ve tampon ayrımı yerine toplam sayı.
# - `hp100`: motor gücü / 100; `litre`: motor hacmi / 1000. İkisi de 1. defterin motor kuralıyla (güç aralığın orta
#   noktası, hacim üst sınırı).
#
# Kimlik kolonları (`segment`, `brand`, `kb_fuel`, `kb_transmission`, `model`) ilan tablosunda olduğu gibi kalıyor.

# %%
reg_table = listings.copy()
reg_table["log_price"] = np.log1p(reg_table["price"])
reg_table["age"] = reg_table["vehicle_age"]
reg_table["km100k"] = reg_table["gb_mileage"] / 1e5
reg_table["heavy_damage"] = reg_table["is_heavy_damaged"]
reg_table["painted"] = reg_table["count_painted"]
reg_table["changed"] = reg_table["count_changed"]
reg_table["hp100"] = reg_table["power_hp_val"] / 100
reg_table["litre"] = reg_table["engine_cc_val"] / 1000

# %% [markdown]
# OLS boş değerle çalışmaz. Motor gücü ya da hacmi boş olan ilanlar (1. defter, 2.2: değeri olmayan ilanlar) bu yüzden
# regresyona giremez. Kaç ilan, hangi yüzden? İkisi birden boş olan ilan toplamda bir kez sayılır.

# %%
has_hp, has_cc = listings["power_hp_val"].notna(), listings["engine_cc_val"].notna()
dropped = pd.Series({"hp": (~has_hp).sum(), "cc": (~has_cc).sum(), "both": (~has_hp & ~has_cc).sum(),
                     "total": (~(has_hp & has_cc)).sum()})

dropped.rename({"hp": "gücü boş", "cc": "hacmi boş", "both": "ikisi de boş", "total": "regresyona girmeyen"}).to_frame(
    "ilan")

# %% [markdown]
# Kalan ilanlarda yaş ve kilometre **medyan araca** ortalanır: medyan yaş ve medyan kilometre çıkarılır. Kare ve
# etkileşim terimleri varken yaşın ve kilometrenin katsayısı böylece medyan araçtaki marjinal etki olur. Ortalamak
# tahminleri ve R²'yi değiştirmez, yalnız katsayıların okunduğu noktayı belirler.

# %%
reg_table = reg_table[has_hp & has_cc]
age_center, km_center = reg_table["age"].median(), reg_table["km100k"].median()
reg_table["age"] = reg_table["age"] - age_center
reg_table["km100k"] = reg_table["km100k"] - km_center

print("Regresyondaki ilan:", len(reg_table), "| araç modeli:", reg_table["model"].nunique())
print(f"Medyan araç: {age_center:g} yaş, {km_center * 1e5:,.0f} km")

# %% [markdown]
# ### 5.2 Formüller
#
# Regresyonlar patsy formülüyle yazılıyor: `~` işaretinin solu hedef, sağı açıklayıcılar. `I(age ** 2)` yaşın karesi,
# `age:km100k` yaş ile kilometrenin çarpımı (etkileşim: yaşın fiyatla ilişkisi kilometreye göre değişebilir). `C(...)`
# kategorik bir kolonun her değeri için bir kukla (0/1) kolon; ilk değer referans. Formüller üç parçadan kuruluyor:
#
# - **durum** (`CONDITION`): yaş, km, kareleri ve etkileşimi, ağır hasar, boyalı ve değişen parça.
# - **motor** (`ENGINE`): güç ve hacim.
# - **kimlik kontrolleri** (`CONTROLS`): segment, marka, yakıt ve vites kuklaları. Daha ince kimlik araç modelinin adı
#   (`C(model)`): her araç modeli için bir kukla.

# %%
CONDITION = "age + I(age ** 2) + km100k + I(km100k ** 2) + age:km100k + heavy_damage + painted + changed"
ENGINE = "hp100 + litre"
CONTROLS = "C(segment) + C(brand) + C(kb_fuel) + C(kb_transmission)"
TERMS = f"{CONDITION} + {ENGINE}"

TERM_TR = {"age": "yaş", "I(age ** 2)": "yaş²", "km100k": "km (100 bin)", "I(km100k ** 2)": "km²",
           "age:km100k": "yaş×km", "heavy_damage": "ağır hasar", "painted": "boyalı parça",
           "changed": "değişen parça", "hp100": "+100 hp", "litre": "+1 litre"}   # terim -> gösterilen ad
TERM_NAMES = list(TERM_TR)

# %% [markdown]
# ### 5.3 Fiyatı ne açıklıyor
#
# Aynı ilanlarda altı regresyon: yalnız kimlik (iki düzeyde), yalnız durum, durum ve motor, sonra ikisi birlikte. Her
# biri için R² (log fiyatın açıklanan payı) iki kez: örneklem içi, yani regresyonun kurulduğu ilanlarda, ve kat dışı.
# Kat dışı için ilanlar `folds.make_folds` ile beş kata bölünüyor (karıştırılmış, tohum 42). Tahmin modelinin katları
# da aynı fonksiyonla kurulur (3. defter), ama orada bütün ilanlar bölünür, burada yalnız regresyona giren ilanlar:
# fonksiyon aynı, bölme aynı değil. `hedonic.oof_predictions` her katın ilanlarını öteki dört katta kurulan
# regresyonla tahmin eder; eğitim katlarında hiç görülmeyen bir araç modelinin kuklası sıfır alır, yani o ilan kalan
# kolonlardan tahmin edilir. İki R² de log fiyatın genel ortalaması etrafındaki kareler toplamına göre.

# %%
LADDER = {
    "identity_segment": CONTROLS,
    "identity_model": "C(model)",
    "condition": CONDITION,
    "condition_engine": TERMS,
    "segment_control": f"{TERMS} + {CONTROLS}",
    "model_control": f"{TERMS} + {CONTROLS} + C(model)",
}
LADDER_TR = {"identity_segment": "yalnız segment, marka, yakıt, vites", "identity_model": "yalnız araç modelinin adı",
             "condition": "yalnız durum", "condition_engine": "durum + motor",
             "segment_control": "durum + motor + segment, marka, yakıt, vites (segment kontrolü)",
             "model_control": "segment kontrolü + araç modelinin adı (araç modeli kontrolü)"}   # yalnız gösterim için

# %%
y = reg_table["log_price"].to_numpy()
total_ss = ((y - y.mean()) ** 2).sum()
cv_folds = folds.make_folds(len(reg_table))

ladder = {}
for name, rhs in LADDER.items():
    X = patsy.dmatrix(rhs, reg_table, return_type="dataframe").to_numpy()
    in_sample = X @ np.linalg.lstsq(X, y, rcond=None)[0]
    out_of_fold = hedonic.oof_predictions(X, y, cv_folds)
    ladder[name] = {"n_columns": X.shape[1], "r2_in_sample": 1 - ((y - in_sample) ** 2).sum() / total_ss,
                    "r2_out_of_fold": 1 - ((y - out_of_fold) ** 2).sum() / total_ss}
ladder = pd.DataFrame.from_dict(ladder, orient="index")

ladder.round(4).rename(index=LADDER_TR, columns={"n_columns": "kolon (sabit dahil)", "r2_in_sample": "R² örneklem içi",
                                                 "r2_out_of_fold": "R² kat dışı"}).rename_axis("regresyon")

# %% [markdown]
# Kat dışı R²'yi üç parçaya ayırıyoruz. **Yalnız durumun kattığı**: araç modeli kontrolü eksi yalnız araç modelinin
# adı. **Yalnız kimliğin kattığı**: araç modeli kontrolü eksi yalnız durum (motor burada kimlik tarafında sayılıyor).
# **Ortak**: geri kalanı; durum ve kimlik birbirinden bağımsız değil, bu kısmı ikisinden biri tek başına da açıklıyor.

# %%
oof_r2 = ladder["r2_out_of_fold"]
parts = pd.Series({"condition_only": oof_r2["model_control"] - oof_r2["identity_model"],
                   "identity_only": oof_r2["model_control"] - oof_r2["condition"]})
parts["shared"] = oof_r2["model_control"] - parts.sum()

parts.round(4).rename({"condition_only": "yalnız durumun kattığı", "identity_only": "yalnız kimliğin kattığı",
                       "shared": "ortak"}).rename_axis("parça").to_frame("R² (kat dışı)")

# %% [markdown]
# **Ne görüyoruz.** Yalnız segment, marka, yakıt ve vites log fiyatın küçük bir kısmını açıklıyor. Yalnız durum çok
# daha fazlasını, yalnız araç modelinin adı ondan da fazlasını açıklıyor. Durum ve motor birlikte kaba kimliğin çok
# üstünde; kimlik ve durum birlikte en yüksek. Üç parçanın en büyüğü ortak kısım: fiyat farklarının çoğunu durum da
# kimlik de tek başına açıklayabiliyor, çünkü ikisi birlikte değişiyor. Örneklem içi R² kat dışından en çok yalnız araç
# modelinin adında yüksek: az ilanlı araç modellerinde kukla kendi ilanlarına uyuyor, görmediği ilanda o kadar
# tutmuyor; okunan sayı bu yüzden kat dışı R².
#
# ### 5.4 Kimlik sabitken etkiler: üç sütun
#
# Aynı on terim üç regresyonda:
#
# - **kimlik yok**: yalnız durum ve motor terimleri.
# - **segment kontrolü**: segment, marka, yakıt ve vites kuklaları eklenir.
# - **araç modeli kontrolü**: bunlara araç modelinin adı da eklenir; her etki aynı araç modelinin ilanları arasında
#   ölçülür.
#
# `hedonic.fit_clustered` regresyonu kurar. Araç modeli kontrolünde araç modelinin kuklaları tasarıma açıkça konmaz:
# log fiyat ve her kolon kendi araç modelinin ortalamasından çıkarılır (grup içi dönüşüm). Bu, kuklalarla aynı
# katsayıları verir ve tasarımı tam ranklı tutar; araç modelinin içinde sabit kalan kolonlar (sabit terim, marka, çoğu
# segment kuklası) düşer. Fonksiyon on terimden biri düşerse ya da tabloda boş değer varsa durur.
#
# Standart hatalar **araç modeline göre kümeli**: aynı araç modelinin ilanları birbirinden bağımsız değil (aynı katalog
# değerleri, benzer fiyatlama), her araç modeli bir küme. %95 aralıklar küme sayısı eksi bir serbestlik dereceli t
# dağılımından. Fonksiyon R²'yi de verir; araç modeli kontrolünde bu, kuklalı regresyonun R²'si.

# %%
COLUMNS = {"no_identity": (f"log_price ~ {TERMS}", None),
           "segment_control": (f"log_price ~ {TERMS} + {CONTROLS}", None),
           "model_control": (f"log_price ~ {TERMS} + {CONTROLS}", "model")}   # sütun -> (formül, emilen grup)
COLUMNS_TR = {"no_identity": "kimlik yok", "segment_control": "segment kontrolü",
              "model_control": "araç modeli kontrolü"}

fits, r2 = {}, {}
for name, (formula, absorb) in COLUMNS.items():
    fits[name], r2[name] = hedonic.fit_clustered(reg_table, formula, cluster="model", absorb=absorb, tracked=TERM_NAMES)

pd.Series(r2).round(4).rename(COLUMNS_TR).rename_axis("sütun").to_frame("R² (örneklem içi)")

# %% [markdown]
# `hedonic.coefficient_table` her terimin katsayısını, kümeli standart hatasını ve %95 aralığını verir; bunları fiyata
# yüzde etkiye çevirir: etki = 100 · (e^katsayı − 1). Tabloda her sütun için etki ve aralığın alt ve üst sınırı (%).

# %%
coefs = {name: hedonic.coefficient_table(fit, TERM_NAMES) for name, fit in fits.items()}
effects = pd.concat({COLUMNS_TR[name]: table[["pct", "pct_lo", "pct_hi"]] for name, table in coefs.items()}, axis=1)

effects.round(2).rename(index=TERM_TR, columns={"pct": "etki %", "pct_lo": "alt", "pct_hi": "üst"}).rename_axis(
    "terim")

# %% [markdown]
# Okuma: yaş satırı medyan araçta yaş bir yıl artınca, km satırı kilometre 100 bin km artınca fiyattaki değişim. Kare
# ve etkileşim terimleri (yaş², km², yaş×km) tek başına okunmaz; eğrinin bükülmesini taşır. Ağır hasar satırı kaydın
# olup olmaması, boyalı ve değişen parça satırları her ek parça; +100 hp ve +1 litre satırları öteki motor terimi
# sabitken.

# %%
fig, ax = plt.subplots(figsize=(8, 5.2))
positions = np.arange(len(TERM_NAMES))
for i, (name, table) in enumerate(coefs.items()):
    errors = [table["pct"] - table["pct_lo"], table["pct_hi"] - table["pct"]]
    ax.errorbar(table["pct"], positions + (i - 1) * 0.25, xerr=errors, fmt="o", ms=4, capsize=3, lw=1,
                label=COLUMNS_TR[name])
ax.axvline(0, color="grey", ls="--", lw=0.8)
ax.set_yticks(positions, [TERM_TR[term] for term in TERM_NAMES])
ax.invert_yaxis()
ax.set_xlabel("fiyata etki (%)")
ax.set_title("Hedonik etkiler: nokta ve araç modeline göre kümeli %95 aralık", fontsize=11)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# Kimlik sabitlenince bir katsayı neden değişir? Araç modeli kontrolü regresyonu log fiyatı üç parçaya ayırır: on
# terimin katkısı, **kimlik bileşeni** (segment, marka, yakıt ve vites kuklaları ile araç modelinin sabit etkisi) ve
# artık. Kimlik bileşeni aynı on terime regreslenince çıkan katsayı, öteki terimler sabitken o terim bir birim
# artınca kimlik bileşeninin ne kadar değiştiğini söyler: artıysa, o terimi yüksek olan ilanlar daha pahalı araç
# modellerindendir. Bu katsayı iki sütunun farkına cebirsel olarak eşittir (dışarıda bırakılan değişken formülü):
# kimliksiz katsayı = araç modeli kontrolündeki katsayı + bu katsayı. Tabloda katsayı, araç modeline göre kümeli %95
# aralığı ve iki sütunun farkı, log ölçekte. Aralığı sıfırı dışlayan terimde iki sütunun farkı örnekleme hatasından
# büyüktür.

# %%
model_fit = fits["model_control"]
term_columns = patsy.dmatrix(TERMS, reg_table, return_type="dataframe")[TERM_NAMES]
identity_part = reg_table["log_price"] - term_columns @ model_fit.params[TERM_NAMES] - model_fit.resid
identity_fit, _ = hedonic.fit_clustered(reg_table.assign(identity_part=identity_part), f"identity_part ~ {TERMS}",
                                        cluster="model", tracked=TERM_NAMES)
identity_link = hedonic.coefficient_table(identity_fit, TERM_NAMES)[["coef", "ci_lo", "ci_hi", "contains_zero"]]
identity_link["column_gap"] = coefs["no_identity"]["coef"] - coefs["model_control"]["coef"]
print("Aralığı sıfırı dışlayan terimler:",
      ", ".join(TERM_TR[term] for term in TERM_NAMES if not identity_link.loc[term, "contains_zero"]))

identity_link.drop(columns="contains_zero").round(4).rename(index=TERM_TR, columns={
    "coef": "kimlik bileşeniyle ilişki", "ci_lo": "alt", "ci_hi": "üst",
    "column_gap": "kimlik yok − araç modeli kontrolü"}).rename_axis("terim")

# %% [markdown]
# **Ne görüyoruz.** Yaş, km, ağır hasar, boyalı ve değişen parça üç sütunda da fiyatla eksi yönde ilişkili ve
# aralıkları dar. Kimlik sabitlenince birkaç terim örnekleme hatasından fazla değişiyor; etki tablosunda en belirgin
# değişen iki terim km ve +100 hp. **Km** kimlik olmadan daha zayıf görünüyor ve son tablo nedenini gösteriyor: km'nin
# kimlik bileşeniyle ilişkisi artıda. Yaşı, hasarı ve motoru aynı ilanlar arasında km'si yüksek olanlar ortalamada
# fiyatı daha yüksek araç modellerinden; kimliksiz katsayı bunu da taşıdığı için km'nin fiyatla ilişkisini küçük
# gösteriyor. **+100 hp** ise kimlik olmadan çok daha büyük: log ölçekte kimliksiz etkinin yarıdan fazlası segment,
# marka, yakıt ve vites girince gidiyor; büyük kısmı motorun kendisi değil, aracın sınıfı. Yaşta, boyalı ve değişen
# parçada da fark sıfırdan ayrılıyor ama etki tablosunda küçük. Motor terimlerinin aralıkları araç modeli kontrolünde
# segment kontrolündekinden geniş: araç modelinin adı motoru büyük ölçüde belirlediği için orada güç ve hacim yalnız
# aynı araç modelinin ilanları arasındaki farktan ölçülüyor (5.9).
#
# ### 5.5 Güven aralıkları ve sütunların farkı
#
# Kümeli standart hatanın ikinci gerekçesi: hata varyansı ilandan ilana eşit değilse sıradan standart hatalar
# yanlıştır. Breusch–Pagan testi bunu segment kontrolü sütununun artıklarında sınar: test istatistiği (LM) büyüdükçe
# p küçülür; küçük p, eşit varyansın reddedilmesi demek. p bilgisayarın gösterebildiği en küçük sayının da altına
# düşerse `0` diye hesaplanır; hücre o durumda sayıyı değil bunu yazar.

# %%
bp_lm, bp_p = het_breuschpagan(fits["segment_control"].resid, fits["segment_control"].model.exog)[:2]
print(f"Breusch-Pagan LM: {bp_lm:.0f} (serbestlik derecesi {fits['segment_control'].model.exog.shape[1] - 1})")
print("p:", f"{bp_p:.3g}" if bp_p > 0 else f"gösterilebilen en küçük sayının ({np.finfo(float).tiny:.0e}) altında")

# %% [markdown]
# Komşu sütunların farkı örnekleme hatasından büyük mü? Regresyonlar aynı ilanlarda kurulduğu için farkları, bağımsız
# iki tahminin farkı gibi sınanamaz. `hedonic.difference_test` her terim için farkın standart hatasını iki regresyonun
# ilan başına etkilerinin (influence) farkından, yine araç modeline göre kümeli olarak hesaplar ve %5 düzeyinde
# sınar. Tabloda kimliğin girdiği iki adım, fark log ölçekte: kimlik yok eksi segment kontrolü, sonra segment kontrolü
# eksi araç modeli kontrolü. İki ucun farkı (kimlik yok eksi araç modeli kontrolü) 5.4'teki son tabloda.

# %%
STEPS = [("no_identity", "segment_control"), ("segment_control", "model_control")]
differences = {step: hedonic.difference_test(fits[step[0]], fits[step[1]], reg_table["model"], TERM_NAMES)
               for step in STEPS}
difference = differences["segment_control", "model_control"]
print("Küme (araç modeli):", reg_table["model"].nunique())

shown = pd.concat({f"{COLUMNS_TR[a]} − {COLUMNS_TR[b]}": table.round({"diff": 5, "se": 5, "t": 2}).assign(
    significant=table["significant"].map({True: "evet", False: "hayır"})) for (a, b), table in differences.items()},
    axis=1)
shown.rename(index=TERM_TR, columns={"diff": "fark (log)", "se": "standart hata", "significant": "anlamlı (%5)"}
             ).rename_axis("terim")

# %% [markdown]
# Duyarlılık: standart hatalar seriye göre kümelenince hangi terimlerin aralığı sıfırı içeriyor? Seri az sayıda ve
# büyüklükleri çok dengesiz olduğu için bu yalnız karşılaştırma için; araç modeline göre kümeyle yan yana. Tablonun
# her hücresi, aralığı sıfırı içeren terimler.

# %%
zero_terms = {}
for name in ["segment_control", "model_control"]:
    formula, absorb = COLUMNS[name]
    by_series, _ = hedonic.fit_clustered(reg_table, formula, cluster="series", absorb=absorb, tracked=TERM_NAMES)
    for cluster, fit in [("model", fits[name]), ("series", by_series)]:
        contains_zero = hedonic.coefficient_table(fit, TERM_NAMES)["contains_zero"]
        zero_terms[name, cluster] = ", ".join(TERM_TR[term] for term in TERM_NAMES if contains_zero[term]) or "yok"
print("Küme (seri):", reg_table["series"].nunique())

pd.Series(zero_terms).unstack().loc[["segment_control", "model_control"]].rename(index=COLUMNS_TR, columns={
    "model": "araç modeline göre küme", "series": "seriye göre küme"}).rename_axis(index="sütun", columns=None)

# %% [markdown]
# **Ne görüyoruz.** Breusch–Pagan eşit varyansı açıkça reddediyor; kümeli standart hata bu yüzden de gerekli. İlk
# adımda, segment, marka, yakıt ve vites girince, km, ağır hasar, boyalı ve değişen parça ile +100 hp örnekleme
# hatasından fazla değişiyor; en büyük iki fark km ve +100 hp'de (5.4). İkinci adımda, araç modelinin adı girince,
# fark yaş, yaş×km, ağır hasar ve boyalı parçada örnekleme hatasından büyük; km, değişen parça ve motor terimlerinde
# değil. İkinci adımın anlamlı farkları etki tablosunda küçük: araç modelini sabitlemek bu terimleri biraz kaydırıyor,
# yönlerini değiştirmiyor. Seriye göre kümelenince aralığı sıfırı içeren terimler artıyor: aynı
# serinin araç modelleri de birbirinden bağımsız değil, bu terimlerin anlamlılığı kümelemenin seçimine bağlı.
#
# ### 5.6 Çoklu bağlantı
#
# VIF (varyans şişme çarpanı) bir terimin tasarımdaki öteki kolonlardan ne kadar okunabildiğini ölçer: 1 hiç; 10'un
# üstü, katsayının belirsizliğinin bu yüzden belirgin büyüdüğü demek. Segment kontrolü sütununun tasarımında her terim
# için iki kez hesaplanıyor: ortalanmış (regresyonun kendisi) ve aynı tasarımın ortalanmamış hâli (ham yaş ve km).

# %%
RHS = f"{TERMS} + {CONTROLS}"
uncentred = reg_table.assign(age=reg_table["vehicle_age"], km100k=reg_table["gb_mileage"] / 1e5)
designs = {"centred": patsy.dmatrix(RHS, reg_table, return_type="dataframe"),
           "uncentred": patsy.dmatrix(RHS, uncentred, return_type="dataframe")}
vif = pd.DataFrame({name: [variance_inflation_factor(x.to_numpy(), x.columns.get_loc(term)) for term in TERM_NAMES]
                    for name, x in designs.items()}, index=TERM_NAMES)

vif.round(2).rename(index=TERM_TR, columns={"centred": "ortalanmış", "uncentred": "ortalanmamış"}).rename_axis("terim")

# %% [markdown]
# **Ne görüyoruz.** Ortalanmış tasarımda bütün VIF'ler 10'un altında. Ortalanmamış tasarımda yaş, km ve onlardan
# türeyen kare ve etkileşim terimlerinin VIF'i çok yüksek: beşi aynı iki değişkenden türediği için yapısal olarak
# bağlı. Medyan araca ortalamak bu bağı büyük ölçüde gideriyor; tahminler ve R² değişmiyor. Motor terimlerinin VIF'i
# ortalamadan etkilenmiyor; güç ve hacim birbirine bağlı (5.9).
#
# ### 5.7 Yaş: yıllık basamaklar
#
# İkinci dereceden terim yaşın fiyatla ilişkisini medyan araçta tek bir sayıya indiriyor; bu sayı seçilen fonksiyon
# biçimine bağlı. Burada yaş her yaş (yıl) için bir kuklayla, kilometre esnek bir eğriyle (`bs(...)`: 6 serbestlik
# dereceli B-spline) giriyor; öteki terimler ve araç modeli kontrolü aynı. Sonra iki yaş arasında log fiyatın yıl
# başına ortalama değişimi hesaplanıyor: (β_bitiş − β_başlangıç) / (bitiş − başlangıç), kümeli %95 aralığıyla
# (`t_test`), fiyata yüzde etki olarak.

# %%
AGE_STEPS = [(1, 5), (5, 10), (10, 15), (15, 20)]
STEP_FORMULA = (f"log_price ~ C(age_years) + bs(km_raw, df=6) + heavy_damage + painted + changed + {ENGINE}"
                f" + {CONTROLS}")

steps_table = reg_table.assign(age_years=reg_table["vehicle_age"], km_raw=reg_table["gb_mileage"] / 1e5)
fit_steps, _ = hedonic.fit_clustered(steps_table, STEP_FORMULA, cluster="model", absorb="model",
                                     tracked=[f"C(age_years)[T.{age}]" for step in AGE_STEPS for age in step])

# %% [markdown]
# Her basamak için yıl başına değişim, aralığı ve iki yaş arasındaki (iki uç dahil) ilan sayısı:

# %%
names = list(fit_steps.params.index)
steps = []
for start, end in AGE_STEPS:
    weights = np.zeros(len(names))
    weights[names.index(f"C(age_years)[T.{end}]")] = 1 / (end - start)
    weights[names.index(f"C(age_years)[T.{start}]")] = -1 / (end - start)
    test = fit_steps.t_test(weights)
    low, high = test.conf_int(hedonic.ALPHA)[0]
    steps.append({"from": start, "to": end, "n": reg_table["vehicle_age"].between(start, end).sum(),
                  "pct": hedonic.pct_effect(test.effect[0]), "pct_lo": hedonic.pct_effect(low),
                  "pct_hi": hedonic.pct_effect(high)})
steps = pd.DataFrame(steps)

steps.round(2).rename(columns={"from": "yaştan", "to": "yaşa", "n": "aradaki ilan", "pct": "yıl başına %",
                               "pct_lo": "alt", "pct_hi": "üst"})

# %% [markdown]
# İkinci dereceden terimin tek sayısı (araç modeli kontrolü sütununda yaş, medyan araçta) ve aralığı; basamakların
# kaçı bu aralığın dışında?

# %%
age_row = coefs["model_control"].loc["age"]
outside = ~steps["pct"].between(age_row["pct_lo"], age_row["pct_hi"])
print(f"İkinci dereceden terim: yılda %{age_row['pct']:.2f} [{age_row['pct_lo']:.2f}, {age_row['pct_hi']:.2f}]")
print("Bu aralığın dışında kalan basamak:", outside.sum(), "/", len(steps))

# %% [markdown]
# **Ne görüyoruz.** Her basamak bir düşüş, ama hızı yaşa göre değişiyor: orta yaşlarda en dik, yaşlı araçlarda en
# yavaş. Basamakların bir kısmı ikinci dereceden terimin tek sayısının aralığının dışında: tek sayı yaşa göre değişen
# bir düşüşü tek bir orana indiriyor. Veri tek dönem: ölçülen, aynı dönemde ilanda olan ve model yılı farklı araçların
# fiyat farkı; bir aracın yaşlandıkça ne kadar değer kaybettiği değil.
#
# ### 5.8 Boyalı ve değişen parça sayısı
#
# Doğrusal terim her ek parçaya aynı etkiyi verir. Burada boyalı (sonra değişen) parça sayısı, doğrusal terim yerine
# her sayı için bir kuklayla giriyor; 5 ve fazlası tek grup. Etki hiç boyalı (değişen) parçası olmayan ilanlara göre,
# aynı araç modeli içinde, öteki terimler sabitken.

# %%
COUNT_TOP = 5
by_count = {}
for column in ["painted", "changed"]:
    formula = f"log_price ~ C(bucket) + {TERMS.replace(f' + {column}', '')} + {CONTROLS}"
    levels = [f"C(bucket)[T.{count}]" for count in range(1, COUNT_TOP + 1)]
    bucketed = reg_table.assign(bucket=reg_table[column].clip(upper=COUNT_TOP))
    fit, _ = hedonic.fit_clustered(bucketed, formula, cluster="model", absorb="model", tracked=levels)
    table = hedonic.coefficient_table(fit, levels)[["pct", "pct_lo", "pct_hi"]].set_axis(range(1, COUNT_TOP + 1))
    table.insert(0, "n", bucketed["bucket"].value_counts().reindex(table.index))
    by_count[column] = table

# %% [markdown]
# Karşılaştırma için iki kolon daha: doğrusal katsayının (araç modeli kontrolü sütunu) o sayıda parçaya katlanmış hâli
# ve ilk parçanın etkisinin her parçada tekrarlandığı hâl, 100 · ((1 + etki/100)^sayı − 1).

# %%
COUNT_TR = {"painted": "boyalı", "changed": "değişen"}
for column, table in by_count.items():
    linear = coefs["model_control"].loc[column, "pct"]
    table["linear"] = 100 * ((1 + linear / 100) ** table.index - 1)
    table["first_repeated"] = 100 * ((1 + table.loc[1, "pct"] / 100) ** table.index - 1)

pd.concat(by_count).round(2).rename(index=COUNT_TR, level=0).rename(index={COUNT_TOP: f"{COUNT_TOP} ve fazlası"},
                                                                     level=1).rename(columns={
    "n": "ilan", "pct": "etki %", "pct_lo": "alt", "pct_hi": "üst", "linear": "doğrusal terimle",
    "first_repeated": "ilk parça tekrarlansa"}).rename_axis(["parça", "sayı"])

# %% [markdown]
# **Ne görüyoruz.** **Boyalı parça her biri sabit bir tutar değil.** Tek boyalı parça, doğrusal katsayının
# söylediğinden belirgin fazla düşürüyor; aralığı doğrusal değeri dışlıyor. 5 ve fazlası ise ilk parçanın etkisi her
# parçada tekrarlansaydı olacağından az düşürüyor. İlk boyalı parça en çok, sonrakiler ortalamada daha az düşürüyor.
# **Değişen parça doğrusala yakın:** doğrusal katsayının katlanmış hâli 2'den 5 ve fazlasına kadar her satırın
# aralığına düşüyor; yalnız ilk değişen parça bundan fazla düşürüyor.
#
# ### 5.9 Motor
#
# Araç modeli kontrolünde güç ve hacim yalnız aynı araç modelinin ilanları arasındaki farktan ölçülüyor. Bu fark ne
# kadar? Güç kaç araç modelinde ilanlar arasında değişiyor, ve araç modeli içindeki yayılımı (her ilanın gücünün kendi
# araç modelinin ortalamasından farkının standart sapması) genel yayılımın yanında ne kadar?

# %%
hp = reg_table["power_hp_val"]
hp_by_model = hp.groupby(reg_table["model"])
hp_sd_within = (hp - hp_by_model.transform("mean")).std()

print(f"Araç modeli: {hp_by_model.ngroups}; gücü ilanlar arasında değişen: {(hp_by_model.nunique() > 1).sum()}")
print(f"Gücün standart sapması: araç modeli içinde {hp_sd_within:.1f} hp, genelde {hp.std():.1f} hp")

# %% [markdown]
# Araç modeli içindeki güç farkının bir kısmı katalog hatası olabilir. Motor tutarlılık kuralı (`spec_rule`; aynı kural
# 4. defterde büyük hatalara bakarken de kullanılıyor): gücü ya da hacmi kendi araç modelinin medyanından 1.5 kattan
# fazla sapan ilan işaretlenir; en az 5 ilanı olan araç modellerinde. Kural bütün ilanlara uygulanıyor; işaretli
# ilanlar regresyon tablosundan çıkarılıp araç modeli kontrolü yeniden kuruluyor.

# %%
spec_off, _ = spec_rule.spec_outliers(listings)
reg_table_spec = reg_table[~spec_off.loc[reg_table.index]]
fit_spec, _ = hedonic.fit_clustered(reg_table_spec, COLUMNS["model_control"][0], cluster="model", absorb="model",
                                    tracked=TERM_NAMES)
print("İşaretli ilan: bütün ilanlarda", spec_off.sum(), "| regresyon tablosunda", len(reg_table) - len(reg_table_spec))

ENGINE_TERMS = ["hp100", "litre"]
engine_effects = pd.concat({"all": coefs["model_control"].loc[ENGINE_TERMS, ["pct", "pct_lo", "pct_hi"]],
                            "without": hedonic.coefficient_table(fit_spec, ENGINE_TERMS)[["pct", "pct_lo", "pct_hi"]]},
                           axis=1)
engine_effects.round(2).rename(index=TERM_TR, columns={"all": "bütün ilanlar", "without": "işaretliler olmadan",
                                                       "pct": "etki %", "pct_lo": "alt", "pct_hi": "üst"})

# %% [markdown]
# Kuralın oranı (1.5) bir seçim. Sonuç ona ne kadar bağlı? Aynı adım daha sıkı (1.25) ve daha gevşek (2) bir oranla:
# regresyon tablosundan çıkan ilan ve +100 hp'nin etkisi.

# %%
by_ratio = {}
for ratio in [1.25, spec_rule.RATIO, 2.0]:
    flagged, _ = spec_rule.spec_outliers(listings, ratio=ratio)
    kept = reg_table[~flagged.loc[reg_table.index]]
    fit, _ = hedonic.fit_clustered(kept, COLUMNS["model_control"][0], cluster="model", absorb="model",
                                   tracked=TERM_NAMES)
    by_ratio[ratio] = {"n": len(reg_table) - len(kept),
                       **hedonic.coefficient_table(fit, ["hp100"]).loc["hp100", ["pct", "pct_lo", "pct_hi"]]}

pd.DataFrame(by_ratio).T.astype({"n": int}).round(2).rename_axis("oran").rename(columns={
    "n": "çıkarılan ilan", "pct": "+100 hp etki %", "pct_lo": "alt", "pct_hi": "üst"})

# %% [markdown]
# Güç ve hacim birbirine bağlı; regresyonda ikisi birlikte. Regresyon tablosunda iki kolonun Pearson korelasyonu:

# %%
print("Güç ile hacmin Pearson korelasyonu:", round(reg_table["power_hp_val"].corr(reg_table["engine_cc_val"]), 3))

# %% [markdown]
# **Ne görüyoruz.** Güç araç modellerinin ancak yarısına yakınında ilanlar arasında değişiyor ve araç modeli içindeki
# yayılımı genel yayılımın küçük bir kesri: araç modeli kontrolünde motor terimleri ince bir farktan ölçülüyor, bu
# yüzden aralıkları geniş. Bu farkın bir kısmı katalog hatası: motor değeri tutarsız az sayıda ilan çıkarılınca +100
# hp'nin noktası belirgin kayıyor, aralığı yine geniş. Kayma oranın seçimine bağlı değil: işaretlenen ilan sayısı
# orana göre çok değişiyor, ama +100 hp'nin noktası üç oranda da aynı yere gidiyor; kaymayı en uçtaki birkaç ilan
# yapıyor. Güç ve hacim güçlü biçimde bağlı: hacmin etkisi güç sabitken kalan kısım; iki katsayı birlikte okunmalı,
# birimleri farklı olduğu için doğrudan kıyaslanmaz.
#
# ### 5.10 Ağır hasar: "belirtilmemiş" ile "yok" aynı mı
#
# Ağır hasar terimi kaydı olan ilanı 1, ötekileri 0 sayıyor. Ötekilerin çoğunda sayfa "yok" demiyor, bilgi
# vermiyor (1. defter, 4.1). Bu iki grubu birleştirmek katsayıyı bozuyor mu? Ham kayıt üç durumlu: var, yok, boş
# (belirtilmemiş). Terim burada 0/1 yerine üç durumla giriyor, referans "yok"; öteki terimler ve üç sütun aynı. Her
# sütun için üç satır: bugünkü terim (var, ötekilerin hepsine karşı), "var" ve "belirtilmemiş" (ikisi de "yok"a karşı).

# %%
record = rows_raw.loc[reg_table.index, "is_heavy_damaged"].astype("boolean")
damage_table = reg_table.assign(damage_record=np.select(
    [record.isna().to_numpy(), record.fillna(False).to_numpy(dtype=bool)], ["unspecified", "yes"], default="no"))
print("Regresyon tablosunda:", damage_table["damage_record"].value_counts().rename(
    {"unspecified": "belirtilmemiş", "no": "yok", "yes": "var"}).to_dict())

DAMAGE_TERM = "C(damage_record, Treatment('no'))"
DAMAGE_LEVELS = {f"{DAMAGE_TERM}[T.yes]": "var (yok'a karşı)",
                 f"{DAMAGE_TERM}[T.unspecified]": "belirtilmemiş (yok'a karşı)"}   # terim -> gösterilen ad
damage_effects = {}
for name, (formula, absorb) in COLUMNS.items():
    fit, _ = hedonic.fit_clustered(damage_table, formula.replace("heavy_damage", DAMAGE_TERM), cluster="model",
                                   absorb=absorb, tracked=list(DAMAGE_LEVELS))
    damage_effects[name] = pd.concat([
        coefs[name].loc[["heavy_damage"]].rename(index={"heavy_damage": "var (ötekilerin hepsine karşı)"}),
        hedonic.coefficient_table(fit, list(DAMAGE_LEVELS)).rename(index=DAMAGE_LEVELS)])[["pct", "pct_lo", "pct_hi"]]

pd.concat(damage_effects).round(2).rename(index=COLUMNS_TR, level=0).rename(columns={
    "pct": "etki %", "pct_lo": "alt", "pct_hi": "üst"}).rename_axis(["sütun", "karşılaştırma"])

# %% [markdown]
# **Ne görüyoruz.** Kimlik sabitken "belirtilmemiş" ilanlar kaydı olmayanlardan neredeyse ayrılmıyor: fark iki
# kontrollü sütunda da ağır hasar kaydının etkisinin yanında çok küçük. "Var"ın etkisi iki kodlamada birbirine yakın,
# aralıkları örtüşüyor. Belirtilmemişi "yok" saymak bu regresyonda katsayıyı bozmuyor. Kimlik olmadan belirtilmemiş
# ilanlar biraz pahalı görünüyor; bu, o ilanların hangi araçlar olduğunu yansıtıyor, kimlik girince küçülüyor.

# %% [markdown]
# ## Özet
#
# - **Bağlılık (1):** araç modeli seriyi, markayı ve segmenti neredeyse tamamen belirliyor, tersi değil; seri araç
#   modelinin kaba hâli, marka ayrı bilgi taşımıyor. Sayısal kolonlarda güçlü bağlar yaş ile km, güç ile hacim, boyalı
#   kapı ile boyalı çamurluk arasında.
# - **Segment (2):** segment seriden, performans ailelerinde araç modeli adından türetiliyor. Sitenin segmenti
#   kullanılmıyor: sitenin "G" segmenti gerçek bir segment değil, MPV kasa tipi.
# - **Hedef (3):** fiyat sağa çarpık, log fiyat simetriğe yakın; tahmin modeli ve bu defterin regresyonları log fiyat
#   üzerinde.
# - **Piyasa (4):** ham fiyat segmentle yükseliyor, yaşla ve kilometreyle düşüyor; BMW ve Audi'nin fiyat dağılımları
#   neredeyse iç içe. Belirgin doğal küme yok.
# - **Hedonik regresyon (5):** kimlik ve durum log fiyatın büyük kısmını birlikte açıklıyor ve bu kısmın çoğu ortak.
#   Aynı araç modeli içinde yaş, km, ağır hasar, boyalı ve değişen parça fiyatla eksi yönde ilişkili. Yaşın yıllık
#   etkisi yaşa göre değişiyor; ilk boyalı parça sonrakilerden çok düşürüyor, değişen parça doğrusala yakın. Kimlik
#   olmadan bakınca km zayıf, güç büyük görünüyor: km'si yüksek ilanlar daha pahalı araç modellerinden. Motor
#   terimleri araç modeli içinde ince bir farktan ölçüldüğü için belirsiz. Ağır hasar kaydı belirtilmemiş ilanlar,
#   kimlik sabitken, kaydı olmayanlardan neredeyse ayrılmıyor.

# %% [markdown]
# ## Başlık sayıları
#
# Defterin sayıları tek tabloda, her biri hesaplandığı bölümle. Tablo iki iş görüyor: sonuçları bir arada gösteriyor
# ve defter yeniden çalıştırıldığında sayıların değişip değişmediği bu tablodan denetleniyor; bu yüzden kısa
# tutulmadı. Anahtarlar kod adı olduğu için İngilizce; sayılar yukarıdaki tablolardaki gibi yuvarlandı. Yüzde etkiler
# araç modeli kontrolü sütunundan, aksi yazılmadıkça.

# %%
effect = coefs["model_control"]["pct"].round(2)
effect_no_identity = coefs["no_identity"]["pct"].round(2)
step_pct = steps.set_index(["from", "to"])["pct"].round(2)
count_pct = {column: table["pct"].round(2) for column, table in by_count.items()}
rows = [
    ("theils_series_given_model", float(round(theils.loc["series", "model"], 3)), "U(seri | araç modeli)", "1.1"),
    ("theils_model_given_series", float(round(theils.loc["model", "series"], 3)), "U(araç modeli | seri)", "1.1"),
    ("spearman_age_mileage", float(round(spearman.loc["vehicle_age", "gb_mileage"], 3)), "Spearman: yaş ile km", "1.2"),
    ("high_spearman_pairs", len(high_pairs), "|Spearman| 0.5'i aşan sayısal çift", "1.2"),
    ("segment_from_model_name", int((segment_path != "series_map").sum()),
     "segmenti seri eşlemesinden gelmeyen ilan", "2.1"),
    ("raw_g_segment_listings", len(g_listings), "sitenin G segmenti dediği ilan", "2.2"),
    ("raw_segment_differs", int(differs.sum()), "sitenin segmentinden ayrılan ilan", "2.2"),
    ("price_median", int(price.median()), "medyan fiyat (₺)", "3"),
    ("skew_price", float(skewness.round(2)["price"]), "fiyatın çarpıklığı", "3"),
    ("skew_log_price", float(skewness.round(2)["log_price"]), "log fiyatın çarpıklığı", "3"),
    ("bmw_median", int(by_brand.round().loc["bmw", "median"]), "BMW medyan fiyatı (₺)", "4.3"),
    ("audi_median", int(by_brand.round().loc["audi", "median"]), "Audi medyan fiyatı (₺)", "4.3"),
    ("cliffs_delta", round(delta, 3), "Cliff's delta, BMW ile Audi", "4.3"),
    ("best_silhouette", float(round(max(silhouette.values()), 3)), "en yüksek silhouette (k = 2 … 8)", "4.4"),
    ("hedonic_listings", len(reg_table), "hedonik regresyondaki ilan", "5.1"),
    ("car_model_clusters", int(reg_table["model"].nunique()), "küme: araç modeli", "5.1"),
]
rows += [(f"r2_oof_{name}", float(round(ladder.loc[name, "r2_out_of_fold"], 4)), f"kat dışı R²: {LADDER_TR[name]}",
          "5.3") for name in LADDER]
rows += [
    ("age_pct", float(effect["age"]), "yaş, yıl başına, medyan araçta (%)", "5.4"),
    ("km100k_pct", float(effect["km100k"]), "+100 bin km, medyan araçta (%)", "5.4"),
    ("heavy_damage_pct", float(effect["heavy_damage"]), "ağır hasar kaydı (%)", "5.4"),
    ("painted_pct", float(effect["painted"]), "boyalı parça başına, doğrusal (%)", "5.4"),
    ("changed_pct", float(effect["changed"]), "değişen parça başına, doğrusal (%)", "5.4"),
    ("hp100_pct", float(effect["hp100"]), "+100 hp (%)", "5.4"),
    ("litre_pct", float(effect["litre"]), "+1 litre (%)", "5.4"),
    ("no_identity_km100k_pct", float(effect_no_identity["km100k"]), "kimlik yok: +100 bin km (%)", "5.4"),
    ("no_identity_hp100_pct", float(effect_no_identity["hp100"]), "kimlik yok: +100 hp (%)", "5.4"),
    ("significant_differences", int(difference["significant"].sum()),
     "iki kontrol sütununun farkı anlamlı terim", "5.5"),
    ("vif_max_centred", float(vif["centred"].round(2).max()), "en yüksek VIF, ortalanmış", "5.6"),
    ("hp100_pct_without_spec_outliers", float(engine_effects.round(2).loc["hp100", ("without", "pct")]),
     "+100 hp, motor değeri tutarsız ilanlar olmadan (%)", "5.9"),
    ("age_step_5_10_pct", float(step_pct[5, 10]), "yaş 5-10 arası, yıl başına (%)", "5.7"),
    ("age_step_15_20_pct", float(step_pct[15, 20]), "yaş 15-20 arası, yıl başına (%)", "5.7"),
    ("painted_1_pct", float(count_pct["painted"][1]), "tek boyalı parça (%)", "5.8"),
    ("painted_5plus_pct", float(count_pct["painted"][COUNT_TOP]), "5 ve fazla boyalı parça (%)", "5.8"),
    ("changed_1_pct", float(count_pct["changed"][1]), "tek değişen parça (%)", "5.8"),
    ("damage_unspecified_pct",
     float(damage_effects["model_control"].round(2).loc["belirtilmemiş (yok'a karşı)", "pct"]),
     "ağır hasar kaydı belirtilmemiş, kaydı olmayana karşı (%)", "5.10"),
]
headline = {key: value for key, value, _, _ in rows}

headline_table = pd.DataFrame(rows, columns=["key", "value", "description", "section"], dtype=object).set_index("key")
headline_table[["value", "section", "description"]].rename_axis("anahtar").rename(
    columns={"value": "değer", "section": "bölüm", "description": "ne"})

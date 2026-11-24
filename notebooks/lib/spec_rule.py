"""
Motor tutarlılık kuralı: gücü ya da hacmi kendi araç modelinin medyanından 1.5 kattan fazla sapan ilan işaretlenir.
Böyle bir değer çoğunlukla katalog eşleşmesinin hatasıdır: ilana aracın sahip olmadığı bir motor yazılmıştır. Kural
en az 5 ilanı olan araç modellerinde sınanır; daha az ilanlı araç modelleri sınanamaz (kör nokta).
Modülü import etmek hiçbir dosya okumaz, yazmaz.
"""
MIN_GROUP, RATIO = 5, 1.5


def spec_outliers(listings, min_group=MIN_GROUP, ratio=RATIO):
    """
    Güç (power_hp_val) ve hacim (engine_cc_val; data.add_features'ın motor kuralıyla), her biri aynı araç modelinin
    (model) medyanına karşı. Boş değer işaretlenmez. Döndürür: (işaretli, kör) ilanlar üzerinde iki mantıksal seri;
    kör = araç modelinin min_group'tan az ilanı var.
    """
    hp, cc = listings["power_hp_val"], listings["engine_cc_val"]
    hp_median = hp.groupby(listings["model"]).transform("median")
    cc_median = cc.groupby(listings["model"]).transform("median")
    n_model = listings.groupby("model")["model"].transform("size")
    off = (hp > ratio * hp_median) | (hp < hp_median / ratio) | (cc > ratio * cc_median) | (cc < cc_median / ratio)
    return (n_model >= min_group) & off, n_model < min_group

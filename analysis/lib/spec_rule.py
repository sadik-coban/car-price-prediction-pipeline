"""
spec_rule.py
EN: The engine-consistency rule shared by 08_large_errors (§8 "listings with an inconsistent engine value") and
    06_hedonic (the §6 sensitivity without those listings): a listing whose power or size is more than 1.5× off the
    median of its own model name (models with at least 5 listings) is flagged — a catalogue mismatch prices an engine
    the car does not have. Models with fewer listings cannot be checked (the blind spot).
TR: 08_large_errors (§8 "motor değeri tutarsız ilanlar") ve 06_hedonic (bu ilanlar çıkarılınca §6 duyarlılığı) ortak
    motor tutarlılık kuralı: gücü ya da hacmi kendi model adının medyanından 1.5 kattan fazla sapan ilan (en az 5
    ilanlı modellerde) işaretlenir — katalog uyuşmazlığı aracın sahip olmadığı bir motoru fiyatlar. Daha az ilanlı
    modeller sınanamaz (kör nokta).
"""
import pandas as pd

MIN_GROUP, RATIO = 5, 1.5


def spec_outlier_mask(listings, min_group=MIN_GROUP, ratio=RATIO):
    """
    EN: Power = mean of the lower and upper bound, size = upper bound (the model's engine rule), each against the
        median of the same model name. Returns: (flagged, blind) boolean arrays over the listings.
    TR: Güç = alt ve üst sınırın ortalaması, hacim = üst sınır (modelin motor kuralı), her biri aynı model adının
        medyanına karşı. Döndürür: ilanlar üzerinde (işaretli, kör) mantıksal diziler.
    """
    hp = listings[["power_hp_low", "power_hp_up"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    cc = pd.to_numeric(listings["engine_cc_up"], errors="coerce")
    g = listings.groupby("model")
    n_model = g["model"].transform("size")
    hp_med, cc_med = hp.groupby(listings["model"]).transform("median"), cc.groupby(listings["model"]).transform("median")
    off = ((n_model >= min_group) & ((hp > ratio * hp_med) | (hp < hp_med / ratio)
                                     | (cc > ratio * cc_med) | (cc < cc_med / ratio))).fillna(False).values
    return off, (n_model < min_group).values

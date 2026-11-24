"""
test_spec_rule.py
notebooks/lib/spec_rule.py: motor tutarlılık kuralı küçük, elle kurulmuş tablolarla: araç modelinin medyanından 1.5
kattan fazla sapan güç ya da hacim işaretlenir; az ilanlı araç modeli kör kalır ve hiç işaretlenmez; boş değer
işaretlenmez. Gerçek veri okunmaz.
"""
import numpy as np
import pandas as pd

from lib import spec_rule

NA = np.nan


def frame():
    """A: 6 ilan (biri güçte, biri hacimde sapıyor, biri gücü boş); B: 4 ilan (sınanamaz, biri çok sapıyor)."""
    return pd.DataFrame({
        "model": ["A"] * 6 + ["B"] * 4,
        "power_hp_val": [150, 150, 160, 400, 150, NA, 200, 200, 200, 600],
        "engine_cc_val": [2000, 2000, 2000, 2000, 1000, 2000, 3000, 3000, 3000, 3000],
    })


def test_spec_outliers_flags_power_and_size():
    flagged, blind = spec_rule.spec_outliers(frame())
    assert flagged.tolist() == [False, False, False, True, True, False] + [False] * 4
    assert blind.tolist() == [False] * 6 + [True] * 4                # B'nin 4 ilanı var, en az 5 gerekiyor


def test_spec_outliers_boundary_is_exclusive():
    # tam 1.5 kat sapma işaretlenmez, üstü işaretlenir
    listings = pd.DataFrame({"model": ["A"] * 5, "power_hp_val": [100, 100, 100, 150, 151],
                             "engine_cc_val": [2000] * 5})
    flagged, _ = spec_rule.spec_outliers(listings)
    assert flagged.tolist() == [False, False, False, False, True]

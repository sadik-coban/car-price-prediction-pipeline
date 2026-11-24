"""
Bağımlılık: iki kategorik kolon arasında Theil's U.
Modülü import etmek hiçbir dosya okumaz, yazmaz.
"""
import numpy as np
import pandas as pd


def entropy(values):
    """Bir kolonun Shannon entropisi (doğal log): -Σ p log p, p her değerin payı."""
    p = values.value_counts(normalize=True).to_numpy()
    return float(-(p * np.log(p)).sum())


def theils_u(a, b):
    """
    Theil's U(a | b): b bilinince a'nın belirsizliğinin (entropisinin) giden payı, 0 ile 1 arasında. Yönlüdür.
    U(a | b) = (H(a) - H(a | b)) / H(a), H(a | b) = H(a, b) - H(b). a tek değerliyse 0.
    """
    h_a = entropy(a)
    if h_a == 0:
        return 0.0
    joint = pd.crosstab(a, b).to_numpy().ravel()
    joint = joint[joint > 0] / joint.sum()
    h_ab = float(-(joint * np.log(joint)).sum())
    return (h_a - (h_ab - entropy(b))) / h_a

"""
conftest.py
EN: Shared helper for the analysis tests: metrics(name) reads metrics/<name>.json. The tests here read the
    published metrics only; they never import analysis/ code (so analysis/lib and db/lib never meet in one
    pytest session) and never run an analysis.
TR: Analiz testleri için ortak yardımcı: metrics(name) metrics/<name>.json'u okur. Buradaki testler yalnız
    yayımlanan metrikleri okur; analysis/ kodunu import etmez (analysis/lib ile db/lib aynı pytest oturumunda
    karşılaşmaz) ve hiçbir analizi koşmaz.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def metrics():
    """
    EN: Returns a reader: metrics("08_residuals") → that script's metrics dict (cached).
    TR: Bir okuyucu döndürür: metrics("08_residuals") → o betiğin metrik sözlüğü (önbellekli).
    """
    cache = {}

    def read(name):
        """EN: The parsed metrics file of a script. / TR: Bir betiğin ayrıştırılmış metrik dosyası."""
        if name not in cache:
            cache[name] = json.loads((ROOT / "metrics" / f"{name}.json").read_text(encoding="utf-8"))
        return cache[name]
    return read

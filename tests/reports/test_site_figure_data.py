"""
test_site_figure_data.py
EN: Every report figure's data reaches the site. The portfolio site draws the report figures itself from
    data/site_data.json; a figure whose data stays out of it can only be shown as the PNG. Each figure of the
    technical and business reports (report_common.build_figures) is drawn once with a view that records what it
    reads (nothing is written: _save only closes the figure), and every read must be something
    builders/build_site_data.py exports:
      - a key of a section the site gets whole (meta, domain, methodology) or of column_labels;
      - a key SITE_KEYS lists for any other section (today: error_drivers for figures 11, 27, 29, 30);
      - a derived report value (report_common.derive) whose source the site gets, listed in FROM_SITE below.
    A new figure, or a figure that starts reading something new, fails here until its data is exported (or the
    derived value is checked and listed). The SHAP report's figures are drawn by analysis/shap/ and are out of
    scope.
TR: Her rapor figürünün verisi siteye gider. Portföy sitesi rapor figürlerini data/site_data.json'dan kendisi
    çiziyor; verisi oraya gitmeyen bir figür yalnız PNG olarak gösterilebilir. Teknik ve iş raporunun her figürü
    (report_common.build_figures), okuduklarını kaydeden bir görünümle bir kez çizilir (hiçbir şey yazılmaz:
    _save yalnız figürü kapatır) ve okunan her şey builders/build_site_data.py'nin gönderdiği bir şey olmalı:
      - siteye bütün giden bir bölümün (meta, domain, methodology) ya da column_labels'ın anahtarı;
      - öteki bir bölüm için SITE_KEYS'te sayılan anahtar (bugün: figür 11, 27, 29, 30 için error_drivers);
      - kaynağı siteye giden türetilmiş bir rapor değeri (report_common.derive), aşağıdaki FROM_SITE'ta sayılı.
    Yeni bir figür ya da yeni bir şey okumaya başlayan figür, verisi gönderilene (ya da türetilmiş değer
    denetlenip sayılana) kadar burada düşer. SHAP raporunun figürlerini analysis/shap/ çizer; kapsam dışı.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "builders"))
import build_site_data as SD  # noqa: E402
from report_lib import report_common as RC  # noqa: E402

FIGURES = sorted(set(RC.TECHNICAL_FIGS) | set(RC.BUSINESS_FIGS))
# EN: derived values (report_common.derive) the figures may read: each comes only from meta / domain, which the
#     site gets whole. Before adding one, check that derive builds it from sections the site gets.
# TR: figürlerin okuyabileceği türetilmiş değerler (report_common.derive): her biri yalnız siteye bütün giden
#     meta / domain'den gelir. Birini eklemeden önce derive'ın onu sitenin aldığı bölümlerden kurduğuna bakın.
FROM_SITE = {"model_mae", "base_mae", "n_dedup", "median"}


class Recorded(dict):
    """
    EN: A dict that notes every key read from it (item, get, membership, iteration) as (name, key) in log.
    TR: Okunan her anahtarı (öğe, get, üyelik, yineleme) log'a (ad, anahtar) olarak yazan bir dict.
    """

    def __init__(self, name, data, log):
        """EN: A copy of data that writes its reads to log. / TR: Okumalarını log'a yazan data kopyası."""
        super().__init__(data)
        self.name, self.log = name, log

    def _note(self, keys):
        """EN: Adds (name, key) for every key to log. / TR: Her anahtar için log'a (ad, anahtar) ekler."""
        self.log.update((self.name, k) for k in keys)

    def __getitem__(self, key):
        """EN: d[key], noted. / TR: d[key], kaydedilir."""
        self._note([key])
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        """EN: d.get(key), noted. / TR: d.get(key), kaydedilir."""
        self._note([key])
        return dict.get(self, key, default)

    def __contains__(self, key):
        """EN: key in d, noted. / TR: key in d, kaydedilir."""
        self._note([key])
        return dict.__contains__(self, key)

    def __iter__(self):
        """EN: Iteration reads every key. / TR: Yineleme her anahtarı okur."""
        self._note(dict.keys(self))
        return dict.__iter__(self)

    def keys(self):
        """EN: Every key, all noted. / TR: Her anahtar, hepsi kaydedilir."""
        self._note(dict.keys(self))
        return dict.keys(self)

    def values(self):
        """EN: Every value, all keys noted. / TR: Her değer, bütün anahtarlar kaydedilir."""
        self._note(dict.keys(self))
        return dict.values(self)

    def items(self):
        """EN: Every pair, all keys noted. / TR: Her çift, bütün anahtarlar kaydedilir."""
        self._note(dict.keys(self))
        return dict.items(self)


def exported(section, key):
    """
    EN: True if build_site_data sends section.key to the site.
    TR: build_site_data section.key'i siteye gönderiyorsa True.
    """
    if section in SD.WHOLE_SECTIONS or section == "column_labels":
        return True
    return key in SD.SITE_KEYS.get(section, ())


@pytest.fixture(scope="module")
def view():
    """
    EN: The real report view and its derived values, as the builders make them.
    TR: Derleyicilerin kurduğu gibi gerçek rapor görünümü ve türetilmiş değerleri.
    """
    d = RC.load_report_view()
    return d, RC.derive(d)


@pytest.fixture(autouse=True)
def no_files(monkeypatch):
    """
    EN: Figures are drawn but never written: _save only closes them.
    TR: Figürler çizilir ama yazılmaz: _save yalnız kapatır.
    """
    def close_only(fig, name):
        """EN: Stands in for _save: closes the figure, writes nothing. / TR: _save yerine: figürü kapatır, hiçbir şey yazmaz."""
        plt.close(fig)
        return f"{name}.png"
    monkeypatch.setattr(RC, "_save", close_only)


def test_site_sections_are_known():
    """
    EN: Every partly exported section is a real metrics section, and the whole ones are not listed twice.
    TR: Kısmen gönderilen her bölüm gerçek bir metrik bölümü; bütün gidenler iki kez sayılmıyor.
    """
    partial = set(SD.SITE_KEYS) - set(SD.WHOLE_SECTIONS)
    assert partial <= set(RC.MV.SECTIONS), partial
    assert set(SD.WHOLE_SECTIONS) <= set(RC.MV.SECTIONS)


@pytest.mark.parametrize("no", FIGURES)
def test_figure_data_reaches_the_site(view, no):
    """
    EN: Figure no reads only what the site gets (see the module docstring).
    TR: no numaralı figür yalnız sitenin aldığını okur (modül açıklamasına bakın).
    """
    d, v = view
    log, vlog = set(), set()
    d_rec = {sec: Recorded(sec, val, log) if isinstance(val, dict) else val for sec, val in d.items()}
    v_rec = Recorded("derived", v, vlog)
    dict.__setitem__(v_rec, "ed", d_rec["error_drivers"])   # EN: as the builders set it | TR: derleyicilerdeki gibi
    drawn = RC.build_figures(d_rec, v_rec, "tr", only=[no])
    assert no in drawn, f"figure {no} was not drawn | figür çizilmedi"
    missing = sorted(f"{sec}.{key}" for sec, key in log if not exported(sec, key))
    assert not missing, (f"figure {no} reads data the site does not get | figür sitenin almadığı veriyi okuyor: "
                         f"{missing} — add it to SITE_KEYS in builders/build_site_data.py")
    unknown = sorted(k for _, k in vlog if k != "ed" and k not in FROM_SITE)
    assert not unknown, (f"figure {no} reads derived values not checked for the site | denetlenmemiş türetilmiş "
                         f"değer: {unknown} — check derive() builds them from sections the site gets, then add "
                         f"them to FROM_SITE")

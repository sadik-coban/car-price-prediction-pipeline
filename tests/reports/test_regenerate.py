"""
test_regenerate.py
EN: Reports are a pure function of the metrics. The four builders are run into a temp folder (CARDATASYS_OUT)
    and their output must equal what is in the repository: the six markdown reports (line ends aside), every
    figure byte for byte, site_data.json and column_labels.json as JSON. A hand-edited report, a report not
    rebuilt after a metrics change, or an orphan figure fails here. The SHAP figures are drawn by
    analysis/shap/, so they are checked against the list the SHAP metrics publish.
TR: Raporlar metriklerin saf fonksiyonudur. Dört derleyici geçici bir klasöre (CARDATASYS_OUT) koşulur ve
    çıktıları depodakiyle aynı olmalı: altı markdown rapor (satır sonları hariç), her figür bayt bayt,
    site_data.json ve column_labels.json JSON olarak. Elle düzenlenmiş rapor, metrik değiştikten sonra yeniden
    üretilmemiş rapor ya da sahipsiz figür burada düşer. SHAP figürlerini analysis/shap/ çizdiği için onlar SHAP
    metriklerinin yayımladığı listeye göre sınanır.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "builders"))
from report_lib import metrics_view as MV  # noqa: E402

BUILDERS = ["build_site_data.py", "build_technical_report.py", "build_business_report.py", "build_shap_report.py"]
HINT = "rebuild | yeniden üret: python builders/<builder>.py (never edit reports by hand | raporu elle düzenleme)"


@pytest.fixture(scope="module")
def rebuilt(tmp_path_factory):
    """
    EN: Runs the four builders into a temp folder; returns that folder.
    TR: Dört derleyiciyi geçici bir klasöre koşar; o klasörü döndürür.
    """
    out = tmp_path_factory.mktemp("rebuilt")
    env = {**os.environ, "CARDATASYS_OUT": str(out), "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    for builder in BUILDERS:
        r = subprocess.run([sys.executable, str(ROOT / "builders" / builder)], env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        assert r.returncode == 0, f"{builder} failed | düştü:\n{r.stderr[-2000:]}"
    return out


def text(path):
    """EN: File bytes with CRLF folded to LF. / TR: CRLF'i LF'e indirilmiş dosya baytları."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def test_markdown_reports(rebuilt):
    """EN: The six reports equal their rebuild. / TR: Altı rapor yeniden üretilenle aynı."""
    built = sorted(p.name for p in (rebuilt / "reports").glob("*.md"))
    assert built == sorted(p.name for p in (ROOT / "reports").glob("*.md")) and len(built) == 6
    differ = [n for n in built if text(rebuilt / "reports" / n) != text(ROOT / "reports" / n)]
    assert not differ, f"reports differ from their rebuild | yeniden üretilenden farklı: {differ}\n{HINT}"


def test_figures(rebuilt):
    """
    EN: Every builder figure is byte-identical, and the repository holds no figure that no one draws.
    TR: Derleyicinin her figürü bayt bayt aynı ve depoda kimsenin çizmediği figür yok.
    """
    built = {p.name: p for p in (rebuilt / "reports" / "figures").glob("*.png")}
    view = MV.load_view(ROOT)
    shap_figs = set(view["shap"]["figures"] + view["shap_case"]["figures"])
    repo = {p.name for p in (ROOT / "reports" / "figures").glob("*.png")}
    differ = [n for n, p in built.items() if not (ROOT / "reports" / "figures" / n).exists()
              or p.read_bytes() != (ROOT / "reports" / "figures" / n).read_bytes()]
    assert not differ, f"figures differ from their rebuild | yeniden çizilenden farklı: {sorted(differ)}\n{HINT}"
    assert repo == set(built) | shap_figs, (f"orphan or missing figures | sahipsiz ya da eksik figür: "
                                            f"orphan {sorted(repo - set(built) - shap_figs)} · "
                                            f"missing {sorted((set(built) | shap_figs) - repo)}")


def test_site_data(rebuilt):
    """EN: site_data.json and column_labels.json equal their rebuild. / TR: İkisi de yeniden üretilenle aynı."""
    for rel in ("data/site_data.json", "data/serving/column_labels.json"):
        repo_file = ROOT / rel
        if not repo_file.exists():
            pytest.skip(f"{rel} not built in this clone | bu klonda yok (data/ git dışı)")
        assert json.loads((rebuilt / rel).read_text(encoding="utf-8")) == \
            json.loads(repo_file.read_text(encoding="utf-8")), f"{rel} differs | farklı\n{HINT}"

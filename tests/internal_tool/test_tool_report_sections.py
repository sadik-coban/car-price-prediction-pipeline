"""
test_tool_report_sections.py
EN: How the internal tool cuts a report into sections and render blocks, on a small made-up report: the intro is
    section 0, "###" stays inside its "##" section, a figure whose alt text runs over two lines is still found, a
    figure's number is read from its file name, and a single "~" is escaped so Streamlit does not strike text through.
TR: İç aracın bir raporu bölümlere ve çizim bloklarına nasıl ayırdığı, küçük uydurma bir raporda: giriş bölüm 0,
    "###" kendi "##" bölümünde kalır, alt metni iki satıra taşan figür yine bulunur, figür numarası dosya adından
    okunur ve tek "~" kaçışlanır, böylece Streamlit metni üstü çizili göstermez.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from internal_tool import report_sections as rs  # noqa: E402

MD = """# Title

Intro text.

## 1. First

Text ~₺10k and ~₺48k.

![A figure whose alt
runs over two lines](figures/en-08-residuals.png)

### Sub

More.
## 2. Second
![SHAP](figures/en-sh-06-waterfall.png)
"""


def test_sections():
    """EN: Intro = 0; ### stays inside. / TR: Giriş = 0; ### içeride kalır."""
    secs = rs.split_sections(MD)
    assert [s["title"] for s in secs] == ["Title", "1. First", "2. Second"]
    assert "### Sub" in secs[1]["text"] and secs[1]["text"].startswith("## 1. First")
    assert "".join(s["text"] for s in secs) == MD


def test_figures_and_keys():
    """EN: Two-line alt found; keys from file names. / TR: İki satırlık alt bulunur; numara dosya adından."""
    secs = rs.split_sections(MD)
    assert rs.figures(secs[1]["text"]) == [{"alt": "A figure whose alt runs over two lines",
                                           "path": "figures/en-08-residuals.png", "key": "08"}]
    assert [f["key"] for f in rs.figures(secs[2]["text"])] == ["sh-06"]
    assert rs.figure_key("figures/tr-30-engine.png") == "30" and rs.figure_key("figures/other.png") is None


def test_blocks_and_escaping():
    """EN: md / img / md in order; "~" escaped once. / TR: Sırayla md / img / md; "~" bir kez kaçışlanır."""
    kinds = [b[0] for b in rs.blocks(rs.split_sections(MD)[1]["text"])]
    assert kinds == ["md", "img", "md"]
    assert rs.escape_markdown("~₺10k … ~₺48k") == "\\~₺10k … \\~₺48k"
    assert rs.escape_markdown("\\~ kept") == "\\~ kept"


def test_report_path():
    """EN: Known reports and languages only. / TR: Yalnız bilinen rapor ve diller."""
    assert rs.report_path("shap", "tr").name == "shap.tr.md"
    with pytest.raises(ValueError):
        rs.report_path("other", "tr")

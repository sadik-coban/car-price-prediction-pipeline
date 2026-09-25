"""
report_sections.py
EN: Reads a generated report (reports/<report>.<lang>.md) and cuts it into sections for the internal tool: section
    0 is everything before the first "## " heading (title and intro), then one section per "## " heading ("###"
    and deeper stay inside their section). A section's text is split into markdown blocks and figure blocks so the
    page can draw the images with st.image (Streamlit does not resolve the md's relative figure paths). Pure: no
    streamlit import; the reports are only read.
TR: Üretilmiş bir raporu (reports/<rapor>.<dil>.md) okur ve iç araç için bölümlere ayırır: bölüm 0, ilk "## "
    başlığından önceki her şeydir (başlık ve giriş); sonra her "## " başlığı için bir bölüm ("###" ve daha derini
    kendi bölümünde kalır). Bölüm metni markdown ve figür bloklarına bölünür; böylece sayfa görselleri st.image ile
    çizer (Streamlit md'deki göreli figür yollarını çözmez). Saf: streamlit import etmez; raporlar yalnız okunur.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
# EN: report id → display name | TR: rapor kimliği → görünen ad
REPORTS = {"technical": "Teknik rapor", "business": "Karar notu", "shap": "SHAP raporu"}
LANGS = ("tr", "en")
SECTION_START = re.compile(r"(?m)^(?=## )")
# EN: a figure; its alt text may run over two lines | TR: figür; alt metni iki satıra taşabilir
FIGURE = re.compile(r"!\[(?P<alt>.*?)\]\((?P<path>figures/[^)\s]+)\)", re.S)
FIGURE_KEY = re.compile(r"^(?:tr|en)-((?:sh-)?\d+)-")


def report_path(report, lang, reports_dir=REPORTS_DIR):
    """EN: The md file of a report in a language. / TR: Bir raporun bir dildeki md dosyası."""
    if report not in REPORTS or lang not in LANGS:
        raise ValueError(f"unknown report or language | bilinmeyen rapor ya da dil: {report}, {lang}")
    return Path(reports_dir) / f"{report}.{lang}.md"


def split_sections(md):
    """
    EN: The sections of a report's markdown. Returns: [{"index", "title", "text"}], index 0 = the part before the
        first "## " heading; title = the first line without its leading "#" marks.
    TR: Bir raporun markdown'ının bölümleri. Döndürür: [{"index", "title", "text"}]; index 0 = ilk "## "
        başlığından önceki kısım; title = baştaki "#" işaretleri atılmış ilk satır.
    """
    parts = SECTION_START.split(md)
    return [{"index": i, "title": part.splitlines()[0].lstrip("#").strip() if part.strip() else "",
             "text": part} for i, part in enumerate(parts)]


def read_sections(report, lang, reports_dir=REPORTS_DIR):
    """EN: split_sections of a report file. / TR: Bir rapor dosyasının split_sections'ı."""
    return split_sections(report_path(report, lang, reports_dir).read_text(encoding="utf-8"))


def figure_key(path):
    """
    EN: The figure number in a figure file name: "figures/tr-08-x.png" → "08", "en-sh-06-y.png" → "sh-06"; None if
        the name does not follow the pattern.
    TR: Figür dosya adındaki numara: "figures/tr-08-x.png" → "08", "en-sh-06-y.png" → "sh-06"; ad kalıba uymazsa None.
    """
    m = FIGURE_KEY.match(Path(path).name)
    return m.group(1) if m else None


def figures(text):
    """
    EN: The figures of a section, in order. Returns: [{"alt", "path", "key"}] (alt on one line).
    TR: Bir bölümün figürleri, sırasıyla. Döndürür: [{"alt", "path", "key"}] (alt tek satırda).
    """
    return [{"alt": " ".join(m["alt"].split()), "path": m["path"], "key": figure_key(m["path"])}
            for m in FIGURE.finditer(text)]


def escape_markdown(text):
    """
    EN: Escapes what Streamlit's markdown would misread: a single "~" (two of them in one paragraph render as
        strikethrough, e.g. "~₺10k … ~₺48k").
    TR: Streamlit markdown'ının yanlış okuyacağını kaçışlar: tek "~" (bir paragrafta iki tanesi üstü çizili olur,
        ör. "~₺10k … ~₺48k").
    """
    return re.sub(r"(?<!\\)~", r"\\~", text)


def blocks(text):
    """
    EN: A section's text as render blocks: ("md", markdown) and ("img", alt, path relative to reports/), in order;
        empty markdown between figures is dropped.
    TR: Bir bölümün metni, çizim blokları olarak: ("md", markdown) ve ("img", alt, reports/'a göreli yol), sırasıyla;
        figürler arasındaki boş markdown atılır.
    """
    out, pos = [], 0
    for m in FIGURE.finditer(text):
        if text[pos:m.start()].strip():
            out.append(("md", escape_markdown(text[pos:m.start()])))
        out.append(("img", " ".join(m["alt"].split()), m["path"]))
        pos = m.end()
    if text[pos:].strip():
        out.append(("md", escape_markdown(text[pos:])))
    return out

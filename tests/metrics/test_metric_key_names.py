"""
test_metric_key_names.py
EN: The mixed Turkish/English metric names must not come back. For every group marked done in
    docs/metric-key-renames.json: its metrics files have no retired (old) key outside the data-valued subtrees and no
    old enum value at a map "values" site, and no code, card or doc still quotes a dotted metric path of that group
    with an old name. With no group done this passes trivially; once every group is done it covers everything.
TR: Karışık Türkçe/İngilizce metrik adları geri gelmemeli. docs/metric-key-renames.json'da bitti işaretli her grup
    için: metrik dosyalarında veri değerli alt ağaçlar dışında emekli (eski) anahtar ve eşlemenin "values"
    yerlerinde eski değer yok; hiçbir kod, kart ya da belge o grubun noktalı metrik yolunu eski adla anmıyor. Hiçbir
    grup bitmemişken kendiliğinden geçer; bütün gruplar bitince her şeyi kapsar.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import metric_renames as MR  # noqa: E402

M = MR.load_map()
DONE = MR.done_groups(M)
RETIRED = MR.retired_names(M, DONE)
DOCS = MR.metrics_docs(ROOT / "metrics")
# EN: text that may quote metric paths; the map, its tool and their tests hold old names on purpose
# TR: metrik yolu anabilecek metinler; eşleme, aracı ve testleri eski adları bilerek taşır
TEXT_GLOBS = ["builders/**/*.py", "analysis/**/*.py", "tests/**/*.py", "tools/*.py", "db/**/*.py",
              "analysis/cards/*.json", "docs/*.md", "README*.md", "db_plan.md", "internal_tool/**/*.py"]
ON_PURPOSE = {"tools/metric_renames.py", "tests/metrics/test_metric_renames.py", "tests/metrics/test_metric_key_names.py",
              "docs/metric-key-renames.json"}


@pytest.mark.parametrize("name", sorted(DOCS))
def test_done_files_use_new_names(name):
    """
    EN: A metrics file of a done group has no retired key and no old enum value.
    TR: Bitmiş bir grubun metrik dosyasında emekli anahtar ve eski değer yok.
    """
    if MR.group_of(name, M) not in DONE:
        return
    assert not MR.key_violations(DOCS[name], M, RETIRED), MR.key_violations(DOCS[name], M, RETIRED)[:20]
    assert not MR.value_violations(DOCS[name], M, DONE), MR.value_violations(DOCS[name], M, DONE)[:20]


def test_no_stale_dotted_paths():
    """
    EN: No code, card or doc quotes a dotted metric path of a done group with an old name.
    TR: Hiçbir kod, kart ya da belge bitmiş bir grubun noktalı metrik yolunu eski adla anmıyor.
    """
    if not DONE:
        return
    owners = MR.top_owners(DOCS, M)
    stale = []
    for pattern in TEXT_GLOBS:
        for p in sorted(ROOT.glob(pattern)):
            rel = p.relative_to(ROOT).as_posix()
            if rel in ON_PURPOSE or "__pycache__" in rel:
                continue
            stale += [f"{rel}: {hit}" for hit in MR.stale_refs(p.read_text(encoding="utf-8"), M, DONE, owners, RETIRED)]
    assert not stale, "old metric paths | eski metrik yolları:\n" + "\n".join(stale[:40])

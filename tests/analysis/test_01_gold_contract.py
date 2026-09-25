"""
test_01_gold_contract.py
EN: Invariants of metrics/01_gold_contract.json, the numbers of the technical report's gold subsection (§1). The
    metrics must describe the rules actually in db/gold_rules.json (same groups, order, values, column counts, the
    dropped column), the counts must add up and stay within what the table can hold, and the script must be in
    run_all.ORDER. Reads files only; nothing is run.
TR: metrics/01_gold_contract.json'un değişmezleri; teknik raporun gold alt bölümünün (§1) sayıları. Metrikler
    db/gold_rules.json'daki kuralları anlatmalı (aynı gruplar, sıra, değerler, kolon sayıları, alınmayan kolon),
    sayımlar toplanmalı ve tablonun alabileceğini aşmamalı, betik run_all.ORDER'da olmalı. Yalnız dosya okur;
    hiçbir şey koşmaz.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "metrics" / "01_gold_contract.json"
RULES = json.loads((ROOT / "db" / "gold_rules.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gold():
    """EN: The published gold section. / TR: Yayımlanan gold bölümü."""
    return json.loads(METRICS.read_text(encoding="utf-8"))["error_drivers"]["gold_contract"]


def test_groups_are_the_rule_file(gold):
    """
    EN: One group per fill rule, in the file's order, with its value and column count; the dropped column too.
    TR: Her doldurma kuralı için bir grup, dosyanın sırasıyla, değeri ve kolon sayısıyla; alınmayan kolon da.
    """
    assert [(g["name"], g["value"], g["n_columns"]) for g in gold["groups"]] == \
        [(r["name"], r["value"], len(r["columns"])) for r in RULES["fill"]]
    assert gold["dropped"] == [d["column"] for d in RULES["drop"]]
    assert gold["gold_columns"] == gold["semi_raw_columns"] - len(RULES["drop"])


def test_counts_add_up(gold):
    """
    EN: The total is the sum of the groups; a group fills at most columns × rows cells, on at most every row,
        and a row it touches holds at least one of its cells; empty descriptions fit in the table.
    TR: Toplam grupların toplamı; bir grup en çok kolon × satır hücre doldurur, en çok her satırda, dokunduğu her
        satırda en az bir hücresi vardır; boş açıklamalar tabloya sığar.
    """
    n = gold["table_rows"]
    assert gold["total_cells"] == sum(g["filled_cells"] for g in gold["groups"])
    for g in gold["groups"]:
        assert 0 <= g["affected_rows"] <= n
        assert g["affected_rows"] <= g["filled_cells"] <= g["n_columns"] * g["affected_rows"], g["name"]
    assert 0 <= gold["empty_description_rows"] <= n


def test_row_rule(gold):
    """
    EN: The row rules are the rule file's; the semi-raw rows are gold's rows plus the dropped ones; a dropped
        listing has at least one dropped row.
    TR: Satır kuralları kural dosyasınınki; yarı ham satırlar gold'un satırları artı düşenler; düşen her ilanın en
        az bir düşen satırı var.
    """
    assert [(r["column"], r["values"]) for r in gold["dropped_rows"]] == \
        [(r["column"], r["values"]) for r in RULES.get("drop_rows", [])]
    assert gold["semi_raw_rows"] == gold["table_rows"] + sum(r["rows"] for r in gold["dropped_rows"])
    assert all(0 < r["listings"] <= r["rows"] for r in gold["dropped_rows"])


def test_panel_flags_fill_whole_panels(gold):
    """
    EN: An unspecified panel leaves all three of its flags NULL, so the panel group fills a multiple of 3 cells.
    TR: Belirtilmemiş panelin üç bayrağı birden NULL; panel grubu 3'ün katı hücre doldurur.
    """
    panels = next(g for g in gold["groups"] if g["name"] == "panel_flags")
    assert panels["filled_cells"] % 3 == 0


def test_in_run_order():
    """EN: run_all runs the script, after the DB is read and before 02. / TR: run_all betiği koşar, 02'den önce."""
    spec = importlib.util.spec_from_file_location("run_all", ROOT / "analysis" / "run_all.py")
    run_all = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_all)
    assert run_all.ORDER.index("01_gold_contract.py") < run_all.ORDER.index("02_missingness.py")

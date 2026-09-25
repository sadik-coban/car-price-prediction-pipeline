"""
test_plans.py
EN: The pre-registration gate (tools/new_plan.py). Every real plan under plans/ must be valid (plan_problems); a
    resolved hypothesis needs evidence that exists (a collected test or a metrics key) and its required checks
    among it; an untested hypothesis does NOT fail the gate (the Stop hook only warns). The validator's rules and
    the new / lock flow are tested on made-up plans in a temp folder.
TR: Ön kayıt kapısı (tools/new_plan.py). plans/ altındaki her gerçek plan geçerli olmalı (plan_problems);
    sonuçlanmış hipotez var olan kanıt (toplanan bir test ya da bir metrik anahtarı) ve zorunlu kontrollerini
    kanıtında ister; sınanmamış hipotez kapıyı DÜŞÜRMEZ (Stop hook'u yalnız uyarır). Doğrulayıcının kuralları ve
    new / lock akışı geçici klasörde uydurma planlarla sınanır.
"""
import copy
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import new_plan as NP  # noqa: E402

REAL = sorted((ROOT / "plans").glob(f"*/{NP.PLAN_FILE}"))
METRIC = "metric:08_residuals:report.err_n"          # a key that exists | var olan bir anahtar


@pytest.fixture(scope="module")
def collected():
    """EN: Node ids of the whole suite (only when real plans exist). / TR: Bütün takımın düğüm kimlikleri."""
    if not REAL:
        return set()
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {line.strip().split("[")[0].replace("\\", "/") for line in r.stdout.splitlines() if "::" in line}


@pytest.mark.parametrize("path", REAL, ids=lambda p: p.parent.name)
def test_real_plans_are_valid(path, collected):
    """EN: Every plan under plans/ passes the validator. / TR: plans/ altındaki her plan doğrulayıcıdan geçer."""
    plan = json.loads(path.read_text(encoding="utf-8"))
    assert plan["id"] == path.parent.name
    problems = NP.plan_problems(plan, ROOT, collected)
    assert not problems, f"{path.parent.name}: " + "; ".join(problems)


def locked_plan(**hyp):
    """EN: A complete locked plan with one hypothesis (fields overridable). / TR: Tek hipotezli tam kilitli plan."""
    plan = NP.template("demo", "Soru?", "Question?", date(2026, 9, 24))
    plan.update(data_slice="load_clean()", split="kfold_oof", target="OOF error",
                comparisons={"count": 1, "correction": "none"}, status="locked", approved_by="owner",
                approved_at="2026-09-24")
    plan["hypotheses"][0].update(statement={"tr": "Hata %10'un altında", "en": "Error below 10%"},
                                 accept_if="median < 10", reject_if="median >= 10", required_checks=[METRIC])
    plan["hypotheses"][0].update(hyp)
    return plan


def test_template_draft_is_valid():
    """EN: A fresh draft is valid (TODOs are allowed before the lock). / TR: Taze taslak geçerli (TODO serbest)."""
    assert NP.plan_problems(NP.template("demo", "Soru?", "Question?")) == []


def test_locked_plan_untested_is_valid():
    """EN: Untested hypotheses do not fail the gate. / TR: Sınanmamış hipotez kapıyı düşürmez."""
    plan = locked_plan()
    assert NP.plan_problems(plan) == [] and NP.untested(plan) == ["H1"]


@pytest.mark.parametrize("change, message", [
    (lambda p: p["hypotheses"][0].update(status="confirmed"), "needs evidence"),
    (lambda p: p["hypotheses"][0].update(status="confirmed", evidence=["metric:08_residuals:no.such.key"]),
     "does not exist"),
    (lambda p: p["hypotheses"][0].update(status="refuted", evidence=["test:tests/x.py::test_y"]), "does not exist"),
    (lambda p: p["hypotheses"][0].update(status="inconclusive", evidence=["metric:07_final_model:meta.repro"]),
     "required checks not in the evidence"),
    (lambda p: p.update(target="TODO: later"), "no TODO"),
    (lambda p: p.update(approved_by=None), "approved_by"),
    (lambda p: p.update(status="draft") or p["hypotheses"][0].update(status="confirmed", evidence=[METRIC]),
     "only a locked plan"),
    (lambda p: p["hypotheses"].append(copy.deepcopy(p["hypotheses"][0])), "unique"),
    (lambda p: p["hypotheses"][0].update(status="maybe"), "status must be"),
])
def test_validator_catches(change, message):
    """EN: Each broken plan is caught with its reason. / TR: Her bozuk plan gerekçesiyle yakalanır."""
    plan = locked_plan()
    change(plan)
    assert any(message in m for m in NP.plan_problems(plan)), NP.plan_problems(plan)


def test_resolved_with_evidence_is_valid():
    """EN: A confirmed hypothesis with existing evidence passes. / TR: Var olan kanıtlı onaylı hipotez geçer."""
    assert NP.plan_problems(locked_plan(status="confirmed", evidence=[METRIC])) == []


def test_new_and_lock_flow(tmp_path, capsys):
    """
    EN: new writes a draft; lock refuses it while incomplete; once filled in, lock sets status and approval; a
        second lock and a second new are refused.
    TR: new taslak yazar; tam değilken lock reddeder; doldurulunca lock durumu ve onayı yazar; ikinci lock ve
        ikinci new reddedilir.
    """
    root = str(tmp_path)
    assert NP.main(["new", "demo", "Soru?", "Question?", "--root", root]) == 0
    assert NP.main(["lock", "demo", "--by", "owner", "--root", root]) == 1
    path = NP.plan_path("demo", tmp_path)
    filled = locked_plan()
    filled.update(status="draft", approved_by=None, approved_at=None)
    path.write_text(json.dumps(filled), encoding="utf-8")
    assert NP.main(["lock", "demo", "--by", "owner", "--root", root]) == 0
    plan = json.loads(path.read_text(encoding="utf-8"))
    assert plan["status"] == "locked" and plan["approved_by"] == "owner" and plan["approved_at"]
    assert NP.main(["lock", "demo", "--by", "owner", "--root", root]) == 1
    assert NP.main(["new", "demo", "a", "b", "--root", root]) == 1
    assert NP.main(["new", "Bad Id", "a", "b", "--root", root]) == 1

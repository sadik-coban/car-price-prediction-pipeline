"""
new_plan.py
EN: Pre-registration of a new analysis question: the plan is written and locked BEFORE the analysis, so the
    question, the data slice, the split and the accept / reject criteria cannot drift after the results are seen.
      new  <id> "soru" "question"   writes plans/<id>/analysis_plan.json as a draft (everything else TODO);
      lock <id> --by "<name>"       checks the draft is complete and locks it (status, approved_by, approved_at).
    Lock only after the owner explicitly approves the plan. Once locked, only hypotheses[].status and
    hypotheses[].evidence may change (the Claude Code hook .claude/hooks/guard_plan.py enforces it); a deviation
    needs a new plan and a note in docs/decisions. plan_problems() is the one validator; tests/plans/ runs it on
    every plan (a resolved hypothesis needs evidence that exists; an untested one only warns).
TR: Yeni bir analiz sorusunun ön kaydı: plan analizden ÖNCE yazılır ve kilitlenir; böylece soru, veri kesiti,
    bölme ve kabul / ret ölçütleri sonuçlar görüldükten sonra kaymaz.
      new  <id> "soru" "question"   plans/<id>/analysis_plan.json'u taslak olarak yazar (geri kalan her şey TODO);
      lock <id> --by "<ad>"         taslağın tam olduğunu sınar ve kilitler (status, approved_by, approved_at).
    Yalnız kullanıcı planı açıkça onayladıktan sonra kilitlenir. Kilitlenince yalnız hypotheses[].status ve
    hypotheses[].evidence değişebilir (Claude Code hook'u .claude/hooks/guard_plan.py zorlar); sapma yeni bir plan
    ve docs/decisions'ta bir not ister. plan_problems() tek doğrulayıcı; tests/plans/ onu her planda koşar
    (sonuçlanmış hipotez var olan kanıt ister; untested yalnız uyarır).
Run / Koşum:
    python tools/new_plan.py new price-history "İlan fiyatları taramalar arasında nasıl değişiyor?" "How do asking prices change?"
    python tools/new_plan.py lock price-history --by "sadik-coban"
"""
import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLANS = ROOT / "plans"
PLAN_FILE = "analysis_plan.json"
PLAN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
H_STATUS = ("untested", "confirmed", "refuted", "inconclusive")
RESOLVED = ("confirmed", "refuted", "inconclusive")
P_STATUS = ("draft", "locked")
TEXT = ("question",)
FIELDS = ("data_slice", "split", "target")


def template(plan_id, q_tr, q_en, today=None):
    """
    EN: A new draft plan: the question filled in, one hypothesis to fill, everything else TODO.
    TR: Yeni bir taslak plan: soru dolu, doldurulacak bir hipotez, geri kalan her şey TODO.
    """
    return {"id": plan_id, "created": (today or date.today()).isoformat(),
            "question": {"tr": q_tr, "en": q_en},
            "data_slice": "TODO: which rows (e.g. load_clean(), all snapshots, a segment)",
            "split": "TODO: none / kfold_oof / temporal / ...",
            "target": "TODO: what is measured",
            "comparisons": {"count": 1, "correction": "TODO: none / holm / bonferroni"},
            "hypotheses": [{"id": "H1", "statement": {"tr": "TODO", "en": "TODO"},
                            "accept_if": "TODO", "reject_if": "TODO", "required_checks": [],
                            "status": "untested", "evidence": []}],
            "out_of_scope": [],
            "status": "draft", "approved_by": None, "approved_at": None}


def evidence_exists(ref, root=ROOT, collected=None):
    """
    EN: Whether an evidence / check reference points at something real:
          "test:tests/<file>.py::<name>"      a test pytest collects (collected: set of node ids);
          "metric:<script>:<dotted.key>"      a key in metrics/<script>.json.
    TR: Bir kanıt / kontrol referansının gerçek bir şeyi gösterip göstermediği:
          "test:tests/<dosya>.py::<ad>"       pytest'in topladığı bir test (collected: düğüm kimlikleri);
          "metric:<betik>:<noktalı.anahtar>"  metrics/<betik>.json'daki bir anahtar.
    """
    kind, _, rest = str(ref).partition(":")
    if kind == "test":
        return collected is not None and rest in collected
    if kind == "metric":
        script, _, key = rest.partition(":")
        path = Path(root) / "metrics" / f"{script}.json"
        if not key or not path.exists():
            return False
        o = json.loads(path.read_text(encoding="utf-8"))
        for k in key.split("."):
            if not isinstance(o, dict) or k not in o:
                return False
            o = o[k]
        return True
    return False


def plan_problems(plan, root=ROOT, collected=None, for_lock=False):
    """
    EN: What is wrong in a plan (empty list = valid). A locked plan (or one about to be locked) must be complete:
        no TODO, every hypothesis with accept / reject criteria and at least one required check, approval fields.
        A resolved hypothesis needs evidence, every required check among it, and every reference must exist.
    TR: Bir planda yanlış olan (boş liste = geçerli). Kilitli (ya da kilitlenecek) plan tam olmalı: TODO yok, her
        hipotezin kabul / ret ölçütü ve en az bir zorunlu kontrolü var, onay alanları dolu. Sonuçlanmış hipotez
        kanıt ister; her zorunlu kontrol kanıtta olmalı ve her referans var olmalı.
    """
    p = []
    if plan.get("status") not in P_STATUS:
        p.append(f"status must be one of {P_STATUS}")
    if not (isinstance(plan.get("question"), dict) and plan["question"].get("tr") and plan["question"].get("en")):
        p.append("question: tr and en needed")
    hyps = plan.get("hypotheses") or []
    if not hyps:
        p.append("at least one hypothesis")
    ids = [h.get("id") for h in hyps]
    if len(set(ids)) != len(ids):
        p.append("hypothesis ids must be unique")
    locked = plan.get("status") == "locked" or for_lock
    if locked:
        if "TODO" in json.dumps({k: v for k, v in plan.items() if k not in ("approved_by", "approved_at")},
                                ensure_ascii=False):
            p.append("a locked plan has no TODO left")
        if plan.get("status") == "locked" and not (plan.get("approved_by") and plan.get("approved_at")):
            p.append("a locked plan needs approved_by and approved_at")
        for f in FIELDS:
            if not str(plan.get(f) or "").strip():
                p.append(f"{f} is empty")
    for h in hyps:
        hid = h.get("id")
        if h.get("status") not in H_STATUS:
            p.append(f"{hid}: status must be one of {H_STATUS}")
            continue
        if locked and not (h.get("accept_if") and h.get("reject_if") and h.get("required_checks")):
            p.append(f"{hid}: accept_if, reject_if and at least one required check needed")
        refs = list(h.get("required_checks") or []) + list(h.get("evidence") or [])
        p += [f"{hid}: reference does not exist: {r}" for r in refs if not evidence_exists(r, root, collected)
              and (h.get("status") in RESOLVED or r in (h.get("evidence") or []))]
        if h.get("status") in RESOLVED:
            if plan.get("status") != "locked":
                p.append(f"{hid}: only a locked plan can resolve a hypothesis")
            if not h.get("evidence"):
                p.append(f"{hid}: {h['status']} needs evidence")
            missing = [c for c in h.get("required_checks") or [] if c not in (h.get("evidence") or [])]
            if missing:
                p.append(f"{hid}: required checks not in the evidence: {missing}")
    return p


def untested(plan):
    """EN: Ids of the untested hypotheses of a locked plan. / TR: Kilitli bir planın sınanmamış hipotezleri."""
    if plan.get("status") != "locked":
        return []
    return [h.get("id") for h in plan.get("hypotheses") or [] if h.get("status") == "untested"]


def plan_path(plan_id, root=ROOT):
    """EN: plans/<id>/analysis_plan.json under root. / TR: root altında plans/<id>/analysis_plan.json."""
    return Path(root) / "plans" / plan_id / PLAN_FILE


def main(argv=None):
    """
    EN: Command line (new / lock). Returns: 0 on success, 1 on a refusal (the reason goes to stderr).
    TR: Komut satırı (new / lock). Döndürür: başarıda 0, retde 1 (sebep stderr'e).
    """
    ap = argparse.ArgumentParser(description="Pre-registered analysis plans | ön kayıtlı analiz planları")
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("id")
    n.add_argument("question_tr")
    n.add_argument("question_en")
    lk = sub.add_parser("lock")
    lk.add_argument("id")
    lk.add_argument("--by", required=True, help="who approved the plan | planı kim onayladı")
    for s in (n, lk):
        s.add_argument("--root", default=str(ROOT))
    args = ap.parse_args(argv)
    path = plan_path(args.id, args.root)
    if not PLAN_ID.match(args.id):
        print(f"FAILED: plan id must be lower-case, digits, - or _ | geçersiz id: {args.id}", file=sys.stderr)
        return 1
    if args.cmd == "new":
        if path.exists():
            print(f"FAILED: already exists | zaten var: {path}", file=sys.stderr)
            return 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(template(args.id, args.question_tr, args.question_en), ensure_ascii=False,
                                   indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"written | yazıldı: {path}\nfill it in, show it to the owner, lock only after their approval | "
              f"doldur, kullanıcıya göster, yalnız onayından sonra kilitle")
        return 0
    if not path.exists():
        print(f"FAILED: no such plan | plan yok: {path}", file=sys.stderr)
        return 1
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("status") == "locked":
        print("FAILED: already locked | zaten kilitli", file=sys.stderr)
        return 1
    problems = plan_problems(plan, args.root, for_lock=True)
    problems = [m for m in problems if "reference does not exist" not in m]   # checks may be written after the lock
    if problems:
        print("FAILED: the plan is not complete | plan tam değil:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    plan.update(status="locked", approved_by=args.by, approved_at=date.today().isoformat())
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"locked | kilitlendi: {path} (by {args.by})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

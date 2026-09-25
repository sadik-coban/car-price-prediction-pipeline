# Ön kayıtlı analiz planları · Pre-registered analysis plans

Yeni bir analiz sorusu, analiz yazılmadan **önce** burada plan olarak yazılır ve kullanıcı onaylayınca
kilitlenir. Böylece soru, veri kesiti, bölme ve kabul / ret ölçütleri sonuçlar görüldükten sonra kaymaz.

```
python tools/new_plan.py new <id> "soru" "question"     # plans/<id>/analysis_plan.json, taslak
# doldur → kullanıcıya göster → açık onaydan sonra:
python tools/new_plan.py lock <id> --by "<onaylayan>"
```

- **Taslak** serbestçe değişir. **Kilitli** planda yalnız `hypotheses[].status` ve `hypotheses[].evidence`
  değişebilir; geri kalan her değişikliği Claude Code hook'u reddeder (`.claude/hooks/guard_plan.py`). Sapma
  gerekirse yeni bir plan açılır ve `docs/decisions.md`'ye not düşülür.
- **Hipotez durumu:** `untested` · `confirmed` · `refuted` · `inconclusive`. Üç sonucun üçü de geçerli.
  Sonuçlanmış hipotez kanıt ister ve her zorunlu kontrolü (`required_checks`) kanıtında bulunmalı.
- **Kanıt ve kontrol biçimi:**
  - `test:tests/<dosya>.py::<test>`: pytest'in topladığı bir test (kapıda koşar, kırmızıysa kapı düşer);
  - `metric:<betik>:<noktalı.anahtar>`: `metrics/<betik>.json`'daki bir anahtar.
- **Kapı** (`tests/plans/test_plans.py`, doğrulayıcı `tools/new_plan.py::plan_problems`):
  - her plan geçerli;
  - kilitli planda TODO yok, onay alanları dolu;
  - kanıt gerçekten var.
  Sınanmamış hipotez kapıyı düşürmez, Stop hook'u her tur sonunda **uyarır**.
- Plan dışı fikirler `backlog/ideas.md`'ye yazılır; o turda uygulanmaz.

---

A new analysis question is written here as a plan **before** the analysis and locked once the owner approves
it, so the question, data slice, split and accept / reject criteria cannot drift after the results are seen.

- A **draft** changes freely. In a **locked** plan only `hypotheses[].status` and `hypotheses[].evidence` may
  change; the Claude Code hook (`.claude/hooks/guard_plan.py`) denies anything else. A deviation needs a new
  plan and a note in `docs/decisions.en.md`.
- **Hypothesis status:** `untested` · `confirmed` · `refuted` · `inconclusive`; all three outcomes are valid. A
  resolved hypothesis needs evidence, with every required check among it.
- **Evidence and checks:** `test:tests/<file>.py::<test>` (a collected test; it runs in the gate) or
  `metric:<script>:<dotted.key>` (a key in `metrics/<script>.json`).
- **Gate** (`tests/plans/test_plans.py`, validator `tools/new_plan.py::plan_problems`):
  - every plan is valid;
  - a locked plan has no TODO and its approval fields are filled;
  - the evidence exists.
  An untested hypothesis does not fail the gate; the Stop hook **warns** at the end of every turn.
- Ideas outside the plan go to `backlog/ideas.md` and are not done in that turn.

"""
snapshot_metrics.py
EN: The metrics baseline of the verify gate. The analysis chain is deterministic on frozen data, so every metric
    must stay exactly what it was, except a short list of documented exemptions (tests/baselines/exemptions.json:
    run time stamps, a wall-clock duration, the native CatBoost SHAP attribution that flips between two states).
    Two files are kept, both generated from metrics/*.json, never typed by hand:
      - metrics_shape.json: every key path and its JSON type (a key added, dropped, renamed or retyped shows up);
      - metrics_fingerprint.json: every scalar value, and length + sha256 of every list.
    A difference is either a mistake or an intended change; an intended change is taken into the baseline only
    with a reason (--accept), and every acceptance is logged in tests/baselines/accept_log.jsonl.
TR: Verify kapısının metrik referansı. Analiz zinciri donuk veride deterministik; bu yüzden her metrik aynen
    kalmalı, yalnız belgelenmiş kısa bir istisna listesi hariç (tests/baselines/exemptions.json: koşum zaman
    damgaları, bir süre ölçümü, iki durum arasında gidip gelen native CatBoost SHAP atfı). İki dosya tutulur,
    ikisi de metrics/*.json'dan üretilir, elle yazılmaz:
      - metrics_shape.json: her anahtar yolu ve JSON tipi (eklenen, düşen, adı ya da tipi değişen anahtar görünür);
      - metrics_fingerprint.json: her skaler değer, her listenin uzunluğu + sha256'sı.
    Bir fark ya hatadır ya bilinçli değişikliktir; bilinçli değişiklik referansa yalnız gerekçeyle girer
    (--accept) ve her onay tests/baselines/accept_log.jsonl'a yazılır.
Run / Koşum:
    python tools/snapshot_metrics.py --diff                       # differences | farklar
    python tools/snapshot_metrics.py --accept "<reason | gerekçe>"
"""
import argparse
import fnmatch
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "metrics"
BASELINE_DIR = ROOT / "tests" / "baselines"
SHAPE_FILE = BASELINE_DIR / "metrics_shape.json"
FINGERPRINT_FILE = BASELINE_DIR / "metrics_fingerprint.json"
EXEMPTIONS_FILE = BASELINE_DIR / "exemptions.json"
LOG_FILE = BASELINE_DIR / "accept_log.jsonl"


def metric_files(metrics_dir=METRICS_DIR):
    """
    EN: The metrics files by name ("01_dedup_leakage", "shap/02_oof_shap") → path.
    TR: Metrik dosyaları, ada göre ("01_dedup_leakage", "shap/02_oof_shap") → yol.
    """
    return {p.relative_to(metrics_dir).with_suffix("").as_posix(): p for p in sorted(metrics_dir.rglob("*.json"))}


def json_type(value):
    """
    EN: The JSON type name of a value (bool is checked before int, since bool is an int in Python).
    TR: Bir değerin JSON tip adı (Python'da bool bir int olduğu için önce bool bakılır).
    """
    if value is None:
        return "null"
    for t, name in ((bool, "bool"), (int, "int"), (float, "float"), (str, "str"), (list, "list"), (dict, "dict")):
        if isinstance(value, t):
            return name
    raise TypeError(f"not a JSON value | JSON değeri değil: {type(value).__name__}")


def leaves(obj, prefix=""):
    """
    EN: Flattens a metrics document into {dotted path: (type, fingerprint)}. Dicts are walked (an empty dict is a
        leaf); a list is one leaf fingerprinted by its length and the sha256 of its canonical JSON; a scalar is
        its own fingerprint.
    TR: Bir metrik belgesini {noktalı yol: (tip, parmak izi)} olarak düzler. Dict'lerin içine girilir (boş dict
        yapraktır); liste tek yapraktır, uzunluğu ve kanonik JSON'unun sha256'sıyla; skaler kendi parmak izidir.
    """
    out = {}
    if isinstance(obj, dict) and obj:
        for key, value in obj.items():
            out.update(leaves(value, f"{prefix}.{key}" if prefix else key))
        return out
    kind = json_type(obj)
    if kind == "list":
        canon = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        fingerprint = {"len": len(obj), "sha256": hashlib.sha256(canon.encode("utf-8")).hexdigest()}
    elif kind == "dict":
        fingerprint = {}
    else:
        fingerprint = obj
    out[prefix] = (kind, fingerprint)
    return out


def snapshot(metrics_dir=METRICS_DIR):
    """
    EN: The current shape and fingerprint of every metrics file. Returns: (shape, fingerprint) dicts by file name.
    TR: Her metrik dosyasının güncel şekli ve parmak izi. Döndürür: dosya adına göre (shape, fingerprint).
    """
    shape, fingerprint = {}, {}
    for name, path in metric_files(metrics_dir).items():
        flat = leaves(json.loads(path.read_text(encoding="utf-8")))
        shape[name] = {k: t for k, (t, _) in flat.items()}
        fingerprint[name] = {k: f for k, (_, f) in flat.items()}
    return shape, fingerprint


def load_exemptions(path=EXEMPTIONS_FILE):
    """
    EN: The documented exemptions: [{"file": glob, "path": glob, "reason": {"en", "tr"}}].
    TR: Belgelenmiş istisnalar: [{"file": glob, "path": glob, "reason": {"en", "tr"}}].
    """
    return json.loads(path.read_text(encoding="utf-8"))["exemptions"]


def is_exempt(name, key, exemptions):
    """
    EN: True if (file, key path) matches an exemption; a pattern also covers everything below it.
    TR: (dosya, anahtar yolu) bir istisnaya uyuyorsa True; desen altındaki her şeyi de kapsar.
    """
    return any(fnmatch.fnmatchcase(name, e["file"]) and fnmatch.fnmatchcase(key, e["path"]) for e in exemptions)


def compare(metrics_dir=METRICS_DIR, baseline_dir=BASELINE_DIR):
    """
    EN: Differences between the current metrics and the baseline, by kind: "files" (a metrics file added or
        missing), "shape" (a key added, dropped or retyped), "values" (a non-exempt value changed).
        Returns: {kind: [message]} — empty lists when everything matches.
    TR: Güncel metriklerle referans arasındaki farklar, türe göre: "files" (eklenen ya da eksik metrik dosyası),
        "shape" (eklenen, düşen ya da tipi değişen anahtar), "values" (istisna dışı değişen değer).
        Döndürür: {tür: [mesaj]} — her şey aynıysa boş listeler.
    """
    base_shape = json.loads((baseline_dir / SHAPE_FILE.name).read_text(encoding="utf-8"))
    base_print = json.loads((baseline_dir / FINGERPRINT_FILE.name).read_text(encoding="utf-8"))
    exemptions = load_exemptions(baseline_dir / EXEMPTIONS_FILE.name)
    shape, fingerprint = snapshot(metrics_dir)
    out = {"files": [], "shape": [], "values": []}
    for name in sorted(set(shape) - set(base_shape)):
        out["files"].append(f"{name}: new metrics file, not in the baseline | referansta yok")
    for name in sorted(set(base_shape) - set(shape)):
        out["files"].append(f"{name}: in the baseline but missing | referansta var, dosya yok")
    for name in sorted(set(shape) & set(base_shape)):
        now, then = shape[name], base_shape[name]
        for key in sorted(set(now) - set(then)):
            out["shape"].append(f"{name}: {key} added | eklendi")
        for key in sorted(set(then) - set(now)):
            out["shape"].append(f"{name}: {key} dropped | düştü")
        for key in sorted(set(now) & set(then)):
            if now[key] != then[key]:
                out["shape"].append(f"{name}: {key} type {then[key]} → {now[key]} | tip değişti")
            elif fingerprint[name][key] != base_print[name][key] and not is_exempt(name, key, exemptions):
                out["values"].append(f"{name}: {key} {_short(base_print[name][key])} → {_short(fingerprint[name][key])}")
    return out


def _short(value):
    """
    EN: A short printable form of a fingerprint for messages.
    TR: Mesajlar için parmak izinin kısa yazımı.
    """
    if isinstance(value, dict) and "sha256" in value:
        return f"list[{value['len']}]#{value['sha256'][:8]}"
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 60 else text[:57] + "..."


def accept(reason, metrics_dir=METRICS_DIR, baseline_dir=BASELINE_DIR, now=None):
    """
    EN: Takes the current metrics as the new baseline, for a stated reason, and appends the acceptance (time,
        reason, what changed) to accept_log.jsonl. Refuses without a reason.
        Returns: the differences that were accepted.
    TR: Güncel metrikleri belirtilen gerekçeyle yeni referans yapar ve onayı (zaman, gerekçe, ne değişti)
        accept_log.jsonl'a ekler. Gerekçesiz reddeder. Döndürür: onaylanan farklar.
    """
    if not reason or not reason.strip():
        raise ValueError("a reason is required | gerekçe gerekli")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    has_baseline = (baseline_dir / SHAPE_FILE.name).exists()
    changed = compare(metrics_dir, baseline_dir) if has_baseline else {"files": ["(first baseline | ilk referans)"]}
    shape, fingerprint = snapshot(metrics_dir)
    for path, doc in ((baseline_dir / SHAPE_FILE.name, shape), (baseline_dir / FINGERPRINT_FILE.name, fingerprint)):
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    entry = {"at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"), "reason": reason.strip(),
             "changed": {k: len(v) for k, v in changed.items()}, "first": [m for v in changed.values() for m in v][:20]}
    with open(baseline_dir / LOG_FILE.name, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return changed


def main(argv=None):
    """
    EN: Command line: --diff prints the differences (exit 1 if any); --accept "<reason>" updates the baseline.
    TR: Komut satırı: --diff farkları basar (varsa çıkış 1); --accept "<gerekçe>" referansı günceller.
    """
    ap = argparse.ArgumentParser(description="Metrics baseline of the verify gate | verify kapısının metrik referansı")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--diff", action="store_true", help="show differences from the baseline")
    group.add_argument("--accept", metavar="REASON", help="take the current metrics as the baseline, with a reason")
    args = ap.parse_args(argv)
    if args.accept is not None:
        if not args.accept.strip():
            ap.error("--accept needs a reason | gerekçe gerekli")
        # EN: only a consistent set (same DB, same model run) may become the baseline; the gate stops otherwise
        # TR: yalnız tutarlı bir küme (aynı DB, aynı model koşumu) referans olabilir; değilse kapı durdurur
        sys.path.insert(0, str(ROOT / "builders"))
        from report_lib import metrics_view
        metrics_view.load_view(ROOT)
        changed = accept(args.accept)
        print("baseline updated | referans güncellendi:", {k: len(v) for k, v in changed.items()})
        return 0
    diff = compare()
    for kind, messages in diff.items():
        for m in messages:
            print(f"[{kind}] {m}")
    total = sum(len(v) for v in diff.values())
    print(f"{total} difference(s) | fark")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())

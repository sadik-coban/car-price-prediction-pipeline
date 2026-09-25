"""
test_observed_values_analysis.py
EN: The analysis code expects only what the data showed, and handles everything it showed — both ways, against the
    silver part of the register of observed values (db/observed_values.json; no data needed):
      segment rule — (a) every series in SEGMENT_MAP, PERF_BASE and MODEL_SEG occurs and every MODEL_SEG prefix
      matches an observed model; (b) every observed (series, model) pair resolves to a segment (the run would stop
      otherwise);
      English labels — (a) every data value CAT_EN translates occurs in kb_transmission / kb_drivetrain / kb_fuel;
      (b) every observed value of those columns has a label.
    segment_rule.py and labels.py are loaded by path (the name `lib` means db/lib in this pytest session).
TR: Analiz kodu yalnız verinin gösterdiğini bekler ve gösterdiği her şeyi ele alır — iki yönde, gözlenen değerler
    kaydının silver parçasına göre (db/observed_values.json; veri gerekmez):
      segment kuralı — (a) SEGMENT_MAP, PERF_BASE ve MODEL_SEG'deki her seri geçiyor ve her MODEL_SEG öneki
      gözlenen bir modele uyuyor; (b) gözlenen her (seri, model) çifti bir segmente çözülüyor (yoksa koşum durur);
      İngilizce etiketler — (a) CAT_EN'in çevirdiği her veri değeri kb_transmission / kb_drivetrain / kb_fuel'de
      geçiyor; (b) o kolonların gözlenen her değerinin etiketi var.
    segment_rule.py ve labels.py yol üzerinden yüklenir (bu pytest oturumunda `lib` adı db/lib demek).
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTER = json.loads((ROOT / "db" / "observed_values.json").read_text(encoding="utf-8"))
SILVER = REGISTER["silver"]
LABEL_COLUMNS = ("kb_transmission", "kb_drivetrain", "kb_fuel")


def load(name, rel):
    """EN: A module loaded from a file path. / TR: Dosya yolundan yüklenen modül."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SR = load("observed_segment_rule", "analysis/lib/segment_rule.py")
LB = load("observed_labels", "analysis/lib/labels.py")


def observed(column):
    """EN: The non-null observed values of a silver column. / TR: Bir silver kolonunun NULL olmayan gözlenen değerleri."""
    return {v for v, _ in SILVER["columns"][column]["values"] if v is not None}


def test_segment_series_are_observed():
    """
    EN: (a) every series the rule names occurs; every MODEL_SEG prefix matches a model of its series.
    TR: (a) kuralın andığı her seri geçiyor; her MODEL_SEG öneki kendi serisinin bir modeline uyuyor.
    """
    series = observed("series")
    named = set(SR.SEGMENT_MAP) | set(SR.PERF_BASE) | set(SR.MODEL_SEG)
    assert named <= series, f"series never observed | hiç görülmemiş seriler: {sorted(named - series)}"
    for s, prefixes in SR.MODEL_SEG.items():
        models = [m for m, _ in SILVER["series_models"][s]]
        for prefix in prefixes:
            assert any(str(m).startswith(prefix) for m in models), (s, prefix)


def test_every_observed_pair_resolves():
    """EN: (b) every observed (series, model) gets a segment. / TR: (b) gözlenen her (seri, model) segment alıyor."""
    unresolved = [(s, m) for s, models in SILVER["series_models"].items() for m, _ in models
                  if SR.resolve(s, m) == (None, None)]
    assert not unresolved, unresolved[:10]


def test_english_labels_both_ways():
    """
    EN: (a) CAT_EN's data values occur; (b) every observed value of the label columns has a label.
    TR: (a) CAT_EN'in veri değerleri geçiyor; (b) etiket kolonlarının gözlenen her değerinin etiketi var.
    """
    seen = set().union(*(observed(c) for c in LABEL_COLUMNS))
    keys = set(LB.CAT_EN) - {"missing"}          # EN: the NULL sentinel, not a data value | TR: NULL işareti
    assert keys <= seen, f"labels for values never observed | hiç görülmemiş değer etiketleri: {sorted(keys - seen)}"
    assert seen <= keys, f"observed values without a label | etiketsiz gözlenen değerler: {sorted(seen - keys)}"

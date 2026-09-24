"""
test_hooks.py
EN: The Claude Code hooks in .claude/hooks/ are part of the gate, so they are tested like code. The path guard
    must deny every protected folder (in every path spelling Windows and Git Bash produce) and let everything
    else through. The Stop gate is run against a throwaway project whose tools/verify.py is a stub, so its
    decisions (block, skip when nothing changed, give up after 3 attempts, recover) are checked without running
    the real suite inside itself. .claude/ is not in git: in a clone without the hooks these tests skip.
TR: .claude/hooks/ içindeki Claude Code hook'ları kapının parçası, bu yüzden kod gibi sınanır. Yol koruması her
    korunan klasörü (Windows'un ve Git Bash'in ürettiği her yol yazımında) reddetmeli, geri kalan her şeyi
    geçirmeli. Stop kapısı, tools/verify.py'si sahte olan geçici bir projede koşulur; böylece kararları (engelle,
    değişiklik yoksa atla, 3 denemeden sonra bırak, toparlan) gerçek test takımını kendi içinde koşmadan sınanır.
    .claude/ git'te yok: hook'ların olmadığı bir klonda bu testler atlanır.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / ".claude" / "hooks"
pytestmark = pytest.mark.skipif(not HOOKS.exists(), reason=".claude/hooks not in this clone | bu klonda yok")

# EN: stub verify: answers according to <root>/mode, counts calls in <root>/calls (neither is watched)
# TR: sahte verify: <root>/mode'a göre cevaplar, çağrıları <root>/calls'a sayar (ikisi de izlenmez)
STUB_VERIFY = '''
"""EN: stub. TR: sahte."""
import json, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
with open(root / "calls", "a") as f:
    f.write("x")
mode = (root / "mode").read_text().strip()
if mode == "green":
    print(json.dumps({"ok": True, "failed": []}))
elif mode == "red":
    print(json.dumps({"ok": False, "failed": ["tests.x::test_y: AssertionError: kirmizi"]}))
else:
    print("not json")
'''


def run_hook(name, event, project):
    """
    EN: Runs a hook script with an event on stdin. Returns: its parsed stdout ({} when silent).
    TR: Bir hook betiğini stdin'deki olayla koşar. Döndürür: ayrıştırılmış stdout'u (sessizse {}).
    """
    r = subprocess.run([sys.executable, str(HOOKS / name)], input=json.dumps(event).encode("utf-8"),
                       capture_output=True, env={**os.environ, "CLAUDE_PROJECT_DIR": str(project)})
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    out = r.stdout.decode("utf-8").strip()
    return json.loads(out) if out else {}


def decision(file_path):
    """EN: The guard's decision for an Edit of a path. / TR: Bir yolun Edit'i için korumanın kararı."""
    out = run_hook("guard_paths.py", {"tool_name": "Edit", "tool_input": {"file_path": file_path}}, ROOT)
    return out.get("hookSpecificOutput", {}).get("permissionDecision", "allow")


DRIVE, REST = str(ROOT)[0], str(ROOT)[3:].replace("\\", "/")


@pytest.mark.parametrize("path", [
    str(ROOT / "metrics" / "07_final_model.json"),
    str(ROOT / "reports" / "figures" / "fig01.png"),
    str(ROOT / "data" / "raw" / "a.json"),
    str(ROOT / "analysis" / "frozen" / "x.csv"),
    str(ROOT / "tests" / "baselines" / "exemptions.json"),
    "metrics/new.json",                                        # EN: relative | TR: göreli
    f"{DRIVE.lower()}:/{REST.lower()}/Reports/x.md",            # EN: other case | TR: farklı harf
    f"/{DRIVE.lower()}/{REST}/data/site_data.json",             # EN: Git Bash form | TR: Git Bash biçimi
])
def test_guard_denies_protected(path):
    """EN: Every protected path spelling is denied. / TR: Korunan yolun her yazımı reddedilir."""
    assert decision(path) == "deny"


@pytest.mark.parametrize("path", [
    str(ROOT / "analysis" / "07_final_model.py"),
    str(ROOT / "builders" / "build_technical_report.py"),
    str(ROOT / "tests" / "metrics" / "test_metrics.py"),
    str(ROOT / "metricsx" / "a.json"),                          # EN: prefix only | TR: yalnız önek
    str(Path.home() / "elsewhere.txt"),                         # EN: outside the repo | TR: depo dışı
])
def test_guard_allows_the_rest(path):
    """EN: Code, tests and paths outside the repo pass. / TR: Kod, testler ve depo dışı geçer."""
    assert decision(path) == "allow"


@pytest.fixture
def project(tmp_path):
    """
    EN: A throwaway project: one watched file, the stub verify, mode=green.
    TR: Geçici bir proje: izlenen bir dosya, sahte verify, mode=green.
    """
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "verify.py").write_text(STUB_VERIFY, encoding="utf-8")
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis" / "x.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "mode").write_text("green")
    return tmp_path


def stop(project, active=False):
    """EN: Runs the Stop gate once. / TR: Stop kapısını bir kez koşar."""
    return run_hook("stop_gate.py", {"hook_event_name": "Stop", "stop_hook_active": active}, project)


def calls(project):
    """EN: How many times the stub verify ran. / TR: Sahte verify kaç kez koştu."""
    return len((project / "calls").read_text()) if (project / "calls").exists() else 0


def change(project, mode):
    """EN: Edits the watched file and sets the stub's answer. / TR: İzlenen dosyayı değiştirir, cevabı ayarlar."""
    with open(project / "analysis" / "x.py", "a", encoding="utf-8") as f:
        f.write("b = 2\n")
    (project / "mode").write_text(mode)


def test_stop_green_then_skip(project):
    """EN: Green passes; with nothing changed verify is not run again. / TR: Yeşil geçer; değişiklik yoksa yeniden koşmaz."""
    assert stop(project) == {} and calls(project) == 1
    assert stop(project) == {} and calls(project) == 1


def test_stop_blocks_then_gives_up_then_recovers(project):
    """
    EN: Red blocks 3 times in a turn, then lets go with UNVERIFIED; a later turn with no change is not re-blocked;
        a fix clears the marker.
    TR: Kırmızı bir turda 3 kez engeller, sonra UNVERIFIED ile bırakır; değişiklik olmayan sonraki tur yeniden
        engellenmez; düzeltme izi temizler.
    """
    state = project / ".claude" / "state"
    change(project, "red")
    for attempt, active in ((1, False), (2, True), (3, True)):
        out = stop(project, active)
        assert out["decision"] == "block" and f"({attempt}/3)" in out["reason"] and "tests.x::test_y" in out["reason"]
    out = stop(project, True)
    assert "decision" not in out and "UNVERIFIED" in out["systemMessage"] and (state / "UNVERIFIED").exists()
    assert calls(project) == 3
    assert stop(project) == {} and calls(project) == 3
    change(project, "green")
    assert stop(project) == {} and not (state / "UNVERIFIED").exists() and not (state / "stop_attempts").exists()


def test_stop_counts_restart_each_turn(project):
    """EN: A new turn starts the attempt count from 1. / TR: Yeni tur deneme sayısını 1'den başlatır."""
    change(project, "red")
    stop(project)
    change(project, "red")
    assert "(2/3)" in stop(project, True)["reason"]
    change(project, "red")
    assert "(1/3)" in stop(project, False)["reason"]


def test_stop_blocks_when_verify_does_not_report(project):
    """EN: A verify that prints no summary counts as red. / TR: Özet basmayan verify kırmızı sayılır."""
    change(project, "garbage")
    out = stop(project)
    assert out["decision"] == "block" and "did not report" in out["reason"]

"""
run_all.py
EN: Runs every analysis script in dependency order, each in its own process (exactly as
    `python analysis/<script>.py` would), then the builders; stops at the first failure and prints the times.
    No analysis here — it only orders.
      python analysis/run_all.py                 everything: analysis scripts + builders
      python analysis/run_all.py --from 07       from the first script whose name starts with 07 (+ builders)
      python analysis/run_all.py --only 08       only the scripts whose name starts with 08 (no builders)
      python analysis/run_all.py --only shap     only analysis/shap/*
      python analysis/run_all.py --no-builders   skip the builders
    Order: 01–06 → 07_model_comparison (writes the OOF artefact) → 07_final_model (serving files) → the rest of
    07, 08, 09, 10 → shap/02 (OOF SHAP artefact) → shap/03–06 → builders.
TR: Bütün analiz betiklerini bağımlılık sırasıyla, her birini kendi sürecinde (`python analysis/<betik>.py`
    nasıl koşuyorsa öyle) koşar, sonra derleyicileri; ilk hatada durur ve süreleri basar. Burada analiz yok —
    yalnız sıralar. Sıra: 01–06 → 07_model_comparison (OOF artefaktını yazar) → 07_final_model (servis
    dosyaları) → 07'nin kalanı, 08, 09, 10 → shap/02 (OOF SHAP artefaktı) → shap/03–06 → derleyiciler.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
ORDER = ["01_dedup_leakage.py", "01_engine_rule.py", "01_unspecified_panels.py", "01_gold_contract.py",
         "02_missingness.py",
         "03_association.py", "03_segment_quality.py", "03_brand_ablation.py", "04_target.py", "05_segmentation.py",
         "06_hedonic.py", "07_model_comparison.py", "07_final_model.py", "07_lofo.py", "07_text_flag.py",
         "08_conformal_coverage.py", "08_residuals.py", "08_large_errors.py", "09_drift.py", "09_backtest.py",
         "10_free_text.py", "shap/02_oof_shap.py", "shap/03_what_sets_price.py", "shap/04_variants.py",
         "shap/06_one_listing.py"]
BUILDERS = ["build_site_data.py", "build_technical_report.py", "build_business_report.py", "build_shap_report.py"]


def check_order_complete():
    """
    EN: Stops if a numbered analysis script exists on disk but is missing from ORDER (or the other way round),
        so a new script cannot be forgotten.
    TR: Diskte olup ORDER'da olmayan (ya da tersi) numaralı analiz betiği varsa durur; yeni betik unutulamaz.
    """
    on_disk = {p.relative_to(ANALYSIS).as_posix() for p in ANALYSIS.rglob("*.py") if p.name[:2].isdigit()}
    if on_disk != set(ORDER):
        raise SystemExit(f"ORDER is out of date | ORDER güncel değil: not listed {sorted(on_disk - set(ORDER))} · "
                         f"missing on disk {sorted(set(ORDER) - on_disk)}")


def select(start=None, only=None):
    """
    EN: The scripts to run: all, from the first one starting with `start`, or only those starting with `only`.
    TR: Koşulacak betikler: hepsi, `start` ile başlayan ilkinden itibaren ya da yalnız `only` ile başlayanlar.
    """
    if only:
        chosen = [s for s in ORDER if s.startswith(only)]
    elif start:
        first = next((i for i, s in enumerate(ORDER) if s.startswith(start)), None)
        if first is None:
            raise SystemExit(f"no script starts with | şununla başlayan betik yok: {start}")
        chosen = ORDER[first:]
    else:
        chosen = list(ORDER)
    if not chosen:
        raise SystemExit(f"nothing selected | hiçbir betik seçilmedi: {only}")
    return chosen


def run(path):
    """
    EN: Runs one script in its own process from the repo root; stops the whole run if it fails.
        Returns: seconds taken.
    TR: Bir betiği repo kökünden kendi sürecinde koşar; başarısız olursa bütün koşumu durdurur.
        Döndürür: geçen saniye.
    """
    t0 = time.time()
    print(f"\n▶ {path.relative_to(ROOT).as_posix()}", flush=True)
    if subprocess.run([sys.executable, str(path)], cwd=ROOT).returncode != 0:
        raise SystemExit(f"FAILED | BAŞARISIZ: {path.relative_to(ROOT).as_posix()} — later scripts were not run")
    return time.time() - t0


def main():
    """
    EN: Parses the options, runs the selected scripts (+ builders) and prints a timing table.
    TR: Seçenekleri okur, seçilen betikleri (+ derleyicileri) koşar ve süre tablosunu basar.
    """
    ap = argparse.ArgumentParser(description="Run the analysis scripts in dependency order | analiz betiklerini sırayla koş")
    ap.add_argument("--from", dest="start", help="start at the first script whose name starts with this")
    ap.add_argument("--only", help="run only the scripts whose name starts with this (no builders)")
    ap.add_argument("--no-builders", action="store_true", help="skip the builders")
    args = ap.parse_args()
    check_order_complete()
    paths = [ANALYSIS / s for s in select(args.start, args.only)]
    if not (args.only or args.no_builders):
        paths += [ROOT / "builders" / b for b in BUILDERS]
    times = [(p.relative_to(ROOT).as_posix(), run(p)) for p in paths]
    print("\n" + "\n".join(f"  {sec:7.1f} s  {name}" for name, sec in times))
    print(f"  {sum(t for _n, t in times) / 60:7.1f} min total | toplam")


if __name__ == "__main__":
    main()

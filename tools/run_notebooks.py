"""
run_notebooks.py
EN: Refreshes the outputs of the notebooks under notebooks/. Each notebook runs top to bottom in a fresh kernel
    (Restart & Run All) whose working folder is notebooks/, and is saved with its outputs. In a notebook the light
    scripts run without saving (analysis/lib/records.py: run) and the heavy ones are read from metrics/, so this
    takes about half a minute and writes no metric. Run it after the chain (analysis/run_all.py, then the builders)
    so the notebooks show the new record. A failing cell stops the run and that notebook is not written.
    Why not `jupyter execute --inplace`: on Windows it reads the file in the console code page (Turkish letters
    break) and it stamps every cell with its run time (a diff on every run).
TR: notebooks/ altındaki defterlerin çıktılarını yeniler. Her defter, çalışma klasörü notebooks/ olan taze bir
    kernel'de yukarıdan aşağı koşar (Restart & Run All) ve çıktılarıyla kaydedilir. Defterde hafif betikler
    kaydetmeden koşar (analysis/lib/records.py: run), ağır olanlar metrics/'ten okunur; bu yüzden yarım dakika
    kadar sürer ve hiçbir metrik yazmaz. Zincirden sonra (analysis/run_all.py, sonra derleyiciler) koşun; defterler
    yeni kaydı gösterir. Düşen bir hücre koşumu durdurur ve o defter yazılmaz.
    Neden `jupyter execute --inplace` değil: Windows'ta dosyayı konsolun kod sayfasıyla okuyor (Türkçe harfler
    bozuluyor) ve her hücreye koşum zamanını yazıyor (her koşumda fark).
Run / Koşum:
    python tools/run_notebooks.py                 every notebook | bütün defterler
    python tools/run_notebooks.py 3_errors_and_time
"""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def notebooks(names=()):
    """
    EN: The notebooks to run: the named ones, or every notebook under notebooks/. Stops on an unknown name.
        Returns: list of Paths.
    TR: Koşulacak defterler: adı verilenler ya da notebooks/ altındaki her defter. Bilinmeyen adda durur.
        Döndürür: Path listesi.
    """
    if not names:
        return sorted(p for p in NOTEBOOKS.glob("*.ipynb"))
    paths = [NOTEBOOKS / f"{n.removesuffix('.ipynb')}.ipynb" for n in names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"no such notebook | böyle bir defter yok: {missing}")
    return paths


def run(path):
    """
    EN: Runs one notebook in a fresh kernel and saves it with its outputs (UTF-8, no run-time stamps).
    TR: Bir defteri taze bir kernel'de koşar ve çıktılarıyla kaydeder (UTF-8, koşum zamanı damgası yok).
    """
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=None, kernel_name="python3", record_timing=False,
                            resources={"metadata": {"path": str(NOTEBOOKS)}})
    try:
        client.execute()
    except CellExecutionError as e:
        raise SystemExit(f"FAILED | BAŞARISIZ: {path.relative_to(ROOT).as_posix()} — not written | yazılmadı\n{e}")
    nbformat.write(nb, path)


def main():
    """
    EN: Parses the names and runs the notebooks one by one.
    TR: Adları okur ve defterleri tek tek koşar.
    """
    ap = argparse.ArgumentParser(description="Refresh notebook outputs | defter çıktılarını yenile")
    ap.add_argument("names", nargs="*", help="notebook names under notebooks/, e.g. 1_data")
    for path in notebooks(ap.parse_args().names):
        run(path)
        print("written | yazıldı:", path.relative_to(ROOT).as_posix(), flush=True)


if __name__ == "__main__":
    main()

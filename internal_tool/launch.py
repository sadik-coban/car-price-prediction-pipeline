"""
launch.py
EN: Starts the internal tool with Streamlit, bound to 127.0.0.1 only, usage statistics off and PYTHONUTF8=1 (the
    Windows console defaults to cp1252). It refuses any address that is not loopback: the tool shows scraped
    listings with their ad_id and url, which must never be served to a network. By default the browser opens;
    --headless keeps it closed (background runs, checks).
TR: İç aracı Streamlit ile başlatır: yalnız 127.0.0.1'e bağlı, kullanım istatistiği kapalı, PYTHONUTF8=1 (Windows
    konsolu varsayılan olarak cp1252). Loopback olmayan adresi reddeder: araç kazınan ilanları ad_id ve url'leriyle
    gösterir, bunlar asla bir ağa sunulmamalı. Varsayılan olarak tarayıcı açılır; --headless kapalı tutar (arka plan
    koşumları, denetimler).
Run / Koşum:
    .venv-tool\\Scripts\\python internal_tool\\launch.py [--port 8501] [--headless]
"""
import argparse
import ipaddress
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "internal_tool" / "app.py"
HOST = "127.0.0.1"
PORT = 8501


def is_loopback(address):
    """
    EN: True when address is a loopback IP (127.0.0.0/8, ::1) or "localhost"; False for anything else, including
        "" and 0.0.0.0 (all interfaces).
    TR: address bir loopback IP'si (127.0.0.0/8, ::1) ya da "localhost" ise True; "" ve 0.0.0.0 (bütün arayüzler)
        dahil başka her şey için False.
    """
    if not address:
        return False
    if address.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def build_command(python, port=PORT, address=HOST, headless=False):
    """
    EN: The Streamlit command line for the app. Raises ValueError for a non-loopback address.
    TR: Uygulamanın Streamlit komut satırı. Loopback olmayan adreste ValueError yükseltir.
    """
    if not is_loopback(address):
        raise ValueError(f"not a loopback address | loopback adresi değil: {address!r}")
    return [str(python), "-m", "streamlit", "run", str(APP),
            "--server.address", address, "--server.port", str(port),
            "--server.headless", "true" if headless else "false",
            "--browser.gatherUsageStats", "false"]


def build_env(base):
    """EN: The app's environment: base plus PYTHONUTF8=1. / TR: Uygulamanın ortamı: base artı PYTHONUTF8=1."""
    env = dict(base)
    env["PYTHONUTF8"] = "1"
    return env


def main(argv=None):
    """
    EN: Command line. Returns: Streamlit's exit code.
    TR: Komut satırı. Döndürür: Streamlit'in çıkış kodu.
    """
    ap = argparse.ArgumentParser(description="Internal tool on 127.0.0.1 | iç araç, yalnız 127.0.0.1")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--headless", action="store_true", help="do not open a browser | tarayıcı açma")
    args = ap.parse_args(argv)
    cmd = build_command(sys.executable, port=args.port, headless=args.headless)
    print(f"internal tool | iç araç: http://{HOST}:{args.port}")
    return subprocess.call(cmd, env=build_env(os.environ), cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())

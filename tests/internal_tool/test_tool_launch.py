"""
test_tool_launch.py
EN: The internal tool's launcher: only loopback addresses are accepted (the tool shows scraped listings with ad_id
    and url, which must never reach a network), the command binds to that address with usage statistics off, and
    the environment forces UTF-8.
TR: İç aracın başlatıcısı: yalnız loopback adresleri kabul edilir (araç kazınan ilanları ad_id ve url'leriyle
    gösterir, bunlar asla bir ağa ulaşmamalı), komut o adrese bağlanır ve kullanım istatistiği kapalıdır, ortam
    UTF-8'i zorlar.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from internal_tool import launch  # noqa: E402


@pytest.mark.parametrize("address, ok", [("127.0.0.1", True), ("127.0.0.2", True), ("::1", True), ("localhost", True),
                                         ("0.0.0.0", False), ("", False), ("192.168.1.10", False), ("::", False),
                                         ("example.com", False), (None, False)])
def test_is_loopback(address, ok):
    """EN: Loopback IPs and localhost only. / TR: Yalnız loopback IP'leri ve localhost."""
    assert launch.is_loopback(address) is ok


def test_command_binds_loopback_without_stats():
    """EN: The command runs app.py on 127.0.0.1, stats off. / TR: Komut app.py'yi 127.0.0.1'de koşar, istatistik kapalı."""
    cmd = launch.build_command("py", port=8600, headless=True)
    pairs = dict(zip(cmd, cmd[1:]))
    assert cmd[:4] == ["py", "-m", "streamlit", "run"] and Path(cmd[4]) == ROOT / "internal_tool" / "app.py"
    assert pairs["--server.address"] == "127.0.0.1" and pairs["--server.port"] == "8600"
    assert pairs["--server.headless"] == "true" and pairs["--browser.gatherUsageStats"] == "false"
    assert dict(zip(*[iter(launch.build_command("py")[5:])] * 2))["--server.headless"] == "false"


@pytest.mark.parametrize("address", ["0.0.0.0", "192.168.1.10", ""])
def test_command_refuses_network_addresses(address):
    """EN: A non-loopback address is refused. / TR: Loopback olmayan adres reddedilir."""
    with pytest.raises(ValueError):
        launch.build_command("py", address=address)


def test_env_forces_utf8():
    """EN: PYTHONUTF8=1 is set, the base is kept. / TR: PYTHONUTF8=1 konur, taban korunur."""
    env = launch.build_env({"PATH": "x", "PYTHONUTF8": "0"})
    assert env == {"PATH": "x", "PYTHONUTF8": "1"}

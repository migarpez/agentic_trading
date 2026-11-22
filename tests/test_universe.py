import subprocess
from pathlib import Path

from agentic_trading.data.universe import exchange_of, to_yahoo


def test_to_yahoo():
    assert to_yahoo("BRK.B", "") == "BRK-B"
    assert to_yahoo("MMM", "") == "MMM"
    assert to_yahoo("BT.A", ".L") == "BT-A.L"
    assert to_yahoo("BP", ".L") == "BP.L"
    assert to_yahoo("AIR.PA", ".DE") == "AIR.PA"
    assert to_yahoo("san.mc", ".MC") == "SAN.MC"
    assert to_yahoo("NESN", ".SW") == "NESN.SW"


def test_exchange_of():
    assert exchange_of("BP.L") == ("GB", "GBp")
    assert exchange_of("NESN.SW") == ("CH", "CHF")
    assert exchange_of("AAPL") == ("US", "USD")


def test_normalize_name():
    from agentic_trading.data.universe import normalize_name
    assert normalize_name("Shell plc") == normalize_name("Shell")
    assert normalize_name("Nestlé SA") == "nestle"
    assert normalize_name("Alphabet Inc. (Class A)") == normalize_name("Alphabet Inc. (Class C)")


def test_equivalent_groups_by_cik():
    import pandas as pd

    from agentic_trading.data.universe import equivalent_groups
    raw = pd.DataFrame({"symbol": ["GOOGL", "GOOG", "AAPL", "SHEL.L", "SHELL.AS"],
                        "cik": ["1652044", "1652044", "320193", None, None]})
    groups = equivalent_groups(raw)
    assert {"GOOGL", "GOOG"} in groups and {"SHEL.L", "SHELL.AS"} in groups
    assert not any("AAPL" in g for g in groups)


def test_make_data_keeps_universe():
    root = Path(__file__).resolve().parents[1]
    run = lambda t: subprocess.run(["make", "-n", t], cwd=root, capture_output=True, text=True,
                                   check=True).stdout
    assert "build_universe.py" not in run("data") and "download_data.py" in run("data")
    assert "build_universe.py" in run("universe")

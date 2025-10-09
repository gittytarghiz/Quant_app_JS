"""
Initializer.py — create initial Level-1 asset classification JSON
Reads tickers from existing txt lists and assigns base asset classes.
"""

import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_PATH = Path(__file__).resolve().parent / "asset_classification.json"

def load_tickers(path):
    """Read tickers from a .txt file (one per line)."""
    if not Path(path).exists():
        return []
    with open(path, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]

def initialize_classification():
    """
    Create a Level-1 classification JSON using existing ticker lists.
    Categories: US Equities, Commodities, Crypto, Forex, Indices, Bonds
    """
    classification = {
        "Equity_US": load_tickers(DATA_DIR / "tickers_nasdaq.txt")
                     + load_tickers(DATA_DIR / "tickers_nyse.txt"),
        "Commodity": ["GC=F", "CL=F", "SI=F", "NG=F"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"],
        "Forex": ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCAD=X"],
        "Index": ["^GSPC", "^NDX", "^DJI", "^VIX"],
        "Bond": ["^TNX", "TLT", "LQD"]
    }

    with open(OUT_PATH, "w") as f:
        json.dump(classification, f, indent=2)

    print(f"✅ Asset classification JSON created at {OUT_PATH}")

if __name__ == "__main__":
    initialize_classification()

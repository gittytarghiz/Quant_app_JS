import json, pandas as pd, yfinance as yf
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA, CLASS = BASE / "data", Path(__file__).resolve().parent / "asset_classification.json"

def classify_yahoo(t):
    """Quick Yahoo-based classification."""
    try:
        q = (yf.Ticker(t).info.get("quoteType", "") or "").lower()
        if "etf" in q: return "ETF"
        if "crypto" in q: return "Crypto"
        if "future" in q or "commodity" in q: return "Commodity"
        if "currency" in q: return "Forex"
        if "equity" in q: return "Equity_US"
    except: pass
    return "Unclassified"

def get_tickers():
    """Collect all tickers from data/ohlc/interval parquet files."""
    tickers = set()
    for ohlc in ["close","open","high","low"]:
        d = DATA/ohlc
        if not d.exists(): continue
        for sub in d.iterdir():
            if not sub.is_dir(): continue
            for f in sub.glob("*.parquet"):
                try:
                    tickers |= {c.upper() for c in pd.read_parquet(f, nrows=1).columns}
                except: pass
    return tickers

def run():
    c = json.load(open(CLASS)) if CLASS.exists() else {}
    all_cls = {k: set(v) for k, v in c.items()}
    known = set().union(*all_cls.values()) if all_cls else set()
    unclassified = get_tickers() - known

    print(f"Scanning... found {len(unclassified)} new tickers.")
    for t in unclassified:
        cls = classify_yahoo(t)
        all_cls.setdefault(cls, set()).add(t)

    json.dump({k: sorted(v) for k,v in all_cls.items()}, open(CLASS,"w"), indent=2)
    print(f"Updated {CLASS.name} ✅")

if __name__ == "__main__":
    run()

    # === Copy updated JSON to frontend ===
    FRONTEND_PATH = BASE.parent / "frontend/public/asset_classification.json"
    try:
        if FRONTEND_PATH.exists():
            FRONTEND_PATH.unlink()  # remove old version
        data = json.load(open(CLASS))
        FRONTEND_PATH.write_text(json.dumps(data, indent=2))
        print(f"📤 Copied to frontend: {FRONTEND_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to copy to frontend: {e}")


import pandas as pd
from typing import Optional, Dict, Any, List


def format_weights(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    """Convert weights DataFrame to list of records (JSON-serializable)."""
    if df is None or df.empty:
        return []
    df = df.copy()
    df.index = pd.to_datetime(df.index)  # ensure datetime index
    df.index.name = "date"
    return df.reset_index().to_dict(orient="records")


def format_pnl(series: Optional[pd.Series]) -> List[Dict[str, Any]]:
    """Convert PnL Series to list of records (JSON-serializable)."""
    if series is None or series.empty:
        return []
    df = pd.DataFrame({"date": pd.to_datetime(series.index), "pnl": series.values})
    return df.to_dict(orient="records")


def normalize_details(details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize optimization details for API response (drop None values)."""
    if not details:
        return {}
    return {k: v for k, v in details.items() if v is not None}

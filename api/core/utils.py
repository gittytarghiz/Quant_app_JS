import pandas as pd
from typing import Optional, Dict, Any

def format_weights(df: pd.DataFrame) -> list[Dict[str, Any]]:
    """Convert weights DataFrame to list of records"""
    if df is None or df.empty:
        return []
    df = df.copy()
    df.index.name = "date"
    return df.reset_index().to_dict("records")

def format_pnl(series: pd.Series) -> list[Dict[str, Any]]:
    """Convert PnL Series to list of records"""
    if series is None or series.empty:
        return []
    df = pd.DataFrame({"date": series.index, "pnl": series.values})
    return df.to_dict("records")

def normalize_details(details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize optimization details for API response"""
    if not details:
        return {}
    return {k: v for k, v in details.items() if v is not None}
from fastapi import APIRouter, Query
from portfolio_optimization.walkforward_user_weights import walkforward_user_weights

router = APIRouter(prefix="/user_weights", tags=["Optimizers"])

@router.get("/")
def user_weights_api(
    tickers: str,
    start: str,
    end: str,
    weights: str,
    dtype: str = "close",
    interval: str = "1d",
    rebalance: str = "monthly",
    leverage: float = 1.0,
    interest_rate: float = 0.04,
):
    """
    Example:
    /user_weights?tickers=AMZN,AAPL,NVDA&start=2020-01-01&end=2025-01-01&weights=AMZN:0.4,AAPL:0.3,NVDA:0.3
    """
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    weights_dict = {}
    for item in weights.split(","):
        if ":" in item:
            k, v = item.split(":")
            weights_dict[k.strip().upper()] = float(v)

    result = walkforward_user_weights(
        tickers=tickers_list,
        start=start,
        end=end,
        weights=weights_dict,
        dtype=dtype,
        interval=interval,
        rebalance=rebalance,
        leverage=leverage,
        interest_rate=interest_rate,
    )
    return result

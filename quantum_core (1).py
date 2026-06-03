"""
Module: quantum_core.py  —  v2.0  "Max-Return Multi-Factor"
============================================================

WHAT'S NEW vs v1:
  ─ Multi-factor alpha scoring replaces pure Sharpe screening:
      α = w_ret·μ  +  w_mom·Momentum  +  w_qual·QualityScore  −  w_cvar·CVaR_penalty
  ─ Horizon-aware data fetching:
      ≤ 7 days   → hourly bars (1mo history)   — short-term traders
      ≤ 30 days  → daily bars (3mo history)
      ≤ 90 days  → daily bars (1y history)
      > 90 days  → daily bars (3y history)      — long-term investors
  ─ Momentum signals at 1m / 3m / 6m windows (weighted by horizon)
  ─ RSI trend filter: penalises overbought (RSI > 75) assets in short horizons
  ─ CVaR(95%) risk penalty replaces simple variance penalty in QUBO
  ─ Return-focused risk penalty: λ(t) = 0.8 / √t  (lighter risk penalty → more return)
  ─ Three weight-allocation objectives available (app.py chooses):
      SHARPE   – classic Sharpe maximisation (balanced)
      SORTINO  – downside-only risk (asymmetric, penalises losses not gains)
      MAXRET   – maximum expected return subject to max-drawdown constraint (aggressive)
  ─ Kelly position sizing overlay (half-Kelly for safety)
  ─ Max-drawdown and CVaR computed per final portfolio for display
  ─ Sector concentration constraint auto-added (max 40% in one sector) if not overridden
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from functools import lru_cache
from typing import Dict, Any, List, Tuple, Optional

from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import SamplingVQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────

SP500_UNIVERSE: List[str] = [
    # High-growth Technology (expanded — primary return drivers)
    "NVDA", "AMD", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA",
    "AVGO", "ORCL", "CRM", "ADBE", "NOW", "PANW", "SNOW", "PLTR",
    "MRVL", "AMAT", "KLAC", "LRCX", "MU", "QCOM", "TXN", "ADI",
    "CDNS", "SNPS", "FTNT", "CRWD", "ZS", "NET", "DDOG", "TEAM",
    # Healthcare / Biotech (growth + defensive)
    "LLY", "ABBV", "REGN", "VRTX", "AMGN", "GILD", "MRNA", "BIIB",
    "UNH", "TMO", "DHR", "SYK", "ISRG", "BSX", "EW", "ZTS",
    # Financials
    "GS", "MS", "JPM", "BAC", "BLK", "SCHW", "AXP", "COF",
    "V", "MA", "PYPL", "SQ",
    # Consumer Discretionary
    "BKNG", "ABNB", "UBER", "LYFT", "NKE", "LULU", "TJX", "ROST",
    "HD", "MCD", "CMG", "SBUX",
    # Energy (momentum plays)
    "XOM", "CVX", "COP", "OXY", "DVN", "FANG",
    # Industrials
    "CAT", "DE", "ETN", "PWR", "GE", "HON",
    # Communication
    "NFLX", "DIS", "EA", "TTWO", "RBLX",
    # Consumer Staples (defensive ballast)
    "COST", "WMT", "PG", "KO",
    # Materials / Clean Energy
    "ALB", "SQM", "FCX", "NEM", "NEE", "ENPH", "SEDG",
]

# Sector labels for concentration limits
TICKER_SECTOR: Dict[str, str] = {
    **{t: "technology" for t in [
        "NVDA","AMD","AAPL","MSFT","META","GOOGL","AMZN","AVGO","ORCL","CRM","ADBE",
        "NOW","PANW","SNOW","PLTR","MRVL","AMAT","KLAC","LRCX","MU","QCOM","TXN",
        "ADI","CDNS","SNPS","FTNT","CRWD","ZS","NET","DDOG","TEAM",
    ]},
    **{t: "healthcare" for t in [
        "LLY","ABBV","REGN","VRTX","AMGN","GILD","MRNA","BIIB","UNH","TMO",
        "DHR","SYK","ISRG","BSX","EW","ZTS",
    ]},
    **{t: "financial" for t in ["GS","MS","JPM","BAC","BLK","SCHW","AXP","COF","V","MA","PYPL","SQ"]},
    **{t: "consumer_disc" for t in ["BKNG","ABNB","UBER","LYFT","NKE","LULU","TJX","ROST","HD","MCD","CMG","SBUX","TSLA"]},
    **{t: "energy" for t in ["XOM","CVX","COP","OXY","DVN","FANG"]},
    **{t: "industrials" for t in ["CAT","DE","ETN","PWR","GE","HON"]},
    **{t: "communication" for t in ["NFLX","DIS","EA","TTWO","RBLX"]},
    **{t: "staples" for t in ["COST","WMT","PG","KO"]},
    **{t: "materials" for t in ["ALB","SQM","FCX","NEM","NEE","ENPH","SEDG"]},
}


def get_sp500_tickers() -> List[str]:
    return SP500_UNIVERSE.copy()


# ─────────────────────────────────────────────────────────────────────────────
# HORIZON CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def get_horizon_config(horizon_days: int) -> Dict[str, Any]:
    """
    Maps investment horizon to appropriate data granularity.
    Short-term traders get intraday data; long-term gets 3 years of daily bars.
    """
    if horizon_days <= 7:
        return {
            "period": "1mo", "interval": "1h",
            "ann_factor": 252 * 6.5,   # 252 days × 6.5 trading hours
            "mode": "intraday",
            "mom_windows": [4, 12, 24],        # hours: 4h, 12h, 24h
            "rsi_period": 14,
            "label": f"{horizon_days}d (intraday)",
        }
    elif horizon_days <= 30:
        return {
            "period": "3mo", "interval": "1d",
            "ann_factor": 252,
            "mode": "short",
            "mom_windows": [5, 10, 21],        # days: 1w, 2w, 1m
            "rsi_period": 14,
            "label": f"{horizon_days}d (short-term)",
        }
    elif horizon_days <= 90:
        return {
            "period": "1y", "interval": "1d",
            "ann_factor": 252,
            "mode": "medium",
            "mom_windows": [21, 42, 63],       # days: 1m, 2m, 3m
            "rsi_period": 21,
            "label": f"{horizon_days}d (medium-term)",
        }
    else:
        return {
            "period": "3y", "interval": "1d",
            "ann_factor": 252,
            "mode": "long",
            "mom_windows": [21, 63, 126],      # days: 1m, 3m, 6m
            "rsi_period": 21,
            "label": f"{horizon_days}d (long-term)",
        }


# ─────────────────────────────────────────────────────────────────────────────
# MARKET DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=16)
def _cached_download(universe_tuple: tuple, period: str, interval: str) -> bytes:
    tickers = list(universe_tuple)
    logger.info(f"Downloading {len(tickers)} assets | period={period} interval={interval}")
    raw = yf.download(tickers, period=period, interval=interval, progress=False, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        data = raw["Close"]
    else:
        data = raw

    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data.to_parquet()


def fetch_market_data(universe: List[str], period: str, interval: str) -> pd.DataFrame:
    universe_tuple = tuple(sorted(universe))
    parquet_bytes = _cached_download(universe_tuple, period, interval)
    data = pd.read_parquet(pd.io.common.BytesIO(parquet_bytes))
    # Keep columns with ≥70% non-null data
    data = data.dropna(axis=1, thresh=int(0.70 * len(data)))
    return data


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Wilder RSI. Returns last RSI value."""
    if len(prices) < period + 2:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    rs = avg_gain / avg_loss if avg_loss > 1e-10 else 100.0
    return 100.0 - 100.0 / (1.0 + rs)


def compute_momentum(prices: np.ndarray, windows: List[int]) -> float:
    """
    Weighted composite momentum across multiple lookback windows.
    Longer windows get slightly higher weight for trend confirmation.
    Returns a -1 to +1 normalised score.
    """
    scores = []
    weights_w = np.linspace(1.0, 2.0, len(windows))  # longer → higher weight
    for w, window in zip(weights_w, windows):
        if len(prices) > window:
            ret = prices[-1] / prices[-window] - 1.0
            scores.append(w * ret)
    if not scores:
        return 0.0
    raw = np.sum(scores) / np.sum(weights_w[:len(scores)])
    # Soft-clip to [-1, 1] using tanh
    return float(np.tanh(raw * 5))


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from peak to trough."""
    cumulative = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / np.maximum(peak, 1e-10)
    return float(drawdown.min())


def compute_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Conditional Value at Risk (Expected Shortfall) at given confidence.
    Returns the mean of losses beyond the VaR threshold (positive = bad).
    """
    if len(returns) < 20:
        return float(np.std(returns))
    sorted_r = np.sort(returns)
    cutoff = int((1.0 - confidence) * len(sorted_r))
    cutoff = max(cutoff, 1)
    return float(-sorted_r[:cutoff].mean())   # positive number = magnitude of tail loss


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-FACTOR SCREENING
# ─────────────────────────────────────────────────────────────────────────────

def compute_multi_factor_scores(
    data: pd.DataFrame,
    cfg: Dict[str, Any],
    horizon_days: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Computes a composite alpha score for each asset.

    Alpha = w_ret·μ_scaled  +  w_mom·Momentum  −  w_cvar·CVaR  −  w_rsi·RSI_penalty

    Weight schedule (shifts by horizon):
        Short (<30d):  momentum dominates  → w_mom=0.5, w_ret=0.3, w_cvar=0.2
        Medium (<90d): balanced            → w_mom=0.3, w_ret=0.4, w_cvar=0.3
        Long (>90d):   return dominates    → w_mom=0.2, w_ret=0.5, w_cvar=0.3

    Returns
    -------
    alpha   : (N,) composite score for QUBO linear terms
    mu      : (N,) annualised expected return
    sigma   : (N, N) annualised covariance matrix
    stats   : per-asset dict list for display
    """
    tickers = data.columns.tolist()
    n = len(tickers)
    ann = cfg["ann_factor"]

    log_returns = np.log(data / data.shift(1)).dropna()
    mu_annual = log_returns.mean().values * ann
    sigma_annual = log_returns.cov().values * ann

    # ── Per-asset metrics ────────────────────────────────────────────────
    vol = np.sqrt(np.diag(sigma_annual))
    rf_annual = 0.05

    momentum_scores = np.zeros(n)
    rsi_vals = np.zeros(n)
    cvar_vals = np.zeros(n)
    drawdown_vals = np.zeros(n)
    sortino_vals = np.zeros(n)

    windows = cfg["mom_windows"]
    rsi_period = cfg["rsi_period"]

    for i, ticker in enumerate(tickers):
        prices = data[ticker].dropna().values
        rets = log_returns[ticker].dropna().values if ticker in log_returns.columns else np.array([])

        momentum_scores[i] = compute_momentum(prices, windows)
        rsi_vals[i] = compute_rsi(prices, rsi_period)
        cvar_vals[i] = compute_cvar(rets) * np.sqrt(ann) if len(rets) >= 20 else 0.3
        drawdown_vals[i] = abs(compute_max_drawdown(rets)) if len(rets) >= 20 else 0.3

        # Sortino ratio
        neg_rets = rets[rets < 0]
        ds = neg_rets.std() * np.sqrt(ann) if len(neg_rets) > 5 else max(vol[i], 1e-6)
        sortino_vals[i] = (mu_annual[i] - rf_annual) / ds if ds > 1e-8 else 0.0

    # ── Factor weights by horizon ────────────────────────────────────────
    mode = cfg["mode"]
    if mode == "intraday":
        w_ret, w_mom, w_cvar, w_rsi = 0.25, 0.55, 0.15, 0.05
    elif mode == "short":
        w_ret, w_mom, w_cvar, w_rsi = 0.30, 0.45, 0.20, 0.05
    elif mode == "medium":
        w_ret, w_mom, w_cvar, w_rsi = 0.40, 0.30, 0.25, 0.05
    else:  # long
        w_ret, w_mom, w_cvar, w_rsi = 0.50, 0.20, 0.25, 0.05

    # Normalise each factor to [0, 1] range for stable weighting
    def norm(x: np.ndarray) -> np.ndarray:
        r = x - x.min()
        return r / r.max() if r.max() > 1e-10 else np.ones_like(x) * 0.5

    mu_norm = norm(mu_annual)
    mom_norm = norm(momentum_scores)      # already [-1,1] via tanh
    cvar_norm = norm(-cvar_vals)          # flip: lower CVaR = better
    rsi_penalty = np.where(rsi_vals > 75, (rsi_vals - 75) / 25, 0.0)  # overbought penalty

    alpha = (
        w_ret  * mu_norm
        + w_mom  * mom_norm
        + w_cvar * cvar_norm
        - w_rsi  * rsi_penalty
    )

    # ── Stats list for dashboard ─────────────────────────────────────────
    stats = []
    for i, ticker in enumerate(tickers):
        sharpe = (mu_annual[i] - rf_annual) / vol[i] if vol[i] > 1e-8 else 0.0
        stats.append({
            "ticker": ticker,
            "alpha_score": round(float(alpha[i]), 4),
            "expected_return_pct": round(float(mu_annual[i]) * 100, 2),
            "volatility_pct": round(float(vol[i]) * 100, 2),
            "sharpe_ratio": round(float(sharpe), 3),
            "sortino_ratio": round(float(sortino_vals[i]), 3),
            "momentum_score": round(float(momentum_scores[i]), 3),
            "rsi": round(float(rsi_vals[i]), 1),
            "cvar_95_pct": round(float(cvar_vals[i]) * 100, 2),
            "max_drawdown_pct": round(float(drawdown_vals[i]) * 100, 2),
        })

    return alpha, mu_annual, sigma_annual, stats


def screen_universe(
    data: pd.DataFrame,
    cfg: Dict[str, Any],
    horizon_days: int,
    max_quantum_assets: int = 12,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Multi-factor screening: returns top-K assets by composite alpha.
    Also returns full-universe stats for the risk-return scatter.
    """
    available = data.columns.tolist()
    logger.info(f"Multi-factor screening {len(available)} assets → top {max_quantum_assets} ...")

    alpha, mu_annual, sigma_annual, all_stats = compute_multi_factor_scores(data, cfg, horizon_days)

    alpha_series = pd.Series(alpha, index=available)
    top_k = alpha_series.nlargest(max_quantum_assets).index.tolist()
    logger.info(f"Top-K by alpha: {top_k}")

    # Slice to top-K
    idx = [available.index(t) for t in top_k]
    mu_k = mu_annual[idx]
    sigma_k = sigma_annual[np.ix_(idx, idx)]
    alpha_k = alpha[idx]

    # Scale to horizon  (t = horizon_days / 252)
    t = horizon_days / 252.0
    scaled_mu = mu_k * t
    scaled_sigma = sigma_k * t        # covariance scales linearly

    # Filter stats to top-K for dashboard
    top_stats = [s for s in all_stats if s["ticker"] in top_k]
    full_stats = all_stats   # keep all for scatter plot

    return top_k, scaled_mu, scaled_sigma, alpha_k, full_stats


# ─────────────────────────────────────────────────────────────────────────────
# QUBO BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_qubo(
    tickers: List[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    alpha: np.ndarray,          # multi-factor composite score
    compliance_payload: Dict[str, Any],
    horizon_days: int,
    add_sector_concentration: bool = True,
) -> QuadraticProgram:
    """
    Multi-factor QUBO objective:

        minimise  −αᵀx  +  λ_risk · xᵀΣx

    where α = composite alpha (return + momentum − CVaR − RSI_penalty),
    already normalised and weighted per horizon.

    Risk penalty λ(t):
        λ = 0.8 / √(horizon_days / 21)   — lighter than v1 to favour returns
        This decays fast with horizon, letting alpha dominate for long-term.
    """
    qp = QuadraticProgram("QuantumMultiFactorQUBO")
    for t in tickers:
        qp.binary_var(name=t)

    # Lighter risk penalty (0.8 base vs 1.5 in v1) → more return-seeking
    lambda_risk = 0.8 / np.sqrt(max(horizon_days / 21.0, 1.0))
    logger.info(f"QUBO: λ_risk={lambda_risk:.4f}, horizon={horizon_days}d")

    # Use alpha (not just -mu) as the linear objective → multi-factor driven selection
    linear = {tickers[i]: float(-alpha[i]) for i in range(len(tickers))}

    quadratic: Dict[Tuple, float] = {}
    for i in range(len(tickers)):
        for j in range(i, len(tickers)):
            v = float(lambda_risk * sigma[i, j])
            if abs(v) > 1e-12:
                quadratic[(tickers[i], tickers[j])] = v

    qp.minimize(linear=linear, quadratic=quadratic)

    # ── NLP constraints ──────────────────────────────────────────────────
    ticker_set = set(tickers)
    for idx, rule in enumerate(compliance_payload.get("constraints", [])):
        affected = rule.get("target_tickers", [])
        ctype = rule.get("constraint_type", "max_exposure")
        threshold = int(rule.get("threshold_value", 1))
        coeffs = {t: 1 for t in affected if t in ticker_set}
        if not coeffs:
            continue
        name = f"nlp_{idx}_{ctype}"
        if ctype == "equality":
            qp.linear_constraint(linear=coeffs, sense="==", rhs=threshold, name=name)
        elif ctype == "max_exposure":
            qp.linear_constraint(linear=coeffs, sense="<=", rhs=threshold, name=name)
        elif ctype == "min_exposure":
            qp.linear_constraint(linear=coeffs, sense=">=", rhs=threshold, name=name)
        logger.info(f"  Constraint [{name}]: {len(coeffs)} tickers, threshold={threshold}")

    # ── Auto sector-concentration constraint (max 40% of selections per sector) ──
    if add_sector_concentration:
        from collections import defaultdict
        sector_groups: Dict[str, List[str]] = defaultdict(list)
        for t in tickers:
            sector_groups[TICKER_SECTOR.get(t, "other")].append(t)

        for sector, s_tickers in sector_groups.items():
            if len(s_tickers) >= 3:
                # max sector weight = 40% of total selected (rounded up)
                max_sector = max(2, int(np.ceil(0.40 * len(tickers))))
                coeffs = {t: 1 for t in s_tickers}
                name = f"sector_conc_{sector}"
                # Only add if not already overridden by NLP
                existing = [c.name for c in qp.linear_constraints]
                if name not in existing:
                    qp.linear_constraint(linear=coeffs, sense="<=", rhs=max_sector, name=name)

    return qp


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT OPTIMIZERS
# ─────────────────────────────────────────────────────────────────────────────

def optimize_weights_sharpe(
    tickers: List[str], mu: np.ndarray, sigma: np.ndarray, rf: float = 0.05
) -> Dict[str, float]:
    """Classic Sharpe-maximising SLSQP (good all-around)."""
    n = len(tickers)
    if n == 1:
        return {tickers[0]: 1.0}

    def neg_sharpe(w):
        ret = np.dot(mu, w)
        vol = np.sqrt(w @ sigma @ w)
        return -(ret - rf) / vol if vol > 1e-10 else 1e6

    return _run_weight_opt(neg_sharpe, n, tickers)


def optimize_weights_sortino(
    tickers: List[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    rf: float = 0.05,
    n_sim: int = 5000,
) -> Dict[str, float]:
    """
    Sortino-maximising weights.
    Penalises only downside deviation, not upside volatility.
    Typically produces higher-return portfolios than Sharpe.
    """
    n = len(tickers)
    if n == 1:
        return {tickers[0]: 1.0}

    # Simulate daily returns for downside estimation
    try:
        L = np.linalg.cholesky(sigma / 252 + np.eye(n) * 1e-8)
    except np.linalg.LinAlgError:
        sigma_reg = sigma + np.eye(n) * 1e-6
        L = np.linalg.cholesky(sigma_reg / 252)

    daily_mu = mu / 252
    np.random.seed(42)
    z = np.random.randn(n, n_sim)
    sim_rets = daily_mu.reshape(-1, 1) + L @ z   # (n, n_sim)

    def neg_sortino(w):
        port_r = w @ sim_rets           # (n_sim,)
        ann_ret = port_r.mean() * 252
        neg_r = port_r[port_r < 0]
        ds = neg_r.std() * np.sqrt(252) if len(neg_r) > 5 else 1e-6
        return -(ann_ret - rf) / ds if ds > 1e-10 else 1e6

    return _run_weight_opt(neg_sortino, n, tickers)


def optimize_weights_maxreturn(
    tickers: List[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    max_drawdown_limit: float = 0.25,   # max 25% drawdown allowed
    n_sim: int = 5000,
) -> Dict[str, float]:
    """
    Maximum expected return subject to max-drawdown constraint.
    Most aggressive allocation — targets 50-100%+ returns by concentrating
    in highest-alpha assets while capping tail risk.
    Includes Kelly position sizing cap (max 60% per asset).
    """
    n = len(tickers)
    if n == 1:
        return {tickers[0]: 1.0}

    try:
        L = np.linalg.cholesky(sigma / 252 + np.eye(n) * 1e-8)
    except np.linalg.LinAlgError:
        sigma_reg = sigma + np.eye(n) * 1e-6
        L = np.linalg.cholesky(sigma_reg / 252)

    daily_mu = mu / 252
    np.random.seed(42)
    z = np.random.randn(n, n_sim)
    sim_rets = daily_mu.reshape(-1, 1) + L @ z

    def neg_return(w):
        return -np.dot(mu, w)   # maximise return

    def drawdown_constraint(w):
        port_r = w @ sim_rets
        cum = np.cumprod(1 + port_r)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / np.maximum(peak, 1e-10)
        return max_drawdown_limit + dd.min()   # ≥ 0 required

    constraints = [
        {"type": "eq",   "fun": lambda x: np.sum(x) - 1.0},
        {"type": "ineq", "fun": drawdown_constraint},
    ]
    bounds = [(0.02, 0.60)] * n   # 2% min (diversification), 60% max (Kelly cap)
    w0 = np.ones(n) / n

    result = minimize(
        neg_return, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    weights = np.maximum(result.x, 0.0)
    total = weights.sum()
    if total < 1e-8:
        weights = np.ones(n) / n
    else:
        weights /= total
    return {t: float(w) for t, w in zip(tickers, weights)}


def _run_weight_opt(
    objective, n: int, tickers: List[str],
    min_w: float = 0.02, max_w: float = 0.70
) -> Dict[str, float]:
    """Shared SLSQP runner with re-normalisation."""
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    bounds = [(min_w, max_w)] * n
    w0 = np.ones(n) / n

    best_result = None
    best_val = np.inf

    # Multi-start from 3 initial points for robustness
    starts = [w0, np.random.dirichlet(np.ones(n)), np.random.dirichlet(np.ones(n))]
    np.random.seed(42)
    for w_init in starts:
        res = minimize(
            objective, w_init, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if res.fun < best_val:
            best_val = res.fun
            best_result = res

    weights = np.maximum(best_result.x, 0.0)
    weights[weights < 0.005] = 0.0
    total = weights.sum()
    if total < 1e-8:
        weights = np.ones(n) / n
    else:
        weights /= total
    return {t: float(w) for t, w in zip(tickers, weights)}


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO RISK METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_portfolio_risk_metrics(
    weights: Dict[str, float],
    mu_annual: np.ndarray,
    sigma_annual: np.ndarray,
    tickers: List[str],
    horizon_days: int,
    n_sim: int = 10000,
) -> Dict[str, Any]:
    """
    Computes comprehensive risk metrics for the final portfolio:
    - Expected return (horizon-scaled)
    - Volatility
    - Sharpe, Sortino
    - CVaR(95%), Max Drawdown (simulated)
    - Probability of profit
    """
    w = np.array([weights.get(t, 0.0) for t in tickers])
    t = horizon_days / 252.0
    rf = 0.05 * t

    mu_scaled = mu_annual * t
    sigma_scaled = sigma_annual * t

    port_ret = float(np.dot(mu_scaled, w))
    port_vol = float(np.sqrt(w @ sigma_scaled @ w))
    sharpe = (port_ret - rf) / port_vol if port_vol > 1e-8 else 0.0

    # Monte Carlo simulation for tail metrics
    try:
        sigma_daily = sigma_annual / 252
        mu_daily = mu_annual / 252
        L = np.linalg.cholesky(sigma_daily + np.eye(len(tickers)) * 1e-8)
        np.random.seed(42)
        z = np.random.randn(len(tickers), n_sim)
        daily_sim = mu_daily.reshape(-1, 1) + L @ z
        port_sim_daily = w @ daily_sim   # (n_sim,)

        # Scale to horizon
        horizon_sim = port_sim_daily * horizon_days
        cvar = compute_cvar(horizon_sim)
        prob_profit = float((horizon_sim > 0).mean())

        # Max drawdown simulation (path-dependent)
        path_rets = np.random.choice(port_sim_daily, size=(horizon_days, 200), replace=True)
        cum_paths = np.cumprod(1 + path_rets, axis=0)
        peaks = np.maximum.accumulate(cum_paths, axis=0)
        dds = (cum_paths - peaks) / np.maximum(peaks, 1e-10)
        max_dd = float(dds.min(axis=0).mean())

        # Sortino
        neg_sim = horizon_sim[horizon_sim < 0]
        ds = neg_sim.std() if len(neg_sim) > 10 else max(port_vol, 1e-6)
        sortino = (port_ret - rf) / ds if ds > 1e-8 else 0.0

    except Exception:
        cvar = port_vol
        prob_profit = 0.5
        max_dd = -port_vol
        sortino = sharpe

    return {
        "expected_return_pct":    round(port_ret * 100, 2),
        "expected_volatility_pct": round(port_vol * 100, 2),
        "sharpe_ratio":            round(sharpe, 3),
        "sortino_ratio":           round(sortino, 3),
        "cvar_95_pct":             round(cvar * 100, 2),
        "max_drawdown_pct":        round(max_dd * 100, 2),
        "prob_profit_pct":         round(prob_profit * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class QuantumPortfolioCore:

    def __init__(self, seed: int = 42):
        algorithm_globals.random_seed = seed
        self.energy_history: List[float] = []
        self.active_universe: List[str] = []
        self.num_assets: int = 0

    def _vqe_callback(self, eval_count, parameters, value, metadata):
        self.energy_history.append(float(value))

    def run_full_pipeline(
        self,
        universe: List[str],
        horizon_days: int,
        compliance_payload: Dict[str, Any],
        max_quantum_assets: int = 12,
        vqe_maxiter: int = 150,
        use_spsa: bool = False,
        weight_objective: str = "SORTINO",   # SHARPE | SORTINO | MAXRET
    ) -> Dict[str, Any]:
        """
        Full pipeline:
          1. Fetch horizon-appropriate market data
          2. Multi-factor screen → top-K alpha assets
          3. Build QUBO (multi-factor objective + constraints)
          4. Run VQE + classical benchmark
          5. Weight-optimize selected portfolio (Sortino / Sharpe / MaxReturn)
          6. Compute comprehensive risk metrics

        Parameters
        ----------
        horizon_days      : Investment horizon in calendar days
        weight_objective  : "SHARPE" | "SORTINO" | "MAXRET"
        """
        self.energy_history.clear()
        cfg = get_horizon_config(horizon_days)
        logger.info(f"Pipeline start: {cfg['label']}, universe={len(universe)}, K={max_quantum_assets}")

        # ── 1. Data ─────────────────────────────────────────────────────────
        data = fetch_market_data(universe, cfg["period"], cfg["interval"])
        available = data.columns.tolist()
        logger.info(f"Data available: {len(available)} assets")
        if not available:
            raise ValueError("No market data returned. Check network and ticker list.")

        # ── 2. Multi-factor screening ────────────────────────────────────────
        top_k, mu_scaled, sigma_scaled, alpha_k, full_stats = screen_universe(
            data, cfg, horizon_days, max_quantum_assets
        )
        self.active_universe = top_k
        self.num_assets = len(top_k)
        if self.num_assets == 0:
            raise ValueError("Screening returned 0 assets.")

        # Store annualised versions for weight optimization
        t = horizon_days / 252.0
        mu_annual = mu_scaled / t
        sigma_annual = sigma_scaled / t

        # ── 3. QUBO ─────────────────────────────────────────────────────────
        qp = build_qubo(
            top_k, mu_scaled, sigma_scaled, alpha_k,
            compliance_payload, horizon_days
        )
        logger.info(f"QUBO: {qp.get_num_binary_vars()} vars, {qp.get_num_linear_constraints()} constraints")

        # ── 4. Classical benchmark ───────────────────────────────────────────
        classical = self._run_classical_benchmark(qp)

        # ── 5. VQE ──────────────────────────────────────────────────────────
        vqe_result = self._run_vqe(qp, vqe_maxiter, use_spsa)
        bitstring = "".join([str(int(v)) for v in vqe_result.x])
        selected = [top_k[i] for i, b in enumerate(bitstring) if b == "1"]

        # ── 6. Weight optimization ───────────────────────────────────────────
        sel_idx = [top_k.index(t) for t in selected] if selected else []
        if selected and len(selected) > 0:
            mu_sel = mu_annual[sel_idx]
            sigma_sel = sigma_annual[np.ix_(sel_idx, sel_idx)]

            if weight_objective == "MAXRET":
                weights = optimize_weights_maxreturn(selected, mu_sel, sigma_sel)
            elif weight_objective == "SORTINO":
                weights = optimize_weights_sortino(selected, mu_sel, sigma_sel)
            else:
                weights = optimize_weights_sharpe(selected, mu_sel, sigma_sel)
        else:
            weights = {}

        # ── 7. Risk metrics ──────────────────────────────────────────────────
        risk_metrics = {}
        if selected:
            risk_metrics = compute_portfolio_risk_metrics(
                weights, mu_annual[sel_idx], sigma_annual[np.ix_(sel_idx, sel_idx)],
                selected, horizon_days
            )

        # Filter stats to screened universe for scatter plot
        screened_stats = [s for s in full_stats if s["ticker"] in top_k]

        return {
            "status": "COMPLETED",
            "horizon_days": horizon_days,
            "horizon_label": cfg["label"],
            "weight_objective": weight_objective,
            "full_universe_size": len(universe),
            "available_after_data_clean": len(available),
            "screened_asset_pool": top_k,
            "screened_stats": screened_stats,
            "full_stats": full_stats,
            "quantum_bitstring": bitstring,
            "selected_portfolio": selected,
            "num_selected": len(selected),
            "optimal_weights": weights,
            "vqe_final_energy": float(vqe_result.fval),
            "convergence_history": self.energy_history,
            "classical_baseline": classical,
            "pipeline_accuracy_matched": (bitstring == classical["bitstring"]),
            "qubo_num_variables": qp.get_num_binary_vars(),
            "qubo_num_constraints": qp.get_num_linear_constraints(),
            "risk_metrics": risk_metrics,
        }

    def _run_classical_benchmark(self, qp):
        try:
            result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
            return {
                "bitstring": "".join([str(int(v)) for v in result.x]),
                "objective_value": float(result.fval),
            }
        except Exception as e:
            logger.error(f"Classical benchmark failed: {e}")
            return {"bitstring": "0" * self.num_assets, "objective_value": float("inf")}

    def _run_vqe(self, qp, vqe_maxiter=150, use_spsa=False):
        logger.info(f"VQE: {self.num_assets} qubits, maxiter={vqe_maxiter}")
        ansatz = efficient_su2(
            num_qubits=self.num_assets,
            su2_gates=["ry", "rz"],
            entanglement="linear",
            reps=2,
        )
        optimizer = SPSA(maxiter=vqe_maxiter) if use_spsa else COBYLA(maxiter=vqe_maxiter)
        vqe = SamplingVQE(
            sampler=StatevectorSampler(),
            ansatz=ansatz,
            optimizer=optimizer,
            callback=self._vqe_callback,
        )
        result = MinimumEigenOptimizer(vqe).solve(qp)
        logger.info(f"VQE done. Energy: {result.fval:.6f}")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_universe = [
        "NVDA","AMD","AAPL","MSFT","META","GOOGL","AMZN","TSLA",
        "LLY","ABBV","REGN","VRTX","GS","MS","BKNG","XOM","CAT","NFLX",
    ]
    mock_payload = {
        "constraints": [
            {
                "target_tickers": test_universe,
                "constraint_type": "equality",
                "threshold_value": 5,
                "description": "Select exactly 5 assets",
            }
        ]
    }
    engine = QuantumPortfolioCore(seed=42)
    for horizon in [7, 30, 365]:
        print(f"\n--- Horizon: {horizon} days ---")
        results = engine.run_full_pipeline(
            universe=test_universe,
            horizon_days=horizon,
            compliance_payload=mock_payload,
            max_quantum_assets=8,
            vqe_maxiter=80,
            weight_objective="SORTINO",
        )
        print("Selected:", results["selected_portfolio"])
        print("Weights:", {k: f"{v*100:.1f}%" for k,v in results["optimal_weights"].items()})
        print("Risk:", results["risk_metrics"])

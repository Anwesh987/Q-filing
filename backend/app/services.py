import os
import re
import sys
import math
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.models import OptimizeRequest, OptimizeResponse, PortfolioResult

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = [
    "NVDA", "AMD", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA",
    "LLY", "ABBV", "REGN", "VRTX", "UNH", "JNJ",
    "XOM", "CVX", "COP", "EOG",
    "JPM", "BAC", "GS", "MS",
    "BKNG", "NFLX", "DIS"
]

SECTOR_TICKERS = {
    "technology": ["NVDA", "AMD", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA"],
    "tech": ["NVDA", "AMD", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA"],
    "healthcare": ["LLY", "ABBV", "REGN", "VRTX", "UNH", "JNJ"],
    "health": ["LLY", "ABBV", "REGN", "VRTX", "UNH", "JNJ"],
    "pharma": ["LLY", "ABBV", "REGN", "VRTX", "UNH", "JNJ"],
    "energy": ["XOM", "CVX", "COP", "EOG"],
    "oil": ["XOM", "CVX", "COP", "EOG"],
    "financial": ["JPM", "BAC", "GS", "MS"],
    "finance": ["JPM", "BAC", "GS", "MS"],
    "bank": ["JPM", "BAC", "GS", "MS"],
    "communication": ["META", "GOOGL", "NFLX", "DIS"],
    "media": ["META", "GOOGL", "NFLX", "DIS"],
    "consumer": ["AMZN", "TSLA", "BKNG", "DIS"],
}


def _clean_universe(universe: List[str] | None) -> List[str]:
    selected = universe or DEFAULT_UNIVERSE
    return list(dict.fromkeys([ticker.strip().upper() for ticker in selected if ticker.strip()]))


def _sector_members(word: str, universe: List[str]) -> List[str]:
    raw = SECTOR_TICKERS.get(word.lower(), [])
    universe_set = set(universe)
    return [ticker for ticker in raw if ticker in universe_set]


def _extract_constraints_regex(text: str, universe: List[str]) -> Dict[str, Any]:
    """Lightweight fallback extractor. Good enough for demo deployment."""
    text_lower = text.lower()
    constraints: List[Dict[str, Any]] = []

    # exactly/select N assets
    for pattern in [
        r"(?:select|hold|include|choose)\s+exactly\s+(\d+)\s+assets?",
        r"exactly\s+(\d+)\s+assets?",
        r"portfolio\s+must\s+select\s+(\d+)\s+assets?",
    ]:
        for m in re.finditer(pattern, text_lower):
            constraints.append({
                "target_tickers": universe,
                "constraint_type": "equality",
                "threshold_value": int(m.group(1)),
                "description": f"Portfolio must select exactly {m.group(1)} assets"
            })

    # Maximum/cap sector exposure
    max_patterns = [
        r"(technology|tech|healthcare|health|pharma|energy|oil|financial|finance|bank|communication|media|consumer)\s+(?:sector\s+)?(?:exposure\s+)?(?:must\s+not\s+exceed|not\s+exceed|capped\s+at|cap(?:ped)?\s+at|maximum|at\s+most|limited\s+to)\s+(\d+)",
        r"(?:maximum|at\s+most|no\s+more\s+than|cap(?:ped)?\s+at|limited\s+to)\s+(\d+)\s+(technology|tech|healthcare|health|pharma|energy|oil|financial|finance|bank|communication|media|consumer)\s+(?:stocks?|assets?|holdings?)",
    ]
    for pattern in max_patterns:
        for m in re.finditer(pattern, text_lower):
            if m.group(1).isdigit():
                threshold, sector = int(m.group(1)), m.group(2)
            else:
                sector, threshold = m.group(1), int(m.group(2))
            tickers = _sector_members(sector, universe)
            if tickers:
                constraints.append({
                    "target_tickers": tickers,
                    "constraint_type": "max_exposure",
                    "threshold_value": threshold,
                    "description": f"{sector.title()} exposure capped at {threshold} assets"
                })

    # Minimum sector exposure
    min_patterns = [
        r"(technology|tech|healthcare|health|pharma|energy|oil|financial|finance|bank|communication|media|consumer)\s+(?:sector\s+)?(?:must\s+represent|minimum|at\s+least|required|min(?:imum)?\s+of)\s+(\d+)",
        r"(?:minimum|at\s+least|min(?:imum)?\s+of)\s+(\d+)\s+(technology|tech|healthcare|health|pharma|energy|oil|financial|finance|bank|communication|media|consumer)\s+(?:stocks?|assets?|holdings?)",
    ]
    for pattern in min_patterns:
        for m in re.finditer(pattern, text_lower):
            if m.group(1).isdigit():
                threshold, sector = int(m.group(1)), m.group(2)
            else:
                sector, threshold = m.group(1), int(m.group(2))
            tickers = _sector_members(sector, universe)
            if tickers:
                constraints.append({
                    "target_tickers": tickers,
                    "constraint_type": "min_exposure",
                    "threshold_value": threshold,
                    "description": f"{sector.title()} requires at least {threshold} assets"
                })

    # De-duplicate constraints by description
    seen = set()
    deduped = []
    for c in constraints:
        key = (tuple(c["target_tickers"]), c["constraint_type"], c["threshold_value"], c["description"])
        if key not in seen:
            deduped.append(c)
            seen.add(key)

    return {
        "constraints": deduped,
        "source_document": "frontend_text_input"
    }


def _build_demo_portfolio(payload: Dict[str, Any], universe: List[str], horizon_days: int, weight_objective: str) -> PortfolioResult:
    constraints = payload.get("constraints", [])
    warnings = []

    # Determine target count
    target_count = 8
    for c in constraints:
        if c.get("constraint_type") == "equality":
            target_count = max(1, min(int(c.get("threshold_value", 8)), len(universe)))
            break

    selected: List[str] = []

    # First satisfy min exposure constraints
    for c in constraints:
        if c.get("constraint_type") == "min_exposure":
            n = int(c.get("threshold_value", 0))
            for ticker in c.get("target_tickers", [])[:n]:
                if ticker not in selected and ticker in universe:
                    selected.append(ticker)

    # Add general assets
    for ticker in universe:
        if len(selected) >= target_count:
            break
        if ticker not in selected:
            selected.append(ticker)

    # Respect max exposure constraints roughly by trimming sector overages
    for c in constraints:
        if c.get("constraint_type") == "max_exposure":
            allowed = int(c.get("threshold_value", len(universe)))
            sector_set = set(c.get("target_tickers", []))
            sector_selected = [t for t in selected if t in sector_set]
            if len(sector_selected) > allowed:
                to_remove = sector_selected[allowed:]
                selected = [t for t in selected if t not in to_remove]
                warnings.append(
                    f"Trimmed {len(to_remove)} assets to respect: {c.get('description')}"
                )

    # Refill if trimming made it too small
    for ticker in universe:
        if len(selected) >= target_count:
            break
        if ticker not in selected:
            selected.append(ticker)

    selected = selected[:target_count] or universe[:min(8, len(universe))]

    # Simple but visually useful allocation
    if weight_objective == "MAXRET":
        raw = [1.0 / (i + 1) for i in range(len(selected))]
    elif weight_objective == "SHARPE":
        raw = [1.15 if i % 2 == 0 else 0.85 for i in range(len(selected))]
    else:  # SORTINO
        raw = [1.0 for _ in selected]

    total = sum(raw)
    weights = {ticker: round(raw[i] / total, 4) for i, ticker in enumerate(selected)}

    # Stable deterministic-ish demo metrics
    horizon_factor = min(max(horizon_days / 365, 0.02), 2.0)
    expected_return = round(0.09 + 0.02 * math.log1p(horizon_factor), 4)
    volatility = round(0.16 + (0.04 if weight_objective == "MAXRET" else 0.0), 4)
    sharpe = round(expected_return / max(volatility, 0.0001), 3)

    risk_metrics = {
        "expected_return_pct": round(expected_return * 100, 2),
        "expected_volatility_pct": round(volatility * 100, 2),
        "sharpe_ratio": sharpe,
        "sortino_ratio": round(sharpe * 1.28, 3),
        "cvar_95_pct": -4.2 if weight_objective != "MAXRET" else -5.6,
        "max_drawdown_pct": -18.5 if weight_objective != "MAXRET" else -24.0,
        "prob_profit_pct": 63.2 if weight_objective != "MAXRET" else 67.5,
    }

    return PortfolioResult(
        selected_assets=selected,
        weights=weights,
        risk_metrics=risk_metrics,
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
        num_constraints_applied=len(constraints),
        constraint_descriptions=[c.get("description", "") for c in constraints],
        vqe_energy=-42.3,
        horizon_days=horizon_days,
        weight_objective=weight_objective,
    )


def _try_existing_pipeline(req: OptimizeRequest, universe: List[str]) -> OptimizeResponse | None:
    """Try to call the existing repo pipeline if user copied integration_pipeline.py into backend/app."""
    if not req.use_existing_pipeline and os.environ.get("USE_EXISTING_PIPELINE", "false").lower() != "true":
        return None

    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    try:
        from integration_pipeline import QuantumComplianceOptimizer  # type: ignore

        optimizer = QuantumComplianceOptimizer(
            universe=universe,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        report = optimizer.run_from_text(
            regulatory_text=req.regulatory_text,
            horizon_days=req.horizon_days,
            weight_objective=req.weight_objective,
            use_quantum_core=os.environ.get("USE_QUANTUM_CORE", "false").lower() == "true",
        )

        portfolio_result = report.portfolio_result
        if hasattr(portfolio_result, "__dict__"):
            portfolio_result = portfolio_result.__dict__

        return OptimizeResponse(
            execution_status=report.execution_status,
            compliance_payload=report.compliance_payload,
            portfolio_result=PortfolioResult(**portfolio_result),
            warnings=report.warnings,
            logs=report.logs,
            mode="existing_pipeline",
        )
    except Exception as e:
        logger.exception("Existing pipeline failed; falling back to deployment demo mode.")
        return None


def optimize_from_text(req: OptimizeRequest) -> OptimizeResponse:
    universe = _clean_universe(req.universe)

    existing_response = _try_existing_pipeline(req, universe)
    if existing_response is not None:
        return existing_response

    logs = [
        "[INFO] Running deployment-safe demo optimizer",
        "[INFO] Regex constraint extraction enabled",
        "[INFO] Heavy quantum core disabled for cloud-stable first deployment"
    ]

    payload = _extract_constraints_regex(req.regulatory_text, universe)
    portfolio_result = _build_demo_portfolio(
        payload=payload,
        universe=universe,
        horizon_days=req.horizon_days,
        weight_objective=req.weight_objective,
    )

    warnings = []
    if not payload.get("constraints"):
        warnings.append("No explicit constraints detected; using default universe allocation.")

    return OptimizeResponse(
        execution_status="SUCCESS",
        compliance_payload=payload,
        portfolio_result=portfolio_result,
        warnings=warnings,
        logs=logs,
        mode="deployment_demo_fallback",
    )

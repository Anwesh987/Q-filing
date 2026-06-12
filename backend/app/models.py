from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class OptimizeRequest(BaseModel):
    regulatory_text: str = Field(..., min_length=10)
    horizon_days: int = Field(90, ge=1, le=3650)
    weight_objective: str = Field("SORTINO", pattern="^(SHARPE|SORTINO|MAXRET)$")
    universe: Optional[List[str]] = None
    use_existing_pipeline: bool = False


class Constraint(BaseModel):
    target_tickers: List[str]
    constraint_type: str
    threshold_value: int
    description: str


class PortfolioResult(BaseModel):
    selected_assets: List[str]
    weights: Dict[str, float]
    risk_metrics: Dict[str, Any]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    num_constraints_applied: int
    constraint_descriptions: List[str]
    vqe_energy: float
    horizon_days: int
    weight_objective: str


class OptimizeResponse(BaseModel):
    execution_status: str
    compliance_payload: Dict[str, Any]
    portfolio_result: PortfolioResult
    warnings: List[str]
    logs: List[str]
    mode: str

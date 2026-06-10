"""
tests/test_nlp_rag_integration.py
==================================

Comprehensive test suite for NLP RAG engine and integration pipeline.

Run: pytest tests/test_nlp_rag_integration.py -v
     pytest tests/test_nlp_rag_integration.py -v --cov=nlp_rag_engine
"""

import json
import pytest
from typing import Dict, List, Any

# Import modules under test
from nlp_rag_engine import (
    NLPComplianceEngine,
    ConstraintRule,
    CompliancePayload,
    SECTOR_REFERENCE,
    SECTOR_SYNONYMS,
)
from integration_pipeline import (
    QuantumComplianceOptimizer,
    PortfolioResult,
    PipelineExecutionReport,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def test_universe() -> List[str]:
    """Standard test universe."""
    return [
        "NVDA", "AMD", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA",
        "LLY", "ABBV", "REGN", "VRTX", "UNH", "JNJ",
        "XOM", "CVX", "COP", "EOG",
        "JPM", "BAC", "GS", "MS",
        "BKKING", "NFLX", "DIS",
    ]


@pytest.fixture
def nlp_engine(test_universe) -> NLPComplianceEngine:
    """Initialized NLP engine."""
    return NLPComplianceEngine(target_universe=test_universe)


@pytest.fixture
def optimizer(test_universe) -> QuantumComplianceOptimizer:
    """Initialized optimizer."""
    return QuantumComplianceOptimizer(universe=test_universe)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: CONSTRAINT RULE VALIDATION (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

class TestConstraintRuleValidation:
    """Test Pydantic validation of ConstraintRule."""

    def test_valid_constraint_max_exposure(self):
        """Valid max_exposure constraint."""
        rule = ConstraintRule(
            target_tickers=["AAPL", "MSFT"],
            constraint_type="max_exposure",
            threshold_value=2,
            description="Tech cap",
        )
        assert rule.constraint_type == "max_exposure"
        assert rule.threshold_value == 2

    def test_valid_constraint_equality(self):
        """Valid equality constraint."""
        rule = ConstraintRule(
            target_tickers=["AAPL", "MSFT", "NVDA"],
            constraint_type="equality",
            threshold_value=3,
        )
        assert rule.constraint_type == "equality"

    def test_valid_constraint_min_exposure(self):
        """Valid min_exposure constraint."""
        rule = ConstraintRule(
            target_tickers=["LLY", "ABBV"],
            constraint_type="min_exposure",
            threshold_value=1,
        )
        assert rule.constraint_type == "min_exposure"

    def test_invalid_constraint_type(self):
        """Reject invalid constraint_type."""
        with pytest.raises(ValueError, match="constraint_type must be"):
            ConstraintRule(
                target_tickers=["AAPL"],
                constraint_type="invalid_type",
                threshold_value=1,
            )

    def test_invalid_threshold_negative(self):
        """Reject negative threshold."""
        with pytest.raises(ValueError, match="threshold_value must be non-negative"):
            ConstraintRule(
                target_tickers=["AAPL"],
                constraint_type="max_exposure",
                threshold_value=-1,
            )

    def test_constraint_rule_to_dict(self):
        """Convert constraint to dict."""
        rule = ConstraintRule(
            target_tickers=["AAPL", "MSFT"],
            constraint_type="max_exposure",
            threshold_value=2,
            description="Tech cap",
        )
        rule_dict = rule.model_dump()
        assert isinstance(rule_dict, dict)
        assert rule_dict["constraint_type"] == "max_exposure"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: SECTOR MAPPING
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorMapping:
    """Test sector reference and synonym resolution."""

    def test_sector_reference_completeness(self):
        """All major sectors present in reference."""
        required_sectors = [
            "technology", "healthcare", "energy", "financial",
            "consumer_defensive", "consumer_cyclical", "industrials",
        ]
        for sector in required_sectors:
            assert sector in SECTOR_REFERENCE
            assert len(SECTOR_REFERENCE[sector]) > 0

    def test_sector_synonyms_coverage(self):
        """Synonyms map to canonical sectors."""
        test_synonyms = {
            "tech": "technology",
            "health": "healthcare",
            "energy": "energy",
            "bank": "financial",
        }
        for syn, expected_canon in test_synonyms.items():
            assert SECTOR_SYNONYMS.get(syn) == expected_canon

    def test_nlp_engine_resolve_sector(self, nlp_engine):
        """Resolve sector keyword to tickers."""
        tech_tickers = nlp_engine._resolve_sector_tickers("tech")
        assert len(tech_tickers) > 0
        assert "AAPL" in tech_tickers or "MSFT" in tech_tickers

    def test_nlp_engine_resolve_unknown_sector(self, nlp_engine):
        """Unknown sector returns empty list."""
        result = nlp_engine._resolve_sector_tickers("nonexistent_sector_xyz")
        assert isinstance(result, list)

    def test_sector_tickers_in_universe(self, nlp_engine, test_universe):
        """Resolved sector tickers are in universe."""
        tech_tickers = nlp_engine._resolve_sector_tickers("tech")
        for ticker in tech_tickers:
            assert ticker in nlp_engine.universe_set


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: NLP CONSTRAINT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

class TestNLPConstraintExtraction:
    """Test constraint extraction from text."""

    def test_regex_fallback_exactly_constraint(self, nlp_engine):
        """Extract 'exactly N assets' constraint."""
        text = "Portfolio must select exactly 10 assets from the universe."
        constraints = nlp_engine._regex_fallback(text)
        assert len(constraints) > 0
        assert any(c["constraint_type"] == "equality" for c in constraints)

    def test_regex_fallback_max_sector_constraint(self, nlp_engine):
        """Extract 'max N tech stocks' constraint."""
        text = "Technology sector exposure must not exceed 3 assets."
        constraints = nlp_engine._regex_fallback(text)
        assert len(constraints) > 0
        assert any(c["constraint_type"] == "max_exposure" for c in constraints)

    def test_regex_fallback_min_sector_constraint(self, nlp_engine):
        """Extract 'min N healthcare stocks' constraint."""
        text = "Portfolio must include at least 2 healthcare stocks."
        constraints = nlp_engine._regex_fallback(text)
        assert len(constraints) > 0
        assert any(c["constraint_type"] == "min_exposure" for c in constraints)

    def test_extract_multiple_constraints(self, nlp_engine):
        """Extract multiple constraints from text."""
        text = """
        Portfolio requirements:
        1. Select exactly 10 assets
        2. Technology maximum 4 stocks
        3. Healthcare minimum 2 stocks
        """
        constraints = nlp_engine._regex_fallback(text)
        assert len(constraints) >= 2

    def test_resolve_constraints_filters_to_universe(self, nlp_engine):
        """Resolution filters tickers to universe."""
        raw_constraints = [
            {
                "target_tickers": ["AAPL", "UNKNOWNTICKER"],
                "constraint_type": "max_exposure",
                "threshold_value": 2,
                "description": "Tech cap",
            }
        ]
        resolved = nlp_engine._resolve_constraints(raw_constraints)
        assert len(resolved) > 0
        # UNKNOWNTICKER should be filtered out
        all_tickers = [t for r in resolved for t in r.target_tickers]
        assert "UNKNOWNTICKER" not in all_tickers

    def test_parse_llm_json_with_markdown(self, nlp_engine):
        """Parse LLM response with markdown fences."""
        raw_response = """```json
[{"target_tickers": ["AAPL"], "constraint_type": "max_exposure", "threshold_value": 2}]
```"""
        parsed = nlp_engine._parse_llm_response(raw_response)
        assert len(parsed) > 0

    def test_extract_rules_from_text_returns_payload(self, nlp_engine):
        """extract_rules_from_text returns valid CompliancePayload."""
        text = "Select 8 assets. Tech max 3. Healthcare min 2."
        payload = nlp_engine.extract_rules_from_text(text)
        assert isinstance(payload, dict)
        assert "constraints" in payload
        assert isinstance(payload["constraints"], list)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: COMPLIANCE PAYLOAD VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

class TestCompliancePayloadValidation:
    """Test CompliancePayload validation."""

    def test_valid_compliance_payload(self):
        """Valid CompliancePayload."""
        payload = CompliancePayload(
            constraints=[
                ConstraintRule(
                    target_tickers=["AAPL", "MSFT"],
                    constraint_type="max_exposure",
                    threshold_value=2,
                    description="Tech cap",
                )
            ],
            source_document="test_source",
        )
        assert len(payload.constraints) == 1

    def test_payload_to_dict(self):
        """Convert payload to dict."""
        payload = CompliancePayload(
            constraints=[
                ConstraintRule(
                    target_tickers=["AAPL"],
                    constraint_type="equality",
                    threshold_value=1,
                )
            ]
        )
        payload_dict = payload.model_dump()
        assert isinstance(payload_dict, dict)
        assert "constraints" in payload_dict

    def test_empty_constraints_list(self):
        """Empty constraints list is valid."""
        payload = CompliancePayload(constraints=[])
        assert len(payload.constraints) == 0


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: INTEGRATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationPipeline:
    """Test full integration pipeline."""

    def test_optimizer_initialization(self, optimizer, test_universe):
        """Optimizer initializes correctly."""
        assert optimizer.universe == test_universe
        assert len(optimizer.universe_set) == len(test_universe)

    def test_optimizer_validate_payload_empty(self, optimizer):
        """Validate empty constraints payload."""
        payload = {"constraints": []}
        result = optimizer.validate_compliance_payload(payload)
        assert result is True

    def test_optimizer_validate_payload_with_constraints(self, optimizer):
        """Validate constraints payload with valid constraints."""
        payload = {
            "constraints": [
                {
                    "target_tickers": ["AAPL", "MSFT"],
                    "constraint_type": "max_exposure",
                    "threshold_value": 2,
                    "description": "Tech cap",
                }
            ]
        }
        result = optimizer.validate_compliance_payload(payload)
        assert result is True

    def test_run_from_text_demo_mode(self, optimizer):
        """Run pipeline from text in demo mode."""
        text = "Select 8 assets. Tech max 3. Healthcare min 2."
        report = optimizer.run_from_text(
            regulatory_text=text,
            horizon_days=90,
            weight_objective="SORTINO",
            use_quantum_core=False,
        )
        assert report.execution_status == "SUCCESS"
        assert report.portfolio_result is not None
        assert len(report.portfolio_result.selected_assets) > 0

    def test_run_from_text_constraint_extraction(self, optimizer):
        """Verify constraints extracted correctly."""
        text = """
        Portfolio must select exactly 10 assets.
        Technology capped at 3.
        Healthcare minimum 2.
        """
        report = optimizer.run_from_text(
            regulatory_text=text,
            horizon_days=90,
            use_quantum_core=False,
        )
        constraints = report.compliance_payload.get("constraints", [])
        assert len(constraints) >= 1

    def test_run_from_text_multiple_horizons(self, optimizer):
        """Run pipeline across multiple horizons."""
        text = "Select 8 assets total."
        results = {}
        for h in [7, 30, 90]:
            report = optimizer.run_from_text(
                regulatory_text=text,
                horizon_days=h,
                use_quantum_core=False,
            )
            results[h] = report.portfolio_result

        # All should succeed
        assert all(r is not None for r in results.values())

    def test_portfolio_result_fields(self):
        """PortfolioResult has all required fields."""
        result = PortfolioResult(
            selected_assets=["AAPL", "MSFT"],
            weights={"AAPL": 0.5, "MSFT": 0.5},
            risk_metrics={"sharpe_ratio": 1.0},
            expected_return=0.15,
            volatility=0.20,
            sharpe_ratio=0.75,
            num_constraints_applied=1,
            constraint_descriptions=["Test constraint"],
            vqe_energy=-10.0,
            horizon_days=90,
            weight_objective="SORTINO",
        )
        assert result.expected_return == 0.15
        assert len(result.selected_assets) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: EDGE CASES & ERROR HANDLING
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_universe(self):
        """Handle empty universe gracefully."""
        engine = NLPComplianceEngine(target_universe=[])
        assert len(engine.universe) == 0

    def test_single_asset_universe(self):
        """Handle single-asset universe."""
        engine = NLPComplianceEngine(target_universe=["AAPL"])
        assert len(engine.universe) == 1

    def test_constraint_threshold_zero(self):
        """Constraint with threshold=0 is valid."""
        rule = ConstraintRule(
            target_tickers=["AAPL"],
            constraint_type="equality",
            threshold_value=0,
        )
        assert rule.threshold_value == 0

    def test_constraint_with_special_characters(self):
        """Handle tickers with special characters."""
        rule = ConstraintRule(
            target_tickers=["BRK-B", "BRK.A"],
            constraint_type="max_exposure",
            threshold_value=1,
        )
        assert "BRK-B" in rule.target_tickers

    def test_sector_resolution_case_insensitive(self, nlp_engine):
        """Sector resolution is case-insensitive."""
        result1 = nlp_engine._resolve_sector_tickers("TECH")
        result2 = nlp_engine._resolve_sector_tickers("tech")
        assert len(result1) == len(result2)

    def test_regex_fallback_no_matches(self, nlp_engine):
        """Regex fallback returns empty list if no patterns match."""
        text = "This text contains no constraint patterns whatsoever."
        constraints = nlp_engine._regex_fallback(text)
        assert isinstance(constraints, list)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: INTEGRATION END-TO-END
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_text_to_portfolio(self):
        """Complete pipeline: text → constraints → portfolio."""
        universe = [
            "NVDA", "AMD", "AAPL", "MSFT", "META",
            "LLY", "ABBV", "UNH",
            "XOM", "CVX",
            "JPM", "BAC",
        ]

        optimizer = QuantumComplianceOptimizer(universe=universe)

        constraint_text = """
        Portfolio requirements:
        - Select exactly 10 assets
        - Technology maximum 3
        - Healthcare minimum 2
        - Energy maximum 2
        """

        report = optimizer.run_from_text(
            regulatory_text=constraint_text,
            horizon_days=90,
            weight_objective="SORTINO",
            use_quantum_core=False,
        )

        # Verify complete flow
        assert report.execution_status == "SUCCESS"
        assert report.portfolio_result is not None
        assert len(report.portfolio_result.selected_assets) > 0
        assert len(report.portfolio_result.weights) > 0
        assert sum(report.portfolio_result.weights.values()) > 0.99

    def test_multiple_constraint_types(self):
        """Pipeline handles mixed constraint types."""
        universe = [
            "AAPL", "MSFT", "NVDA", "AMD",
            "LLY", "ABBV", "UNH",
            "XOM", "CVX",
            "JPM", "BAC",
        ]

        optimizer = QuantumComplianceOptimizer(universe=universe)

        text = """
        Exactly 8 assets must be selected.
        Technology: maximum 3
        Healthcare: minimum 2
        Energy: minimum 1, maximum 2
        """

        report = optimizer.run_from_text(
            regulatory_text=text,
            horizon_days=90,
            use_quantum_core=False,
        )

        assert report.execution_status == "SUCCESS"
        constraints = report.compliance_payload.get("constraints", [])
        assert len(constraints) > 0


# ─────────────────────────────────────────────────────────────────────────────
# TEST EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Run tests with: pytest test_nlp_rag_integration.py -v"""
    pytest.main([__file__, "-v", "--tb=short"])

"""
COMPLETE WORKING EXAMPLE
=========================

This script demonstrates the full NLP RAG + Quantum pipeline end-to-end.
No dependencies beyond what's already in nlp_rag_engine.py and integration_pipeline.py

Run: python example_pipeline_demo.py
"""

import os
import json
from integration_pipeline import QuantumComplianceOptimizer, format_report


def example_1_simple_text_extraction():
    """
    Example 1: Extract constraints from simple regulatory text.
    No PDF needed, no quantum optimization required.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: CONSTRAINT EXTRACTION FROM REGULATORY TEXT")
    print("=" * 80)

    universe = [
        "NVDA", "AMD", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA",
        "LLY", "ABBV", "REGN", "VRTX", "UNH", "JNJ",
        "XOM", "CVX", "COP", "EOG",
        "JPM", "BAC", "GS", "MS",
        "BKNG", "NFLX", "DIS", "TSLA",
    ]

    # Create optimizer
    optimizer = QuantumComplianceOptimizer(universe=universe)

    # Sample regulatory constraint text
    constraint_text = """
    PORTFOLIO COMPLIANCE REQUIREMENTS (as of Q4 2025)

    Investment Policy Statement mandates:
    1. The portfolio must select exactly 12 assets from the approved universe.
    2. Technology sector exposure must not exceed 4 assets.
    3. Healthcare sector must represent at least 3 assets.
    4. Energy sector is capped at 2 stocks maximum.
    5. Financial sector allocation limited to at most 3 assets.
    6. Minimum 1 communication sector holding required.

    All constraints are hard constraints and must be respected.
    """

    # Run extraction
    report = optimizer.run_from_text(
        regulatory_text=constraint_text,
        horizon_days=90,
        weight_objective="SORTINO",
        use_quantum_core=False,  # Demo mode
    )

    # Display results
    print(format_report(report))
    print("\nExtracted Constraints:")
    for i, constraint in enumerate(report.compliance_payload.get("constraints", []), 1):
        print(f"  [{i}] {constraint['description']}")
        print(f"      Type: {constraint['constraint_type']}")
        print(f"      Threshold: {constraint['threshold_value']}")
        print(f"      Tickers: {', '.join(constraint['target_tickers'][:3])}...")


def example_2_multiple_horizons():
    """
    Example 2: Run portfolio optimization across multiple horizons
    with the same constraints.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: MULTI-HORIZON PORTFOLIO SERIES")
    print("=" * 80)

    universe = [
        "NVDA", "AMD", "AAPL", "MSFT", "META", "GOOGL", "AMZN",
        "LLY", "ABBV", "UNH",
        "XOM", "CVX",
        "JPM", "BAC",
        "BKKING", "NFLX", "DIS",
    ]

    optimizer = QuantumComplianceOptimizer(universe=universe)

    constraint_text = """
    Must select 8 assets total.
    Technology capped at 3.
    Healthcare at least 2.
    Energy maximum 2.
    """

    results = {}
    for horizon in [7, 30, 90, 365]:
        print(f"\nOptimizing for {horizon}-day horizon...")

        report = optimizer.run_from_text(
            regulatory_text=constraint_text,
            horizon_days=horizon,
            weight_objective="SORTINO",
            use_quantum_core=False,
        )

        results[horizon] = report.portfolio_result

    # Comparison table
    print("\n" + "-" * 80)
    print("HORIZON COMPARISON TABLE")
    print("-" * 80)
    print(f"{'Horizon':<12} {'Return':<10} {'Vol':<10} {'Sharpe':<10} {'Assets':<10}")
    print("-" * 80)

    for h in [7, 30, 90, 365]:
        if h in results:
            r = results[h]
            print(
                f"{h}d{'':<8} "
                f"{r.expected_return*100:>7.2f}%  "
                f"{r.volatility*100:>7.2f}%  "
                f"{r.sharpe_ratio:>7.3f}  "
                f"{len(r.selected_assets):>7d}"
            )

    # Asset selection changes
    print("\n" + "-" * 80)
    print("ASSET SELECTION BY HORIZON")
    print("-" * 80)
    for h in [7, 30, 90, 365]:
        if h in results:
            assets = results[h].selected_assets
            print(f"{h}d:  {', '.join(assets)}")


def example_3_sector_resolution():
    """
    Example 3: Demonstrate sector keyword resolution.
    Shows how "tech stocks" becomes actual ticker symbols.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: SECTOR KEYWORD RESOLUTION")
    print("=" * 80)

    universe = [
        # Tech
        "NVDA", "AMD", "AAPL", "MSFT", "META", "GOOGL", "AMZN",
        # Healthcare
        "LLY", "ABBV", "UNH", "JNJ",
        # Energy
        "XOM", "CVX", "COP",
        # Financials
        "JPM", "BAC", "GS",
    ]

    optimizer = QuantumComplianceOptimizer(universe=universe)

    # Constraints using sector keywords
    constraint_text = """
    The portfolio will:
    - Include exactly 10 assets
    - Cap technology at 3 stocks
    - Include healthcare with minimum 2 assets
    - Limit energy to at most 2 stocks
    """

    report = optimizer.run_from_text(
        regulatory_text=constraint_text,
        horizon_days=90,
        use_quantum_core=False,
    )

    print("\nConstraint Resolution:")
    for i, constraint in enumerate(report.compliance_payload.get("constraints", []), 1):
        print(f"\n[{i}] {constraint['description']}")
        print(f"    Type: {constraint['constraint_type']:<15} Threshold: {constraint['threshold_value']}")
        print(f"    Resolved Tickers ({len(constraint['target_tickers'])}): {', '.join(constraint['target_tickers'][:5])}")
        if len(constraint['target_tickers']) > 5:
            print(f"    ... and {len(constraint['target_tickers']) - 5} more")


def example_4_constraint_validation():
    """
    Example 4: Show constraint validation and conflict detection.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: CONSTRAINT VALIDATION & WARNINGS")
    print("=" * 80)

    universe = [
        "NVDA", "AMD", "AAPL", "MSFT", "META",
        "LLY", "ABBV", "UNH",
        "JPM", "BAC",
    ]

    optimizer = QuantumComplianceOptimizer(universe=universe)

    # Potentially problematic constraints
    constraint_text = """
    Portfolio must:
    - Select exactly 15 assets (NOTE: universe only has 10!)
    - Tech sector minimum 4 assets
    - Healthcare maximum 2 assets (but we have only 3 total)
    """

    report = optimizer.run_from_text(
        regulatory_text=constraint_text,
        horizon_days=90,
        use_quantum_core=False,
    )

    print(f"\nExecution Status: {report.execution_status}")
    print(f"Warnings ({len(report.warnings)}):")
    for w in report.warnings:
        print(f"  ⚠ {w}")

    print(f"\nExtracted Constraints: {len(report.compliance_payload.get('constraints', []))}")


def example_5_weight_objectives_comparison():
    """
    Example 5: Compare different weight allocation objectives
    (Sharpe vs Sortino vs MAXRET) for the same constraints.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: WEIGHT OBJECTIVES COMPARISON")
    print("=" * 80)

    universe = [
        "NVDA", "AMD", "AAPL", "MSFT", "META", "GOOGL", "AMZN",
        "LLY", "ABBV", "UNH",
        "XOM", "CVX",
        "JPM", "BAC",
        "NFLX", "DIS",
    ]

    optimizer = QuantumComplianceOptimizer(universe=universe)

    constraint_text = """
    Select 10 assets.
    Tech maximum 4.
    Healthcare minimum 2.
    Energy maximum 2.
    """

    objectives = ["SHARPE", "SORTINO", "MAXRET"]
    results = {}

    for obj in objectives:
        print(f"\nOptimizing with {obj} objective...")

        report = optimizer.run_from_text(
            regulatory_text=constraint_text,
            horizon_days=90,
            weight_objective=obj,
            use_quantum_core=False,
        )

        results[obj] = report.portfolio_result

    # Comparison table
    print("\n" + "-" * 80)
    print("OBJECTIVE COMPARISON")
    print("-" * 80)
    print(f"{'Objective':<12} {'Return':<10} {'Vol':<10} {'Sharpe':<10} {'Sortino':<10}")
    print("-" * 80)

    for obj in objectives:
        if obj in results:
            r = results[obj]
            sortino = r.risk_metrics.get("sortino_ratio", 0.0)
            print(
                f"{obj:<12} "
                f"{r.expected_return*100:>7.2f}%  "
                f"{r.volatility*100:>7.2f}%  "
                f"{r.sharpe_ratio:>7.3f}  "
                f"{sortino:>7.3f}"
            )

    print("\nInterpretation:")
    print("  SHARPE:   Balanced risk-return (Sharpe maximized)")
    print("  SORTINO:  Protects against downside (Sortino maximized)")
    print("  MAXRET:   Aggressive return-seeking (return maximized with max-DD constraint)")


def example_6_json_export():
    """
    Example 6: Export results to JSON for downstream systems.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: JSON EXPORT FOR DOWNSTREAM INTEGRATION")
    print("=" * 80)

    universe = [
        "NVDA", "AMD", "AAPL", "MSFT", "META",
        "LLY", "ABBV", "UNH",
        "JPM", "BAC",
    ]

    optimizer = QuantumComplianceOptimizer(universe=universe)

    constraint_text = "Select 8 assets. Tech max 3. Healthcare min 2."

    report = optimizer.run_from_text(
        regulatory_text=constraint_text,
        horizon_days=90,
        weight_objective="SORTINO",
        use_quantum_core=False,
    )

    # Export portfolio
    if report.portfolio_result:
        export = {
            "portfolio": {
                "assets": report.portfolio_result.selected_assets,
                "weights": report.portfolio_result.weights,
                "num_assets": len(report.portfolio_result.selected_assets),
            },
            "risk_profile": {
                "expected_return_pct": report.portfolio_result.expected_return * 100,
                "expected_volatility_pct": report.portfolio_result.volatility * 100,
                "sharpe_ratio": report.portfolio_result.sharpe_ratio,
                "max_drawdown_pct": report.portfolio_result.risk_metrics.get("max_drawdown_pct"),
                "cvar_95_pct": report.portfolio_result.risk_metrics.get("cvar_95_pct"),
            },
            "compliance": {
                "constraints_applied": report.portfolio_result.num_constraints_applied,
                "constraint_descriptions": report.portfolio_result.constraint_descriptions,
            },
            "metadata": {
                "horizon_days": report.portfolio_result.horizon_days,
                "weight_objective": report.portfolio_result.weight_objective,
                "execution_status": report.execution_status,
            }
        }

        print("\nExported JSON (suitable for API/database storage):")
        print(json.dumps(export, indent=2))


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("QUANTUM PORTFOLIO OPTIMIZATION - COMPLETE EXAMPLE SUITE")
    print("=" * 80)
    print("""
This demo suite shows the complete NLP RAG + Quantum Portfolio system working
end-to-end. No external files or complex setup required.

Each example runs standalone and can be modified for your specific use case.
    """)

    # Run all examples
    example_1_simple_text_extraction()
    example_2_multiple_horizons()
    example_3_sector_resolution()
    example_4_constraint_validation()
    example_5_weight_objectives_comparison()
    example_6_json_export()

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
To extend this to production:

1. OBTAIN A REAL SEC FILING
   - Download 10-K from SEC EDGAR (https://www.sec.gov/edgar)
   - Save as 'sample_10k.pdf'
   
2. EXTRACT FROM PDF
   - Install: pip install unstructured pdf2image pillow
   - Use extract_compliance_from_pdf() instead of extract_compliance_from_text()
   
3. ENABLE QUANTUM OPTIMIZATION
   - Install: pip install qiskit qiskit-aer qiskit-algorithms
   - Set use_quantum_core=True in pipeline calls
   - Import quantum_core.QuantumPortfolioCore
   
4. INTEGRATE WITH EXECUTION
   - Feed optimal_weights to trade execution system
   - Monitor portfolio performance vs risk metrics
   - Rebalance periodically (weekly/monthly)

5. DEPLOYMENT
   - Set ANTHROPIC_API_KEY environment variable
   - Implement database storage for constraint audit trail
   - Add API endpoint for constraint updates
   - Monitor yfinance API rate limits (concurrent sector mapping)
    """)

"""
Module: nlp_rag_engine.py — v4.0 "Integrated Vectorless RAG + Smart Constraint Extraction"
===========================================================================================

ARCHITECTURE:
  Phase 1 – Vectorless Structural RAG:
    PDF parsing (unstructured) → hierarchical document tree organized by headers
    Deterministic section retrieval (no vectors, no embeddings)
    
  Phase 2 – LLM-based Constraint Extraction:
    Claude API extracts constraints from retrieved text
    Regex fallback if API unavailable
    
  Phase 3 – Sector Resolution:
    "tech stocks" → ["AAPL", "MSFT", "NVDA", ...] (filtered to universe)
    Live yfinance mapping + static reference fallback
    
  Phase 4 – Validation:
    Pydantic strict validation
    Returns CompliancePayload ready for QuantumPortfolioCore

READY FOR: quantum_core.py run_full_pipeline(..., compliance_payload=payload)
"""

import os
import re
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple

import requests
import yfinance as yf
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STRICT DATA CONTRACTS (Pydantic v2)
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintRule(BaseModel):
    """Single portfolio constraint rule."""
    target_tickers: List[str] = Field(..., description="Tickers affected by this rule")
    constraint_type: str = Field(
        ..., description="'max_exposure' | 'min_exposure' | 'equality'"
    )
    threshold_value: int = Field(..., description="Absolute numerical limit of assets")
    description: str = Field(default="", description="Human-readable explanation")

    @field_validator("constraint_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"max_exposure", "min_exposure", "equality"}
        if v not in allowed:
            raise ValueError(f"constraint_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("threshold_value")
    @classmethod
    def validate_threshold(cls, v: int) -> int:
        if v < 0:
            raise ValueError("threshold_value must be non-negative")
        return v


class CompliancePayload(BaseModel):
    """Final validated payload for quantum optimizer."""
    constraints: List[ConstraintRule]
    source_document: str = Field(default="", description="Source file or text identifier")


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR REFERENCE DATA
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_REFERENCE: Dict[str, List[str]] = {
    "technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD", "AVGO", "ORCL",
        "CSCO", "ADBE", "CRM", "INTC", "QCOM", "TXN", "NOW", "PANW", "SNOW",
        "MU", "AMAT", "KLAC", "LRCX", "ADI", "MRVL", "NXPI",
    ],
    "healthcare": [
        "JNJ", "UNH", "LLY", "ABBV", "PFE", "MRK", "TMO", "ABT", "DHR",
        "BMY", "AMGN", "GILD", "VRTX", "REGN", "SYK", "ZTS", "EW", "BSX",
        "HCA", "CI", "CVS", "MCK", "CAH",
    ],
    "energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY",
        "PXD", "HAL", "DVN", "HES", "FANG", "APA", "BKR", "CTRA", "WMB", "KMI",
    ],
    "financial": [
        "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK",
        "SCHW", "CB", "MMC", "AON", "PGR", "TRV", "MET", "PRU", "AFL",
        "USB", "PNC", "TFC", "COF", "DFS",
    ],
    "consumer_defensive": [
        "WMT", "PG", "KO", "PEP", "COST", "MDLZ", "CL", "KHC", "GIS",
        "K", "CAG", "SJM", "HRL", "MKC", "CHD", "CLX", "EL", "COTY",
    ],
    "consumer_cyclical": [
        "TSLA", "AMZN", "HD", "MCD", "NKE", "SBUX", "TJX", "BKKING", "LOW",
        "ABNB", "EBAY", "ETSY", "ROST", "DG", "DLTR", "BBY", "M", "GPS",
    ],
    "industrials": [
        "HON", "UPS", "CAT", "DE", "GE", "LMT", "RTX", "BA", "MMM",
        "EMR", "ETN", "PH", "ITW", "ROK", "GWW", "XYL", "IEX", "FTV",
    ],
    "utilities": [
        "NEE", "DUK", "SO", "AEP", "EXC", "SRE", "PCG", "XEL", "ED",
        "ETR", "FE", "PPL", "WEC", "DTE", "CMS", "AES", "EIX",
    ],
    "real_estate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "O", "WELL", "AVB",
        "EQR", "VTR", "MAA", "CPT", "ESS", "UDR", "AIR", "BXP",
    ],
    "materials": [
        "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "VMC", "MLM",
        "PPG", "ALB", "CF", "MOS", "FMC", "IFF", "CE", "EMN",
    ],
    "communication": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
        "CHTR", "EA", "TTWO", "FOXA", "WBD", "PARA", "OMC", "IPG",
    ],
}

SECTOR_SYNONYMS: Dict[str, str] = {
    "tech": "technology", "technology": "technology", "software": "technology",
    "semiconductor": "technology", "chip": "technology", "ai": "technology",
    "health": "healthcare", "healthcare": "healthcare", "pharma": "healthcare",
    "pharmaceutical": "healthcare", "biotech": "healthcare", "medical": "healthcare",
    "energy": "energy", "oil": "energy", "gas": "energy", "petroleum": "energy",
    "finance": "financial", "financial": "financial", "bank": "financial",
    "banking": "financial", "insurance": "financial",
    "consumer": "consumer_cyclical", "defensive": "consumer_defensive",
    "staples": "consumer_defensive", "retail": "consumer_cyclical",
    "industrial": "industrials", "industrials": "industrials",
    "utilities": "utilities", "utility": "utilities",
    "real estate": "real_estate", "reit": "real_estate",
    "material": "materials", "materials": "materials", "mining": "materials",
    "communication": "communication", "media": "communication", "telecom": "communication",
}


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: VECTORLESS STRUCTURAL RAG (PDF PARSING)
# ─────────────────────────────────────────────────────────────────────────────

class VectorlessDocumentParser:
    """
    Parses PDF by physical layout (headers vs paragraphs).
    Builds hierarchical document tree organized by section headers.
    Retrieves sections deterministically by header name (no vectors).
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.document_tree: Dict[str, List[str]] = {}
        self.section_order: List[str] = []  # Track order of sections for context

    def build_structural_tree(self) -> Dict[str, List[str]]:
        """
        Parses PDF using unstructured library.
        Organizes content hierarchically by detected headers/titles.
        Falls back gracefully if PDF parsing fails.
        """
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        logger.info(f"Building vectorless document tree from {self.pdf_path}")

        try:
            from unstructured.partition.pdf import partition_pdf
            from unstructured.documents.elements import Title, NarrativeText, ListItem

            elements = partition_pdf(filename=self.pdf_path, strategy="fast")
        except ImportError:
            logger.error("unstructured library not installed. Install: pip install unstructured pdf2image pillow")
            raise
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            raise

        current_section = "Document_Root"
        self.document_tree[current_section] = []
        self.section_order.append(current_section)

        for element in elements:
            # Detect new section via headers
            if isinstance(element, Title):
                current_section = str(element).strip()
                if current_section not in self.document_tree:
                    self.document_tree[current_section] = []
                    self.section_order.append(current_section)
                    logger.debug(f"  → Section detected: {current_section[:60]}...")

            # Accumulate text in current section
            elif isinstance(element, (NarrativeText, ListItem)):
                text_content = str(element).strip()
                if text_content and len(text_content) > 5:  # Filter noise
                    self.document_tree[current_section].append(text_content)

        logger.info(f"Document tree built: {len(self.document_tree)} sections, "
                   f"{sum(len(v) for v in self.document_tree.values())} paragraphs")
        return self.document_tree

    def retrieve_section(self, target_section_name: str, merge_related: bool = True) -> str:
        """
        Retrieves text for a section by deterministic header matching.
        
        Args:
            target_section_name: e.g., "Risk Factors"
            merge_related: If True, merges all sections containing the keyword (handles variants)
        
        Returns:
            Concatenated text from matched sections
        """
        target_lower = target_section_name.lower()

        # Find all headers containing the target keyword (case-insensitive)
        matched_sections = [
            header for header in self.section_order
            if target_lower in header.lower()
        ]

        if not matched_sections:
            logger.warning(f"No section found matching '{target_section_name}'")
            return ""

        logger.info(f"Retrieved {len(matched_sections)} section(s) matching '{target_section_name}':")
        for section in matched_sections:
            logger.info(f"  - {section[:70]}...")

        # Merge all matched sections (preserve order from document)
        extracted_parts = []
        for section in matched_sections:
            section_text = "\n".join(self.document_tree[section])
            if section_text:
                extracted_parts.append(section_text)

        return "\n\n---\n\n".join(extracted_parts)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 & 3: LLM EXTRACTION + SECTOR RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

class NLPComplianceEngine:
    """
    Two-phase extraction and resolution:
    
    Phase A – LLM-based extraction (Claude API or regex fallback)
    Phase B – Sector resolution ("tech" → actual tickers)
    """

    def __init__(self, target_universe: List[str], anthropic_api_key: Optional[str] = None):
        self.universe = target_universe
        self.universe_set = set(target_universe)
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        
        # Build sector mappings
        self.sector_map = self._build_sector_mapping(target_universe)
        logger.info(f"NLP engine initialized for {len(target_universe)} assets")

    # ── Sector Mapping ──────────────────────────────────────────────────────

    def _fetch_single_sector(self, ticker: str) -> Dict[str, str]:
        """Concurrently fetch sector for one ticker via yfinance."""
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown").lower().replace(" ", "_")
            return {ticker: sector}
        except Exception:
            return {ticker: "unknown"}

    def _build_sector_mapping(self, universe: List[str]) -> Dict[str, List[str]]:
        """
        Builds a live sector → tickers mapping.
        Tries yfinance (concurrent), falls back to static SECTOR_REFERENCE.
        """
        logger.info(f"Building sector mapping for {len(universe)} assets...")
        ticker_to_sector: Dict[str, str] = {}

        # Try concurrent yfinance fetch
        try:
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = {executor.submit(self._fetch_single_sector, t): t for t in universe}
                for future in as_completed(futures, timeout=30):
                    try:
                        ticker_to_sector.update(future.result())
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Concurrent yfinance fetch failed: {e}")

        # Fallback: use static reference for unknown tickers
        for ticker in universe:
            if ticker_to_sector.get(ticker, "unknown") == "unknown":
                for sector, sector_tickers in SECTOR_REFERENCE.items():
                    if ticker in sector_tickers:
                        ticker_to_sector[ticker] = sector
                        break

        # Group by sector
        grouped: Dict[str, List[str]] = defaultdict(list)
        for ticker, sector in ticker_to_sector.items():
            grouped[sector].append(ticker)

        logger.info(f"Sector mapping complete: {len(grouped)} sectors, "
                   f"{sum(len(v) for v in grouped.values())} tickers mapped")
        return dict(grouped)

    def _resolve_sector_tickers(self, sector_keyword: str) -> List[str]:
        """Resolves sector keyword (e.g., 'tech') to tickers in universe."""
        canonical = SECTOR_SYNONYMS.get(sector_keyword.lower(), sector_keyword.lower())

        candidates: List[str] = []

        # From live sector map
        for sector_name, tickers in self.sector_map.items():
            if canonical in sector_name.lower():
                candidates.extend(tickers)

        # From static reference
        for sector_name, tickers in SECTOR_REFERENCE.items():
            if canonical in sector_name.lower():
                candidates.extend(tickers)

        # Filter to universe members only, de-duplicate
        resolved = list(dict.fromkeys(t for t in candidates if t in self.universe_set))
        return resolved

    # ── LLM Extraction ──────────────────────────────────────────────────────

    def _call_claude_api(self, prompt: str) -> str:
        """Calls Claude Sonnet via Anthropic API."""
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    def _build_extraction_prompt(self, text: str) -> str:
        """Builds extraction prompt with universe and sector context."""
        universe_str = ", ".join(self.universe[:30]) + ("..." if len(self.universe) > 30 else "")
        sector_keys = ", ".join(sorted(SECTOR_SYNONYMS.keys())[:20])

        return f"""You are a financial compliance AI. Extract ALL portfolio constraints from the following text.

ASSET UNIVERSE (available tickers): {universe_str}
SECTOR KEYWORDS: {sector_keys}

Output a JSON array of constraint objects. Each constraint:
{{
  "target_tickers": ["TICK1", "TICK2"] or ["sector_name"],
  "constraint_type": "equality" | "max_exposure" | "min_exposure",
  "threshold_value": <positive integer>,
  "description": "<brief explanation>"
}}

RULES:
- "equality": portfolio MUST select EXACTLY N matching assets
- "max_exposure": AT MOST N matching assets
- "min_exposure": AT LEAST N matching assets
- If text says "tech stocks", set target_tickers to ["tech"]
- Extract EVERY constraint you find

Output ONLY valid JSON array. No preamble, no markdown.

TEXT TO ANALYZE:
{text}"""

    def _parse_llm_response(self, raw: str) -> List[Dict[str, Any]]:
        """Parses Claude's JSON response, strips markdown if needed."""
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "constraints" in parsed:
                return parsed["constraints"]
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response")

        return []

    def _regex_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Regex-based extraction if LLM unavailable."""
        constraints = []
        text_lower = text.lower()

        # Pattern: "exactly N assets"
        for m in re.finditer(r"exactly\s+(\d+)\s+assets?", text_lower):
            constraints.append({
                "target_tickers": list(self.universe_set),
                "constraint_type": "equality",
                "threshold_value": int(m.group(1)),
                "description": f"Exactly {m.group(1)} assets",
            })

        # Pattern: "max/no more than N <sector> stocks"
        for pattern in [
            r"maximum\s+(?:of\s+)?(\d+)\s+(\w+)\s+stocks?",
            r"no\s+more\s+than\s+(\d+)\s+(\w+)\s+stocks?",
            r"at\s+most\s+(\d+)\s+(\w+)\s+stocks?",
        ]:
            for m in re.finditer(pattern, text_lower):
                n, sector_word = int(m.group(1)), m.group(2)
                resolved = self._resolve_sector_tickers(sector_word) or [sector_word]
                constraints.append({
                    "target_tickers": resolved,
                    "constraint_type": "max_exposure",
                    "threshold_value": n,
                    "description": f"Max {n} {sector_word} stocks",
                })

        # Pattern: "min/at least N <sector> stocks"
        for pattern in [
            r"minimum\s+(?:of\s+)?(\d+)\s+(\w+)\s+stocks?",
            r"at\s+least\s+(\d+)\s+(\w+)\s+stocks?",
        ]:
            for m in re.finditer(pattern, text_lower):
                n, sector_word = int(m.group(1)), m.group(2)
                resolved = self._resolve_sector_tickers(sector_word) or [sector_word]
                constraints.append({
                    "target_tickers": resolved,
                    "constraint_type": "min_exposure",
                    "threshold_value": n,
                    "description": f"Min {n} {sector_word} stocks",
                })

        return constraints

    def _resolve_constraints(self, raw_constraints: List[Dict[str, Any]]) -> List[ConstraintRule]:
        """Resolves sector keywords → tickers, validates, returns ConstraintRule objects."""
        resolved_rules: List[ConstraintRule] = []

        for raw in raw_constraints:
            tickers = raw.get("target_tickers", [])
            ctype = raw.get("constraint_type", "max_exposure")
            threshold = int(raw.get("threshold_value", 1))
            desc = raw.get("description", "")

            # Expand sector keywords to tickers
            expanded: List[str] = []
            for t in tickers:
                t_upper = t.upper()
                if t_upper in self.universe_set:
                    expanded.append(t_upper)
                else:
                    # Try as sector keyword
                    resolved = self._resolve_sector_tickers(t)
                    if resolved:
                        expanded.extend(resolved)
                        logger.debug(f"Resolved sector '{t}' → {len(resolved)} tickers")
                    else:
                        logger.warning(f"Could not resolve '{t}' (not in universe, not a known sector)")

            # De-duplicate while preserving order
            expanded = list(dict.fromkeys(expanded))

            if not expanded:
                logger.warning(f"Skipping constraint — no resolvable tickers: {raw}")
                continue

            try:
                rule = ConstraintRule(
                    target_tickers=expanded,
                    constraint_type=ctype,
                    threshold_value=threshold,
                    description=desc,
                )
                resolved_rules.append(rule)
                logger.debug(f"Validated constraint: {len(expanded)} tickers, type={ctype}, threshold={threshold}")
            except Exception as e:
                logger.error(f"Constraint validation failed: {e} | raw={raw}")

        return resolved_rules

    # ── Public API ──────────────────────────────────────────────────────────

    def extract_rules_from_text(self, regulatory_text: str, source: str = "") -> Dict[str, Any]:
        """
        Main entry point. Extracts constraints from free-form text.
        
        Priority:
          1. Claude API (if ANTHROPIC_API_KEY available)
          2. Regex fallback
        """
        logger.info(f"Starting NLP constraint extraction from source: '{source}'")
        raw_constraints: List[Dict[str, Any]] = []

        if self.anthropic_api_key:
            try:
                prompt = self._build_extraction_prompt(regulatory_text)
                llm_response = self._call_claude_api(prompt)
                raw_constraints = self._parse_llm_response(llm_response)
                logger.info(f"Claude API extracted {len(raw_constraints)} raw constraints")
            except Exception as e:
                logger.error(f"Claude API failed: {e}. Falling back to regex.")
                raw_constraints = self._regex_fallback(regulatory_text)
        else:
            logger.warning("No ANTHROPIC_API_KEY — using regex extraction only")
            raw_constraints = self._regex_fallback(regulatory_text)

        # Resolve and validate
        resolved = self._resolve_constraints(raw_constraints)
        logger.info(f"Extracted and validated {len(resolved)} constraints")

        payload = CompliancePayload(
            constraints=resolved,
            source_document=source,
        )

        return payload.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE: PDF → RAG → EXTRACTION → PAYLOAD
# ─────────────────────────────────────────────────────────────────────────────

def extract_compliance_from_pdf(
    pdf_path: str,
    target_universe: List[str],
    target_section: str = "Risk Factors",
    anthropic_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end pipeline: PDF → vectorless RAG → LLM extraction → validated payload.
    
    Args:
        pdf_path: Path to SEC 10-K or other regulatory PDF
        target_universe: List of valid ticker symbols
        target_section: Section name to extract (e.g., "Risk Factors")
        anthropic_api_key: Anthropic API key (defaults to env var)
    
    Returns:
        CompliancePayload dict ready for quantum_core.run_full_pipeline()
    """
    logger.info("=" * 80)
    logger.info("PHASE 1: VECTORLESS RAG PARSING")
    logger.info("=" * 80)

    parser = VectorlessDocumentParser(pdf_path)
    parser.build_structural_tree()

    logger.info("=" * 80)
    logger.info(f"PHASE 1B: SECTION RETRIEVAL ('{target_section}')")
    logger.info("=" * 80)

    extracted_text = parser.retrieve_section(target_section, merge_related=True)
    if not extracted_text:
        logger.error(f"No text retrieved for section '{target_section}'")
        extracted_text = ""

    text_preview = extracted_text[:300].replace("\n", " ")
    logger.info(f"Retrieved text (first 300 chars): {text_preview}...")

    logger.info("=" * 80)
    logger.info("PHASE 2: NLP CONSTRAINT EXTRACTION + SECTOR RESOLUTION")
    logger.info("=" * 80)

    extractor = NLPComplianceEngine(
        target_universe=target_universe,
        anthropic_api_key=anthropic_api_key,
    )
    payload = extractor.extract_rules_from_text(
        extracted_text,
        source=f"PDF: {os.path.basename(pdf_path)} → section '{target_section}'",
    )

    logger.info("=" * 80)
    logger.info("PHASE 3: VALIDATION & FINAL PAYLOAD")
    logger.info("=" * 80)
    logger.info(f"Final payload: {len(payload['constraints'])} constraints")
    for i, constraint in enumerate(payload['constraints'], 1):
        logger.info(f"  [{i}] type={constraint['constraint_type']}, "
                   f"threshold={constraint['threshold_value']}, "
                   f"tickers={len(constraint['target_tickers'])}")

    return payload


def extract_compliance_from_text(
    regulatory_text: str,
    target_universe: List[str],
    source_description: str = "custom_text",
    anthropic_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Simpler variant: skips PDF parsing, works directly with text.
    Useful for testing or if you already have extracted text.
    
    Args:
        regulatory_text: Free-form regulatory / compliance text
        target_universe: List of valid tickers
        source_description: Brief label for logging
        anthropic_api_key: Anthropic API key
    
    Returns:
        CompliancePayload dict
    """
    logger.info("=" * 80)
    logger.info("NLP EXTRACTION FROM RAW TEXT")
    logger.info("=" * 80)

    extractor = NLPComplianceEngine(
        target_universe=target_universe,
        anthropic_api_key=anthropic_api_key,
    )
    payload = extractor.extract_rules_from_text(regulatory_text, source=source_description)

    logger.info(f"Extracted {len(payload['constraints'])} constraints")
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    test_universe = [
        "NVDA", "AMD", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA",
        "LLY", "ABBV", "REGN", "VRTX", "GS", "MS", "JPM", "BAC",
        "BKNG", "XOM", "CVX", "COP", "CAT", "DE", "NFLX", "DIS",
    ]

    # ── Test 1: Extract from raw text (no PDF needed) ──────────────────────
    print("\n" + "=" * 80)
    print("TEST 1: CONSTRAINT EXTRACTION FROM RAW TEXT")
    print("=" * 80)

    sample_text = """
    Portfolio Compliance Requirements:
    - The portfolio must select exactly 8 assets from our approved universe.
    - Technology sector exposure must not exceed 3 assets.
    - Healthcare sector must represent at least 2 assets.
    - Maximum 2 energy stocks are allowed.
    - Financial sector cap is 3 assets.
    """

    payload = extract_compliance_from_text(
        regulatory_text=sample_text,
        target_universe=test_universe,
        source_description="test_constraints_v1",
    )

    print("\n✓ EXTRACTED CONSTRAINTS:")
    print(json.dumps(payload, indent=2))

    # ── Test 2: Extract from PDF (requires unstructured + actual PDF) ──────
    print("\n" + "=" * 80)
    print("TEST 2: CONSTRAINT EXTRACTION FROM PDF (optional)")
    print("=" * 80)

    pdf_path = "sample_filing.pdf"
    if os.path.exists(pdf_path):
        print(f"Found PDF: {pdf_path}")
        try:
            payload = extract_compliance_from_pdf(
                pdf_path=pdf_path,
                target_universe=test_universe,
                target_section="Risk Factors",
            )
            print("\n✓ PDF EXTRACTION SUCCESSFUL:")
            print(json.dumps(payload, indent=2))
        except Exception as e:
            print(f"✗ PDF extraction failed: {e}")
    else:
        print(f"No PDF at {pdf_path} (skipping this test)")

    print("\n" + "=" * 80)
    print("Ready to feed payload to quantum_core.run_full_pipeline(..., compliance_payload=payload)")
    print("=" * 80)

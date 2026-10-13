"""Liquid-token target-price valuation engine.

Compute logic for the liquid-valuation template notebook lives here so the
notebook stays orchestration + display only. All deal-specific inputs (slug,
TAM, scenarios, peer sets, labels) come from the notebook's config cell; these
modules carry no deal literals.

  marks    -- live DeFiLlama pull, on-disk snapshot cache, FDV helper
  model    -- scenario projection (revenue build -> income stmt) + target-price valuation
  comps    -- comparable-set pull (yfinance equities + DeFiLlama tokens)
  display  -- formatting helpers for the notebook output tables
  pipeline -- formats notebook objects into the one-pager NUMBERS dict
"""

from . import marks, model, comps, display  # noqa: F401

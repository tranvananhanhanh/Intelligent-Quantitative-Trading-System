"""
Base Strategy Module
====================

Defines base classes and data containers used by all strategy implementations:
- StrategyConfig: configuration dataclass
- StrategyResult: result dataclass holding weights and metadata
- BaseStrategy: abstract base class for all strategies
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    name: str
    description: str = ""
    max_position_weight: float = 0.20  # max single-stock weight (20%)
    min_stocks: int = 1


@dataclass
class StrategyResult:
    """Result returned by a strategy's generate_weights call."""
    strategy_name: str
    weights: pd.DataFrame  # columns include at least ['gvkey', 'weight']
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply_risk_limits(self, weights_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply per-stock weight caps and re-normalize so weights sum to 1.

        Args:
            weights_df: DataFrame containing at least a 'weight' column.

        Returns:
            DataFrame with risk limits applied.
        """
        if weights_df.empty or 'weight' not in weights_df.columns:
            return weights_df

        result = weights_df.copy()
        max_w = self.config.max_position_weight

        # Cap individual weights
        result['weight'] = result['weight'].clip(upper=max_w)

        # Re-normalize
        total = result['weight'].sum()
        if total > 0:
            result['weight'] = result['weight'] / total

        return result

    @abstractmethod
    def generate_weights(self, data: Dict[str, pd.DataFrame],
                         **kwargs) -> StrategyResult:
        """Generate portfolio weights from input data."""
        raise NotImplementedError

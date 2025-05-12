"""
Factory class for creating metric instances.
"""

from typing import Dict, List, Type

from translations.metrics.metric import Metric
from translations.metrics.llm_score_metric import LLMScoreMetric
from translations.metrics.basic_metrics import (
    BLEUMetric,
    ExactMatchMetric,
    TokenOverlapMetric,
    PrecisionRecallMetric,
    LevenshteinDistanceMetric,
)


class MetricFactory:
    """Factory class for creating metric instances."""

    _metrics: Dict[str, Type[Metric]] = {
        "exact_match": ExactMatchMetric,
        "token_overlap": TokenOverlapMetric,
        "precision_recall": PrecisionRecallMetric,
        "bleu": BLEUMetric,
        "levenshtein": LevenshteinDistanceMetric,
        "llm_score": LLMScoreMetric,
    }

    @classmethod
    def get_metric(
        cls,
        name: str,
        **kwargs,
    ) -> Metric:
        """
        Get a metric instance by name.

        Args:
            name: Name of the metric
            **kwargs: Additional arguments for the metric

        Returns:
            Metric instance

        Raises:
            ValueError: If the metric is not found
        """
        if name not in cls._metrics:
            raise ValueError(f"Unknown metric: {name}")

        metric_class = cls._metrics[name]
        return metric_class(**kwargs)

    @classmethod
    def get_available_metrics(cls) -> List[str]:
        """
        Get a list of available metrics.

        Returns:
            List of available metric names
        """
        return list(cls._metrics.keys())

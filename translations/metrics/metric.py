"""
Base classes for translation metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MetricsResult:
    """
    Data class to store the results of multiple metrics.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    additional_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_metric(
        self,
        name: str,
        value: float,
    ) -> None:
        """
        Add a metric result.

        Args:
            name: Name of the metric
            value: Value of the metric
        """
        self.metrics[name] = value

    def get_metric(
        self,
        name: str,
    ) -> Optional[float]:
        """
        Get a specific metric result.

        Args:
            name: Name of the metric

        Returns:
            Metric value or None if not found
        """
        return self.metrics.get(name)

    def add_metric_info(
        self,
        name: str,
        key: str,
        value: Any,
    ) -> None:
        """
        Add additional information for a metric.

        Args:
            name: Name of the metric
            key: Key for the additional information
            value: Value of the additional information
        """
        if name not in self.additional_info:
            self.additional_info[name] = {}

        self.additional_info[name][key] = value

    def get_metric_info(
        self,
        name: str,
        key: str,
    ) -> Optional[Any]:
        """
        Get additional information for a metric.

        Args:
            name: Name of the metric
            key: Key for the additional information

        Returns:
            The value of the additional information or None if not found
        """
        if name not in self.additional_info:
            return None

        return self.additional_info[name].get(key)

    def __str__(self) -> str:
        """Format metrics results as a string."""
        if not self.metrics:
            return "No metrics available"

        result = "\n".join([f"{name}: {value: .4f}" for name, value in self.metrics.items()])

        # Add any reasoning information for LLM metrics
        for name, info in self.additional_info.items():
            if "reasoning" in info:
                result += f"\n\n{name} reasoning: {info['reasoning']}"

        return result


@dataclass
class TestCase:
    """
    Data class to store a translation test case with expected and actual results.
    """

    original_text: str
    expected_translation: Optional[str] = None
    actual_translation: Optional[str] = None
    metrics_results: MetricsResult = field(default_factory=MetricsResult)

    def __str__(self) -> str:
        """Get a string representation of the test case."""
        result = f"Original: {self.original_text}\n"
        if self.expected_translation:
            result += f"Expected: {self.expected_translation}\n"
        if self.actual_translation:
            result += f"Translated: {self.actual_translation}\n"
        if self.metrics_results.metrics:
            result += f"Metrics: \n{self.metrics_results}"
        return result


class Metric(ABC):
    """Base class for all translation metrics"""

    @abstractmethod
    def compute(
        self,
        hypothesis: str,
        reference: str,
    ) -> float:
        """
        Compute the metric for a single translation pair.

        Args:
            hypothesis: The translation being evaluated
            reference: The reference translation

        Returns:
            The metric score
        """
        pass

    def compute_batch(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> Dict[str, Any]:
        """
        Compute the metric for a batch of translation pairs.

        Args:
            hypotheses: The translations being evaluated
            references: The reference translations

        Returns:
            Dictionary with overall score and any additional statistics
        """
        if len(hypotheses) != len(references):
            raise ValueError("Number of hypotheses and references must match")

        scores = [
            self.compute(
                hypothesis=hyp,
                reference=ref,
            )
            for hyp, ref in zip(hypotheses, references)
        ]

        return {
            "score": sum(scores) / len(scores) if scores else 0.0,
            "individual_scores": scores,
        }

    @property
    def name(self) -> str:
        """Get the name of the metric."""
        return getattr(self, "_name", self.__class__.__name__)

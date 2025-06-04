"""
Evaluator class for translation metrics.
"""

from typing import Any, Dict, List, Union, Optional

from translations.data.management import TranslationDataset
from translations.metrics.metric_factory import MetricFactory
from translations.metrics.llm_score_metric import LLMScoreMetric
from translations.metrics.metric import Metric, TestCase, MetricsResult


class TranslationEvaluator:
    """Evaluator class for translation metrics."""

    def __init__(
        self,
        metrics: List[Union[str, Metric]] = None,
    ) -> None:
        """
        Initialize the evaluator with metrics.

        Args:
            metrics: List of metric names or instances
        """
        self.metrics: List[Metric] = []

        if metrics:
            for metric in metrics:
                if isinstance(metric, str):
                    self.add_metric_by_name(
                        name=metric,
                    )
                else:
                    self.add_metric(
                        metric=metric,
                    )

    def add_metric(
        self,
        metric: Metric,
    ) -> None:
        """
        Add a metric to the evaluator.

        Args:
            metric: Metric instance
        """
        self.metrics.append(metric)

    def add_metric_by_name(
        self,
        name: str,
        **kwargs,
    ) -> None:
        """
        Add a metric to the evaluator by name.

        Args:
            name: Name of the metric
            **kwargs: Additional arguments for the metric
        """
        metric = MetricFactory.get_metric(
            name=name,
            **kwargs,
        )
        self.add_metric(
            metric=metric,
        )

    def evaluate_case(
        self,
        test_case: TestCase,
    ) -> MetricsResult:
        """
        Evaluate a single test case with all metrics.

        Args:
            test_case: Test case to evaluate

        Returns:
            Metrics results
        """
        if not test_case.actual_translation or not test_case.expected_translation:
            raise ValueError("Test case must have both actual and expected translations")

        for metric in self.metrics:
            # For LLMScoreMetric, pass the original text as well for context
            if isinstance(metric, LLMScoreMetric):
                score = metric.compute(
                    hypothesis=test_case.actual_translation,
                    reference=test_case.expected_translation,
                    source_text=test_case.original_text,
                )

                # Get the reasoning and add it to the results
                reasoning = metric.get_reasoning(
                    hypothesis=test_case.actual_translation,
                    reference=test_case.expected_translation,
                    source_text=test_case.original_text,
                )

                test_case.metrics_results.add_metric(name=metric.name, value=score)
                test_case.metrics_results.add_metric_info(
                    name=metric.name,
                    key="reasoning",
                    value=reasoning,
                )
            else:
                score = metric.compute(
                    hypothesis=test_case.actual_translation,
                    reference=test_case.expected_translation,
                )
                test_case.metrics_results.add_metric(name=metric.name, value=score)

        return test_case.metrics_results

    def evaluate_batch(
        self,
        hypotheses: List[str],
        references: List[str],
        source_texts: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate a batch of translations with all metrics.

        Args:
            hypotheses: List of translations to evaluate
            references: List of reference translations
            source_texts: Optional list of source texts for context

        Returns:
            Dictionary of metric results
        """
        if len(hypotheses) != len(references):
            raise ValueError("Number of hypotheses and references must match")

        if source_texts and len(source_texts) != len(hypotheses):
            raise ValueError("Number of source texts must match number of hypotheses")

        results = {}

        for metric in self.metrics:
            if isinstance(metric, LLMScoreMetric) and source_texts:
                metric_result = metric.compute_batch(
                    hypotheses=hypotheses,
                    references=references,
                    source_texts=source_texts,
                )
            else:
                metric_result = metric.compute_batch(
                    hypotheses=hypotheses,
                    references=references,
                )

            results[metric.name] = metric_result

        return results

    def evaluate_dataset(
        self,
        dataset: TranslationDataset,
        expected_dataset: Optional[TranslationDataset] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate a dataset with all metrics.

        Args:
            dataset: Dataset with translations to evaluate
            expected_dataset: Dataset with reference translations (default: None)

        Returns:
            Dictionary of metric results
        """
        if expected_dataset:
            if len(dataset) != len(expected_dataset):
                raise ValueError("Number of translations and references must match")

            hypotheses = dataset.reference
            references = expected_dataset.reference
            source_texts = dataset.source  # Use source texts for context
        else:
            hypotheses = dataset.reference
            references = dataset.source
            source_texts = None  # No separate source texts

        return self.evaluate_batch(
            hypotheses=hypotheses,
            references=references,
            source_texts=source_texts,
        )

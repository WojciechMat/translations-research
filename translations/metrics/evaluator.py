from typing import Dict, List, Union, Optional

from translations.data.management import TranslationDataset
from translations.metrics.metric_factory import MetricFactory
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
        self.metrics = []

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

        results = MetricsResult()

        for metric in self.metrics:
            score = metric.compute(
                hypothesis=test_case.actual_translation,
                reference=test_case.expected_translation,
            )

            results.add_metric(
                name=metric.name,
                value=score,
            )

        # Update the test case with results
        test_case.metrics_results = results

        return results

    def evaluate_batch(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a batch of translations with all metrics.

        Args:
            hypotheses: List of translations to evaluate
            references: List of reference translations

        Returns:
            Dictionary of metric results
        """
        if len(hypotheses) != len(references):
            raise ValueError("Number of hypotheses and references must match")

        results = {}

        for metric in self.metrics:
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
    ) -> Dict[str, Dict[str, float]]:
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
        else:
            hypotheses = dataset.reference
            references = dataset.source

        return self.evaluate_batch(
            hypotheses=hypotheses,
            references=references,
        )

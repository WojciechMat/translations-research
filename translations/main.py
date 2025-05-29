#!/usr/bin/env python3
"""
English-Polish Translation Pipeline
"""

import os
import logging
from typing import List, Optional

import hydra
from dotenv import load_dotenv
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

from translations.metrics.metric import TestCase
from translations.metrics.evaluator import TranslationEvaluator
from translations.sheets_uploader import TranslationSheetsUploader
from translations.models.moses.moses_translator import MosesTranslator
from translations.models.brutal.brutal_translator import BrutalTranslator
from translations.models.brutal.dictionary_utils import prepare_dictionary
from translations.data.management import TranslationDataset, TranslationDatasetManager

load_dotenv()


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationPipeline:
    """Pipeline for translation tasks"""

    def __init__(
        self,
        translator,
        data_manager: TranslationDatasetManager,
        slice_spec: Optional[str] = None,
    ) -> None:
        """
        Initialize the translation pipeline.

        Args:
            translator: The translator to use
            data_manager: The data manager
            slice_spec: Slice specification (e.g., ":50", "-50:", "10:20")
        """
        self.translator = translator
        self.data_manager = data_manager
        self.slice_spec = slice_spec

    def run(
        self,
        split: str = "test",
    ) -> TranslationDataset:
        """
        Run the translation pipeline.

        Args:
            split: Which data split to use ("train", "val", or "test")

        Returns:
            Dataset with translated texts
        """
        # Get the dataset for the specified split
        if split == "train":
            dataset = self.data_manager.load_train_data()
        elif split == "val":
            dataset = self.data_manager.load_val_data()
        else:  # Default to test
            dataset = self.data_manager.load_test_data()

        # Apply slicing if specified
        if self.slice_spec:
            dataset = self._apply_slice(dataset, self.slice_spec)

        # Preprocess dataset
        preprocessed_dataset = self.data_manager.preprocess_dataset(
            dataset=dataset,
            lowercase=True,
            strip=True,
        )

        # Translate dataset
        translated_dataset = self.translator.translate_batch(preprocessed_dataset)

        return translated_dataset

    def _apply_slice(
        self,
        dataset: TranslationDataset,
        slice_spec: str,
    ) -> TranslationDataset:
        """
        Apply a slice specification to the dataset.

        Args:
            dataset: Dataset to slice
            slice_spec: Slice specification (e.g., ":50", "-50:", "10:20")

        Returns:
            Sliced dataset
        """
        if slice_spec == ":":
            return dataset

        # Parse the slice specification
        if ":" not in slice_spec:
            try:
                # Single index
                idx = int(slice_spec)
                return TranslationDataset(
                    source=[dataset.source[idx]],
                    reference=[dataset.reference[idx]],
                )
            except ValueError:
                raise ValueError(f"Invalid slice specification: {slice_spec}")

        # Handle start:stop format
        parts = slice_spec.split(":")
        if len(parts) > 2:
            raise ValueError(f"Invalid slice specification: {slice_spec}")

        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None

        return TranslationDataset(
            source=dataset.source[start:stop],
            reference=dataset.reference[start:stop],
        )

    def evaluate(
        self,
        results: TranslationDataset,
        metrics: List[str] = None,
        evaluator: Optional[TranslationEvaluator] = None,
    ) -> List[TestCase]:
        """
        Evaluate translation results using specified metrics.

        Args:
            results: Dataset with translated texts
            metrics: List of metric names to use
            evaluator: Evaluator instance (optional)

        Returns:
            List of evaluated test cases
        """
        if evaluator is None:
            evaluator = TranslationEvaluator(metrics=metrics or ["bleu", "levenshtein", "precision_recall"])

        test_cases = []

        for i in range(len(results)):
            pair = results[i]
            test_case = TestCase(
                original_text=pair.source,
                expected_translation=self.data_manager.load_test_data().reference[i],
                actual_translation=pair.target,
            )

            # Evaluate the test case
            evaluator.evaluate_case(test_case=test_case)
            test_cases.append(test_case)

        return test_cases

    def display_results(
        self,
        results: TranslationDataset,
        n_examples: int = 5,
    ) -> None:
        """
        Display translation results.

        Args:
            results: Dataset with translated texts
            n_examples: Number of examples to display
        """
        print(f"\nShowing {min(n_examples, len(results))} translation examples: ")
        for i in range(min(n_examples, len(results))):
            pair = results[i]
            print(f"Example {i+1}: ")
            print(f"  Source (EN):      {pair.source}")
            print(f"  Reference (PL):   {pair.target}")  # This shows the actual translation
            print("-" * 80)


@hydra.main(config_path="config", config_name="brutal")
def main(cfg: DictConfig) -> None:
    """Main entry point using Hydra configuration"""
    print(OmegaConf.to_yaml(cfg))

    # Initialize data manager
    data_manager = TranslationDatasetManager(
        source_lang=cfg.data.source_lang,
        target_lang=cfg.data.target_lang,
        random_seed=cfg.data.random_seed,
        cache_dir=cfg.data.cache_dir,
    )

    # Load dataset
    dataset_path = to_absolute_path(cfg.data.dataset_path)
    data_manager.load_dataset(dataset_path=dataset_path)

    # Print dataset statistics if requested
    if cfg.data.calculate_stats:
        stats = data_manager.get_dataset_stats()
        print("\nDataset Statistics: ")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # Create translator based on configuration
    translator = None
    if cfg.translator == "brutal":
        # Prepare dictionary for brutal translator
        dictionary_path = prepare_dictionary(
            cfg=cfg,
            data_manager=data_manager,
        )

        # Create BrutalTranslator
        translator = BrutalTranslator(
            dictionary_path=dictionary_path,
            keep_unknown=cfg.dictionary.keep_unknown,
            lowercase=cfg.dictionary.lowercase,
        )
    elif cfg.translator == "moses":
        # Create MosesTranslator with configured server URL
        translator = MosesTranslator(
            server_url=cfg.moses.server_url
            if hasattr(cfg, "moses") and hasattr(cfg.moses, "server_url")
            else "http://localhost:8080/RPC2",
        )
    else:
        raise ValueError(f"Unsupported translator type: {cfg.translator}")

    # Set up and run the translation pipeline
    pipeline = TranslationPipeline(
        translator=translator,
        data_manager=data_manager,
        slice_spec=cfg.slice_spec,
    )

    # Run the pipeline
    results = pipeline.run(split="test")

    # Create evaluator with default metrics
    metrics = ["bleu", "levenshtein", "precision_recall", "token_overlap", "llm_score"]
    if hasattr(cfg, "metrics") and cfg.metrics:
        metrics = cfg.metrics

    evaluator = TranslationEvaluator(metrics=metrics)

    # Display results with metrics
    print(f"\nShowing {min(5, len(results))} translation examples with metrics: ")
    for i in range(min(5, len(results))):
        pair = results[i]
        test_case = TestCase(
            original_text=pair.source,
            expected_translation=data_manager.load_test_data().reference[i],
            actual_translation=pair.target,
        )

        # Evaluate the test case
        evaluator.evaluate_case(
            test_case=test_case,
        )

        print(f"Example {i+1}: ")
        print(f"  Original:    {test_case.original_text}")
        print(f"  Expected:    {test_case.expected_translation}")
        print(f"  Translated:  {test_case.actual_translation}")
        print(f"  Metrics: \n    {test_case.metrics_results}")
        print("-" * 80)

    # Run a separate test case with metrics
    test_case = TestCase(
        original_text="Hello world, this is a test translation.",
        expected_translation="Cześć świat, to jest test tłumaczenie.",
    )

    test_case.actual_translation = translator.translate(test_case.original_text)

    # Evaluate the test case
    evaluator.evaluate_case(
        test_case=test_case,
    )

    print("\nTest Case: ")
    print(test_case)

    # Evaluate all results and get test cases
    test_cases = pipeline.evaluate(
        results=results,
        metrics=metrics,
        evaluator=evaluator,
    )

    # Check if Google Sheets upload is enabled
    use_sheets = False
    if hasattr(cfg, "use_sheets"):
        use_sheets = cfg.use_sheets

    # Initialize and use Google Sheets uploader if requested and environment variable is set
    if use_sheets and os.getenv("SHEET"):
        logger.info("Uploading translation results to Google Sheets...")

        # Initialize the uploader
        uploader = TranslationSheetsUploader()

        # Upload results
        uploader.upload_evaluation_results(evaluator, test_cases)

        logger.info(f"Successfully uploaded {len(test_cases)} test cases to Google Sheets")
    elif use_sheets and not os.getenv("SHEET"):
        logger.warning("SHEET environment variable not set. Skipping Google Sheets upload.")
    else:
        logger.info("Google Sheets upload not enabled. Set 'use_sheets=true' in config to enable.")


if __name__ == "__main__":
    main()

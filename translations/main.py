#!/usr/bin/env python3
"""
English-Polish Translation Pipeline
"""

import os
from typing import List, Optional

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

from translations.metrics.metric import TestCase
from translations.models.brutal.brutal_translator import BrutalTranslator
from translations.data.management import TranslationDataset, EuroparlDataManager
from translations.models.brutal.dictionary_utils import (
    create_test_dictionary,
    build_dictionary_from_diki,
    generate_word_list_from_dataset,
)


class TranslationPipeline:
    """Pipeline for translation tasks"""

    def __init__(
        self,
        translator,
        data_manager: EuroparlDataManager,
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
    ) -> dict:
        """
        Evaluate translation results using specified metrics.

        Args:
            results: Dataset with translated texts
            metrics: List of metric names to use

        Returns:
            Dictionary of metric results
        """
        # For now, just a placeholder
        print("Evaluation would happen here with these metrics:", metrics)
        return {"placeholder": 0.0}

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
        print(f"\nShowing {min(n_examples, len(results))} translation examples:")
        for i in range(min(n_examples, len(results))):
            pair = results[i]
            print(f"Example {i+1}:")
            print(f"  Source (EN):      {pair.source}")
            print(f"  Reference (PL):   {pair.target}")  # This shows the actual translation
            print("-" * 80)


def prepare_dictionary(cfg: DictConfig) -> str:
    """
    Prepare the dictionary based on configuration.

    Args:
        cfg: Configuration

    Returns:
        Path to the dictionary file
    """
    # Create dictionaries directory if it doesn't exist
    os.makedirs(to_absolute_path("tmp/dictionaries"), exist_ok=True)

    dictionary_path = to_absolute_path(cfg.dictionary.path)

    # Build dictionary if requested
    if cfg.dictionary.build:
        if cfg.dictionary.source == "diki":
            # Load dataset first to generate word list
            data_manager = EuroparlDataManager(
                source_lang=cfg.data.source_lang,
                target_lang=cfg.data.target_lang,
                random_seed=cfg.data.random_seed,
                cache_dir=to_absolute_path(cfg.data.cache_dir),
            )
            dataset_path = to_absolute_path(cfg.data.dataset_path)
            data_manager.load_dataset(dataset_path=dataset_path)

            # Generate word list
            generate_word_list_from_dataset(
                data_manager=data_manager,
                output_path=to_absolute_path("tmp/dictionaries/word_list.txt"),
                max_words=50000,
            )

            # Build dictionary from diki
            build_dictionary_from_diki(
                word_list_path=to_absolute_path("tmp/dictionaries/word_list.txt"),
                output_path=dictionary_path,
                batch_size=50,
                delay=1.0,
            )
        else:
            raise NotImplementedError(f"Wrong dictionary source: {cfg.dictionary.source}")

    # Create test dictionary if no dictionary exists
    if not os.path.exists(dictionary_path):
        print(f"Dictionary file {dictionary_path} not found. Creating a test dictionary.")
        create_test_dictionary(output_path=dictionary_path)

    return dictionary_path


@hydra.main(config_path="config", config_name="brutal")
def main(cfg: DictConfig) -> None:
    """Main entry point using Hydra configuration"""
    print(OmegaConf.to_yaml(cfg))

    # Prepare dictionary
    dictionary_path = prepare_dictionary(cfg)

    # Initialize data manager
    data_manager = EuroparlDataManager(
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
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # Create translator based on configuration
    translator = BrutalTranslator(
        dictionary_path=dictionary_path,
        keep_unknown=cfg.dictionary.keep_unknown,
        lowercase=cfg.dictionary.lowercase,
    )

    # Set up and run the translation pipeline
    pipeline = TranslationPipeline(
        translator=translator,
        data_manager=data_manager,
        slice_spec=cfg.slice_spec,
    )

    # Run the pipeline
    results = pipeline.run(split="test")

    # Display results with clear labels
    print(f"\nShowing {min(5, len(results))} translation examples:")
    for i in range(min(5, len(results))):
        pair = results[i]
        test_case = TestCase(
            original_text=pair.source,
            expected_translation=data_manager.load_test_data().reference[i],
            actual_translation=pair.target,
        )
        print(f"Example {i+1}:")
        print(f"  Original:    {test_case.original_text}")
        print(f"  Expected:    {test_case.expected_translation}")
        print(f"  Translated:  {test_case.actual_translation}")
        print("-" * 80)

    # Run a separate test case
    test_case = TestCase(
        original_text="Hello world, this is a test translation.",
        expected_translation="Cześć świat, to jest test tłumaczenie.",
    )

    test_case.actual_translation = translator.translate(test_case.original_text)
    print("\nTest Case:")
    print(test_case)


if __name__ == "__main__":
    main()

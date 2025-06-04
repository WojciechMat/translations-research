"""
Data management component for the English-Polish translation project.
Handles dataset loading, preprocessing, and splitting for multiple datasets.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import Value, Dataset, Features, Translation, load_dataset, concatenate_datasets


@dataclass
class TranslationPair:
    """
    Data class to store a source-target translation pair.
    """

    source: str
    target: str


@dataclass
class TranslationDataset:
    """
    Data class to store translation datasets with source and reference texts.
    """

    source: List[str]
    reference: List[str]

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> TranslationPair:
        return TranslationPair(
            source=self.source[idx],
            target=self.reference[idx],
        )


class TranslationDatasetManager:
    """
    Manages multiple translation datasets (Europarl and OPUS-100) for English-Polish translation tasks.
    Can load datasets individually or concatenate them together.
    """

    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "pl",
        random_seed: int = 42,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the translation dataset manager.

        Args:
            source_lang: Source language code (default: "en")
            target_lang: Target language code (default: "pl")
            random_seed: Random seed for reproducibility (default: 42)
            cache_dir: Directory to cache the dataset (default: None)
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.random_seed = random_seed
        self.cache_dir = cache_dir
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_dataset(
        self,
        dataset_path: Optional[str] = None,
        include_opus100: bool = True,
        opus100_sample_size: Optional[int] = None,
    ) -> None:
        """
        Load translation datasets. Can load Europarl only, OPUS-100 only, or both concatenated.

        Args:
            dataset_path: Path to the Europarl dataset script (None to skip Europarl)
            include_opus100: Whether to include OPUS-100 dataset (default: True)
            opus100_sample_size: Maximum number of samples from OPUS-100 per split (None for all)
        """
        config_name = f"{self.source_lang}-{self.target_lang}"
        datasets_to_combine = {"train": [], "validation": [], "test": []}

        # Load Europarl dataset if path provided
        if dataset_path:
            try:
                print(f"Loading Europarl dataset from {dataset_path}...")
                europarl_dataset = load_dataset(
                    dataset_path,
                    name=config_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                )

                # Add Europarl splits to combination
                for split in ["train", "validation", "test"]:
                    if split in europarl_dataset:
                        datasets_to_combine[split].append(europarl_dataset[split])
                        print(f"  Added Europarl {split}: {len(europarl_dataset[split])} examples")

            except Exception as e:
                print(f"Warning: Could not load Europarl dataset: {e}")

        # Load OPUS-100 dataset
        if include_opus100:
            try:
                print("Loading OPUS-100 dataset...")
                opus100_dataset = load_dataset(
                    "Helsinki-NLP/opus-100",
                    name=config_name,
                    cache_dir=self.cache_dir,
                )

                # Sample if requested
                if opus100_sample_size:
                    print(f"  Sampling {opus100_sample_size} examples per split from OPUS-100...")
                    for split in opus100_dataset:
                        if len(opus100_dataset[split]) > opus100_sample_size:
                            opus100_dataset[split] = (
                                opus100_dataset[split].shuffle(seed=self.random_seed).select(range(opus100_sample_size))
                            )

                # OPUS-100 already has the correct format: {"translation": {"en": "text", "pl": "text"}}
                # Add source identifier and standardize features to match Europarl
                def standardize_opus100_features(example):
                    return {
                        "translation": {
                            self.source_lang: example["translation"][self.source_lang],
                            self.target_lang: example["translation"][self.target_lang],
                        },
                        "dataset_source": "opus-100",
                    }

                # Apply standardization and add to combination
                for split in ["train", "validation", "test"]:
                    if split in opus100_dataset:
                        # Remove all columns except translation, then add our standardized version
                        standardized_split = opus100_dataset[split].map(
                            standardize_opus100_features, remove_columns=opus100_dataset[split].column_names
                        )
                        datasets_to_combine[split].append(standardized_split)
                        print(f"  Added OPUS-100 {split}: {len(standardized_split)} examples")

            except Exception as e:
                print(f"Warning: Could not load OPUS-100 dataset: {e}")

        # If we have Europarl, standardize its features as well for consistency
        if dataset_path and any(datasets_to_combine.values()):

            def standardize_europarl_features(example):
                result = {
                    "translation": {
                        self.source_lang: example["translation"][self.source_lang],
                        self.target_lang: example["translation"][self.target_lang],
                    },
                    "dataset_source": "europarl",
                }
                return result

            # Apply standardization to Europarl datasets (those without dataset_source)
            for split in ["train", "validation", "test"]:
                if datasets_to_combine[split]:
                    # Find and standardize Europarl datasets
                    standardized_datasets = []
                    for ds in datasets_to_combine[split]:
                        if "dataset_source" not in ds.column_names:
                            # This is Europarl - standardize it
                            standardized_ds = ds.map(standardize_europarl_features, remove_columns=ds.column_names)
                            standardized_datasets.append(standardized_ds)
                        else:
                            # This is already standardized (OPUS-100)
                            standardized_datasets.append(ds)
                    datasets_to_combine[split] = standardized_datasets

        # Concatenate datasets for each split
        self.dataset = {}

        # Define a standard feature schema to ensure compatibility
        standard_features = Features(
            {
                "translation": Translation(languages=(self.source_lang, self.target_lang)),
                "dataset_source": Value(dtype="string"),
            }
        )

        for split, dataset_list in datasets_to_combine.items():
            if dataset_list:
                if len(dataset_list) == 1:
                    # Cast single dataset to standard features
                    self.dataset[split] = dataset_list[0].cast(standard_features)
                    print(f"Single {split} split: {len(self.dataset[split])} examples")
                else:
                    # Debug: print features of each dataset before concatenation
                    print(f"Concatenating {len(dataset_list)} datasets for {split} split:")
                    for i, ds in enumerate(dataset_list):
                        print(f"  Dataset {i}: {len(ds)} examples, features: {ds.features}")

                    # Cast all datasets to the same feature schema before concatenation
                    standardized_datasets = []
                    for ds in dataset_list:
                        standardized_ds = ds.cast(standard_features)
                        standardized_datasets.append(standardized_ds)

                    # Concatenate multiple datasets
                    self.dataset[split] = concatenate_datasets(standardized_datasets)
                    print(f"Combined {split} split: {len(self.dataset[split])} examples")
            else:
                print(f"No datasets found for {split} split")

        # If no datasets were loaded, raise an error
        if not self.dataset:
            raise ValueError("No datasets were successfully loaded. Check your dataset paths and configurations.")

        # Process the available splits
        self._prepare_splits()

    def _prepare_splits(self) -> None:
        """
        Prepare train/validation/test splits from the loaded dataset.
        """
        # Process train split
        if "train" in self.dataset:
            self.train_dataset = self._convert_to_translation_dataset(self.dataset["train"])
            print(f"Loaded combined train split with {len(self.train_dataset)} examples")
        else:
            print("Warning: No train split found in the dataset")
            self.train_dataset = TranslationDataset(source=[], reference=[])

        # Process validation split
        if "validation" in self.dataset:
            self.val_dataset = self._convert_to_translation_dataset(self.dataset["validation"])
            print(f"Loaded combined validation split with {len(self.val_dataset)} examples")
        else:
            print("Warning: No validation split found in the dataset")
            # Create an empty validation dataset or subset from train
            if len(self.train_dataset) > 0:
                val_size = min(int(len(self.train_dataset) * 0.1), 1000)  # 10% or max 1000 examples
                random.seed(self.random_seed)
                indices = random.sample(range(len(self.train_dataset)), val_size)

                self.val_dataset = TranslationDataset(
                    source=[self.train_dataset.source[i] for i in indices],
                    reference=[self.train_dataset.reference[i] for i in indices],
                )
                print(f"Created validation split with {len(self.val_dataset)} examples from train split")
            else:
                self.val_dataset = TranslationDataset(source=[], reference=[])

        # Process test split
        if "test" in self.dataset:
            self.test_dataset = self._convert_to_translation_dataset(self.dataset["test"])
            print(f"Loaded combined test split with {len(self.test_dataset)} examples")
        else:
            print("Warning: No test split found in the dataset")
            # Create an empty test dataset or subset from validation
            if len(self.val_dataset) > 0:
                self.test_dataset = self.val_dataset
                print(f"Using validation split as test split with {len(self.test_dataset)} examples")
            else:
                self.test_dataset = TranslationDataset(source=[], reference=[])

    def _convert_to_translation_dataset(
        self,
        hf_dataset: Dataset,
    ) -> TranslationDataset:
        """
        Convert a HuggingFace dataset to our TranslationDataset format.

        Args:
            hf_dataset: HuggingFace dataset to convert

        Returns:
            TranslationDataset with source and reference texts
        """
        source_texts = []
        reference_texts = []

        for example in hf_dataset:
            source_texts.append(example["translation"][self.source_lang])
            reference_texts.append(example["translation"][self.target_lang])

        return TranslationDataset(
            source=source_texts,
            reference=reference_texts,
        )

    def load_train_data(self) -> TranslationDataset:
        """
        Get the training dataset.

        Returns:
            Training dataset
        """
        if self.train_dataset is None:
            raise ValueError("Dataset not loaded yet. Call load_dataset() first.")
        return self.train_dataset

    def load_val_data(self) -> TranslationDataset:
        """
        Get the validation dataset.

        Returns:
            Validation dataset
        """
        if self.val_dataset is None:
            raise ValueError("Dataset not loaded yet. Call load_dataset() first.")
        return self.val_dataset

    def load_test_data(self) -> TranslationDataset:
        """
        Get the test dataset.

        Returns:
            Test dataset
        """
        if self.test_dataset is None:
            raise ValueError("Dataset not loaded yet. Call load_dataset() first.")
        return self.test_dataset

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            raise ValueError("Dataset not loaded yet. Call load_dataset() first.")

        train_source_lens = [len(text.split()) for text in self.train_dataset.source]
        train_target_lens = [len(text.split()) for text in self.train_dataset.reference]

        stats = {
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "test_size": len(self.test_dataset),
            "total_size": len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset),
            "train_source_avg_len": sum(train_source_lens) / len(train_source_lens) if train_source_lens else 0,
            "train_target_avg_len": sum(train_target_lens) / len(train_target_lens) if train_target_lens else 0,
            "train_source_max_len": max(train_source_lens) if train_source_lens else 0,
            "train_target_max_len": max(train_target_lens) if train_target_lens else 0,
        }

        # Add per-dataset statistics if we have the combined dataset with source info
        if hasattr(self, "dataset") and self.dataset and "train" in self.dataset:
            if "dataset_source" in self.dataset["train"].column_names:
                # Count examples by source dataset
                train_sources = self.dataset["train"]["dataset_source"]
                europarl_count = sum(1 for source in train_sources if source == "europarl")
                opus100_count = sum(1 for source in train_sources if source == "opus-100")

                stats["europarl_train_size"] = europarl_count
                stats["opus100_train_size"] = opus100_count

                print(f"Dataset composition - Europarl: {europarl_count}, OPUS-100: {opus100_count}")

        return stats

    def preprocess_text(
        self,
        text: str,
        lowercase: bool = True,
        strip: bool = True,
    ) -> str:
        """
        Preprocess text for translation.

        Args:
            text: Text to preprocess
            lowercase: Whether to lowercase the text (default: True)
            strip: Whether to strip whitespace (default: True)

        Returns:
            Preprocessed text
        """
        if strip:
            text = text.strip()

        if lowercase:
            text = text.lower()

        return text

    def preprocess_dataset(
        self,
        dataset: TranslationDataset,
        lowercase: bool = True,
        strip: bool = True,
    ) -> TranslationDataset:
        """
        Preprocess a translation dataset.

        Args:
            dataset: Dataset to preprocess
            lowercase: Whether to lowercase the text (default: True)
            strip: Whether to strip whitespace (default: True)

        Returns:
            Preprocessed dataset
        """
        preprocessed_source = [self.preprocess_text(text, lowercase=lowercase, strip=strip) for text in dataset.source]

        preprocessed_reference = [
            self.preprocess_text(text, lowercase=lowercase, strip=strip) for text in dataset.reference
        ]

        return TranslationDataset(
            source=preprocessed_source,
            reference=preprocessed_reference,
        )

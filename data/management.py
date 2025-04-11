"""
Data management component for the English-Polish translation project.
Handles dataset loading, preprocessing, and splitting.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset


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


class EuroparlDataManager:
    """
    Manages the Europarl-ST dataset for English-Polish translation tasks.
    """

    def __init__(
        self,
        source_lang: str = "en",
        target_lang: str = "pl",
        random_seed: int = 42,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the data manager.

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
    ) -> None:
        """
        Load the Europarl dataset for English-Polish translation.

        Args:
            use_local_script: Whether to use a local script (default: True)
            dataset_path: Path to the local dataset script (default: None)
        """
        config_name = f"{self.source_lang}-{self.target_lang}"

        self.dataset = load_dataset(
            dataset_path,
            name=config_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        print(f"Successfully loaded dataset with language pair: {config_name}")

        # Process the available splits
        self._prepare_splits()

    def _prepare_splits(self) -> None:
        """
        Prepare train/validation/test splits from the loaded dataset.
        """
        # Process train split
        if "train" in self.dataset:
            self.train_dataset = self._convert_to_translation_dataset(self.dataset["train"])
            print(f"Loaded train split with {len(self.train_dataset)} examples")
        else:
            print("Warning: No train split found in the dataset")
            self.train_dataset = TranslationDataset(source=[], reference=[])

        # Process validation split
        if "validation" in self.dataset:
            self.val_dataset = self._convert_to_translation_dataset(self.dataset["validation"])
            print(f"Loaded validation split with {len(self.val_dataset)} examples")
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
            print(f"Loaded test split with {len(self.test_dataset)} examples")
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

        return {
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "test_size": len(self.test_dataset),
            "train_source_avg_len": sum(train_source_lens) / len(train_source_lens) if train_source_lens else 0,
            "train_target_avg_len": sum(train_target_lens) / len(train_target_lens) if train_target_lens else 0,
            "train_source_max_len": max(train_source_lens) if train_source_lens else 0,
            "train_target_max_len": max(train_target_lens) if train_target_lens else 0,
        }

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

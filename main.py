"""
Example script to demonstrate the usage of Europarl English-Polish dataset.
"""

import hydra
from omegaconf import DictConfig

from data.management import EuroparlDataManager


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Example of loading and using the Europarl EN-PL dataset.
    """
    # Initialize the data manager
    data_manager = EuroparlDataManager(
        source_lang=cfg.data.source_lang,
        target_lang=cfg.data.target_lang,
        random_seed=cfg.data.random_seed,
    )

    # Load the dataset using the local script
    data_manager.load_dataset(
        dataset_path=cfg.data.dataset_path,
    )

    # Get dataset statistics
    stats = data_manager.get_dataset_stats()
    print("Dataset Statistics: ")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Load and preprocess the training data
    train_data = data_manager.load_train_data()
    preprocessed_train_data = data_manager.preprocess_dataset(
        dataset=train_data,
        lowercase=True,
        strip=True,
    )

    # Print some examples from the training data
    print("\nTraining Data Examples: ")
    for i in range(min(5, len(preprocessed_train_data))):
        example = preprocessed_train_data[i]
        print(f"  Example {i+1}: ")
        print(f"    Source (EN): {example.source}")
        print(f"    Target (PL): {example.target}")
        print("-" * 80)

    # Load validation data
    val_data = data_manager.load_val_data()
    print(f"\nValidation data size: {len(val_data)}")

    # Load test data
    test_data = data_manager.load_test_data()
    print(f"\nTest data size: {len(test_data)}")


if __name__ == "__main__":
    main()

# scripts/export_corpus.py
import os

from translations.data.management import EuroparlDataManager


def export_corpus(data_manager, output_dir, split, src_lang, tgt_lang):
    """Export corpus to plain text files."""
    if split == "train":
        dataset = data_manager.load_train_data()
    elif split == "val":
        dataset = data_manager.load_val_data()
    else:  # test
        dataset = data_manager.load_test_data()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create output files
    src_file = os.path.join(output_dir, f"{split}.{src_lang}")
    tgt_file = os.path.join(output_dir, f"{split}.{tgt_lang}")

    # Write source sentences
    with open(src_file, "w", encoding="utf-8") as f:
        for sentence in dataset.source:
            f.write(f"{sentence.strip()}\n")

    # Write target sentences
    with open(tgt_file, "w", encoding="utf-8") as f:
        for sentence in dataset.reference:
            f.write(f"{sentence.strip()}\n")

    print(f"Exported {len(dataset)} sentences to {src_file} and {tgt_file}")
    return src_file, tgt_file


# Initialize data manager
data_manager = EuroparlDataManager(
    source_lang="en",
    target_lang="pl",
    random_seed=42,
    cache_dir="./tmp/data/cache",
)

# Load dataset
data_manager.load_dataset(dataset_path="translations/hf_datasets/EuroparlEnPl")

# Export splits
os.makedirs("./corpus", exist_ok=True)
export_corpus(data_manager, "./corpus", "train", "en", "pl")
export_corpus(data_manager, "./corpus", "val", "en", "pl")
export_corpus(data_manager, "./corpus", "test", "en", "pl")

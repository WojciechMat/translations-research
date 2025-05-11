# flake8: noqa
import os
import argparse
from typing import Tuple

from translations.data.management import TranslationDataset, EuroparlDataManager


def dataset_to_tmx(dataset: TranslationDataset, output_file: str, src_lang: str, tgt_lang: str) -> None:
    """
    Convert a TranslationDataset to a TMX file.

    Args:
        dataset: Dataset containing source and reference texts
        output_file: Path to output TMX file
        src_lang: Source language code (e.g., 'en')
        tgt_lang: Target language code (e.g., 'pl')
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as tmx:
        # Write TMX header
        tmx.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        tmx.write('<!DOCTYPE tmx SYSTEM "tmx14.dtd">\n')
        tmx.write('<tmx version="1.4">\n')
        tmx.write(f'  <header creationtool="EuroparlTMXCreator" creationtoolversion="1.0" ')
        tmx.write(f'segtype="sentence" o-tmf="PlainText" adminlang="{src_lang}" ')
        tmx.write(f'srclang="{src_lang}" datatype="plaintext">\n')
        tmx.write("  </header>\n")
        tmx.write("  <body>\n")

        # Process each translation pair
        for src, tgt in zip(dataset.source, dataset.reference):
            src = src.strip()
            tgt = tgt.strip()

            # Skip empty pairs
            if not src or not tgt:
                continue

            # Escape XML special characters
            src = src.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            tgt = tgt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            # Write translation unit
            tmx.write("    <tu>\n")
            tmx.write(f'      <tuv xml:lang="{src_lang}"><seg>{src}</seg></tuv>\n')
            tmx.write(f'      <tuv xml:lang="{tgt_lang}"><seg>{tgt}</seg></tuv>\n')
            tmx.write("    </tu>\n")

        # Close TMX file
        tmx.write("  </body>\n")
        tmx.write("</tmx>\n")

    print(f"Created TMX file with {len(dataset)} translation units: {output_file}")


def create_tmx_files(
    data_manager: EuroparlDataManager,
    output_dir: str,
    src_lang: str,
    tgt_lang: str,
    train_ratio: float = 0.9,
    use_val_for_tuning: bool = True,
) -> Tuple[str, str]:
    """
    Create TMX files for Moses training and tuning from Europarl data.

    Args:
        data_manager: EuroparlDataManager instance
        output_dir: Directory to save TMX files
        src_lang: Source language code
        tgt_lang: Target language code
        train_ratio: Ratio of training data if splitting the train set (ignored if use_val_for_tuning=True)
        use_val_for_tuning: Whether to use validation set for tuning (True) or split training set (False)

    Returns:
        Tuple of paths to the created TMX files (train_tmx, tune_tmx)
    """
    # Create output directories
    train_dir = os.path.join(output_dir, "tmx-train")
    tune_dir = os.path.join(output_dir, "tmx-tune")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(tune_dir, exist_ok=True)

    # Output TMX file paths
    train_tmx = os.path.join(train_dir, f"europarl-{src_lang}-{tgt_lang}.tmx")
    tune_tmx = os.path.join(tune_dir, f"europarl-{src_lang}-{tgt_lang}.tmx")

    if use_val_for_tuning:
        # Use training data for training and validation data for tuning
        train_dataset = data_manager.load_train_data()
        tune_dataset = data_manager.load_val_data()
    else:
        # Split training data into training and tuning sets
        train_dataset = data_manager.load_train_data()
        split_idx = int(len(train_dataset) * train_ratio)

        # Create tuning dataset from portion of training data
        tune_dataset = TranslationDataset(
            source=train_dataset.source[split_idx:], reference=train_dataset.reference[split_idx:]
        )

        # Update training dataset to exclude tuning data
        train_dataset = TranslationDataset(
            source=train_dataset.source[:split_idx], reference=train_dataset.reference[:split_idx]
        )

    # Create TMX files
    dataset_to_tmx(train_dataset, train_tmx, src_lang, tgt_lang)
    dataset_to_tmx(tune_dataset, tune_tmx, src_lang, tgt_lang)

    return train_tmx, tune_tmx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create TMX files from Europarl data")
    parser.add_argument("--source-lang", default="en", help="Source language code")
    parser.add_argument("--target-lang", default="pl", help="Target language code")
    parser.add_argument("--output-dir", default="./moses-smt", help="Output directory for TMX files")
    parser.add_argument(
        "--dataset-path", default="translations/hf_datasets/EuroparlEnPl", help="Path to Europarl dataset"
    )
    parser.add_argument("--cache-dir", default="./tmp/data/cache", help="Cache directory")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use-val-for-tuning", action="store_true", help="Use validation set for tuning (otherwise split training set)"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.9, help="Ratio of training data if splitting the train set"
    )

    args = parser.parse_args()

    # Initialize data manager
    data_manager = EuroparlDataManager(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        random_seed=args.random_seed,
        cache_dir=args.cache_dir,
    )

    # Load dataset
    data_manager.load_dataset(dataset_path=args.dataset_path)

    # Create TMX files
    train_tmx, tune_tmx = create_tmx_files(
        data_manager=data_manager,
        output_dir=args.output_dir,
        src_lang=args.source_lang,
        tgt_lang=args.target_lang,
        train_ratio=args.train_ratio,
        use_val_for_tuning=args.use_val_for_tuning,
    )

    print("\nTMX files created successfully:")
    print(f"Training TMX: {train_tmx}")
    print(f"Tuning TMX: {tune_tmx}")
    print("\nReady for Moses SMT training!")

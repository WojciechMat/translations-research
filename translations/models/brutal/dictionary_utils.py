import os
import gzip
import json
import time
from typing import Dict

import requests
from tqdm import tqdm


def load_dictionary(
    dictionary_path: str,
) -> Dict[str, str]:
    """
    Load a dictionary from a JSON file.

    Args:
        dictionary_path: Path to the dictionary file

    Returns:
        Dictionary mapping source words to target words
    """
    with open(dictionary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dictionary(
    dictionary: Dict[str, str],
    output_path: str,
) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        dictionary: Dictionary to save
        output_path: Path to save the dictionary to
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)


def generate_word_list_from_dataset(
    data_manager,
    output_path: str,
    max_words: int = 5000,
) -> None:
    """
    Generate a list of unique English words from the dataset.

    Args:
        data_manager: EuroparlDataManager instance
        output_path: Path to save the word list
        max_words: Maximum number of words to include
    """
    # Load the dataset
    train_dataset = data_manager.load_train_data()

    # Extract unique words
    all_words = set()
    for text in train_dataset.source:
        # Simple tokenization by splitting on whitespace and removing punctuation
        words = [word.strip(".,!?;:\"'()[]{}").lower() for word in text.split()]  # noqa
        all_words.update(words)

    # Filter out very short words and sort by frequency
    filtered_words = [word for word in all_words if len(word) > 1]

    # Take the most common words up to max_words
    word_list = sorted(filtered_words)[:max_words]

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the word list
    with open(output_path, "w", encoding="utf-8") as f:
        for word in word_list:
            f.write(f"{word}\n")

    print(f"Generated word list with {len(word_list)} unique words.")


def build_dictionary_from_wiktionary(
    word_list_path: str,
    output_path: str,
    batch_size: int = 50,
    delay: float = 0.5,
) -> None:
    """
    Build an English-Polish dictionary using Wiktionary API.

    Args:
        word_list_path: Path to a text file with English words (one per line)
        output_path: Path to save the resulting dictionary
        batch_size: Number of words to process in a batch
        delay: Delay between batches to avoid rate limiting
    """
    # Load English words
    with open(word_list_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    dictionary = {}

    # Process words in batches
    for i in tqdm(range(0, len(words), batch_size)):
        batch = words[i : i + batch_size]

        for word in batch:
            try:
                # Query Wiktionary API
                url = f"https://en.wiktionary.org/api/rest_v1/page/definition/{word}"  # noqa
                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()

                    # Look for Polish translations
                    polish_translation = None

                    if "en" in data:
                        for entry in data["en"]:
                            if "translations" in entry and "pl" in entry["translations"]:
                                polish_translation = entry["translations"]["pl"][0]
                                break

                    if polish_translation:
                        dictionary[word] = polish_translation

            except Exception as e:
                print(f"Error processing word '{word}': {e}")

        # Save progress
        save_dictionary(dictionary, output_path)

        # Delay to avoid rate limiting
        time.sleep(delay)

    print(f"Dictionary built with {len(dictionary)} entries.")

def create_test_dictionary(
    output_path: str = "dictionaries/test_dictionary.json",
) -> None:
    """
    Create a small test dictionary for development purposes.

    Args:
        output_path: Path to save the test dictionary
    """
    test_dictionary = {
        "hello": "cześć",
        "world": "świat",
        "this": "to",
        "is": "jest",
        "a": "a",
        "test": "test",
        "translation": "tłumaczenie",
        "i": "ja",
        "you": "ty",
        "he": "on",
        "she": "ona",
        "we": "my",
        "they": "oni",
        "cat": "kot",
        "dog": "pies",
        "house": "dom",
        "car": "samochód",
        "book": "książka",
        "read": "czytać",
        "write": "pisać",
        "love": "kochać",
        "hate": "nienawidzić",
        "eat": "jeść",
        "drink": "pić",
        "sleep": "spać",
        "work": "pracować",
        "play": "grać",
        "go": "iść",
        "come": "przyjść",
        "see": "widzieć",
        "hear": "słyszeć",
        "speak": "mówić",
        "listen": "słuchać",
    }

    save_dictionary(test_dictionary, output_path)
    print(f"Test dictionary created with {len(test_dictionary)} entries.")

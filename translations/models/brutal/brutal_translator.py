import os
import re
import json

from translations.models.base_translator import Translator
from translations.models.brutal.dictionary_utils import load_dictionary


class BrutalTranslator(Translator):
    """Word-for-word translation using a dictionary lookup approach"""

    def __init__(
        self,
        dictionary_path: str,
        keep_unknown: bool = True,
        lowercase: bool = True,
    ) -> None:
        """
        Initialize the brutal translator.

        Args:
            dictionary_path: Path to the English-Polish dictionary JSON file
            keep_unknown: Whether to keep unknown words in the original form
            lowercase: Whether to lowercase input text for translation
        """
        # Ensure the dictionary file exists
        if not os.path.exists(dictionary_path):
            raise FileNotFoundError(f"Dictionary file not found: {dictionary_path}")

        self.dictionary = self._load_dictionary(dictionary_path)
        self.keep_unknown = keep_unknown
        self.lowercase = lowercase

        # Print some dictionary entries to debug
        print(f"Loaded dictionary with {len(self.dictionary)} entries")
        sample_entries = list(self.dictionary.items())[:5]
        print("Sample dictionary entries:", sample_entries)

    def _load_dictionary(self, dictionary_path: str) -> dict:
        """Load the dictionary from a file with error handling."""
        try:
            with open(dictionary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {dictionary_path}. Creating a basic dictionary.")
            return {
                "hello": "cześć",
                "world": "świat",
                "this": "to",
                "is": "jest",
                "a": "a",
                "test": "test",
                "translation": "tłumaczenie",
            }

    def translate(
        self,
        text: str,
    ) -> str:
        """
        Translate a text word by word using the dictionary.

        Args:
            text: Source text to translate

        Returns:
            Translated text
        """
        # Preprocessing based on configuration
        processed_text = text
        if self.lowercase:
            processed_text = processed_text.lower()

        # Split into words while preserving punctuation
        words = re.findall(r"\b\w+\b|[^\w\s]", processed_text)

        # Translate each word
        translated_words = []
        for word in words:
            # Check if it's a word or punctuation
            if re.match(r"\w+", word):
                # It's a word - look up in dictionary
                if word in self.dictionary:
                    translated_words.append(self.dictionary[word])
                else:
                    # Handle unknown words
                    if self.keep_unknown:
                        translated_words.append(word)
                    else:
                        # Skip unknown words
                        pass
            else:
                # It's punctuation - keep as is
                translated_words.append(word)

        return " ".join(translated_words)

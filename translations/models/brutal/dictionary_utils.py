# flake8: noqa
import os
import re
import json
import time
from typing import Dict, List, Optional

from tqdm import tqdm
from hydra.utils import to_absolute_path

from translations.llm_client.gemini_client import GeminiClient


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


def is_valid_dictionary_word(word: str) -> bool:
    """
    Check if a word is a valid dictionary word (not a number, date, etc.)

    Args:
        word: Word to check

    Returns:
        True if the word is a valid dictionary word
    """
    # Skip numbers, percentages, dates, etc.
    if re.match(r"^[0-9]+$", word):  # Pure numbers
        print(f"Word not found: {word} (number)")
        return False
    if re.match(r"^[0-9]+[%]$", word):  # Percentages
        print(f"Word not found: {word} (percentage)")
        return False
    if re.match(r"^[0-9]+[-][0-9]+$", word):  # Ranges like 2-3
        print(f"Word not found: {word} (range)")
        return False
    if re.match(r"^[0-9]+[.][0-9]+$", word):  # Decimals
        print(f"Word not found: {word} (decimal)")
        return False
    if re.match(r"^[0-9]+(st|nd|rd|th)$", word):  # Ordinals like 1st
        print(f"Word not found: {word} (ordinal)")
        return False
    if re.match(r"^[0-9]+[-][a-z]+$", word):  # Combinations like 18-month
        print(f"Word not found: {word} (combination)")
        return False
    if len(word) <= 1:  # Single characters
        print(f"Word not found: {word} (too short)")
        return False
    return True


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

    # Filter out invalid dictionary words
    filtered_words = [word for word in all_words if is_valid_dictionary_word(word)]

    print(f"Found {len(all_words)} unique words, {len(filtered_words)} valid dictionary words")

    # Take the most common words up to max_words
    word_list = sorted(filtered_words)[:max_words]

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the word list
    with open(output_path, "w", encoding="utf-8") as f:
        for word in word_list:
            f.write(f"{word}\n")

    print(f"Generated word list with {len(word_list)} unique words.")


def batch_translate_with_gemini(
    gemini_client: GeminiClient,
    word_batch: List[str],
    retry_count: int = 3,
    retry_delay: float = 5.0,
) -> Dict[str, str]:
    """
    Translate a batch of words using Gemini API.

    Args:
        gemini_client: Initialized GeminiClient
        word_batch: List of words to translate
        retry_count: Number of retries on failure
        retry_delay: Delay between retries

    Returns:
        Dictionary of English to Polish translations
    """
    # Create the prompt for Gemini
    prompt = f"""
Task: Translate the following English words to Polish.

Instructions:
1. For each word, provide the most common and appropriate Polish translation.
2. If a word has multiple meanings, choose the most generally applicable translation.
3. Respond with ONLY a valid JSON dictionary where keys are the English words and values are their Polish translations.
4. Do not include any explanations, notes, or markdown formatting in the response - only the JSON dictionary.
5. If you cannot translate a specific word, exclude it from the JSON response.
6. Use the exact translation, for example "accompanies": "towarzyszy"
7. Each original word (key) should be mapped to exactly one word (value), without any additional notes.

Words to translate:
{', '.join(word_batch)}

Example response format:
{{
  "hello": "cześć",
  "world": "świat",
  "computer": "komputer"
}}

Remember to provide a valid json, it has to be parsable! Answer with a json only!
"""

    # Try to get translations with retries
    for attempt in range(retry_count):
        try:
            response = gemini_client.prompt(prompt)

            # Extract the JSON dictionary from the response
            json_text = response.strip()

            # Remove any markdown code block formatting if present
            if json_text.startswith("```json"):
                json_text = json_text.replace("```json", "", 1)
            if json_text.startswith("```"):
                json_text = json_text.replace("```", "", 1)
            if json_text.endswith("```"):
                json_text = json_text[:-3]

            json_text = json_text.strip()

            # Parse the JSON
            translations = json.loads(json_text)

            return translations

        except Exception as e:
            print(f"Error translating batch (attempt {attempt+1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    # If all retries failed, return an empty dictionary
    print(f"Failed to translate batch after {retry_count} attempts")
    return {}


def build_dictionary_with_gemini(
    word_list_path: str,
    output_path: str,
    generation_cfg: Dict[str, any],
    batch_size: int = 100,
    max_batches: Optional[int] = None,
) -> None:
    """
    Build an English-Polish dictionary using Gemini API.

    Args:
        word_list_path: Path to a text file with English words (one per line)
        output_path: Path to save the resulting dictionary
        generation_cfg: Configuration for the Gemini client
        batch_size: Number of words to process in a batch
        max_batches: Maximum number of batches to process (None for all)
    """
    # Initialize Gemini client
    gemini_client = GeminiClient(generation_cfg)

    # Load English words
    with open(word_list_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    # If there are no words, log and exit
    if not words:
        print("No words found in the word list. Exiting.")
        return

    print(f"Loaded {len(words)} words from {word_list_path}")

    # Try to load existing dictionary to continue from previous run
    dictionary = {}
    if os.path.exists(output_path):
        try:
            dictionary = load_dictionary(output_path)
            print(f"Loaded existing dictionary with {len(dictionary)} entries")

            # Remove already processed words
            words = [word for word in words if word not in dictionary]
            print(f"Remaining words to process: {len(words)}")
        except Exception as e:
            print(f"Error loading existing dictionary: {e}")

    # Split words into batches
    batches = [words[i : i + batch_size] for i in range(0, len(words), batch_size)]

    if max_batches:
        batches = batches[:max_batches]
        print(f"Processing {len(batches)} batches (limited by max_batches)")
    else:
        print(f"Processing {len(batches)} batches")

    # Process each batch
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        print(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} words")

        # Translate the batch
        batch_translations = batch_translate_with_gemini(gemini_client, batch)

        # Update the dictionary
        dictionary.update(batch_translations)

        # Save progress after each batch
        save_dictionary(dictionary, output_path)
        print(f"Dictionary now has {len(dictionary)} entries")

        # Short delay between batches to avoid API rate limits
        time.sleep(1.0)

    # Final save
    save_dictionary(dictionary, output_path)

    if len(dictionary) < 50:
        print(f"Warning: Dictionary built with only {len(dictionary)} entries. Using fallback dictionary.")
        create_en_pl_dictionary_for_europarl(output_path)
    else:
        print(f"Dictionary built with {len(dictionary)} entries.")


def create_en_pl_dictionary_for_europarl(
    output_path: str = "dictionaries/en_pl_dictionary.json",
) -> None:
    """
    Create a comprehensive English-Polish dictionary focused on Europarl vocabulary.

    Args:
        output_path: Path to save the dictionary
    """
    # Load the basic dictionary
    dictionary = {
        # Basic vocabulary
        "hello": "cześć",
        "world": "świat",
        "this": "to",
        "is": "jest",
        "a": "a",
        "test": "test",
        "translation": "tłumaczenie",
        # Pronouns
        "i": "ja",
        "you": "ty",
        "he": "on",
        "she": "ona",
        "we": "my",
        "they": "oni",
        "it": "to",
        # Common nouns
        "house": "dom",
        "water": "woda",
        "problem": "problem",
        "work": "praca",
        "report": "raport",
        "president": "prezydent",
        "committee": "komitet",
        "environment": "środowisko",
        "opinion": "opinia",
        "agriculture": "rolnictwo",
        "development": "rozwój",
        "resources": "zasoby",
        "management": "zarządzanie",
        "drought": "susza",
        "scarcity": "niedobór",
        "countries": "kraje",
        "union": "unia",
        "european": "europejski",
        "climate": "klimat",
        "dimension": "wymiar",
        "consequences": "konsekwencje",
        "observatory": "obserwatorium",
        "reality": "rzeczywistość",
        # Verbs
        "would": "chciałby",
        "like": "lubić",
        "congratulate": "gratulować",
        "done": "zrobione",
        "picks": "wybiera",
        "expressed": "wyrażone",
        "are": "są",
        "have": "mieć",
        "ceased": "przestało",
        "be": "być",
        "am": "jestem",
        "pleased": "zadowolony",
        "incorporates": "zawiera",
        "included": "zawarte",
        "was": "był",
        "play": "grać",
        "includes": "zawiera",
        "push": "pchać",
        "think": "myśleć",
        "highlight": "podkreślić",
        "keep": "zachować",
        "mind": "umysł",
        "establishing": "ustanowienie",
        "mentioned": "wspomniany",
        "hope": "nadzieja",
        "become": "stać się",
        # Adjectives
        "first": "pierwszy",
        "firstly": "po pierwsze",
        "many": "wiele",
        "crucial": "kluczowy",
        "whole": "cały",
        "southern": "południowy",
        "sustainable": "zrównoważony",
        "agricultural": "rolniczy",
        "available": "dostępny",
        "important": "ważny",
        "public": "publiczny",
        "health": "zdrowie",
        "safety": "bezpieczeństwo",
        "current": "obecny",
        "economic": "ekonomiczny",
        "environmental": "środowiskowy",
        # Prepositions and articles
        "in": "w",
        "on": "na",
        "of": "z",
        "for": "dla",
        "with": "z",
        "by": "przez",
        "to": "do",
        "from": "z",
        "as": "jako",
        "up": "w górę",
        "the": "",
        # Other words
        "mr": "pan",
        "because": "ponieważ",
        "regarding": "dotyczący",
        "which": "który",
        "now": "teraz",
        "only": "tylko",
        "some": "niektóre",
        "so": "więc",
        "also": "także",
        "not": "nie",
        "and": "i",
        "that": "że",
        "one": "jeden",
        "day": "dzień",
        # Europarl-specific vocabulary
        "parliament": "parlament",
        "deputy": "poseł",
        "delegate": "delegat",
        "commission": "komisja",
        "directive": "dyrektywa",
        "regulation": "rozporządzenie",
        "legislation": "ustawodawstwo",
        "resolution": "rezolucja",
        "proposal": "propozycja",
        "amendment": "poprawka",
        "vote": "głosowanie",
        "debate": "debata",
        "session": "sesja",
        "council": "rada",
        "member": "członek",
        "state": "państwo",
        "citizens": "obywatele",
        "policy": "polityka",
        "budget": "budżet",
        "sustainable": "zrównoważony",
        "economic": "ekonomiczny",
        "social": "społeczny",
        "political": "polityczny",
        "democratic": "demokratyczny",
        "representative": "przedstawiciel",
        "chairman": "przewodniczący",
        "vice": "wice",
        "rapporteur": "sprawozdawca",
        "party": "partia",
        "group": "grupa",
        "coalition": "koalicja",
        "opposition": "opozycja",
        "majority": "większość",
        "minority": "mniejszość",
        "procedure": "procedura",
        "agreement": "porozumienie",
        "treaty": "traktat",
        "cooperation": "współpraca",
        "integration": "integracja",
        "implementation": "wdrożenie",
        "framework": "ramy",
        "strategy": "strategia",
        "initiative": "inicjatywa",
        "programme": "program",
        "fund": "fundusz",
        "financial": "finansowy",
        "fiscal": "fiskalny",
        "investment": "inwestycja",
        "trade": "handel",
        "market": "rynek",
        "industry": "przemysł",
        "sector": "sektor",
        "transport": "transport",
        "energy": "energia",
        "climate": "klimat",
        "change": "zmiana",
        "emissions": "emisje",
        "reduction": "redukcja",
        "protection": "ochrona",
        "conservation": "konserwacja",
        "security": "bezpieczeństwo",
        "defense": "obrona",
        "border": "granica",
        "migration": "migracja",
        "immigration": "imigracja",
        "refugee": "uchodźca",
        "crisis": "kryzys",
        "emergency": "nagły wypadek",
        "response": "odpowiedź",
        "action": "działanie",
        "measure": "środek",
        "standard": "standard",
        "quality": "jakość",
        "efficiency": "efektywność",
        "effectiveness": "skuteczność",
        "transparency": "przejrzystość",
        "accountability": "odpowiedzialność",
        "corruption": "korupcja",
        "ethics": "etyka",
        "rights": "prawa",
        "freedom": "wolność",
        "justice": "sprawiedliwość",
        "equality": "równość",
        "discrimination": "dyskryminacja",
        "gender": "płeć",
        "women": "kobiety",
        "men": "mężczyźni",
        "youth": "młodzież",
        "elderly": "osoby starsze",
        "disabled": "niepełnosprawni",
        "minorities": "mniejszości",
        "indigenous": "rdzenni",
        "rural": "wiejski",
        "urban": "miejski",
        "regional": "regionalny",
        "national": "krajowy",
        "international": "międzynarodowy",
        "global": "globalny",
        "local": "lokalny",
        "central": "centralny",
        "federal": "federalny",
        "bilateral": "dwustronny",
        "multilateral": "wielostronny",
        # Continuous-related words (commonly missing)
        "continued": "kontynuowany",
        "continues": "kontynuuje",
        "continuing": "kontynuując",
        "continuity": "ciągłość",
        "continuous": "ciągły",
        "continuously": "nieprzerwanie",
        "continuum": "kontinuum",
        "contours": "kontury",
        "contraception": "antykoncepcja",
        "contract": "umowa",
    }

    # Save the enhanced dictionary
    save_dictionary(dictionary, output_path)
    print(f"Enhanced Europarl dictionary created with {len(dictionary)} entries.")


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
        "mr": "pan",
        "because": "ponieważ",
        "president": "prezydent",
        "congratulate": "gratulować",
        "work": "praca",
        "done": "zrobiony",
        "report": "raport",
        "many": "wiele",
        "concerns": "obawy",
        "expressed": "wyrażone",
        "in": "w",
        "house": "dom",
        "regarding": "dotyczący",
        "problems": "problemy",
        "crucial": "kluczowy",
        "for": "dla",
        "whole": "cały",
        "european": "europejski",
        "union": "unia",
    }

    save_dictionary(test_dictionary, output_path)
    print(f"Test dictionary created with {len(test_dictionary)} entries.")


def prepare_dictionary(
    cfg,
    data_manager=None,
) -> str:
    """
    Prepare the dictionary based on configuration.

    Args:
        cfg: Configuration object
        data_manager: Data manager instance (optional)

    Returns:
        Path to the dictionary file
    """
    # Create dictionaries directory if it doesn't exist
    os.makedirs(to_absolute_path("tmp/dictionaries"), exist_ok=True)

    dictionary_path = to_absolute_path(cfg.dictionary.path)

    # Build dictionary if requested
    if cfg.dictionary.build:
        if cfg.dictionary.source == "gemini":
            # Generate word list if data_manager is provided
            if data_manager:
                word_list_path = to_absolute_path("tmp/dictionaries/word_list.txt")
                generate_word_list_from_dataset(
                    data_manager=data_manager,
                    output_path=word_list_path,
                    max_words=cfg.dictionary.max_words if hasattr(cfg.dictionary, "max_words") else 50000,
                )
            else:
                # Use existing word list
                word_list_path = cfg.dictionary.word_list_path
                if not os.path.exists(word_list_path):
                    raise ValueError(f"Word list not found at {word_list_path}")

            # Configure Gemini client
            generation_cfg = {
                "model_path": cfg.gemini.model_path if hasattr(cfg, "gemini") else "gemini-1.5-flash",
                "max_new_tokens": cfg.gemini.max_new_tokens if hasattr(cfg, "gemini") else 500000,
                "temperature": cfg.gemini.temperature if hasattr(cfg, "gemini") else 0.2,
            }

            # Build dictionary using Gemini
            build_dictionary_with_gemini(
                word_list_path=word_list_path,
                output_path=dictionary_path,
                generation_cfg=generation_cfg,
                batch_size=cfg.dictionary.batch_size if hasattr(cfg.dictionary, "batch_size") else 500,
                max_batches=cfg.dictionary.max_batches if hasattr(cfg.dictionary, "max_batches") else None,
            )
        else:  # fallback to manual dictionary
            create_en_pl_dictionary_for_europarl(output_path=dictionary_path)

    # Create test dictionary if no dictionary exists
    if not os.path.exists(dictionary_path):
        print(f"Dictionary file {dictionary_path} not found. Creating a test dictionary.")
        create_test_dictionary(output_path=dictionary_path)

    return dictionary_path

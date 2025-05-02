import os
import re
import json
import time
from typing import Set, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from bs4 import BeautifulSoup


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
    max_words: int = 50000,
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


def fetch_translation_from_diki(word: str, known_missing: Set[str] = None) -> Tuple[Optional[Tuple[str, str]], bool]:
    """
    Fetch translation for a word from diki.pl

    Args:
        word: Word to translate from English to Polish
        known_missing: Set of words known to be missing

    Returns:
        Tuple of (translation tuple or None, bool indicating if word should be retried)
    """
    # Skip words already known to be missing
    if known_missing and word in known_missing:
        return None, False

    # Skip invalid dictionary words
    if not is_valid_dictionary_word(word):
        return None, False

    # Clean up the word
    clean_word = word.strip().lower()

    # Construct URL for Diki.pl
    url = f"https://www.diki.pl/slownik-angielskiego?q={clean_word}"  # noqa

    try:
        # Add headers to mimic a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",  # noqa
            "Accept-Language": "en-US,en;q=0.9,pl;q=0.8",
        }

        # Send the request
        response = requests.get(url, headers=headers, timeout=10)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse HTML content
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the first translation in the proper section
            # Each meaning is in a div with class="foreignToNativeMeanings"
            meanings_div = soup.select_one(".foreignToNativeMeanings")

            if meanings_div:
                # Find the first Polish translation (in span with class="hw")
                translation_element = meanings_div.select_one(".hw")

                if translation_element:
                    polish_translation = translation_element.text.strip()
                    print(f"Found translation for '{word}': '{polish_translation}'")
                    return (word, polish_translation), False

            # No translation found
            print(f"No Polish translation found for '{word}'")
            return None, False

        elif response.status_code == 429:
            # Too Many Requests - should retry
            print(f"Rate limited for '{word}', will retry later")
            return None, True

        else:
            print(f"Failed to fetch translation for '{word}': Status code {response.status_code}")
            return None, False

    except requests.exceptions.Timeout:
        # Timeout error - should retry
        print(f"Timeout for '{word}', will retry later")
        return None, True

    except Exception as e:
        print(f"Error processing word '{word}': {e}")
        return None, False


def build_dictionary_from_diki(
    word_list_path: str,
    output_path: str,
    max_workers: int = 4,
    batch_size: int = 20,
    delay: float = 1.0,
) -> None:
    """
    Build an English-Polish dictionary by scraping Diki.pl using multithreading.

    Args:
        word_list_path: Path to a text file with English words (one per line)
        output_path: Path to save the resulting dictionary
        max_workers: Maximum number of worker threads
        batch_size: Size of word batches for processing
        delay: Delay between requests to avoid rate limiting
    """
    # Load English words
    with open(word_list_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    # If there are no words, log and exit
    if not words:
        print("No words found in the word list. Exiting.")
        return

    print(f"Loaded {len(words)} words from {word_list_path}")

    # Track words that are not found or failed
    known_missing = set()
    retry_words = set()

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

    # Process words in batches with ThreadPoolExecutor
    for i in range(0, len(words), batch_size):
        batch = words[i : i + batch_size]

        # Process this batch
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit translation tasks
            future_to_word = {
                executor.submit(fetch_translation_from_diki, word, known_missing): word
                for word in batch
                if word not in known_missing
            }

            # Process results as they complete
            for future in as_completed(future_to_word):
                word = future_to_word[future]
                try:
                    result, should_retry = future.result()

                    if result:
                        # Add translation to dictionary
                        english_word, polish_translation = result
                        dictionary[english_word] = polish_translation
                    elif should_retry:
                        # Add to retry list
                        retry_words.add(word)
                    else:
                        # Add to missing list
                        known_missing.add(word)

                except Exception as e:
                    print(f"Error processing word '{word}': {e}")
                    retry_words.add(word)

                # Short delay to avoid overwhelming the server
                time.sleep(delay / max_workers)

        # Save progress after each batch
        save_dictionary(dictionary, output_path)
        print(
            f"Progress: {len(dictionary)} words translated, "
            f"{len(known_missing)} words not found, "
            f"{len(retry_words)} words to retry"
        )

        # Add a longer delay between batches
        time.sleep(delay * 2)

    # Process retry words if any
    if retry_words:
        print(f"Retrying {len(retry_words)} words with longer delays...")
        for word in tqdm(retry_words):
            result, _ = fetch_translation_from_diki(word, known_missing)
            if result:
                english_word, polish_translation = result
                dictionary[english_word] = polish_translation
            else:
                known_missing.add(word)

            # Longer delay for retries
            time.sleep(delay * 3)

            # Save after each retry
            if len(dictionary) % 10 == 0:
                save_dictionary(dictionary, output_path)

    # Final save
    save_dictionary(dictionary, output_path)

    if len(dictionary) < 50:
        print(f"Warning: Dictionary built with only {len(dictionary)} entries. Using fallback dictionary.")
        create_en_pl_dictionary_for_europarl(output_path)
    else:
        print(f"Dictionary built with {len(dictionary)} entries.")
        print(f"Words not found: {len(known_missing)}")


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

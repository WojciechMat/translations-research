from abc import ABC, abstractmethod

from translations.models.base_translator import Translator
from translations.data.management import TranslationDataset
from translations.llm_client.gemini_client import GeminiClient


class GeminiTranslator(Translator):
    def __init__(self):
        generation_cfg = {
            "model_path": "gemini-2.0-flash-001",
            "max_new_tokens": 1024,
            "temperature": 0.2,
        }
        self.llm_client = GeminiClient(generation_cfg=generation_cfg)

    def translate(
        self,
        text: str,
    ) -> str:
        prompt = f"""You are a professional english to polish translator.
        Your task is to read a sentence in english and translate it to polis the best you can.
        Keep the original meaning and style of the english text.
        The english text to translate is:
        ```
        {text}
        ```
        Answer with ONLY the translated text in polish."""
        response = self.llm_client.prompt(prompt=prompt).strip().strip("```").strip()
        return response

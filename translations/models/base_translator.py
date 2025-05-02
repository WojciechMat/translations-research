from abc import ABC, abstractmethod

from translations.data.management import TranslationDataset


class Translator(ABC):
    """Base class for all translation methods"""

    @abstractmethod
    def translate(
        self,
        text: str,
    ) -> str:
        """
        Translate a single text.

        Args:
            text: Text to translate

        Returns:
            Translated text
        """
        pass

    def translate_batch(
        self,
        dataset: TranslationDataset,
    ) -> TranslationDataset:
        """
        Translate a batch of texts.
        
        Args:
            dataset: Dataset containing texts to translate
            
        Returns:
            Dataset with translated texts
        """
        translated_texts = []
        
        for i in range(len(dataset)):
            pair = dataset[i]
            translated = self.translate(pair.source)
            translated_texts.append(translated)
            
        # Create a new dataset with the source texts and translations
        return TranslationDataset(
            source=dataset.source,
            reference=translated_texts,  # Put the translations in the reference field
        )

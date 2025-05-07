import xmlrpc.client
from typing import Dict, Any

from abc import ABC, abstractmethod
from translations.models.base_translator import Translator
from translations.data.management import TranslationDataset

class MosesTranslator(Translator):
    """
    Translator implementation that uses Moses SMT server via XML-RPC
    """
    
    def __init__(self, server_url: str = "http://localhost:8080/RPC2"):
        """
        Initialize Moses translator with server URL
        
        Args:
            server_url: URL to the Moses XML-RPC server
        """
        self.server = xmlrpc.client.ServerProxy(server_url)
        
    def translate(self, text: str) -> str:
        """
        Translate a single text using Moses SMT server
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        # Prepare the parameters for Moses server
        # Moses server expects a dictionary with 'text' key
        params = {"text": text}
        
        try:
            # Send the translation request to Moses server
            result = self.server.translate(params)
            
            # Moses returns a dictionary with 'text' key containing the translation
            if isinstance(result, Dict) and 'text' in result:
                return result['text']
            else:
                # Fallback in case of unexpected response format
                return str(result)
                
        except Exception as e:
            # Handle potential errors (connection issues, server errors, etc.)
            print(f"Translation error: {e}")
            # Return original text as fallback
            return text
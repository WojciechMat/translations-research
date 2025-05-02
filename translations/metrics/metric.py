from typing import Optional
from dataclasses import dataclass


@dataclass
class TestCase:
    """
    Data class to store a translation test case with expected and actual results.
    """

    original_text: str
    expected_translation: Optional[str] = None
    actual_translation: Optional[str] = None

    def __str__(self) -> str:
        """Get a string representation of the test case."""
        result = f"Original:   {self.original_text}\n"
        if self.expected_translation:
            result += f"Expected:   {self.expected_translation}\n"
        if self.actual_translation:
            result += f"Translated: {self.actual_translation}"
        return result

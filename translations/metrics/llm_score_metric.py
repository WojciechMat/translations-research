"""
LLM-based scoring metric for translation evaluation.
This metric uses Gemini to score translations.
"""
# flake8: noqa
import json
import time
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from translations.metrics.metric import Metric
from translations.llm_client.gemini_client import GeminiClient


class LLMScoreMetric(Metric):
    """
    Metric that uses an LLM (Gemini) to score translations on a scale of 1-10,
    normalized to 0-1 for consistency with other metrics.
    """

    def __init__(
        self,
        generation_cfg: Optional[DictConfig] = None,
        retry_count: int = 3,
        retry_delay: float = 2.0,
        cache_results: bool = True,
    ) -> None:
        """
        Initialize the LLM-based scoring metric.

        Args:
            generation_cfg: Configuration for the Gemini client
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
            cache_results: Whether to cache results to avoid repeated API calls
        """
        self._name = "llm_score"  # Override the default class name for the metric factory

        # Default configuration if none provided
        if generation_cfg is None:
            generation_cfg = {
                "model_path": "gemini-2.0-flash-001",
                "max_new_tokens": 1024,
                "temperature": 0.2,
            }

        # Initialize the Gemini client
        self.client = GeminiClient(generation_cfg)

        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.cache_results = cache_results
        self.cache = {}  # Simple in-memory cache

        # Track reasoning for the score
        self.reasoning_cache = {}

    def compute(
        self,
        hypothesis: str,
        reference: str,
        source_text: Optional[str] = None,
    ) -> float:
        """
        Compute an LLM-based score for translation quality.

        Args:
            hypothesis: The translation being evaluated
            reference: The reference translation
            source_text: Optional source text for context

        Returns:
            Normalized score between 0.0 and 1.0
        """
        # Create a cache key
        cache_key = f"{hypothesis}|{reference}"
        if source_text:
            cache_key += f"|{source_text}"

        # Check cache if enabled
        if self.cache_results and cache_key in self.cache:
            return self.cache[cache_key]

        # Create the prompt for Gemini
        if source_text:
            prompt = self._create_prompt_with_source(
                hypothesis=hypothesis,
                reference=reference,
                source_text=source_text,
            )
        else:
            prompt = self._create_prompt(
                hypothesis=hypothesis,
                reference=reference,
            )

        # Call Gemini with retries
        raw_score, reasoning = self._call_llm_with_retry(prompt)

        # Normalize score to 0-1 range
        normalized_score = raw_score / 10.0

        # Cache the result if caching is enabled
        if self.cache_results:
            self.cache[cache_key] = normalized_score
            self.reasoning_cache[cache_key] = reasoning

        return normalized_score

    def get_reasoning(
        self,
        hypothesis: str,
        reference: str,
        source_text: Optional[str] = None,
    ) -> str:
        """
        Get the reasoning for the most recent score for this translation pair.

        Args:
            hypothesis: The translation being evaluated
            reference: The reference translation
            source_text: Optional source text for context

        Returns:
            Reasoning text for the score
        """
        cache_key = f"{hypothesis}|{reference}"
        if source_text:
            cache_key += f"|{source_text}"

        if cache_key in self.reasoning_cache:
            return self.reasoning_cache[cache_key]
        else:
            # If no cached reasoning, compute the score to generate the reasoning
            _ = self.compute(hypothesis, reference, source_text)
            return self.reasoning_cache.get(cache_key, "No reasoning available.")

    def compute_batch(
        self,
        hypotheses: List[str],
        references: List[str],
        source_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute scores for a batch of translations.

        Args:
            hypotheses: The translations being evaluated
            references: The reference translations
            source_texts: Optional source texts for context

        Returns:
            Dictionary with overall score and individual scores
        """
        if len(hypotheses) != len(references):
            raise ValueError("Number of hypotheses and references must match")

        if source_texts and len(source_texts) != len(hypotheses):
            raise ValueError("Number of source texts must match hypotheses")

        scores = []
        reasonings = []

        for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
            src = source_texts[i] if source_texts else None
            score = self.compute(hyp, ref, src)
            scores.append(score)

            # Get reasoning for this score
            reasoning = self.get_reasoning(hyp, ref, src)
            reasonings.append(reasoning)

        return {
            "score": sum(scores) / len(scores) if scores else 0.0,
            "individual_scores": scores,
            "reasonings": reasonings,
        }

    def _create_prompt(
        self,
        hypothesis: str,
        reference: str,
    ) -> str:
        """
        Create a prompt for Gemini to score a translation.

        Args:
            hypothesis: The translation being evaluated
            reference: The reference translation

        Returns:
            Prompt string
        """
        return f"""
Task: Rate the quality of a Polish translation on a scale of 1-10.

Reference (correct Polish translation):
{reference}

Machine Translation (to be evaluated):
{hypothesis}

Instructions:
1. Compare the machine translation to the reference translation.
2. Evaluate accuracy, fluency, and preservation of meaning.
3. Rate the translation on a scale of 1-10 where:
   - 1-3: Poor translation with major errors
   - 4-6: Acceptable translation with some issues
   - 7-8: Good translation with minor issues
   - 9-10: Excellent translation, nearly perfect

4. Provide a brief explanation of your rating (2-3 sentences).
5. Format your response as a JSON object with "score" (integer 1-10) and "reasoning" (string) fields.

Example response format:
{{
  "score": 7,
  "reasoning": "The translation captures the main meaning but has minor grammatical errors. Word choice is mostly appropriate with one awkward phrase. Overall, it's comprehensible and conveys the intended message well."
}}

Respond ONLY with a valid, parsable JSON object. Do not include any other text.
"""

    def _create_prompt_with_source(
        self,
        hypothesis: str,
        reference: str,
        source_text: str,
    ) -> str:
        """
        Create a prompt for Gemini to score a translation with source text.

        Args:
            hypothesis: The translation being evaluated
            reference: The reference translation
            source_text: Source text in English

        Returns:
            Prompt string
        """
        return f"""
Task: Rate the quality of a Polish translation on a scale of 1-10.

Original English Text:
{source_text}

Reference (correct Polish translation):
{reference}

Machine Translation (to be evaluated):
{hypothesis}

Instructions:
1. Compare the machine translation to the reference translation.
2. Consider how well both capture the meaning of the original English text.
3. Evaluate accuracy, fluency, and preservation of meaning.
4. Rate the translation on a scale of 1-10 where:
   - 1-3: Poor translation with major errors
   - 4-6: Acceptable translation with some issues
   - 7-8: Good translation with minor issues
   - 9-10: Excellent translation, nearly perfect

5. Provide a brief explanation of your rating (2-3 sentences).
6. Format your response as a JSON object with "score" (integer 1-10) and "reasoning" (string) fields.

Example response format:
{{
  "score": 7,
  "reasoning": "The translation captures the main meaning but has minor grammatical errors. Word choice is mostly appropriate with one awkward phrase. Overall, it's comprehensible and conveys the intended message well."
}}

Respond ONLY with a valid, parsable JSON object. Do not include any other text.
"""

    def _call_llm_with_retry(
        self,
        prompt: str,
    ) -> tuple[int, str]:
        """
        Call the LLM with retry logic.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Tuple of (score, reasoning)
        """
        for attempt in range(self.retry_count):
            try:
                response = self.client.prompt(prompt)
                # Parse the JSON respons
                response = response.split("```")[1].strip("json").strip()
                response_json = json.loads(response.strip())

                # Extract score and reasoning
                score = int(response_json.get("score", 0))
                reasoning = response_json.get("reasoning", "No reasoning provided.")

                # Validate score
                if not 1 <= score <= 10:
                    raise ValueError(f"Score out of range: {score}")

                return score, reasoning

            except Exception as e:
                print(f"Error calling LLM (attempt {attempt+1}/{self.retry_count}): {e}")
                if attempt < self.retry_count - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)

        # If all retries fail, return a default score
        print(f"Failed to get LLM score after {self.retry_count} attempts, returning default score of 5")
        return 5, "Failed to get response from LLM after multiple attempts."

    @property
    def name(self) -> str:
        """Get the name of the metric."""
        return self._name

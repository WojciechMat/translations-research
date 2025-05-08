from collections import Counter
from typing import Dict, List, Tuple

from translations.metrics.metric import Metric


class ExactMatchMetric(Metric):
    """
    Metric that calculates the percentage of exact translation matches.
    """

    def compute(
        self,
        hypothesis: str,
        reference: str,
    ) -> float:
        """
        Compute exact match score (1.0 if exact match, 0.0 otherwise).

        Args:
            hypothesis: The translation being evaluated
            reference: The reference translation

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return 1.0 if hypothesis.strip() == reference.strip() else 0.0


class TokenOverlapMetric(Metric):
    """
    Metric that calculates token-level overlap between hypothesis and reference.
    """

    def compute(
        self,
        hypothesis: str,
        reference: str,
    ) -> float:
        """
        Compute token overlap as the Jaccard similarity.

        Args:
            hypothesis: The translation being evaluated
            reference: The reference translation

        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        # Simple tokenization by splitting on whitespace
        hypothesis_tokens = set(hypothesis.lower().split())
        reference_tokens = set(reference.lower().split())

        if not hypothesis_tokens and not reference_tokens:
            return 1.0  # Both empty means perfect match

        # Calculate Jaccard similarity: intersection over union
        intersection = len(hypothesis_tokens.intersection(reference_tokens))
        union = len(hypothesis_tokens.union(reference_tokens))

        return intersection / union if union > 0 else 0.0


class PrecisionRecallMetric(Metric):
    """
    Metric that calculates precision, recall, and F1 score for token overlap.
    """

    def __init__(
        self,
        lowercase: bool = True,
    ) -> None:
        """
        Initialize the metric.

        Args:
            lowercase: Whether to lowercase tokens before comparison
        """
        self.lowercase = lowercase

    def compute(
        self,
        hypothesis: str,
        reference: str,
    ) -> float:
        """
        Compute F1 score for token overlap.

        Args:
            hypothesis: The translation being evaluated
            reference: The reference translation

        Returns:
            F1 score (0.0 to 1.0)
        """
        precision, recall, f1 = self._compute_precision_recall_f1(
            hypothesis=hypothesis,
            reference=reference,
        )

        return f1

    def compute_batch(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute precision, recall, and F1 for a batch of translations.

        Args:
            hypotheses: The translations being evaluated
            references: The reference translations

        Returns:
            Dictionary with precision, recall, F1, and individual scores
        """
        if len(hypotheses) != len(references):
            raise ValueError("Number of hypotheses and references must match")

        precision_sum = 0.0
        recall_sum = 0.0
        f1_sum = 0.0
        individual_scores = []

        for hyp, ref in zip(hypotheses, references):
            precision, recall, f1 = self._compute_precision_recall_f1(
                hypothesis=hyp,
                reference=ref,
            )

            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
            individual_scores.append(f1)

        n = len(hypotheses)

        return {
            "precision": precision_sum / n if n > 0 else 0.0,
            "recall": recall_sum / n if n > 0 else 0.0,
            "f1": f1_sum / n if n > 0 else 0.0,
            "score": f1_sum / n if n > 0 else 0.0,
            "individual_scores": individual_scores,
        }

    def _compute_precision_recall_f1(
        self,
        hypothesis: str,
        reference: str,
    ) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1 score.

        Args:
            hypothesis: The translation being evaluated
            reference: The reference translation

        Returns:
            Tuple of (precision, recall, F1)
        """
        # Tokenize and optionally lowercase
        if self.lowercase:
            hyp_tokens = hypothesis.lower().split()
            ref_tokens = reference.lower().split()
        else:
            hyp_tokens = hypothesis.split()
            ref_tokens = reference.split()

        # Convert to sets for overlap calculation
        hyp_set = set(hyp_tokens)
        ref_set = set(ref_tokens)

        # Calculate intersection
        intersection = hyp_set.intersection(ref_set)

        # Calculate precision and recall
        precision = len(intersection) / len(hyp_set) if hyp_set else 0.0
        recall = len(intersection) / len(ref_set) if ref_set else 0.0

        # Calculate F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1


class BLEUMetric(Metric):
    """
    Simplified BLEU (Bilingual Evaluation Understudy) metric implementation.
    BLEU measures the similarity between a machine translation and a reference translation.
    """

    def __init__(
        self,
        max_n: int = 4,
        weights: List[float] = None,
        smooth: bool = True,
    ) -> None:
        """
        Initialize the BLEU metric.

        Args:
            max_n: Maximum n-gram size to consider
            weights: Weights for each n-gram size (default is uniform)
            smooth: Whether to apply smoothing for zero counts
        """
        self.max_n = max_n
        self.weights = weights or [1.0 / max_n] * max_n
        self.smooth = smooth

    def compute(
        self,
        hypothesis: str,
        reference: str,
    ) -> float:
        """
        Compute BLEU score for a single translation pair.

        Args:
            hypothesis: The translation being evaluated
            reference: The reference translation

        Returns:
            BLEU score (0.0 to 1.0)
        """
        # Tokenize
        hyp_tokens = hypothesis.lower().split()
        ref_tokens = reference.lower().split()

        # If either is empty, return 0
        if not hyp_tokens or not ref_tokens:
            return 0.0

        # Calculate brevity penalty
        bp = self._brevity_penalty(
            hyp_length=len(hyp_tokens),
            ref_length=len(ref_tokens),
        )

        # Calculate n-gram precision for each n
        precisions = []
        for n in range(1, self.max_n + 1):
            precision = self._ngram_precision(
                hyp_tokens=hyp_tokens,
                ref_tokens=ref_tokens,
                n=n,
            )
            precisions.append(precision)

        # Apply smoothing if needed
        if self.smooth:
            precisions = [max(p, 1e-10) for p in precisions]

        # Calculate weighted log probability
        log_score = sum(w * self._log(p) for w, p in zip(self.weights, precisions))

        # Apply brevity penalty and convert to score
        bleu = bp * self._exp(log_score)

        return bleu

    def _brevity_penalty(
        self,
        hyp_length: int,
        ref_length: int,
    ) -> float:
        """
        Calculate brevity penalty.

        Args:
            hyp_length: Length of hypothesis
            ref_length: Length of reference

        Returns:
            Brevity penalty factor
        """
        if hyp_length >= ref_length:
            return 1.0

        return self._exp(1 - ref_length / hyp_length)

    def _ngram_precision(
        self,
        hyp_tokens: List[str],
        ref_tokens: List[str],
        n: int,
    ) -> float:
        """
        Calculate n-gram precision.

        Args:
            hyp_tokens: Tokens from hypothesis
            ref_tokens: Tokens from reference
            n: n-gram size

        Returns:
            n-gram precision score
        """
        # Generate n-grams
        hyp_ngrams = self._get_ngrams(
            tokens=hyp_tokens,
            n=n,
        )
        ref_ngrams = self._get_ngrams(
            tokens=ref_tokens,
            n=n,
        )

        # If no n-grams, return 0
        if not hyp_ngrams:
            return 0.0

        # Count matches
        matches = 0
        hyp_counts = Counter(hyp_ngrams)
        ref_counts = Counter(ref_ngrams)

        for ngram, count in hyp_counts.items():
            matches += min(count, ref_counts.get(ngram, 0))

        # Calculate precision
        return matches / sum(hyp_counts.values())

    def _get_ngrams(
        self,
        tokens: List[str],
        n: int,
    ) -> List[Tuple[str, ...]]:
        """
        Generate n-grams from a list of tokens.

        Args:
            tokens: List of tokens
            n: n-gram size

        Returns:
            List of n-grams
        """
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _log(
        self,
        x: float,
    ) -> float:
        """Safe logarithm function that handles zero."""
        import math

        return math.log(max(x, 1e-10))

    def _exp(
        self,
        x: float,
    ) -> float:
        """Exponential function."""
        import math

        return math.exp(x)


class LevenshteinDistanceMetric(Metric):
    """
    Metric based on the Levenshtein (edit) distance between strings.
    Normalized to range from 0.0 (completely different) to 1.0 (identical).
    """

    def compute(
        self,
        hypothesis: str,
        reference: str,
    ) -> float:
        """
        Compute normalized Levenshtein similarity.

        Args:
            hypothesis: The translation being evaluated
            reference: The reference translation

        Returns:
            Normalized Levenshtein similarity (0.0 to 1.0)
        """
        # Calculate raw Levenshtein distance
        distance = self._levenshtein_distance(
            s1=hypothesis,
            s2=reference,
        )

        # Normalize by max length to get a score between 0 and 1
        max_len = max(len(hypothesis), len(reference))

        # Convert distance to similarity score (1.0 is perfect match)
        if max_len == 0:
            return 1.0  # Both strings empty

        return 1.0 - (distance / max_len)

    def _levenshtein_distance(
        self,
        s1: str,
        s2: str,
    ) -> int:
        """
        Calculate the Levenshtein distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Levenshtein distance (number of edits needed)
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(
                s1=s2,
                s2=s1,
            )

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

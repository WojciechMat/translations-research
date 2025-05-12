"""
Google Sheets uploader for translation evaluation with LLM reasoning support.
"""

import os
import logging
from typing import List

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from translations.metrics.metric import TestCase
from translations.metrics.evaluator import TranslationEvaluator

logger = logging.getLogger(__name__)


class TranslationSheetsUploader:
    """Class to upload translation evaluation results to Google Sheets row by row."""

    def __init__(self):
        """Initialize the Google Sheets uploader using the SHEET environment variable."""
        self.sheet_url = os.getenv("SHEET")
        self.client = None

        if not self.sheet_url:
            logger.warning("No Google Sheet URL provided. Set the SHEET environment variable.")
            return

        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]

        # Get credentials from environment variable or default path
        creds_file = os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS",
            "google_credentials.json",
        )
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            filename=creds_file,
            scopes=scopes,
        )

        # Authorize
        worksheet_id = self.sheet_url.split("gid=")[-1]
        self.client = gspread.authorize(credentials)
        self.sheet = self.client.open_by_url(self.sheet_url)
        self.worksheet = self.sheet.get_worksheet_by_id(worksheet_id)

        logger.info("Successfully connected to Google Sheets")

    def _column_letter(self, column_number):
        """Convert a column number to a column letter (e.g., 1 -> A, 27 -> AA)."""
        result = ""
        while column_number > 0:
            column_number, remainder = divmod(column_number - 1, 26)
            result = chr(65 + remainder) + result
        return result

    def add_headers(self, metrics: List[str], include_llm_reasoning: bool = False):
        """
        Add headers to the Google Sheet based on the metrics.

        Args:
            metrics: List of metric names
            include_llm_reasoning: Whether to include a column for LLM reasoning
        """
        headers = [
            "Original Text",
            "Expected Translation",
            "Actual Translation",
            "Overall Score",
        ] + metrics

        # Add LLM reasoning column if needed
        if include_llm_reasoning:
            headers.append("LLM Reasoning")

        self.worksheet.append_row(headers)

        # Get the total number of columns
        num_columns = len(headers)

        # Apply formatting
        header_range = f"A1:{self._column_letter(num_columns)}1"  # noqa

        # Format headers (bold)
        bold_format = {"textFormat": {"bold": True}}
        self.worksheet.format(header_range, bold_format)

        # Format source text area (light green)
        light_green_format = {"backgroundColor": {"red": 0.91, "green": 1.0, "blue": 0.9}}
        self.worksheet.format("A2:A1000", light_green_format)  # noqa

        # Format expected translation (light blue)
        light_blue_format = {"backgroundColor": {"red": 0.9, "green": 0.9, "blue": 1.0}}
        self.worksheet.format("B2:B1000", light_blue_format)  # noqa

        # Format actual translation (light yellow)
        light_yellow_format = {"backgroundColor": {"red": 1.0, "green": 1.0, "blue": 0.9}}
        self.worksheet.format("C2:C1000", light_yellow_format)  # noqa

        # Format LLM reasoning if present (light gray)
        if include_llm_reasoning:
            light_gray_format = {"backgroundColor": {"red": 0.95, "green": 0.95, "blue": 0.95}}
            reasoning_col = self._column_letter(num_columns)
            self.worksheet.format(f"{reasoning_col}2:{reasoning_col}1000", light_gray_format)  # noqa

        logger.info("Added and formatted headers in Google Sheets")

    def append_test_case(
        self,
        test_case: TestCase,
        metric_names: List[str],
        include_llm_reasoning: bool = False,
    ):
        """
        Append a single translation test case to the Google Sheet.

        Args:
            test_case: The translation test case to upload
            metric_names: List of metric names to include in the sheet
            include_llm_reasoning: Whether to include LLM reasoning
        """
        if not self.client:
            return

        for retry in range(5):
            try:
                # Prepare metric scores
                metric_scores = {}
                for name in metric_names:
                    value = test_case.metrics_results.get_metric(name)
                    metric_scores[name] = value if value is not None else ""

                # Calculate overall score (average of all metrics)
                overall_score = (
                    sum(v for v in metric_scores.values() if v != "") / len(metric_scores) if metric_scores else 0
                )

                # Prepare row data
                row_data = [
                    test_case.original_text[:49999] if test_case.original_text else "",
                    test_case.expected_translation[:49999] if test_case.expected_translation else "",
                    test_case.actual_translation[:49999] if test_case.actual_translation else "",
                    overall_score,
                ]

                # Add each individual metric score
                for metric_name in metric_names:
                    row_data.append(metric_scores.get(metric_name, ""))

                # Add LLM reasoning if applicable
                if include_llm_reasoning:
                    llm_reasoning = ""
                    # Look for reasoning in additional_info if it exists
                    if hasattr(test_case.metrics_results, "additional_info"):
                        # Look for reasoning in any of the LLM metrics
                        for name in metric_names:
                            if "llm" in name.lower():
                                reasoning = test_case.metrics_results.additional_info.get(name, {}).get("reasoning", "")
                                if reasoning:
                                    llm_reasoning = reasoning
                                    break
                    row_data.append(llm_reasoning)

                # Append the row to the sheet
                self.worksheet.append_row(row_data)

                # Apply conditional formatting for metrics
                self.format_conditionally(metric_names, overall_score, include_llm_reasoning)

                logger.info(f"Added and formatted test case to Google Sheets: {test_case.original_text[:50]}...")
                break
            except Exception as e:
                logger.error(f"Failed to upload test case to Google Sheets: {e}")
                if retry == 4:  # Last retry
                    logger.error(f"Failed after 5 retries: {e}")
                break

    def format_conditionally(
        self,
        metric_names: List[str],
        overall_score: float,
        include_llm_reasoning: bool = False,
    ):
        """
        Apply conditional formatting to the metrics cells based on their values.

        Args:
            metric_names: List of metric names
            overall_score: The overall average score
            include_llm_reasoning: Whether to include LLM reasoning
        """
        row_index = len(self.worksheet.get_all_records()) + 2  # +2 because of header and 1-indexing

        # Define color formats
        excellent_format = {"backgroundColor": {"red": 0.75, "green": 0.9, "blue": 0.75}}  # Light green
        good_format = {"backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.75}}  # Light yellow
        poor_format = {"backgroundColor": {"red": 1.0, "green": 0.80, "blue": 0.85}}  # Light red

        # Format overall score (column D)
        overall_cell = f"D{row_index}"
        if overall_score >= 0.8:
            self.worksheet.format(overall_cell, excellent_format)
        elif overall_score >= 0.5:
            self.worksheet.format(overall_cell, good_format)
        else:
            self.worksheet.format(overall_cell, poor_format)

        # Format individual metric scores
        col_offset = 5  # Metrics start from column E (5)

        for i, metric_name in enumerate(metric_names):
            col = col_offset + i
            cell = f"{self._column_letter(col)}{row_index}"

            # Get the value from the sheet since it's already been written
            try:
                value = float(self.worksheet.cell(row_index, col).value or 0)

                if value >= 0.8:
                    self.worksheet.format(cell, excellent_format)
                elif value >= 0.5:
                    self.worksheet.format(cell, good_format)
                else:
                    self.worksheet.format(cell, poor_format)
            except (ValueError, TypeError):
                # Skip formatting if the value is not a valid number
                pass

    def upload_evaluation_results(
        self,
        evaluator: TranslationEvaluator,
        test_cases: List[TestCase],
    ):
        """
        Upload all evaluation results to Google Sheets.

        Args:
            evaluator: The TranslationEvaluator instance that contains metrics
            test_cases: List of TestCase instances with evaluation results
        """
        if not self.client:
            logger.warning("Google Sheets client not initialized. Cannot upload results.")
            return

        # Get metric names from evaluator
        metric_names = [metric.name for metric in evaluator.metrics]

        # Check if we need to include LLM reasoning
        include_llm_reasoning = any("llm" in metric.name.lower() for metric in evaluator.metrics)

        # Add headers
        self.add_headers(metric_names, include_llm_reasoning)

        # Upload each test case
        for test_case in test_cases:
            self.append_test_case(
                test_case=test_case,
                metric_names=metric_names,
                include_llm_reasoning=include_llm_reasoning,
            )

        logger.info(f"Successfully uploaded {len(test_cases)} test cases to Google Sheets")

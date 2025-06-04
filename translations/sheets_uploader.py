"""
Google Sheets uploader for translation evaluation with LLM reasoning support.
Optimized for rate limiting and batch operations.
"""

import os
import time
import logging
from typing import List

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from translations.metrics.metric import TestCase
from translations.metrics.evaluator import TranslationEvaluator

logger = logging.getLogger(__name__)


class TranslationSheetsUploader:
    """Class to upload translation evaluation results to Google Sheets with batch operations."""

    def __init__(self):
        """Initialize the Google Sheets uploader using the SHEET environment variable."""
        self.sheet_url = os.getenv("SHEET")
        self.client = None
        self.batch_size = 20  # Number of rows to upload at once
        self.rate_limit_delay = 1.5  # Seconds to between batch operations

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

    def _rate_limit_sleep(self):
        """Sleep to respect rate limits."""
        time.sleep(self.rate_limit_delay)

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

        # Clear existing content first
        self.worksheet.clear()
        self._rate_limit_sleep()

        # Add headers
        self.worksheet.append_row(headers)
        self._rate_limit_sleep()

        # Get the total number of columns
        num_columns = len(headers)

        # Apply formatting in batches
        self._apply_header_formatting(num_columns, include_llm_reasoning)

        logger.info("Added and formatted headers in Google Sheets")

    def _apply_header_formatting(self, num_columns: int, include_llm_reasoning: bool):
        """Apply formatting to headers and columns."""
        try:
            # Format headers (bold)
            header_range = f"A1:{self._column_letter(num_columns)}1"
            bold_format = {"textFormat": {"bold": True}}
            self.worksheet.format(header_range, bold_format)
            self._rate_limit_sleep()

            # Format source text area (light green)
            light_green_format = {"backgroundColor": {"red": 0.91, "green": 1.0, "blue": 0.9}}
            self.worksheet.format("A2:A1000", light_green_format)
            self._rate_limit_sleep()

            # Format expected translation (light blue)
            light_blue_format = {"backgroundColor": {"red": 0.9, "green": 0.9, "blue": 1.0}}
            self.worksheet.format("B2:B1000", light_blue_format)
            self._rate_limit_sleep()

            # Format actual translation (light yellow)
            light_yellow_format = {"backgroundColor": {"red": 1.0, "green": 1.0, "blue": 0.9}}
            self.worksheet.format("C2:C1000", light_yellow_format)
            self._rate_limit_sleep()

            # Format LLM reasoning if present (light gray)
            if include_llm_reasoning:
                light_gray_format = {"backgroundColor": {"red": 0.95, "green": 0.95, "blue": 0.95}}
                reasoning_col = self._column_letter(num_columns)
                self.worksheet.format(f"{reasoning_col}2:{reasoning_col}1000", light_gray_format)
                self._rate_limit_sleep()

        except Exception as e:
            logger.warning(f"Failed to apply some formatting: {e}")

    def _prepare_test_case_row(
        self,
        test_case: TestCase,
        metric_names: List[str],
        include_llm_reasoning: bool = False,
    ) -> List:
        """
        Prepare a single test case row for batch upload.

        Args:
            test_case: The translation test case to prepare
            metric_names: List of metric names to include in the sheet
            include_llm_reasoning: Whether to include LLM reasoning

        Returns:
            List of values for the row
        """
        # Prepare metric scores
        metric_scores = {}
        for name in metric_names:
            value = test_case.metrics_results.get_metric(name)
            metric_scores[name] = value if value is not None else ""

        # Calculate overall score (average of all metrics)
        overall_score = sum(v for v in metric_scores.values() if v != "") / len(metric_scores) if metric_scores else 0

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

        return row_data

    def upload_test_cases_batch(
        self,
        test_cases: List[TestCase],
        metric_names: List[str],
        include_llm_reasoning: bool = False,
    ):
        """
        Upload multiple test cases in batches to reduce API calls.

        Args:
            test_cases: List of test cases to upload
            metric_names: List of metric names
            include_llm_reasoning: Whether to include LLM reasoning
        """
        if not self.client:
            logger.warning("Google Sheets client not initialized. Cannot upload results.")
            return

        # Prepare all rows
        all_rows = []
        for test_case in test_cases:
            row_data = self._prepare_test_case_row(test_case, metric_names, include_llm_reasoning)
            all_rows.append(row_data)

        # Upload in batches
        for i in range(0, len(all_rows), self.batch_size):
            batch = all_rows[i : i + self.batch_size]

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Use batch update for better performance
                    start_row = i + 2  # +2 because of header and 1-indexing
                    end_row = start_row + len(batch) - 1

                    range_name = f"A{start_row}:{self._column_letter(len(batch[0]))}{end_row}"

                    self.worksheet.update(range_name, batch, value_input_option="USER_ENTERED")

                    logger.info(f"Uploaded batch {i//self.batch_size + 1}: rows {start_row}-{end_row}")

                    # Apply conditional formatting for this batch
                    self._apply_batch_conditional_formatting(batch, start_row, metric_names, include_llm_reasoning)

                    # Rate limiting
                    self._rate_limit_sleep()
                    break

                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        wait_time = (attempt + 1) * 30  # Exponential backoff
                        logger.warning(
                            f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to upload batch {i//self.batch_size + 1}: {e}")
                        break
            else:
                logger.error(f"Failed to upload batch {i//self.batch_size + 1} after {max_retries} attempts")

        logger.info(f"Successfully uploaded {len(test_cases)} test cases to Google Sheets in batches")

    def _apply_all_conditional_formatting(
        self,
        all_rows: List[List],
        metric_names: List[str],
        include_llm_reasoning: bool = False,
    ):
        """
        Apply conditional formatting to all rows using range-based formatting for better efficiency.

        Args:
            all_rows: All row data
            metric_names: List of metric names
            include_llm_reasoning: Whether LLM reasoning is included
        """
        if not all_rows:
            return

        try:
            # Define color formats
            excellent_format = {"backgroundColor": {"red": 0.75, "green": 0.9, "blue": 0.75}}
            good_format = {"backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.75}}
            poor_format = {"backgroundColor": {"red": 1.0, "green": 0.80, "blue": 0.85}}

            # Group cells by their format to apply range-based formatting
            excellent_ranges = []
            good_ranges = []
            poor_ranges = []

            start_row = 2  # Data starts at row 2

            for i, row_data in enumerate(all_rows):
                row_index = start_row + i

                # Process overall score (column D)
                if len(row_data) > 3:
                    try:
                        overall_score = float(row_data[3]) if row_data[3] != "" else 0
                        cell_range = f"D{row_index}"

                        if overall_score >= 0.8:
                            excellent_ranges.append(cell_range)
                        elif overall_score >= 0.5:
                            good_ranges.append(cell_range)
                        else:
                            poor_ranges.append(cell_range)
                    except (ValueError, TypeError):
                        pass

                # Process individual metric scores
                for j, metric_name in enumerate(metric_names):
                    metric_index = 4 + j  # Metrics start at index 4
                    col = 5 + j  # Columns start at E (5)

                    if metric_index < len(row_data):
                        try:
                            value = float(row_data[metric_index]) if row_data[metric_index] != "" else 0
                            cell_range = f"{self._column_letter(col)}{row_index}"

                            if value >= 0.8:
                                excellent_ranges.append(cell_range)
                            elif value >= 0.5:
                                good_ranges.append(cell_range)
                            else:
                                poor_ranges.append(cell_range)
                        except (ValueError, TypeError, IndexError):
                            pass

            # Apply formatting in batches by color
            self._apply_format_to_ranges(excellent_ranges, excellent_format, "excellent")
            self._apply_format_to_ranges(good_ranges, good_format, "good")
            self._apply_format_to_ranges(poor_ranges, poor_format, "poor")

        except Exception as e:
            logger.warning(f"Failed to apply conditional formatting: {e}")

    def _apply_format_to_ranges(self, ranges: List[str], format_dict: dict, format_name: str):
        """
        Apply formatting to a list of ranges in batches.

        Args:
            ranges: List of cell ranges (e.g., ["D2", "E2", "F3"])
            format_dict: Format to apply
            format_name: Name for logging purposes
        """
        if not ranges:
            return

        # Apply formatting in smaller batches to avoid overwhelming the API
        batch_size = 20
        total_batches = (len(ranges) + batch_size - 1) // batch_size

        logger.info(f"Applying {format_name} formatting to {len(ranges)} cells in {total_batches} batches")

        for i in range(0, len(ranges), batch_size):
            batch_ranges = ranges[i : i + batch_size]

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Apply formatting to each range in the batch
                    for cell_range in batch_ranges:
                        self.worksheet.format(cell_range, format_dict)

                    logger.debug(f"Applied {format_name} formatting to batch {i//batch_size + 1}/{total_batches}")

                    # Short delay between batches
                    time.sleep(0.8)
                    break

                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        wait_time = (attempt + 1) * 10
                        logger.warning(f"Rate limit hit during formatting, waiting {wait_time} seconds")
                        time.sleep(wait_time)
                    else:
                        logger.warning(f"Failed to apply {format_name} formatting to batch: {e}")
                        break
            else:
                logger.warning(f"Failed to apply {format_name} formatting to batch after {max_retries} attempts")

    def upload_evaluation_results(
        self,
        evaluator: TranslationEvaluator,
        test_cases: List[TestCase],
    ):
        """
        Upload all evaluation results to Google Sheets using batch operations.

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

        # Upload all test cases in batches
        self.upload_test_cases_batch(
            test_cases=test_cases,
            metric_names=metric_names,
            include_llm_reasoning=include_llm_reasoning,
        )

        logger.info(f"Successfully uploaded {len(test_cases)} test cases to Google Sheets using batch operations")

    # Keep the old method for backwards compatibility
    def append_test_case(
        self,
        test_case: TestCase,
        metric_names: List[str],
        include_llm_reasoning: bool = False,
    ):
        """
        Append a single translation test case to the Google Sheet.
        DEPRECATED: Use upload_test_cases_batch for better performance.
        """
        logger.warning("append_test_case is deprecated. Use upload_test_cases_batch for better performance.")
        self.upload_test_cases_batch([test_case], metric_names, include_llm_reasoning)

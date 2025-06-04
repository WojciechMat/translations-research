"""
Translation Evaluation Dashboard

A Streamlit dashboard for visualizing and analyzing translation metrics
from Google Sheets data.
"""
import os
import logging
from typing import Dict, List, Optional

import gspread
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
import plotly.graph_objects as go
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

# Load environment variables
load_dotenv()

creds_file = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "google_credentials.json",
)

CREDENTIALS = Credentials.from_service_account_file(
    creds_file,
    scopes=[
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Translation Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .metric-card {
        background-color: #524e4e;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .highlight-text {
        font-weight: bold;
        color: #1f77b4;
    }
    .good-score {
        color: #2ca02c;
        font-weight: bold;
    }
    .medium-score {
        color: #ff7f0e;
        font-weight: bold;
    }
    .poor-score {
        color: #d62728;
        font-weight: bold;
    }
    .st-emotion-cache-1hyzxq4 {
        max-width: 100%;
    }
</style>
""",
    unsafe_allow_html=True,
)


class SheetsDataLoader:
    """Class to load data from Google Sheets for the dashboard."""

    def __init__(self):
        """Initialize the Google Sheets data loader."""
        self.sheet_url = os.getenv("SHEET")
        self.client = None
        self.all_dataframes = {}
        self.worksheet_names = []
        self.IGNORE_WORKSHEETS = ["entrypoint"]

        if not self.sheet_url:
            st.error("No Google Sheet URL provided. Set the SHEET environment variable.")
            return

        try:
            self.credentials = CREDENTIALS

            self.client = gspread.authorize(self.credentials)
            self.spreadsheet = self.client.open_by_url(self.sheet_url)

            # Get all worksheet names (filtering out ignored ones)
            all_ws_names = [ws.title for ws in self.spreadsheet.worksheets()]
            self.worksheet_names = [
                name for name in all_ws_names if name.lower() not in [n.lower() for n in self.IGNORE_WORKSHEETS]
            ]

            # Also initialize the Drive API to get sheet metadata
            self.drive_service = build("drive", "v3")

            logger.info(f"Successfully connected to Google Sheets. Found {len(self.worksheet_names)} valid worksheets.")
        except Exception as e:
            st.error(f"Error connecting to Google Sheets: {e}")
            logger.error(f"Error connecting to Google Sheets: {e}")

    def get_worksheet_names(self) -> List[str]:
        """Get the list of worksheet names."""
        return self.worksheet_names

    @st.cache_data(ttl=600)
    def _fetch_spreadsheet_metadata(sheet_url: str) -> Dict:
        """
        Fetch metadata about the spreadsheet (cached).

        Args:
            sheet_url: URL of the spreadsheet

        Returns:
            Dictionary with metadata
        """
        try:
            drive_service = build("drive", "v3")
            sheet_id = sheet_url.split("/d/")[1].split("/")[0]
            response = drive_service.files().get(fileId=sheet_id, fields="name,createdTime,modifiedTime").execute()

            return {
                "name": response.get("name", "Unknown"),
                "created": response.get("createdTime", "Unknown"),
                "modified": response.get("modifiedTime", "Unknown"),
            }
        except Exception as e:
            logger.error(f"Error fetching spreadsheet metadata: {e}")
            return {}

    def get_spreadsheet_metadata(self) -> Dict:
        """Get metadata about the spreadsheet."""
        if not self.client:
            return {}

        metadata = SheetsDataLoader._fetch_spreadsheet_metadata(self.sheet_url)

        metadata["num_worksheets"] = len(self.worksheet_names)
        metadata["worksheet_names"] = self.worksheet_names

        return metadata

    @st.cache_data(ttl=300)  # Cache data for 5 minutes
    def _fetch_worksheet_data(sheet_url: str, worksheet_name: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from a specific worksheet (cached).

        Args:
            sheet_url: URL of the spreadsheet
            worksheet_name: Name of the worksheet to load

        Returns:
            DataFrame with the worksheet data or None if an error occurs
        """
        try:
            client = gspread.authorize(CREDENTIALS)
            spreadsheet = client.open_by_url(sheet_url)
            worksheet = spreadsheet.worksheet(worksheet_name)
            data = worksheet.get_all_records()

            if not data:
                logger.warning(f"No data found in worksheet '{worksheet_name}'")
                return None

            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error fetching worksheet '{worksheet_name}': {e}")
            return None

    def load_worksheet_data(self, worksheet_name: str) -> Optional[pd.DataFrame]:
        """
        Load data from a specific worksheet.

        Args:
            worksheet_name: Name of the worksheet to load

        Returns:
            DataFrame with the worksheet data or None if an error occurs
        """
        if not self.client or worksheet_name not in self.worksheet_names:
            return None

        if worksheet_name in self.all_dataframes and self.all_dataframes[worksheet_name] is not None:
            return self.all_dataframes[worksheet_name]

        # Use the cached function to get data
        df = SheetsDataLoader._fetch_worksheet_data(self.sheet_url, worksheet_name)

        if df is not None:
            # Store in instance cache
            self.all_dataframes[worksheet_name] = df

        return df

    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def _fetch_all_worksheets(sheet_url: str, worksheet_names: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from all worksheets in the spreadsheet (cached).

        Args:
            sheet_url: URL of the spreadsheet
            worksheet_names: List of worksheet names to load

        Returns:
            Dictionary mapping worksheet names to DataFrames
        """
        all_dataframes = {}
        client = gspread.authorize(CREDENTIALS)
        spreadsheet = client.open_by_url(sheet_url)

        for worksheet_name in worksheet_names:
            try:
                worksheet = spreadsheet.worksheet(worksheet_name)
                data = worksheet.get_all_records()

                if data:
                    all_dataframes[worksheet_name] = pd.DataFrame(data)
                else:
                    logger.warning(f"No data found in worksheet '{worksheet_name}'")
                    all_dataframes[worksheet_name] = None
            except Exception as e:
                logger.error(f"Error fetching worksheet '{worksheet_name}': {e}")
                all_dataframes[worksheet_name] = None

        return all_dataframes

    def load_all_worksheets(self) -> Dict[str, pd.DataFrame]:
        """
        Load data from all worksheets in the spreadsheet.

        Returns:
            Dictionary mapping worksheet names to DataFrames
        """
        if not self.client:
            return {}

        # Return cached data if we already loaded everything and have all worksheets
        if len(self.all_dataframes) == len(self.worksheet_names) and all(
            name in self.all_dataframes for name in self.worksheet_names
        ):
            return {k: v for k, v in self.all_dataframes.items() if v is not None}

        all_dataframes = SheetsDataLoader._fetch_all_worksheets(
            self.sheet_url,
            self.worksheet_names,
        )

        self.all_dataframes.update(all_dataframes)

        return {k: v for k, v in self.all_dataframes.items() if v is not None}


def format_score(score: float) -> str:
    """Format a score with color based on its value."""
    if score >= 0.8:
        return f'<span class="good-score">{score: .4f}</span>'
    elif score >= 0.5:
        return f'<span class="medium-score">{score: .4f}</span>'
    else:
        return f'<span class="poor-score">{score: .4f}</span>'


def calculate_aggregate_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate aggregate metrics from a worksheet DataFrame.

    Args:
        df: DataFrame containing translation metrics

    Returns:
        Dictionary of aggregate metrics
    """
    metrics = {}

    # Get metric columns (skip the first 4 columns which are metadata)
    metric_columns = df.columns[4:]

    numeric_metric_columns = []
    for col in metric_columns:
        if col != "LLM Reasoning" and "reasoning" not in col.lower():
            try:
                # Check if column has numeric values
                df[col] = pd.to_numeric(df[col], errors="coerce")
                numeric_metric_columns.append(col)
            except Exception as e:
                print(e)
                pass

    if not numeric_metric_columns:
        return {}

    for col in numeric_metric_columns:
        metrics[col] = {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "min": df[col].min(),
            "max": df[col].max(),
            "std": df[col].std(),
        }

    if "Overall Score" in df.columns:
        metrics["overall"] = {
            "mean": df["Overall Score"].mean(),
            "median": df["Overall Score"].median(),
            "min": df["Overall Score"].min(),
            "max": df["Overall Score"].max(),
            "std": df["Overall Score"].std(),
        }

    return metrics


def display_worksheet_selection(
    worksheet_names: List[str],
) -> str:
    """
    Display a selection widget for worksheets.

    Args:
        worksheet_names: List of worksheet names

    Returns:
        Selected worksheet name
    """
    st.sidebar.header("Worksheet Selection")

    selected_worksheet = st.sidebar.selectbox(
        "Choose a worksheet to analyze: ", worksheet_names, index=0 if worksheet_names else None
    )

    return selected_worksheet


def display_spreadsheet_info(metadata: Dict):
    """Display information about the spreadsheet."""
    st.sidebar.header("Spreadsheet Info")

    if not metadata:
        st.sidebar.warning("No spreadsheet metadata available")
        return

    st.sidebar.markdown(f"**Name: ** {metadata.get('name', 'Unknown')}")
    st.sidebar.markdown(f"**Worksheets: ** {metadata.get('num_worksheets', 0)}")

    created = metadata.get("created", "Unknown")
    if created != "Unknown":
        created = pd.to_datetime(created).strftime("%Y-%m-%d")

    modified = metadata.get("modified", "Unknown")
    if modified != "Unknown":
        modified = pd.to_datetime(modified).strftime("%Y-%m-%d")

    st.sidebar.markdown(f"**Created: ** {created}")
    st.sidebar.markdown(f"**Last Modified: ** {modified}")


def display_aggregate_metrics(metrics: Dict, worksheet_name: str):
    """Display aggregate metrics for a worksheet."""
    if not metrics:
        st.info(f"No metrics available for worksheet '{worksheet_name}'.")
        return

    st.markdown(f"## Metrics Summary: {worksheet_name}")

    # Create metrics visualization
    col1, col2, col3 = st.columns(3)

    # Display overall score if available
    if "overall" in metrics:
        overall = metrics["overall"]
        col1.metric("Overall Score (Avg)", f"{overall['mean']: .4f}", f"{overall['std']: .4f} σ")
        col2.metric("Min Overall Score", f"{overall['min']: .4f}")
        col3.metric("Max Overall Score", f"{overall['max']: .4f}")

    # Extract individual metrics (excluding overall)
    individual_metrics = {k: v for k, v in metrics.items() if k != "overall"}

    if not individual_metrics:
        st.info("No individual metrics found for analysis.")
        return

    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame(
        {
            "Metric": individual_metrics.keys(),
            "Mean": [v["mean"] for v in individual_metrics.values()],
            "Median": [v["median"] for v in individual_metrics.values()],
            "Min": [v["min"] for v in individual_metrics.values()],
            "Max": [v["max"] for v in individual_metrics.values()],
            "StdDev": [v["std"] for v in individual_metrics.values()],
        }
    )

    # Create a bar chart of mean metrics values
    fig = px.bar(
        metrics_df,
        x="Metric",
        y="Mean",
        error_y="StdDev",
        title=f"Average Metrics for {worksheet_name}",
        color="Mean",
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
    )

    fig.update_layout(xaxis_title="Metric", yaxis_title="Score (0-1)", yaxis_range=[0, 1.1], showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    # Display metrics table with formatting
    st.markdown("### Detailed Metrics")

    # Format the metrics DataFrame for display
    formatted_df = metrics_df.copy()
    formatted_df["Mean"] = formatted_df["Mean"].map(lambda x: f"{x: .4f}")
    formatted_df["Median"] = formatted_df["Median"].map(lambda x: f"{x: .4f}")
    formatted_df["Min"] = formatted_df["Min"].map(lambda x: f"{x: .4f}")
    formatted_df["Max"] = formatted_df["Max"].map(lambda x: f"{x: .4f}")
    formatted_df["StdDev"] = formatted_df["StdDev"].map(lambda x: f"{x: .4f}")

    st.dataframe(formatted_df, use_container_width=True)


def display_translation_examples(df: pd.DataFrame, worksheet_name: str):
    """Display translation examples from a worksheet."""
    if df is None or df.empty:
        st.info(f"No translation examples available for worksheet '{worksheet_name}'")
        return

    # Check if the DataFrame has the expected columns
    required_columns = ["Original Text", "Expected Translation", "Actual Translation"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.warning(f"The worksheet '{worksheet_name}' is missing these required columns: {', '.join(missing_columns)}")
        st.write("Columns found: ", list(df.columns))
        return

    st.markdown(f"## Translation Examples: {worksheet_name}")

    num_samples = len(df)

    # Initialize session state for sample index if it doesn't exist
    if "sample_idx" not in st.session_state:
        st.session_state.sample_idx = 0

    # Ensure sample_idx is within bounds
    if st.session_state.sample_idx >= num_samples:
        st.session_state.sample_idx = 0
    elif st.session_state.sample_idx < 0:
        st.session_state.sample_idx = 0

    # Add navigation buttons
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col1:
        if st.button("⬅️ Previous", disabled=st.session_state.sample_idx <= 0):
            st.session_state.sample_idx = max(0, st.session_state.sample_idx - 1)

    with col2:
        if st.button("Next ➡️", disabled=st.session_state.sample_idx >= num_samples - 1):
            st.session_state.sample_idx = min(num_samples - 1, st.session_state.sample_idx + 1)

    with col3:
        st.markdown(f"**Example {st.session_state.sample_idx + 1} of {num_samples}**")

    with col4:
        if st.button("First"):
            st.session_state.sample_idx = 0

    with col5:
        if st.button("Last"):
            st.session_state.sample_idx = num_samples - 1

    # Optional: Add a slider that syncs with session state
    slider_value = st.slider(
        "Or jump to example: ",
        min_value=1,
        max_value=num_samples if num_samples > 0 else 1,
        value=st.session_state.sample_idx + 1,
        key="example_slider",
    )

    # Update session state when slider changes
    if slider_value - 1 != st.session_state.sample_idx:
        st.session_state.sample_idx = slider_value - 1

    if num_samples == 0:
        st.info("No translation examples found")
        return

    # Get the selected sample using session state
    sample = df.iloc[st.session_state.sample_idx]

    # Display the sample in a nice format
    st.markdown("### Translation Example")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Source Text (English)")
        st.markdown(f"<div class='metric-card'>{sample['Original Text']}</div>", unsafe_allow_html=True)

        st.markdown("#### Expected Translation (Polish)")
        st.markdown(f"<div class='metric-card'>{sample['Expected Translation']}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("#### Actual Translation (Polish)")
        st.markdown(f"<div class='metric-card'>{sample['Actual Translation']}</div>", unsafe_allow_html=True)

        # Display overall score if available
        if "Overall Score" in df.columns:
            st.markdown("#### Overall Score")
            overall_score = sample.get("Overall Score", 0)
            try:
                overall_score = float(overall_score)
                st.markdown(f"<div class='metric-card'>{format_score(overall_score)}</div>", unsafe_allow_html=True)
            except (ValueError, TypeError):
                st.markdown(f"<div class='metric-card'>{overall_score}</div>", unsafe_allow_html=True)

    # Display individual metrics
    st.markdown("#### Metrics")

    # Get metric columns (skip the first 4 columns which are metadata)
    if len(df.columns) > 4:
        metric_columns = [col for col in df.columns[4:] if col != "LLM Reasoning" and "reasoning" not in col.lower()]

        if metric_columns:
            # Create metrics grid with appropriate number of columns
            cols_per_row = 4

            for row in range(0, len(metric_columns), cols_per_row):
                metric_cols = st.columns(min(cols_per_row, len(metric_columns) - row))

                for i, col_idx in enumerate(range(row, min(row + cols_per_row, len(metric_columns)))):
                    metric = metric_columns[col_idx]
                    with metric_cols[i]:
                        try:
                            score = float(sample[metric])
                            st.markdown(
                                f"<div class='metric-card'><b>{metric}: </b> {format_score(score)}</div>",
                                unsafe_allow_html=True,
                            )
                        except (ValueError, TypeError) as e:
                            print(e)
                            st.markdown(
                                f"<div class='metric-card'><b>{metric}: </b> {sample[metric]}</div>",
                                unsafe_allow_html=True,
                            )
        else:
            st.info("No numeric metrics found in this worksheet")

    # Display LLM reasoning if available
    if "LLM Reasoning" in df.columns and not pd.isna(sample["LLM Reasoning"]) and sample["LLM Reasoning"]:
        st.markdown("#### LLM Reasoning")
        st.markdown(f"<div class='metric-card'>{sample['LLM Reasoning']}</div>", unsafe_allow_html=True)


def display_worksheet_comparison(all_data: Dict[str, pd.DataFrame]):
    """Display comparison of metrics across worksheets."""
    if not all_data:
        st.warning("No worksheet data available for comparison")
        return

    st.markdown("## Worksheet Comparison")

    worksheet_metrics = {}
    for name, df in all_data.items():
        metrics = calculate_aggregate_metrics(df)
        if metrics:
            worksheet_metrics[name] = metrics

    if not worksheet_metrics:
        st.warning("No metrics available for comparison")
        return

    # Create a DataFrame for comparison
    comparison_data = []

    for worksheet, metrics in worksheet_metrics.items():
        row = {"Worksheet": worksheet}

        # Add overall score if available
        if "overall" in metrics:
            row["Overall"] = metrics["overall"]["mean"]

        # Add individual metrics
        for metric_name, metric_values in metrics.items():
            if metric_name != "overall":
                row[metric_name] = metric_values["mean"]

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Get all metric columns (excluding Worksheet)
    metric_columns = [col for col in comparison_df.columns if col != "Worksheet"]

    if not metric_columns:
        st.warning("No common metrics found for comparison")
        return

    # Create a heatmap of metrics across worksheets
    metric_to_display = st.selectbox(
        "Select a metric to compare across worksheets: ",
        ["Overall"] + [col for col in metric_columns if col != "Overall"],
        index=0 if "Overall" in metric_columns else 0,
    )

    if metric_to_display not in comparison_df.columns:
        st.warning(f"Metric '{metric_to_display}' not available for comparison")
        return

    # Sort by the selected metric
    comparison_df = comparison_df.sort_values(by=metric_to_display, ascending=False)

    # Create comparison chart
    fig = px.bar(
        comparison_df,
        x="Worksheet",
        y=metric_to_display,
        title=f"Comparison of {metric_to_display} Across Worksheets",
        color=metric_to_display,
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
    )

    fig.update_layout(
        xaxis_title="Worksheet", yaxis_title=f"{metric_to_display} Score (0-1)", yaxis_range=[0, 1.1], showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Create a heatmap for all metrics
    st.markdown("### All Metrics Comparison")

    # Melt the DataFrame for heatmap
    heatmap_df = comparison_df.melt(
        id_vars=["Worksheet"], value_vars=[col for col in metric_columns], var_name="Metric", value_name="Score"
    )

    fig = px.density_heatmap(
        heatmap_df,
        x="Metric",
        y="Worksheet",
        z="Score",
        title="Metrics Heatmap Across Worksheets",
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
    )

    fig.update_layout(xaxis_title="Metric", yaxis_title="Worksheet", showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    # Display the comparison table
    st.markdown("### Metrics Comparison Table")

    # Format the comparison DataFrame for display
    formatted_df = comparison_df.copy()
    for col in metric_columns:
        formatted_df[col] = formatted_df[col].map(lambda x: f"{x: .4f}" if pd.notna(x) else "N/A")

    st.dataframe(formatted_df, use_container_width=True)


def display_translation_length_analysis(df: pd.DataFrame, worksheet_name: str):
    """Analyze and display correlation between translation length and scores."""
    if df is None or df.empty:
        return

    st.markdown(f"## Translation Length Analysis: {worksheet_name}")

    # Calculate lengths
    df["Original Length"] = df["Original Text"].str.len()
    df["Expected Length"] = df["Expected Translation"].str.len()
    df["Actual Length"] = df["Actual Translation"].str.len()
    df["Length Difference"] = df["Actual Length"] - df["Expected Length"]
    df["Length Ratio"] = df["Actual Length"] / df["Expected Length"]

    # Get metric columns
    metric_columns = [col for col in df.columns[4:] if col != "LLM Reasoning" and "reasoning" not in col.lower()]

    # Create correlation analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Length Distribution")
        fig = go.Figure()

        fig.add_trace(go.Histogram(x=df["Original Length"], name="Original", marker_color="blue", opacity=0.6))

        fig.add_trace(go.Histogram(x=df["Expected Length"], name="Expected", marker_color="green", opacity=0.6))

        fig.add_trace(go.Histogram(x=df["Actual Length"], name="Actual", marker_color="red", opacity=0.6))

        fig.update_layout(
            title="Distribution of Text Lengths",
            xaxis_title="Text Length (characters)",
            yaxis_title="Count",
            barmode="overlay",
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Length Difference vs. Scores")

        # Choose metric for correlation
        selected_metric = st.selectbox(
            "Select metric to correlate with length: ",
            ["Overall Score"] + [col for col in metric_columns if col != "Overall Score"],
            index=0,
        )

        if selected_metric in df.columns:
            fig = px.scatter(
                df,
                x="Length Difference",
                y=selected_metric,
                color="Length Ratio",
                color_continuous_scale="RdYlGn",
                range_color=[0.5, 1.5],
                hover_data=["Original Text", "Actual Translation"],
                title=f"{selected_metric} vs. Length Difference",
            )

            fig.update_layout(xaxis_title="Length Difference (Actual - Expected)", yaxis_title=selected_metric)

            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main function to run the Streamlit dashboard."""
    st.title("Translation Evaluation Dashboard")
    st.markdown(
        """
    This dashboard visualizes translation metrics from Google Sheets.
    Select a worksheet from the sidebar to analyze its translation examples and metrics.
    """
    )

    data_loader = SheetsDataLoader()

    # Check if we have worksheets
    worksheet_names = data_loader.get_worksheet_names()
    if not worksheet_names:
        st.error("No worksheets found. Please check your Google Sheets URL and credentials.")
        return

    # Get spreadsheet metadata and display in sidebar
    metadata = data_loader.get_spreadsheet_metadata()
    display_spreadsheet_info(metadata)

    # Add sidebar navigation
    st.sidebar.header("Navigation")
    view_mode = st.sidebar.radio("View Mode: ", ["Single Worksheet Analysis", "Cross-Worksheet Comparison"])

    if view_mode == "Single Worksheet Analysis":
        selected_worksheet = display_worksheet_selection(worksheet_names)

        if selected_worksheet:
            # Load the selected worksheet data
            df = data_loader.load_worksheet_data(selected_worksheet)

            if df is None or df.empty:
                st.error(f"No data found in worksheet '{selected_worksheet}'")
                return

            # Calculate aggregate metrics
            metrics = calculate_aggregate_metrics(df)

            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Metrics Summary", "Translation Examples", "Length Analysis"])

            with tab1:
                display_aggregate_metrics(metrics, selected_worksheet)

            with tab2:
                display_translation_examples(df, selected_worksheet)

            with tab3:
                display_translation_length_analysis(df, selected_worksheet)

    else:
        # Cross-Worksheet Comparison
        all_data = data_loader.load_all_worksheets()

        if not all_data:
            st.error("No data found in any worksheets")
            return

        # Display cross-worksheet comparison
        display_worksheet_comparison(all_data)


if __name__ == "__main__":
    main()

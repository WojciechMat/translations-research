"""
Translation Analysis Plot Generator

Generates matplotlib plots analyzing translation metrics from Google Sheets data.
Creates organized folders with different types of analysis plots.
"""
import os
import logging
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

import gspread
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from google.oauth2.service_account import Credentials

# Load environment variables
load_dotenv()

# Configure matplotlib
plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Sheets credentials
creds_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "google_credentials.json")
CREDENTIALS = Credentials.from_service_account_file(
    creds_file,
    scopes=[
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ],
)


class TranslationAnalysisPlotter:
    """Class to generate analysis plots from translation evaluation data."""

    def __init__(self):
        """Initialize the plotter with Google Sheets connection."""
        self.sheet_url = os.getenv("SHEET")
        self.client = None
        self.all_dataframes = {}
        self.worksheet_names = []
        self.IGNORE_WORKSHEETS = ["entrypoint"]

        if not self.sheet_url:
            raise ValueError("No Google Sheet URL provided. Set the SHEET environment variable.")

        try:
            self.client = gspread.authorize(CREDENTIALS)
            self.spreadsheet = self.client.open_by_url(self.sheet_url)

            # Get all worksheet names (filtering out ignored ones)
            all_ws_names = [ws.title for ws in self.spreadsheet.worksheets()]
            self.worksheet_names = [
                name for name in all_ws_names if name.lower() not in [n.lower() for n in self.IGNORE_WORKSHEETS]
            ]

            logger.info(f"Connected to Google Sheets. Found {len(self.worksheet_names)} worksheets.")

        except Exception as e:
            logger.error(f"Error connecting to Google Sheets: {e}")
            raise

    def load_all_worksheets(self) -> Dict[str, pd.DataFrame]:
        """Load data from all worksheets."""
        logger.info("Loading all worksheet data...")

        for worksheet_name in self.worksheet_names:
            try:
                worksheet = self.spreadsheet.worksheet(worksheet_name)
                data = worksheet.get_all_records()

                if data:
                    df = pd.DataFrame(data)
                    # Convert numeric columns
                    for col in df.columns[4:]:  # Skip first 4 metadata columns
                        if col != "LLM Reasoning" and "reasoning" not in col.lower():
                            try:
                                df[col] = pd.to_numeric(df[col], errors="coerce")
                            except:
                                pass

                    self.all_dataframes[worksheet_name] = df
                    logger.info(f"Loaded {len(df)} rows from '{worksheet_name}'")
                else:
                    logger.warning(f"No data found in worksheet '{worksheet_name}'")

            except Exception as e:
                logger.error(f"Error loading worksheet '{worksheet_name}': {e}")

        return self.all_dataframes

    def get_metric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get numeric metric columns from DataFrame."""
        metric_columns = []
        for col in df.columns[4:]:  # Skip first 4 metadata columns
            if col != "LLM Reasoning" and "reasoning" not in col.lower():
                if df[col].dtype in ["float64", "int64"] or pd.api.types.is_numeric_dtype(df[col]):
                    metric_columns.append(col)
        return metric_columns

    def create_output_directories(self):
        """Create output directory structure."""
        base_dir = "tmp/results"
        directories = [
            f"{base_dir}/method_comparison",
            f"{base_dir}/correlation_analysis",
            f"{base_dir}/notable_results",
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    def plot_method_comparison(self):
        """Generate plots comparing scores across all methods (worksheets)."""
        logger.info("Generating method comparison plots...")

        if not self.all_dataframes:
            logger.warning("No data available for method comparison")
            return

        # Collect aggregate metrics for each method
        method_metrics = {}
        for method_name, df in self.all_dataframes.items():
            if df is None or df.empty:
                continue

            metric_columns = self.get_metric_columns(df)
            if not metric_columns:
                continue

            method_metrics[method_name] = {}
            for col in metric_columns:
                method_metrics[method_name][col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "median": df[col].median(),
                }

        if not method_metrics:
            logger.warning("No numeric metrics found for comparison")
            return

        # Get all unique metrics across methods
        all_metrics = set()
        for method in method_metrics.values():
            all_metrics.update(method.keys())
        all_metrics = sorted(list(all_metrics))

        # Create comparison plots
        for metric in all_metrics:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            methods = []
            means = []
            stds = []
            medians = []

            for method_name, metrics in method_metrics.items():
                if metric in metrics:
                    methods.append(method_name)
                    means.append(metrics[metric]["mean"])
                    stds.append(metrics[metric]["std"])
                    medians.append(metrics[metric]["median"])

            if not methods:
                continue

            # Bar plot with error bars (mean ± std)
            bars = ax1.bar(methods, means, yerr=stds, capsize=5, color=plt.cm.Set3(np.linspace(0, 1, len(methods))))
            ax1.set_title(f"{metric} - Mean Scores Across Methods")
            ax1.set_ylabel("Score")
            ax1.set_ylim(0, 1.1)
            ax1.tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, mean_val in zip(bars, means):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{mean_val:.3f}",
                    ha="center",
                    va="bottom",
                )

            # Box plot showing distribution
            data_for_boxplot = []
            labels_for_boxplot = []
            for method_name, df in self.all_dataframes.items():
                if df is not None and not df.empty and metric in df.columns:
                    metric_data = df[metric].dropna()
                    if len(metric_data) > 0:
                        data_for_boxplot.append(metric_data)
                        labels_for_boxplot.append(method_name)

            if data_for_boxplot:
                bp = ax2.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True)
                colors = plt.cm.Set3(np.linspace(0, 1, len(bp["boxes"])))
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)

                ax2.set_title(f"{metric} - Score Distributions")
                ax2.set_ylabel("Score")
                ax2.set_ylim(0, 1.1)
                ax2.tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.savefig(
                f'tmp/results/method_comparison/{metric.replace(" ", "_").lower()}_comparison.png',
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # Create overall comparison heatmap
        self._create_comparison_heatmap(method_metrics, all_metrics)

        logger.info("Method comparison plots generated successfully")

    def _create_comparison_heatmap(self, method_metrics: Dict, all_metrics: List[str]):
        """Create a heatmap comparing all methods and metrics."""
        # Create matrix for heatmap
        methods = list(method_metrics.keys())
        matrix_data = []

        for method in methods:
            row = []
            for metric in all_metrics:
                if metric in method_metrics[method]:
                    row.append(method_metrics[method][metric]["mean"])
                else:
                    row.append(np.nan)
            matrix_data.append(row)

        matrix_df = pd.DataFrame(matrix_data, index=methods, columns=all_metrics)

        plt.figure(figsize=(14, 8))
        mask = matrix_df.isnull()
        sns.heatmap(
            matrix_df, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, mask=mask, cbar_kws={"label": "Score"}
        )
        plt.title("Methods vs Metrics Comparison Heatmap")
        plt.xlabel("Metrics")
        plt.ylabel("Methods")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("tmp/results/method_comparison/overall_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_correlation_analysis(self):
        """Generate correlation plots between LLM scores and other metrics."""
        logger.info("Generating correlation analysis plots...")

        # First, create individual worksheet correlation plots
        for method_name, df in self.all_dataframes.items():
            if df is None or df.empty:
                continue

            metric_columns = self.get_metric_columns(df)
            if len(metric_columns) < 2:
                continue

            # Find LLM-related columns
            llm_columns = [col for col in metric_columns if "llm" in col.lower() or "gpt" in col.lower()]
            other_columns = [col for col in metric_columns if col not in llm_columns]

            if not llm_columns or not other_columns:
                continue

            # Create correlation matrix
            corr_matrix = df[metric_columns].corr()

            # Plot correlation heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".3f",
                cmap="RdBu_r",
                center=0,
                mask=mask,
                square=True,
                cbar_kws={"label": "Correlation"},
            )
            plt.title(f"Metric Correlations - {method_name}")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(
                f'tmp/results/correlation_analysis/{method_name.replace(" ", "_").lower()}_correlation_heatmap.png',
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # Create scatter plots for LLM vs other metrics
            for llm_col in llm_columns:
                for other_col in other_columns:
                    if llm_col != other_col:
                        self._create_correlation_scatter(df, llm_col, other_col, method_name)

        # Now create concatenated analysis with all data combined
        self._plot_concatenated_correlation_analysis()

        logger.info("Correlation analysis plots generated successfully")

    def _create_correlation_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, method_name: str):
        """Create violin plot showing correlation between two metrics."""
        # Remove NaN values
        clean_df = df[[x_col, y_col]].dropna()
        if len(clean_df) < 3:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left plot: Traditional scatter with trend line for reference
        ax1.scatter(clean_df[x_col], clean_df[y_col], alpha=0.6, s=50)
        z = np.polyfit(clean_df[x_col], clean_df[y_col], 1)
        p = np.poly1d(z)
        ax1.plot(clean_df[x_col].sort_values(), p(clean_df[x_col].sort_values()), "r--", alpha=0.8, linewidth=2)
        correlation = clean_df[x_col].corr(clean_df[y_col])
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col)
        ax1.set_title(f"Scatter: {x_col} vs {y_col}")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1.05)
        ax1.set_ylim(0, 1.05)

        # Right plot: Violin plots showing distributions
        # Create bins for x_col to show violin plots across different x ranges
        x_bins = pd.cut(clean_df[x_col], bins=5, include_lowest=True)
        violin_data = []
        violin_labels = []

        for bin_label in x_bins.cat.categories:
            bin_mask = x_bins == bin_label
            y_values = clean_df.loc[bin_mask, y_col]
            if len(y_values) > 2:  # Need at least 3 points for violin
                violin_data.append(y_values)
                bin_center = (bin_label.left + bin_label.right) / 2
                violin_labels.append(f"{bin_center:.2f}")

        if violin_data:
            parts = ax2.violinplot(
                violin_data, positions=range(len(violin_data)), widths=0.6, showmeans=True, showmedians=True
            )

            # Customize violin plot colors
            for pc in parts["bodies"]:
                pc.set_facecolor("lightblue")
                pc.set_alpha(0.7)

            ax2.set_xlabel(f"{x_col} (binned)")
            ax2.set_ylabel(y_col)
            ax2.set_title(f"Distribution: {y_col} across {x_col} ranges")
            ax2.set_xticks(range(len(violin_labels)))
            ax2.set_xticklabels(violin_labels, rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.05)

        # Add correlation text box to main plot
        textstr = f"r = {correlation:.3f}\nn = {len(clean_df)}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment="top", bbox=props)

        plt.suptitle(f"{method_name}: {x_col} vs {y_col}\nCorrelation: {correlation:.3f}", fontsize=14)
        plt.tight_layout()
        filename = f"{method_name}_{x_col}_vs_{y_col}".replace(" ", "_").replace("/", "_").lower()
        plt.savefig(f"tmp/results/correlation_analysis/{filename}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_concatenated_correlation_analysis(self):
        """Generate correlation plots using all data concatenated from all worksheets."""
        logger.info("Generating concatenated correlation analysis...")

        # Concatenate all dataframes, adding method column for identification
        all_data_list = []
        for method_name, df in self.all_dataframes.items():
            if df is None or df.empty:
                continue

            df_copy = df.copy()
            df_copy["Method"] = method_name
            all_data_list.append(df_copy)

        if not all_data_list:
            logger.warning("No data available for concatenated correlation analysis")
            return

        # Concatenate all data
        combined_df = pd.concat(all_data_list, ignore_index=True)
        logger.info(f"Combined dataset has {len(combined_df)} total translations")

        # Get metric columns from combined data
        metric_columns = self.get_metric_columns(combined_df)
        if len(metric_columns) < 2:
            logger.warning("Not enough metrics for concatenated correlation analysis")
            return

        # Find LLM-related columns
        llm_columns = [col for col in metric_columns if "llm" in col.lower() or "gpt" in col.lower()]
        other_columns = [col for col in metric_columns if col not in llm_columns]

        # Create overall correlation matrix heatmap
        corr_matrix = combined_df[metric_columns].corr()

        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            center=0,
            mask=mask,
            square=True,
            cbar_kws={"label": "Correlation"},
        )
        plt.title(f"Metric Correlations - All Methods Combined (n={len(combined_df)})")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            "tmp/results/correlation_analysis/all_methods_combined_correlation_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create scatter plots for all metric pairs with method colors
        if llm_columns and other_columns:
            for llm_col in llm_columns:
                for other_col in other_columns:
                    if llm_col != other_col:
                        self._create_concatenated_correlation_scatter(combined_df, llm_col, other_col)

        # Create comprehensive correlation scatter matrix
        self._create_correlation_scatter_matrix(combined_df, metric_columns)

        # Create method-wise correlation comparison
        self._create_method_correlation_comparison(combined_df, metric_columns)

    def _create_concatenated_correlation_scatter(self, df: pd.DataFrame, x_col: str, y_col: str):
        """Create violin plot with all data colored by method."""
        # Remove NaN values
        clean_df = df[[x_col, y_col, "Method"]].dropna()
        if len(clean_df) < 3:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Top left: Traditional scatter plot with method colors for reference
        methods = clean_df["Method"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

        for method, color in zip(methods, colors):
            method_data = clean_df[clean_df["Method"] == method]
            ax1.scatter(method_data[x_col], method_data[y_col], label=method, alpha=0.6, s=30, color=color)

        # Add overall trend line
        z = np.polyfit(clean_df[x_col], clean_df[y_col], 1)
        p = np.poly1d(z)
        ax1.plot(
            clean_df[x_col].sort_values(),
            p(clean_df[x_col].sort_values()),
            "r--",
            alpha=0.8,
            linewidth=2,
            label="Overall Trend",
        )

        correlation = clean_df[x_col].corr(clean_df[y_col])
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col)
        ax1.set_title(f"Scatter Plot by Method")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1.05)
        ax1.set_ylim(0, 1.05)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Top right: Violin plots across x-axis bins for all data
        x_bins = pd.cut(clean_df[x_col], bins=6, include_lowest=True)
        violin_data = []
        violin_labels = []

        for bin_label in x_bins.cat.categories:
            bin_mask = x_bins == bin_label
            y_values = clean_df.loc[bin_mask, y_col]
            if len(y_values) > 2:
                violin_data.append(y_values)
                bin_center = (bin_label.left + bin_label.right) / 2
                violin_labels.append(f"{bin_center:.2f}")

        if violin_data:
            parts = ax2.violinplot(
                violin_data, positions=range(len(violin_data)), widths=0.7, showmeans=True, showmedians=True
            )

            for pc in parts["bodies"]:
                pc.set_facecolor("lightcoral")
                pc.set_alpha(0.7)

            ax2.set_xlabel(f"{x_col} (binned ranges)")
            ax2.set_ylabel(y_col)
            ax2.set_title(f"{y_col} Distribution Across {x_col} Ranges")
            ax2.set_xticks(range(len(violin_labels)))
            ax2.set_xticklabels(violin_labels, rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.05)

        # Bottom left: Method comparison violin plots
        method_violin_data = []
        method_labels = []
        for method in methods:
            method_data = clean_df[clean_df["Method"] == method]
            if len(method_data) > 2:
                method_violin_data.append(method_data[y_col])
                method_labels.append(method[:15] + "..." if len(method) > 15 else method)

        if method_violin_data:
            parts = ax3.violinplot(
                method_violin_data,
                positions=range(len(method_violin_data)),
                widths=0.6,
                showmeans=True,
                showmedians=True,
            )

            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)

            ax3.set_xlabel("Method")
            ax3.set_ylabel(y_col)
            ax3.set_title(f"{y_col} Distribution by Method")
            ax3.set_xticks(range(len(method_labels)))
            ax3.set_xticklabels(method_labels, rotation=45, ha="right")
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1.05)

        # Bottom right: X metric distribution by method
        method_x_violin_data = []
        if method_violin_data:  # Reuse the same method filtering
            method_x_violin_data = []
            for method in methods:
                method_data = clean_df[clean_df["Method"] == method]
                if len(method_data) > 2:
                    method_x_violin_data.append(method_data[x_col])

            parts = ax4.violinplot(
                method_x_violin_data,
                positions=range(len(method_x_violin_data)),
                widths=0.6,
                showmeans=True,
                showmedians=True,
            )

            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)

            ax4.set_xlabel("Method")
            ax4.set_ylabel(x_col)
            ax4.set_title(f"{x_col} Distribution by Method")
            ax4.set_xticks(range(len(method_labels)))
            ax4.set_xticklabels(method_labels, rotation=45, ha="right")
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1.05)

        # Add correlation info
        textstr = f"Overall Correlation: r = {correlation:.3f}\nTotal samples: n = {len(clean_df)}"
        fig.suptitle(f"All Methods: {x_col} vs {y_col}\n{textstr}", fontsize=14, y=0.98)

        plt.tight_layout()
        filename = f"all_methods_{x_col}_vs_{y_col}".replace(" ", "_").replace("/", "_").lower()
        plt.savefig(f"tmp/results/correlation_analysis/{filename}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _create_correlation_scatter_matrix(self, df: pd.DataFrame, metric_columns: List[str]):
        """Create a matrix showing correlations with violin plots."""
        if len(metric_columns) > 6:  # Limit to avoid overcrowded plots
            # Select most important metrics
            important_metrics = ["Overall Score"] if "Overall Score" in metric_columns else []
            llm_metrics = [col for col in metric_columns if "llm" in col.lower() or "gpt" in col.lower()]
            other_metrics = [col for col in metric_columns if col not in llm_metrics and col not in important_metrics]

            selected_metrics = important_metrics + llm_metrics[:2] + other_metrics[:3]
            metric_columns = selected_metrics[:6]

        n_metrics = len(metric_columns)
        if n_metrics < 2:
            return

        fig, axes = plt.subplots(n_metrics, n_metrics, figsize=(18, 18))

        for i, col1 in enumerate(metric_columns):
            for j, col2 in enumerate(metric_columns):
                ax = axes[i, j] if n_metrics > 1 else axes

                if i == j:
                    # Diagonal: violin plot by method
                    if "Method" in df.columns:
                        methods = df["Method"].unique()
                        violin_data = []
                        method_labels = []

                        for method in methods:
                            method_data = df[df["Method"] == method][col1].dropna()
                            if len(method_data) > 2:
                                violin_data.append(method_data)
                                method_labels.append(method[:8] + "..." if len(method) > 8 else method)

                        if violin_data:
                            parts = ax.violinplot(violin_data, positions=range(len(violin_data)), widths=0.6)
                            for k, pc in enumerate(parts["bodies"]):
                                pc.set_facecolor(plt.cm.tab10(k / len(violin_data)))
                                pc.set_alpha(0.7)

                            ax.set_xticks(range(len(method_labels)))
                            ax.set_xticklabels(method_labels, rotation=45, ha="right", fontsize=8)

                    ax.set_title(f"{col1}", fontsize=10)
                    ax.set_ylabel("Score", fontsize=8)

                elif i > j:
                    # Lower triangle: violin plot showing y distribution across x bins
                    clean_df = df[[col1, col2]].dropna()
                    if len(clean_df) > 10:
                        try:
                            x_bins = pd.cut(clean_df[col2], bins=4, include_lowest=True)
                            violin_data = []

                            for bin_label in x_bins.cat.categories:
                                bin_mask = x_bins == bin_label
                                y_values = clean_df.loc[bin_mask, col1]
                                if len(y_values) > 2:
                                    violin_data.append(y_values)

                            if violin_data and len(violin_data) > 1:
                                parts = ax.violinplot(violin_data, positions=range(len(violin_data)), widths=0.6)
                                for pc in parts["bodies"]:
                                    pc.set_facecolor("lightblue")
                                    pc.set_alpha(0.6)

                                correlation = clean_df[col1].corr(clean_df[col2])
                                ax.text(
                                    0.05,
                                    0.95,
                                    f"r={correlation:.2f}",
                                    transform=ax.transAxes,
                                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                                    fontsize=8,
                                )
                                ax.set_xticks(range(len(violin_data)))
                                ax.set_xticklabels([f"Q{i+1}" for i in range(len(violin_data))], fontsize=8)
                            else:
                                # Fallback to scatter if violin doesn't work
                                ax.scatter(clean_df[col2], clean_df[col1], alpha=0.5, s=10)
                                correlation = clean_df[col1].corr(clean_df[col2])
                                ax.text(
                                    0.05,
                                    0.95,
                                    f"r={correlation:.2f}",
                                    transform=ax.transAxes,
                                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                                    fontsize=8,
                                )
                        except:
                            # Fallback to scatter if binning fails
                            ax.scatter(clean_df[col2], clean_df[col1], alpha=0.5, s=10)

                else:
                    # Upper triangle: correlation coefficient as text
                    clean_df = df[[col1, col2]].dropna()
                    if len(clean_df) > 0:
                        correlation = clean_df[col1].corr(clean_df[col2])

                        # Color based on correlation strength
                        color = (
                            "darkgreen" if abs(correlation) > 0.7 else "orange" if abs(correlation) > 0.4 else "gray"
                        )

                        ax.text(
                            0.5,
                            0.5,
                            f"{correlation:.3f}",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=12,
                            weight="bold",
                            color=color,
                        )
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Set labels only on edges
                if i == n_metrics - 1 and i > j:
                    ax.set_xlabel(col2, rotation=45, ha="right", fontsize=9)
                if j == 0 and i > 0:
                    ax.set_ylabel(col1, rotation=90, ha="right", fontsize=9)

        plt.suptitle("Correlation Matrix with Violin Plots - All Methods Combined", fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig("tmp/results/correlation_analysis/correlation_violin_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _create_method_correlation_comparison(self, df: pd.DataFrame, metric_columns: List[str]):
        """Create comparison of correlations across different methods."""
        methods = df["Method"].unique()

        # Calculate correlations for each method
        method_correlations = {}
        for method in methods:
            method_df = df[df["Method"] == method]
            if len(method_df) > 3:  # Need at least a few samples
                corr_matrix = method_df[metric_columns].corr()
                method_correlations[method] = corr_matrix

        if len(method_correlations) < 2:
            return

        # Create comparison visualization
        fig, axes = plt.subplots(2, len(method_correlations), figsize=(6 * len(method_correlations), 12))
        if len(method_correlations) == 1:
            axes = axes.reshape(-1, 1)

        # Plot individual correlation matrices
        for idx, (method, corr_matrix) in enumerate(method_correlations.items()):
            ax = axes[0, idx]
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                mask=mask,
                square=True,
                ax=ax,
                cbar=idx == 0,
            )
            ax.set_title(f"{method}")
            if idx == 0:
                ax.set_ylabel("Individual Method Correlations")

        # Plot correlation differences from overall
        overall_corr = df[metric_columns].corr()
        for idx, (method, corr_matrix) in enumerate(method_correlations.items()):
            ax = axes[1, idx]
            diff_matrix = corr_matrix - overall_corr
            mask = np.triu(np.ones_like(diff_matrix, dtype=bool))
            sns.heatmap(
                diff_matrix,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                mask=mask,
                square=True,
                ax=ax,
                cbar=idx == 0,
                vmin=-0.5,
                vmax=0.5,
            )
            ax.set_title(f"{method} - Overall")
            if idx == 0:
                ax.set_ylabel("Difference from Overall")

        plt.suptitle("Method-wise Correlation Comparison", fontsize=16)
        plt.tight_layout()
        plt.savefig("tmp/results/correlation_analysis/method_correlation_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def find_notable_results(self):
        """Find and display best and worst translation examples across all methods."""
        logger.info("Finding notable translation results...")

        # First, create a mapping of original text to all method translations
        text_to_translations = {}

        for method_name, df in self.all_dataframes.items():
            if df is None or df.empty:
                continue

            required_cols = ["Original Text", "Expected Translation", "Actual Translation"]
            if not all(col in df.columns for col in required_cols):
                continue

            metric_columns = self.get_metric_columns(df)
            if not metric_columns:
                continue

            # Calculate overall score if not present
            if "Overall Score" in df.columns:
                overall_col = "Overall Score"
            else:
                # Use mean of all metrics as overall score
                df["Calculated_Overall"] = df[metric_columns].mean(axis=1)
                overall_col = "Calculated_Overall"

            for idx, row in df.iterrows():
                if pd.notna(row[overall_col]):
                    original_text = str(row["Original Text"]).strip()

                    if original_text not in text_to_translations:
                        text_to_translations[original_text] = {
                            "original": original_text,
                            "expected": row["Expected Translation"],
                            "methods": {},
                        }

                    text_to_translations[original_text]["methods"][method_name] = {
                        "actual": row["Actual Translation"],
                        "overall_score": row[overall_col],
                        "row_data": row,
                        "method_index": idx,
                    }

        if not text_to_translations:
            logger.warning("No translation data found for notable results")
            return

        # Find texts that have translations from multiple methods
        multi_method_texts = {text: data for text, data in text_to_translations.items() if len(data["methods"]) > 1}

        if not multi_method_texts:
            logger.warning("No texts found with multiple method translations")
            # Fall back to single method examples
            multi_method_texts = text_to_translations

        # Create scored examples for ranking
        scored_examples = []
        for original_text, data in multi_method_texts.items():
            # Find best and worst method for this text
            method_scores = [(method, info["overall_score"]) for method, info in data["methods"].items()]
            method_scores.sort(key=lambda x: x[1])

            if method_scores:
                worst_method, worst_score = method_scores[0]
                best_method, best_score = method_scores[-1]
                avg_score = sum(score for _, score in method_scores) / len(method_scores)

                scored_examples.append(
                    {
                        "original_text": original_text,
                        "expected": data["expected"],
                        "methods": data["methods"],
                        "best_method": best_method,
                        "best_score": best_score,
                        "worst_method": worst_method,
                        "worst_score": worst_score,
                        "avg_score": avg_score,
                        "score_range": best_score - worst_score,
                    }
                )

        if not scored_examples:
            logger.warning("No scored examples found for notable results")
            return

        # Sort by different criteria to get diverse examples
        examples_by_best = sorted(scored_examples, key=lambda x: x["best_score"], reverse=True)
        examples_by_worst = sorted(scored_examples, key=lambda x: x["worst_score"])
        examples_by_range = sorted(scored_examples, key=lambda x: x["score_range"], reverse=True)

        # Get best examples (highest best scores)
        best_examples = examples_by_best[:5]

        # Get worst examples (lowest worst scores)
        worst_examples = examples_by_worst[:5]

        # Get most controversial examples (highest score range between methods)
        controversial_examples = examples_by_range[:5]

        # Create notable results report
        self._create_notable_results_report(best_examples, worst_examples, controversial_examples)

        logger.info("Notable results analysis completed")

    def _create_notable_results_report(
        self, best_examples: List[Dict], worst_examples: List[Dict], controversial_examples: List[Dict]
    ):
        """Create a comprehensive report of notable translation results."""

        # Create PDF report
        with PdfPages("tmp/results/notable_results/notable_translations_report.pdf") as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.7, "Notable Translation Results", ha="center", va="center", fontsize=24, weight="bold")
            ax.text(0.5, 0.6, f"Cross-Method Translation Comparison", ha="center", va="center", fontsize=16)
            ax.text(
                0.5,
                0.5,
                f'Analysis Date: {pd.Timestamp.now().strftime("%Y-%m-%d")}',
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.text(
                0.5,
                0.4,
                f"Best Examples: {len(best_examples)} | Worst Examples: {len(worst_examples)} | Most Controversial: {len(controversial_examples)}",
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

            # Best examples
            self._create_cross_method_examples_pages(pdf, best_examples, "Best Translation Examples")

            # Worst examples
            self._create_cross_method_examples_pages(pdf, worst_examples, "Worst Translation Examples")

            # Controversial examples (biggest disagreement between methods)
            self._create_cross_method_examples_pages(pdf, controversial_examples, "Most Controversial Examples")

        # Create individual text files for easy reference
        self._save_cross_method_examples_text(best_examples, "best")
        self._save_cross_method_examples_text(worst_examples, "worst")
        self._save_cross_method_examples_text(controversial_examples, "controversial")

        # Create comparison visualization
        self._create_cross_method_results_visualization(best_examples, worst_examples, controversial_examples)

    def _create_cross_method_examples_pages(self, pdf: PdfPages, examples: List[Dict], title: str):
        """Create PDF pages for cross-method translation examples."""
        for i, example in enumerate(examples):
            fig, ax = plt.subplots(figsize=(8.5, 11))

            # Title
            ax.text(0.5, 0.95, f"{title} - Example {i+1}", ha="center", va="top", fontsize=16, weight="bold")

            # Score summary
            score_text = f'Best: {example["best_method"]} ({example["best_score"]:.3f}) | '
            score_text += f'Worst: {example["worst_method"]} ({example["worst_score"]:.3f}) | '
            score_text += f'Range: {example["score_range"]:.3f}'
            ax.text(0.5, 0.90, score_text, ha="center", va="top", fontsize=10, weight="bold")

            # Original text
            ax.text(0.05, 0.84, "Original Text (English):", ha="left", va="top", fontsize=12, weight="bold")
            ax.text(0.05, 0.80, self._wrap_text(example["original_text"], 80), ha="left", va="top", fontsize=10)

            # Expected translation
            ax.text(0.05, 0.72, "Expected Translation (Polish):", ha="left", va="top", fontsize=12, weight="bold")
            ax.text(0.05, 0.68, self._wrap_text(example["expected"], 80), ha="left", va="top", fontsize=10)

            # All method translations
            ax.text(0.05, 0.60, "Translations by Method:", ha="left", va="top", fontsize=12, weight="bold")

            y_pos = 0.56
            # Sort methods by score for this example
            sorted_methods = sorted(example["methods"].items(), key=lambda x: x[1]["overall_score"], reverse=True)

            for method_name, method_data in sorted_methods:
                # Method name and score
                score_color = (
                    "green"
                    if method_name == example["best_method"]
                    else "red"
                    if method_name == example["worst_method"]
                    else "black"
                )

                ax.text(
                    0.05,
                    y_pos,
                    f'{method_name} (Score: {method_data["overall_score"]:.3f}):',
                    ha="left",
                    va="top",
                    fontsize=11,
                    weight="bold",
                    color=score_color,
                )
                y_pos -= 0.025

                # Translation
                ax.text(0.05, y_pos, self._wrap_text(method_data["actual"], 75), ha="left", va="top", fontsize=9)
                y_pos -= 0.05

                # Detailed metrics for this method
                if y_pos > 0.15:  # Only if space allows
                    metrics_text = []
                    row_data = method_data["row_data"]
                    for col in row_data.index[4:]:  # Skip first 4 metadata columns
                        if col != "LLM Reasoning" and "reasoning" not in col.lower():
                            try:
                                if pd.notna(row_data[col]) and isinstance(row_data[col], (int, float)):
                                    metrics_text.append(f"{col}: {row_data[col]:.3f}")
                            except:
                                pass

                    if metrics_text and len(metrics_text) <= 6:  # Limit to avoid overcrowding
                        metrics_line = " | ".join(metrics_text[:6])
                        ax.text(0.07, y_pos, metrics_line, ha="left", va="top", fontsize=8, style="italic")
                        y_pos -= 0.03

                y_pos -= 0.02  # Extra space between methods

                if y_pos < 0.1:  # If running out of space
                    break

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

    def _save_cross_method_examples_text(self, examples: List[Dict], category: str):
        """Save cross-method examples to text files."""
        filename = f"tmp/results/notable_results/{category}_examples.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"{category.title()} Translation Examples - Cross-Method Comparison\n")
            f.write("=" * 70 + "\n\n")

            for i, example in enumerate(examples):
                f.write(f"Example {i+1}\n")
                f.write("-" * 30 + "\n")
                f.write(f'Score Range: {example["score_range"]:.3f} ')
                f.write(f'(Best: {example["best_score"]:.3f}, Worst: {example["worst_score"]:.3f})\n')
                f.write(f'Best Method: {example["best_method"]}\n')
                f.write(f'Worst Method: {example["worst_method"]}\n\n')

                f.write(f'Original Text:\n{example["original_text"]}\n\n')
                f.write(f'Expected Translation:\n{example["expected"]}\n\n')

                f.write("Translations by Method (sorted by score):\n")
                sorted_methods = sorted(example["methods"].items(), key=lambda x: x[1]["overall_score"], reverse=True)

                for method_name, method_data in sorted_methods:
                    f.write(f'\n  {method_name} (Score: {method_data["overall_score"]:.4f}):\n')
                    f.write(f'  {method_data["actual"]}\n')

                f.write("\n" + "=" * 70 + "\n\n")

    def _create_cross_method_results_visualization(
        self, best_examples: List[Dict], worst_examples: List[Dict], controversial_examples: List[Dict]
    ):
        """Create visualization comparing cross-method results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Best examples - score ranges
        best_ranges = [ex["score_range"] for ex in best_examples]
        best_avg_scores = [ex["avg_score"] for ex in best_examples]

        bars1 = ax1.bar(range(len(best_ranges)), best_avg_scores, color="green", alpha=0.7, label="Average Score")
        ax1.errorbar(
            range(len(best_ranges)),
            best_avg_scores,
            yerr=[ex["score_range"] / 2 for ex in best_examples],
            fmt="none",
            color="black",
            capsize=5,
            label="Score Range",
        )
        ax1.set_title("Best Examples - Score Distribution")
        ax1.set_xlabel("Example Index")
        ax1.set_ylabel("Score")
        ax1.set_xticks(range(len(best_ranges)))
        ax1.set_xticklabels([f"{i+1}" for i in range(len(best_ranges))])
        ax1.legend()
        ax1.set_ylim(0, 1.05)

        # Worst examples - score ranges
        worst_ranges = [ex["score_range"] for ex in worst_examples]
        worst_avg_scores = [ex["avg_score"] for ex in worst_examples]

        bars2 = ax2.bar(range(len(worst_ranges)), worst_avg_scores, color="red", alpha=0.7, label="Average Score")
        ax2.errorbar(
            range(len(worst_ranges)),
            worst_avg_scores,
            yerr=[ex["score_range"] / 2 for ex in worst_examples],
            fmt="none",
            color="black",
            capsize=5,
            label="Score Range",
        )
        ax2.set_title("Worst Examples - Score Distribution")
        ax2.set_xlabel("Example Index")
        ax2.set_ylabel("Score")
        ax2.set_xticks(range(len(worst_ranges)))
        ax2.set_xticklabels([f"{i+1}" for i in range(len(worst_ranges))])
        ax2.legend()
        ax2.set_ylim(0, 1.05)

        # Most controversial examples
        controversial_ranges = [ex["score_range"] for ex in controversial_examples]
        controversial_avg_scores = [ex["avg_score"] for ex in controversial_examples]

        bars3 = ax3.bar(
            range(len(controversial_ranges)), controversial_avg_scores, color="orange", alpha=0.7, label="Average Score"
        )
        ax3.errorbar(
            range(len(controversial_ranges)),
            controversial_avg_scores,
            yerr=[ex["score_range"] / 2 for ex in controversial_examples],
            fmt="none",
            color="black",
            capsize=5,
            label="Score Range",
        )
        ax3.set_title("Most Controversial Examples - Score Distribution")
        ax3.set_xlabel("Example Index")
        ax3.set_ylabel("Score")
        ax3.set_xticks(range(len(controversial_ranges)))
        ax3.set_xticklabels([f"{i+1}" for i in range(len(controversial_ranges))])
        ax3.legend()
        ax3.set_ylim(0, 1.05)

        # Score range comparison across categories
        all_ranges = {
            "Best Examples": best_ranges,
            "Worst Examples": worst_ranges,
            "Most Controversial": controversial_ranges,
        }

        bp = ax4.boxplot(
            [best_ranges, worst_ranges, controversial_ranges],
            labels=["Best", "Worst", "Controversial"],
            patch_artist=True,
        )

        colors = ["green", "red", "orange"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax4.set_title("Score Range Distribution by Category")
        ax4.set_ylabel("Score Range")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("tmp/results/notable_results/cross_method_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _wrap_text(self, text: str, width: int) -> str:
        """Wrap text to specified width."""
        import textwrap

        return "\n".join(textwrap.wrap(str(text), width=width))

    def _save_examples_text(self, examples: List[Dict], category: str):
        """Save examples to text files - legacy method for backward compatibility."""
        # This is kept for any remaining calls, but redirects to the new cross-method version
        self._save_cross_method_examples_text(examples, category)

    def _create_notable_results_visualization(self, best_examples: List[Dict], worst_examples: List[Dict]):
        """Create visualization comparing best and worst results - legacy method for backward compatibility."""
        # This is kept for any remaining calls, but redirects to the new cross-method version
        self._create_cross_method_results_visualization(best_examples, worst_examples, [])

    def generate_summary_report(self):
        """Generate a summary report of all analyses."""
        logger.info("Generating summary report...")

        summary = {
            "total_worksheets": len(self.worksheet_names),
            "worksheets_with_data": len(self.all_dataframes),
            "total_translations": sum(len(df) for df in self.all_dataframes.values() if df is not None),
        }

        # Save summary
        with open("tmp/results/analysis_summary.txt", "w") as f:
            f.write("Translation Analysis Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f'Total Worksheets: {summary["total_worksheets"]}\n')
            f.write(f'Worksheets with Data: {summary["worksheets_with_data"]}\n')
            f.write(f'Total Translations Analyzed: {summary["total_translations"]}\n\n')

            f.write("Generated Files:\n")
            f.write("- Method comparison plots in method_comparison/\n")
            f.write("- Correlation analysis plots in correlation_analysis/\n")
            f.write("- Notable results analysis in notable_results/\n")

        logger.info("Summary report generated")

    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        logger.info("Starting full translation analysis...")

        # Create output directories
        self.create_output_directories()

        # Load all data
        self.load_all_worksheets()

        if not self.all_dataframes:
            logger.error("No data loaded. Analysis cannot proceed.")
            return

        # Generate all plots and analyses
        self.plot_method_comparison()
        self.plot_correlation_analysis()
        self.find_notable_results()
        self.generate_summary_report()

        logger.info("Full analysis completed successfully!")
        logger.info("Results saved in tmp/results/ directory")


def main():
    """Main function to run the analysis."""
    try:
        plotter = TranslationAnalysisPlotter()
        plotter.run_full_analysis()

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()

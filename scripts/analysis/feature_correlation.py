import pandas as pd
import matplotlib.pyplot as plt
from logger import logger

from config import (
    PREPARED_DIR,
    FEATURE_ANALYSIS_DIR,
)
class FeatureCorrelation:

    def __init__(self):

        FEATURE_ANALYSIS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
    
    def load_dataset(self):

        dfs = []

        csv_files = sorted(
            PREPARED_DIR.glob("*.csv")
        )

        logger.info(
            f"Loading {len(csv_files)} stations..."
        )

        for csv_file in csv_files:

            df = pd.read_csv(csv_file)

            dfs.append(df)

        return pd.concat(
            dfs,
            ignore_index=True,
        )
    
    def save_statistics(self, numeric_df):

        stats = numeric_df.describe()

        stats.to_csv(

            FEATURE_ANALYSIS_DIR /
            "feature_statistics.csv"
        )

    def save_correlation(self, numeric_df):
        corr = numeric_df.corr(
            method="pearson"
        )

        corr.to_csv(

            FEATURE_ANALYSIS_DIR /
            "correlation_matrix.csv"
            )
        
        target_corr = (
            corr["target_pm2_5"]
            .sort_values(
                ascending=False
            )
        )

        target_corr = target_corr.drop(
            "target_pm2_5"
        )

        target_corr.to_csv(

            FEATURE_ANALYSIS_DIR /
            "target_correlation.csv"

        )
        return corr, target_corr

    def plot_heatmap(self, corr): 
        plt.figure(
            figsize=(12,10)
        )

        plt.imshow(
            corr,
            aspect="auto",
        )

        plt.colorbar()

        plt.xticks(
            range(len(corr.columns)),
            corr.columns,
            rotation=90,
        )

        plt.yticks(
            range(len(corr.columns)),
            corr.columns,
        )

        plt.tight_layout()

        plt.savefig(

            FEATURE_ANALYSIS_DIR /
            "correlation_heatmap.png"

        )

        plt.close()

        plt.figure(
            figsize=(8,6)
        )

    def plot_target_correlation(self, target_corr):
        target_corr.plot.bar()

        plt.tight_layout()

        plt.savefig(

            FEATURE_ANALYSIS_DIR /
            "target_correlation_bar.png"

        )

        plt.close()

    def run(self):

        df = self.load_dataset()

        numeric_df = df.select_dtypes(
            include="number"
        )

        self.save_statistics(
            numeric_df
        )

        corr, target_corr = self.save_correlation(numeric_df)

        self.plot_heatmap(
            corr
        )

        self.plot_target_correlation(
            target_corr
        )

        logger.info(
            "Feature correlation analysis completed."
        )
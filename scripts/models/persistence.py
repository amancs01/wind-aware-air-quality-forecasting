import pandas as pd
from logger import logger
from config import(
    TEST_DIR,
    PERSISTENCE_RESULTS_DIR,
)
from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )
class PersistenceModel:

    def __init__(self):
        self.results = []

    def predict(self, df):

        return df["pm2_5"]
        
    
    def evaluate_station(self,csv_file):
        test_df = pd.read_csv(csv_file)
        predictions = self.predict(test_df)

        target = test_df["target_pm2_5"]

        mae = mean_absolute_error(
            target,
            predictions,
        )

        rmse = (
            mean_squared_error(
                target,
                predictions,
            ) ** 0.5
        )

        r2 = r2_score(
            target,
            predictions,
        )

        self.results.append({

            "station": csv_file.stem,

            "mae": mae,

            "rmse": rmse,

            "r2": r2,

        })

    def save_results(self):
        results_df = pd.DataFrame(self.results)

        results_df.to_csv(

            PERSISTENCE_RESULTS_DIR /
            "metrics.csv",

            index=False,

        )

    def print_summary(self):
        results_df = pd.DataFrame(self.results)
        logger.info("=" * 50)
        logger.info("Overall Persistence Baseline")
        logger.info("=" * 50)

        logger.info(
            f"Average MAE  : {results_df['mae'].mean():.3f}"
        )

        logger.info(
            f"Average RMSE : {results_df['rmse'].mean():.3f}"
        )

        logger.info(
            f"Average R²   : {results_df['r2'].mean():.3f}"
        )

    def run(self):

        csv_files = sorted(
            TEST_DIR.glob("*.csv")
        )

        logger.info(
            f"Evaluating {len(csv_files)} stations..."
        )

        for csv_file in csv_files:

            self.evaluate_station(csv_file)

        self.save_results()

        self.print_summary()
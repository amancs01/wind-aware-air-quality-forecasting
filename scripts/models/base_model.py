import pandas as pd
from logger import logger
from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from config import(
    TEST_DIR,
    MODEL_FEATURE_COLUMNS
)

class BaseModel:

    def __init__(self, results_dir):

        self.results_dir = results_dir

        self.results = []

        self.results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    def load_dataset(self, csv_file):

        return pd.read_csv(csv_file)
    
    def prepare_features(self, df):
        df = df.dropna(subset=MODEL_FEATURE_COLUMNS + ["target_pm2_5"])
        X = df[MODEL_FEATURE_COLUMNS]
        y = df["target_pm2_5"]
        return X, y
    
    def evaluate(self, target, prediction):

        mae = mean_absolute_error(
            target,
            prediction,
        )

        rmse = (
            mean_squared_error(
                target,
                prediction,
            ) ** 0.5
        )

        r2 = r2_score(
            target,
            prediction,
        )

        return mae, rmse, r2
    
    def save_results(self):

        results_df = pd.DataFrame(
            self.results
        )

        results_df.to_csv(

            self.results_dir /
            "metrics.csv",

            index=False,
        )

    def print_summary(self):

        results_df = pd.DataFrame(self.results)
        logger.info("=" * 50)
        logger.info(f"Overall {self.__class__.__name__}")
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

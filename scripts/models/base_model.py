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
        self.prediction_rows = []
        self.pooled_targets = []
        self.pooled_predictions = []

        self.results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load_dataset(self, csv_file):

        return pd.read_csv(csv_file)

    def prepare_evaluation_frame(self, df, required_features=None):
        required_features = required_features or MODEL_FEATURE_COLUMNS
        required_columns = required_features + ["target_pm2_5"]

        evaluation_df = df.dropna(
            subset=required_columns,
        ).copy()

        evaluation_df["source_index"] = evaluation_df.index

        return evaluation_df

    def split_features_target(self, evaluation_df, required_features=None):
        required_features = required_features or MODEL_FEATURE_COLUMNS
        X = evaluation_df[required_features]
        y = evaluation_df["target_pm2_5"]

        return X, y

    def prepare_features(self, df):
        evaluation_df = self.prepare_evaluation_frame(df)

        return self.split_features_target(evaluation_df)

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

    def record_evaluation(
            self,
            station,
            original_rows,
            evaluation_df,
            prediction,
    ):
        if evaluation_df.empty:
            raise ValueError("no rows match the evaluation frame")

        target = evaluation_df["target_pm2_5"]

        mae, rmse, r2 = self.evaluate(
            target,
            prediction,
        )

        evaluated_rows = len(evaluation_df)
        removed_rows = original_rows - evaluated_rows
        coverage_percent = (
            evaluated_rows / original_rows * 100
            if original_rows
            else 0.0
        )

        self.results.append({
            "station": station,
            "original_rows": original_rows,
            "evaluated_rows": evaluated_rows,
            "removed_rows": removed_rows,
            "evaluation_coverage_percent": coverage_percent,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        })

        predictions_df = pd.DataFrame({
            "station": station,
            "source_index": evaluation_df["source_index"].to_numpy(),
            "target_pm2_5": target.to_numpy(),
            "prediction": prediction,
        })

        if "timestamp" in evaluation_df.columns:
            predictions_df.insert(
                2,
                "timestamp",
                evaluation_df["timestamp"].to_numpy(),
            )

        self.prediction_rows.append(predictions_df)
        self.pooled_targets.extend(target.to_list())
        self.pooled_predictions.extend(pd.Series(prediction).to_list())

    def build_summary(self, results_df):
        if results_df.empty:
            return pd.DataFrame([{
                "model": self.__class__.__name__,
                "datasets": 0,
                "original_rows": 0,
                "evaluated_rows": 0,
                "removed_rows": 0,
                "evaluation_coverage_percent": 0.0,
                "macro_mae": None,
                "macro_rmse": None,
                "macro_mean_r2": None,
                "macro_median_r2": None,
                "pooled_mae": None,
                "pooled_rmse": None,
                "pooled_r2": None,
                "negative_r2_datasets": 0,
                "positive_r2_datasets": 0,
            }])

        pooled_mae, pooled_rmse, pooled_r2 = self.evaluate(
            self.pooled_targets,
            self.pooled_predictions,
        )

        original_rows = int(results_df["original_rows"].sum())
        evaluated_rows = int(results_df["evaluated_rows"].sum())
        removed_rows = int(results_df["removed_rows"].sum())
        coverage_percent = (
            evaluated_rows / original_rows * 100
            if original_rows
            else 0.0
        )

        return pd.DataFrame([{
            "model": self.__class__.__name__,
            "datasets": len(results_df),
            "original_rows": original_rows,
            "evaluated_rows": evaluated_rows,
            "removed_rows": removed_rows,
            "evaluation_coverage_percent": coverage_percent,
            "macro_mae": results_df["mae"].mean(),
            "macro_rmse": results_df["rmse"].mean(),
            "macro_mean_r2": results_df["r2"].mean(),
            "macro_median_r2": results_df["r2"].median(),
            "pooled_mae": pooled_mae,
            "pooled_rmse": pooled_rmse,
            "pooled_r2": pooled_r2,
            "negative_r2_datasets": int((results_df["r2"] < 0).sum()),
            "positive_r2_datasets": int((results_df["r2"] >= 0).sum()),
        }])

    def save_results(self):

        results_df = pd.DataFrame(
            self.results
        )

        results_df.to_csv(

            self.results_dir /
            "metrics.csv",

            index=False,
        )

        predictions_df = pd.concat(
            self.prediction_rows,
            ignore_index=True,
        ) if self.prediction_rows else pd.DataFrame()

        predictions_df.to_csv(

            self.results_dir /
            "predictions.csv",

            index=False,
        )

        summary_df = self.build_summary(results_df)

        summary_df.to_csv(

            self.results_dir /
            "summary.csv",

            index=False,
        )

    def print_summary(self):

        results_df = pd.DataFrame(self.results)
        summary_df = self.build_summary(results_df)
        summary = summary_df.iloc[0]

        logger.info("=" * 50)
        logger.info(f"Overall {self.__class__.__name__}")
        logger.info("=" * 50)

        if results_df.empty:
            logger.warning("No evaluation rows were recorded.")
            return

        logger.info(
            f"Datasets       : {summary['datasets']}"
        )

        logger.info(
            f"Evaluated rows : {summary['evaluated_rows']}"
        )

        logger.info(
            f"Macro mean MAE : {summary['macro_mae']:.3f}"
        )

        logger.info(
            f"Macro mean RMSE: {summary['macro_rmse']:.3f}"
        )

        logger.info(
            f"Macro mean R2  : {summary['macro_mean_r2']:.3f}"
        )

        logger.info(
            f"Macro median R2: {summary['macro_median_r2']:.3f}"
        )

        logger.info(
            f"Pooled MAE     : {summary['pooled_mae']:.3f}"
        )

        logger.info(
            f"Pooled RMSE    : {summary['pooled_rmse']:.3f}"
        )

        logger.info(
            f"Pooled R2      : {summary['pooled_r2']:.3f}"
        )

        logger.info(
            f"Negative R2    : {summary['negative_r2_datasets']}"
        )

    def run(self):

        csv_files = sorted(
            TEST_DIR.glob("*.csv")
        )

        logger.info(
            f"Evaluating {len(csv_files)} stations..."
        )

        for csv_file in csv_files:

            try:
                self.evaluate_station(csv_file)
            except ValueError as exc:
                logger.warning(
                    f"Skipping {csv_file.stem}: {exc}"
                )

        self.save_results()

        self.print_summary()

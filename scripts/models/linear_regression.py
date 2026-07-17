import pandas as pd
from sklearn.linear_model import Ridge
from models.base_model import BaseModel
from logger import logger
from config import(
TRAIN_DIR,
TEST_DIR,
LINEAR_RESULTS_DIR,
)
class LinearRegressionModel(BaseModel):

    def __init__(self):

        super().__init__(
            LINEAR_RESULTS_DIR
        )

        self.model = Ridge(alpha=10.0)

    def fit(self, X_train, y_train):

        self.model.fit(
            X_train,
            y_train,
        )

    def predict(self, X):

        return self.model.predict(X)
    
    def evaluate_station(self, csv_file):
        train_df = self.load_dataset(
            TRAIN_DIR / csv_file.name
        )

        test_df = self.load_dataset(
            TEST_DIR / csv_file.name
        )
        X_train, y_train = self.prepare_features(train_df)

        X_test, y_test = self.prepare_features(test_df)
        self.fit(
            X_train,
            y_train,
        )
        predictions = self.predict(X_test)
        mae, rmse, r2 = self.evaluate(
            y_test,
            predictions,
        )
        self.results.append({

            "station": csv_file.stem,

            "mae": mae,

            "rmse": rmse,

            "r2": r2,

        })
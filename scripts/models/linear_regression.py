from sklearn.linear_model import Ridge
from models.base_model import BaseModel
from config import(
TRAIN_DIR,
TEST_DIR,
LINEAR_RESULTS_DIR,
LINEAR_BASELINE_ALPHA,
)
class LinearRegressionModel(BaseModel):

    def __init__(self):

        super().__init__(
            LINEAR_RESULTS_DIR
        )

        self.model = Ridge(alpha=LINEAR_BASELINE_ALPHA)

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

        test_evaluation_df = self.prepare_evaluation_frame(test_df)
        X_test, _ = self.split_features_target(test_evaluation_df)

        self.fit(
            X_train,
            y_train,
        )
        predictions = self.predict(X_test)

        self.record_evaluation(
            csv_file.stem,
            len(test_df),
            test_evaluation_df,
            predictions,
        )

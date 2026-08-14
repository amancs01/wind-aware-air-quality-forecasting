from sklearn.ensemble import RandomForestRegressor

from models.base_model import BaseModel
from config import(
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_MAX_FEATURES,
    RANDOM_FOREST_MIN_SAMPLES_LEAF,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_FOREST_N_JOBS,
    RANDOM_FOREST_RANDOM_STATE,
    RANDOM_FOREST_RESULTS_DIR,
    TEST_DIR,
    TRAIN_DIR,
)


class RandomForestModel(BaseModel):

    def __init__(self):

        super().__init__(
            RANDOM_FOREST_RESULTS_DIR
        )

        self.model = RandomForestRegressor(
            n_estimators=RANDOM_FOREST_N_ESTIMATORS,
            max_depth=RANDOM_FOREST_MAX_DEPTH,
            min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
            max_features=RANDOM_FOREST_MAX_FEATURES,
            random_state=RANDOM_FOREST_RANDOM_STATE,
            n_jobs=RANDOM_FOREST_N_JOBS,
        )

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

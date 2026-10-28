from xgboost import XGBRegressor

from models.base_model import BaseModel
from config import(
    ML_VALIDATION_DIR,
    TEST_DIR,
    TRAIN_DIR,
    XGBOOST_COLSAMPLE_BYTREE,
    XGBOOST_EARLY_STOPPING_ROUNDS,
    XGBOOST_EVAL_METRIC,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MAX_DEPTH,
    XGBOOST_MIN_CHILD_WEIGHT,
    XGBOOST_N_ESTIMATORS,
    XGBOOST_N_JOBS,
    XGBOOST_OBJECTIVE,
    XGBOOST_RANDOM_STATE,
    XGBOOST_REG_ALPHA,
    XGBOOST_REG_LAMBDA,
    XGBOOST_RESULTS_DIR,
    XGBOOST_SUBSAMPLE,
    XGBOOST_TREE_METHOD,
)


class XGBoostModel(BaseModel):

    def __init__(self):

        super().__init__(
            XGBOOST_RESULTS_DIR
        )

        self.model = self.build_model()

    def build_model(self):

        return XGBRegressor(
            n_estimators=XGBOOST_N_ESTIMATORS,
            learning_rate=XGBOOST_LEARNING_RATE,
            max_depth=XGBOOST_MAX_DEPTH,
            min_child_weight=XGBOOST_MIN_CHILD_WEIGHT,
            subsample=XGBOOST_SUBSAMPLE,
            colsample_bytree=XGBOOST_COLSAMPLE_BYTREE,
            reg_alpha=XGBOOST_REG_ALPHA,
            reg_lambda=XGBOOST_REG_LAMBDA,
            objective=XGBOOST_OBJECTIVE,
            tree_method=XGBOOST_TREE_METHOD,
            eval_metric=XGBOOST_EVAL_METRIC,
            early_stopping_rounds=XGBOOST_EARLY_STOPPING_ROUNDS,
            random_state=XGBOOST_RANDOM_STATE,
            n_jobs=XGBOOST_N_JOBS,
        )

    def fit(self, X_train, y_train, X_validation, y_validation):

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_validation, y_validation)],
            verbose=False,
        )

    def predict(self, X):

        return self.model.predict(X)

    def evaluate_station(self, csv_file):
        train_df = self.load_dataset(
            TRAIN_DIR / csv_file.name
        )

        validation_df = self.load_dataset(
            ML_VALIDATION_DIR / csv_file.name
        )

        test_df = self.load_dataset(
            TEST_DIR / csv_file.name
        )

        train_evaluation_df = self.prepare_evaluation_frame(train_df)
        validation_evaluation_df = self.prepare_evaluation_frame(
            validation_df,
        )
        test_evaluation_df = self.prepare_evaluation_frame(test_df)

        X_train, y_train = self.split_features_target(train_evaluation_df)
        X_validation, y_validation = self.split_features_target(
            validation_evaluation_df,
        )
        X_test, _ = self.split_features_target(test_evaluation_df)

        self.model = self.build_model()
        self.fit(
            X_train,
            y_train,
            X_validation,
            y_validation,
        )
        predictions = self.predict(X_test)

        self.record_evaluation(
            csv_file.stem,
            len(test_df),
            test_evaluation_df,
            predictions,
        )
        self.results[-1]["best_iteration"] = getattr(
            self.model,
            "best_iteration",
            None,
        )
        self.results[-1]["best_score"] = getattr(
            self.model,
            "best_score",
            None,
        )

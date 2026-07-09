import pandas as pd
from logger import logger
from models.base_model import BaseModel
from config import(
    PERSISTENCE_RESULTS_DIR
)

class PersistenceModel(BaseModel):

    def __init__(self):
        super().__init__(
            PERSISTENCE_RESULTS_DIR
        )

    def predict(self, df):

        return df["pm2_5"]
        
    
    def evaluate_station(self,csv_file):
        test_df = self.load_dataset(csv_file)
        prediction = self.predict(test_df)

        target = test_df["target_pm2_5"]

        mae, rmse, r2 = self.evaluate(
            target,
            prediction,
        )

        self.results.append({

            "station": csv_file.stem,

            "mae": mae,

            "rmse": rmse,

            "r2": r2,

        })

    
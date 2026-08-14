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
        evaluation_df = self.prepare_evaluation_frame(test_df)

        prediction = self.predict(evaluation_df)

        self.record_evaluation(
            csv_file.stem,
            len(test_df),
            evaluation_df,
            prediction,
        )

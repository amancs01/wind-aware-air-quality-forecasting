import pandas as pd

from logger import logger

from config import (
    PREPARED_DIR,
    TRAIN_DIR,
    ML_VALIDATION_DIR,
    TEST_DIR,
    SPLIT_DIR,
)


class DatasetSplitter:

    def __init__(self):

        TRAIN_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        ML_VALIDATION_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        TEST_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.summary = []


    def split_station(self, csv_file):
        df = pd.read_csv(csv_file)
        if df.empty:
            logger.warning(
                f"Skipping {csv_file.stem}: no data available."
            )
            return
        rows = len(df)

        train_end = int(rows * 0.70)

        validation_end = int(rows * 0.85)

        train_df = df.iloc[:train_end]

        validation_df = df.iloc[
            train_end:validation_end
        ]

        test_df = df.iloc[
            validation_end:
        ]

        train_df.to_csv(
            TRAIN_DIR / csv_file.name,
            index=False,
        )

        validation_df.to_csv(
            ML_VALIDATION_DIR / csv_file.name,
            index=False,
        )

        test_df.to_csv(
            TEST_DIR / csv_file.name,
            index=False,
        )

        logger.info(csv_file.stem)

        logger.info(
            f"Train      : {len(train_df)}"
        )

        logger.info(
            f"Validation : {len(validation_df)}"
        )

        logger.info(
            f"Test        : {len(test_df)}"
        )

        self.summary.append({

        "station": csv_file.stem,

        "train_rows": len(train_df),

        "validation_rows": len(validation_df),

        "test_rows": len(test_df),

        "total_rows": len(df),

        })

    def save_summary(self):

        summary_df = pd.DataFrame(self.summary)

        output_file = (
            SPLIT_DIR /
            "split_summary.csv"
        )

        summary_df.to_csv(
            output_file,
            index=False,
        )

        logger.info(
            f"Saved {output_file}"
        )


    def run(self):

        csv_files = sorted(
            PREPARED_DIR.glob("*.csv")
        )

        logger.info(
            f"Splitting {len(csv_files)} stations..."
        )

        for csv_file in csv_files:

            self.split_station(csv_file)
        
        self.save_summary()
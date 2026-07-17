import pandas as pd

from logger import logger

from config import (
    TRAIN_DIR,
    ML_VALIDATION_DIR,
    TEST_DIR,
)


class SplitVerifier:

    def verify_station(self, csv_name):
        train = pd.read_csv(TRAIN_DIR / csv_name)
        if train.empty:
            logger.warning(
                f"{csv_name}: empty training dataset."
            )
            return
        validation = pd.read_csv(
            ML_VALIDATION_DIR / csv_name
        )
        if validation.empty:
            logger.warning(
                f"{csv_name}: empty validation dataset."
            )
            return
        test = pd.read_csv(TEST_DIR / csv_name)
        if test.empty:
            logger.warning(
                f"{csv_name}: empty test dataset."
            )
            return
        
        logger.info(
            f"Missing values: "
            f"{train.isna().sum().sum()} | "
            f"{validation.isna().sum().sum()} | "
            f"{test.isna().sum().sum()}"
        )

        logger.info(
            f"Train : "
            f"{train['timestamp'].iloc[0]}"
            f" -> "
            f"{train['timestamp'].iloc[-1]}"
        )

        logger.info(
            f"Validation : "
            f"{validation['timestamp'].iloc[0]}"
            f" -> "
            f"{validation['timestamp'].iloc[-1]}"
        )

        logger.info(
            f"Test : "
            f"{test['timestamp'].iloc[0]}"
            f" -> "
            f"{test['timestamp'].iloc[-1]}"
        )

        assert (
            train["timestamp"].iloc[-1]
            <
            validation["timestamp"].iloc[0]
        )

        assert (
            validation["timestamp"].iloc[-1]
            <
            test["timestamp"].iloc[0]
        )

    def run(self):

        csv_files = sorted(
            TRAIN_DIR.glob("*.csv")
        )

        logger.info(
            f"Verifying {len(csv_files)} stations..."
        )

        for csv_file in csv_files:

            logger.info("=" * 50)

            logger.info(csv_file.stem)

            self.verify_station(csv_file.name)
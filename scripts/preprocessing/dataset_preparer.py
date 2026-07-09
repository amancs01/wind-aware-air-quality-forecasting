import pandas as pd

from logger import logger

from config import (
    FEATURED_DIR,
    PREPARED_DIR,
)


class DatasetPreparer:

    def __init__(self):

        PREPARED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    def prepare_dataset(self, csv_file):
        df = pd.read_csv(csv_file)
        df["target_pm2_5"] = df["pm2_5"].shift(-1)
        
        original_rows = len(df)

        df = (
            df
            .dropna()
            .reset_index(drop=True)
        )
        if df.empty:
            logger.warning(
                f"{csv_file.stem}: no usable rows after preprocessing."
            )
            return
        removed_rows = original_rows - len(df)

        missing = df.isna().sum().sum()

        if missing > 0:

            logger.warning(
                f"{csv_file.name}: {missing} missing values remain."
            )

        output = PREPARED_DIR / csv_file.name

        df.to_csv(
            output,
            index=False,
        )

        logger.info(
            f"{csv_file.stem}"
        )
        logger.info(
            f"Rows removed : {removed_rows}"
        )
        logger.info(
            f"Final rows   : {len(df)}"
        )

    def run(self):

        csv_files = sorted(
            FEATURED_DIR.glob("*.csv")
        )

        logger.info(
            f"Preparing {len(csv_files)} stations..."
        )

        for csv_file in csv_files:

            self.prepare_dataset(csv_file)
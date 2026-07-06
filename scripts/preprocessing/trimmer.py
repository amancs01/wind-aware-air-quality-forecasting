from pathlib import Path

import pandas as pd

from logger import logger

from config import (
    MERGED_DIR,
    FINAL_DIR,
)


class DataTrimmer:

    def __init__(self):

        FINAL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
    
    def trim_dataframe(self, df):

        first_valid = df["pm2_5"].first_valid_index()

        if first_valid is None:

            return None

        return (
            df
            .loc[first_valid:]
            .reset_index(drop=True)
        )
    
    def run(self):

        for csv_file in sorted(MERGED_DIR.glob("*.csv")):

            logger.info(f"Trimming {csv_file.stem}")

            df = pd.read_csv(csv_file)

            print(df["pm2_5"].first_valid_index())

            print(
                df.loc[
                    df["pm2_5"].first_valid_index(),
                    "timestamp"
                ]
            )
            trimmed = self.trim_dataframe(df)

            if trimmed is None:

                logger.warning(
                    f"No PM2.5 found for {csv_file.stem}"
                )

                continue

            output = FINAL_DIR / csv_file.name

            trimmed.to_csv(
                output,
                index=False,
            )

            logger.info(
                f"Saved {output}"
            )
    
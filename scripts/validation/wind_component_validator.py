import numpy as np
import pandas as pd

from config import FEATURED_DIR
from logger import logger


class WindComponentValidator:

    CARDINAL_CASES = pd.DataFrame({
        "wind_speed": [10.0, 10.0, 10.0, 10.0],
        "wind_direction": [0.0, 90.0, 180.0, 270.0],
        "expected_wind_u": [0.0, -10.0, 0.0, 10.0],
        "expected_wind_v": [-10.0, 0.0, 10.0, 0.0],
    })

    def __init__(self, tolerance=1e-9):
        self.tolerance = tolerance

    @staticmethod
    def calculate_components(wind_speed, wind_direction):
        wind_rad = np.deg2rad(wind_direction)

        wind_u = (
            -wind_speed *
            np.sin(wind_rad)
        )

        wind_v = (
            -wind_speed *
            np.cos(wind_rad)
        )

        return wind_u, wind_v

    def validate_cardinal_cases(self):
        cases = self.CARDINAL_CASES.copy()
        cases["wind_u"], cases["wind_v"] = self.calculate_components(
            cases["wind_speed"],
            cases["wind_direction"],
        )

        cases["u_abs_error"] = (
            cases["wind_u"] -
            cases["expected_wind_u"]
        ).abs()

        cases["v_abs_error"] = (
            cases["wind_v"] -
            cases["expected_wind_v"]
        ).abs()

        max_error = max(
            cases["u_abs_error"].max(),
            cases["v_abs_error"].max(),
        )

        if max_error > self.tolerance:
            raise ValueError(
                "Wind cardinal sanity check failed with max error "
                f"{max_error}"
            )

        return cases

    def validate_featured_magnitude(self):
        max_error = 0.0
        rows_checked = 0
        files_checked = 0

        for csv_file in sorted(FEATURED_DIR.glob("*.csv")):
            df = pd.read_csv(
                csv_file,
                usecols=["wind_speed", "wind_u", "wind_v"],
            )

            magnitude = np.sqrt(
                df["wind_u"] ** 2 +
                df["wind_v"] ** 2
            )

            station_max_error = (
                magnitude -
                df["wind_speed"]
            ).abs().max()

            if pd.notna(station_max_error):
                max_error = max(max_error, float(station_max_error))

            rows_checked += len(df)
            files_checked += 1

        return {
            "files_checked": files_checked,
            "rows_checked": rows_checked,
            "max_abs_magnitude_error": max_error,
        }

    def run(self):
        cardinal_results = self.validate_cardinal_cases()
        magnitude_results = self.validate_featured_magnitude()

        logger.info("Wind cardinal sanity cases:")
        logger.info(
            cardinal_results[
                [
                    "wind_direction",
                    "wind_speed",
                    "wind_u",
                    "wind_v",
                ]
            ].to_string(index=False)
        )

        logger.info(
            "Wind magnitude validation: "
            f"{magnitude_results['files_checked']} files, "
            f"{magnitude_results['rows_checked']} rows, max error "
            f"{magnitude_results['max_abs_magnitude_error']}"
        )

        return cardinal_results, magnitude_results

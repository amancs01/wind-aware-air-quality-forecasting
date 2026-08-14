import argparse

from logger import logger
from validation.hourly_air_quality_validator import (
    HourlyAirQualityValidator,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate canonical local-clock hourly PM2.5 data."
    )
    parser.add_argument(
        "--stations",
        nargs="*",
        help="Optional exact canonical station file stems to validate.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    validator = HourlyAirQualityValidator()

    detail, summary = validator.run(
        stations_filter=args.stations,
    )

    logger.info("=" * 50)
    logger.info("Hourly AQ Validation Summary")
    logger.info("=" * 50)

    if not summary.empty:
        row = summary.iloc[0]
        logger.info(f"Stations checked       : {row['stations_checked']}")
        logger.info(f"Files                  : {row['files']}")
        logger.info(f"Rows                   : {row['rows']:,}")
        logger.info(
            "Duplicate timestamp rows: "
            f"{row['duplicate_timestamp_rows']:,}"
        )
        logger.info(
            "Non-clock-hour rows     : "
            f"{row['non_clock_hour_rows']:,}"
        )
        logger.info(
            "Invalid intervals       : "
            f"{row['invalid_one_hour_intervals']:,}"
        )
        logger.info(f"Missing PM2.5 rows     : {row['missing_pm25']:,}")
        logger.info(f"Unsorted stations      : {row['unsorted_stations']}")

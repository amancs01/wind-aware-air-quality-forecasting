import argparse

from logger import logger
from preprocessing.hourly_air_quality_preparer import (
    HourlyAirQualityPreparer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare canonical local-clock hourly PM2.5 data."
    )
    parser.add_argument(
        "--stations",
        nargs="*",
        help="Optional exact sanitized station folder names to prepare.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    preparer = HourlyAirQualityPreparer()

    summary = preparer.run(
        stations_filter=args.stations,
    )

    logger.info("=" * 50)
    logger.info("Hourly AQ Preparation Summary")
    logger.info("=" * 50)
    logger.info(f"Stations prepared : {len(summary)}")

    if not summary.empty:
        logger.info(f"Raw rows          : {summary['raw_rows'].sum():,}")
        logger.info(
            "Canonical rows    : "
            f"{summary['canonical_rows'].sum():,}"
        )
        logger.info(
            "Dropped non-clock : "
            f"{summary['dropped_non_clock_hour_rows'].sum():,}"
        )

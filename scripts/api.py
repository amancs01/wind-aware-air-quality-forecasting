import calendar
import time
from clients.http_client import get_json

def fetch_all_measurements(
    sensor_id,
    year,
    base_url,
    headers,
    timeout,
):
    all_results = []
    url=f"{base_url}/sensors/{int(sensor_id)}/measurements"

    # Loop through all 12 months
    for month in range(1, 13):

        last_day = calendar.monthrange(year, month)[1]

        datetime_from = f"{year}-{month:02d}-01T00:00:00Z"
        datetime_to = f"{year}-{month:02d}-{last_day}T23:59:59Z"

        params = {
            "datetime_from": datetime_from,
            "datetime_to": datetime_to,
            "limit": 1000,
        }

        success = False


        data = get_json(
            url=url,
            headers=headers,
            params=params,
            timeout=timeout,
        )

        results = data["results"]
        if results:
            print(
                f"Month {month}: "
                f"{results[0]['period']['datetimeFrom']['local']}  -->  "
                f"{results[-1]['period']['datetimeFrom']['local']}"
            )

        print(f"Month {month}: {len(results)} rows")

        all_results.extend(results)

    return all_results


def _has_more_pages(meta, page, limit, result_count):
    found = meta.get("found")

    if isinstance(found, int):
        return page * limit < found

    if isinstance(found, str) and found.startswith(">"):
        return result_count == limit

    return result_count == limit


def fetch_all_hourly_measurements(
    sensor_id,
    year,
    base_url,
    headers,
    timeout,
):
    all_results = []
    url = f"{base_url}/sensors/{int(sensor_id)}/hours"
    limit = 1000

    for month in range(1, 13):

        last_day = calendar.monthrange(year, month)[1]

        datetime_from = f"{year}-{month:02d}-01T00:00:00Z"
        datetime_to = f"{year}-{month:02d}-{last_day}T23:59:59Z"

        month_results = []
        page = 1

        while True:

            params = {
                "datetime_from": datetime_from,
                "datetime_to": datetime_to,
                "limit": limit,
                "page": page,
            }

            data = get_json(
                url=url,
                headers=headers,
                params=params,
                timeout=timeout,
                max_retries=6,
            )

            results = data.get("results", [])
            meta = data.get("meta", {})

            month_results.extend(results)

            if not _has_more_pages(
                meta=meta,
                page=page,
                limit=limit,
                result_count=len(results),
            ):
                break

            page += 1
            time.sleep(0.2)

        if month_results:
            first_period = month_results[0]["period"]
            last_period = month_results[-1]["period"]
            print(
                f"Month {month}: "
                f"{first_period['datetimeFrom']['local']}  -->  "
                f"{last_period['datetimeTo']['local']}"
            )

        print(f"Month {month}: {len(month_results)} hourly rows")

        all_results.extend(month_results)

    return all_results

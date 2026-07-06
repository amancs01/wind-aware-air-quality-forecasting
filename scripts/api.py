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
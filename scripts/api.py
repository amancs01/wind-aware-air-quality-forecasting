import calendar
import time
import requests


def fetch_all_measurements(
    sensor_id,
    year,
    base_url,
    headers,
    timeout,
):
    all_results = []

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

        # Retry loop
        for attempt in range(3):

            try:

                response = requests.get(
                    f"{base_url}/sensors/{int(sensor_id)}/measurements",
                    headers=headers,
                    params=params,
                    timeout=timeout,
                )

                if response.status_code == 200:
                    success = True
                    break

                print(
                    f"HTTP {response.status_code} "
                    f"(Month {month}, Attempt {attempt + 1})"
                )

            except requests.exceptions.RequestException as e:

                print(
                    f"Network Error "
                    f"(Month {month}, Attempt {attempt + 1})"
                )
                print(e)

            time.sleep(2 ** attempt)

        if not success:
            print(f"Skipping Month {month}")
            continue

        data = response.json()
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
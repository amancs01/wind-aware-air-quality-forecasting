import time
import requests

def get_json(
    url,
    headers=None,
    params=None,
    timeout=30,
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=timeout,
            )
            if response.status_code == 200:
                return response.json()
            
            print(
                f"HTTP{response.status_code}"
                f"(Attempt {attempt + 1}/{max_retries})"
            )

            time.sleep(2 ** attempt)

        except requests.exceptions.RequestException as e:

            print(
                f"Network error "
                f"(Attempt {attempt + 1}/{max_retries})"
            )

            print(e)

            time.sleep(2 ** attempt)
        
        raise Exception(
            f"Failed after {max_retries} attemts."
        )

        
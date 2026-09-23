import threading
import time
from collections import deque

import requests

BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
MAX_REQUESTS_PER_SECOND = 80

_request_times = deque()
_rate_limit_lock = threading.Lock()


def _throttle():
    """Block until sending another request keeps us under MAX_REQUESTS_PER_SECOND."""
    with _rate_limit_lock:
        now = time.monotonic()
        while _request_times and now - _request_times[0] >= 1:
            _request_times.popleft()

        if len(_request_times) >= MAX_REQUESTS_PER_SECOND:
            sleep_for = 1 - (now - _request_times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            while _request_times and now - _request_times[0] >= 1:
                _request_times.popleft()

        _request_times.append(now)


def fetch_object(object_id, timeout=10):
    """Fetch a single object from the Met Collection API.

    Returns a dict with keys "status_code" and "data" on success (data is
    the parsed JSON body, or None if the response wasn't valid JSON), or
    "status_code": None and an "error" message if the request itself failed.
    """
    _throttle() 

    url = BASE_URL.format(object_id=object_id)
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        return {"status_code": None, "data": None, "error": str(e)}

    try:
        data = response.json()
    except ValueError:
        data = None

    return {"status_code": response.status_code, "data": data, "error": None}

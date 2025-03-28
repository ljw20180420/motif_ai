from requests.exceptions import SSLError
import requests
import sys
import time


def requests_until_success(method, url, **kwargs):
    while True:
        try:
            response = requests.request(method, url, allow_redirects=False, **kwargs)
            break
        except SSLError:
            sys.stdout.write(f"ssl error for {url}, retry")
            time.sleep(5)
    return response

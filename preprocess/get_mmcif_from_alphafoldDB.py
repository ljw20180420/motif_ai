#!/usr/bin/env python

import os

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))


import json
import sys
from requests.exceptions import SSLError
import requests
import time
import pandas as pd


def requests_until_success(method, url, **kwargs):
    while True:
        try:
            response = requests.request(method, url, allow_redirects=False, **kwargs)
            break
        except SSLError:
            sys.stdout.write(f"ssl error for {url}, retry")
            time.sleep(5)
    return response


df = pd.read_table(sys.stdin, header=0, sep="\t", na_filter=False)
for accession in df["Entry"]:
    if os.path.exists(f"get_mmcif_from_alphafoldDB/{accession}.pdb"):
        continue
    sys.stderr.write(f"download mmcif for {accession}\n")
    response = requests_until_success(
        "GET", f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
    )
    model_list = json.loads(response.text)
    if len(model_list) == 0:
        sys.stderr.write(f"cannot find mmcif for {accession}\n")
        continue
    response = requests_until_success("GET", model_list[0]["cifUrl"])
    with open(f"get_mmcif_from_alphafoldDB/{accession}.mmcif", "wb") as fd:
        fd.write(response.content)
    response = requests_until_success("GET", model_list[0]["pdbUrl"])
    with open(f"get_mmcif_from_alphafoldDB/{accession}.pdb", "wb") as fd:
        fd.write(response.content)

#!/usr/bin/env python

import os

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))


import json
import sys
from utils import requests_until_success

for file in os.listdir("convert_any_accession_to_uniprotKB_accession"):
    with open(f"convert_any_accession_to_uniprotKB_accession/{file}") as fd:
        for line in fd:
            accession = line.strip().split()[1]
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

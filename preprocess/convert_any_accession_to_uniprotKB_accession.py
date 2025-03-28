#!/usr/bin/env python

import os

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
import time
import sys
import re
from utils import requests_until_success

response = requests_until_success(
    "GET", "https://rest.uniprot.org/configure/idmapping/fields"
)

database_dict = json.loads(response.text)

database_list = []
for group in database_dict["groups"]:
    for item in group["items"]:
        database_list.append(item["name"])

with open("zf.acc") as fd:
    # accesions = [re.sub(r"\.\d$", "", line) for line in fd]
    accesions = [line for line in fd]

todo_database_list = database_list
while len(todo_database_list) > 0:
    new_todo_database_list = []
    for database in todo_database_list:
        sys.stdout.write(f"request job for {database}\n")
        response = requests_until_success(
            "POST",
            "https://rest.uniprot.org/idmapping/run",
            data={"ids": ",".join(accesions), "from": database, "to": "UniProtKB"},
        )
        text = response.text
        if text.find("parameters is invalid") != -1:
            sys.stdout.write(f"request job for {database} is invalid, skip\n")
            continue
        jobId = json.loads(text)["jobId"]
        sys.stdout.write(f"request job for {database} success\n")
        time.sleep(20)

        sys.stdout.write(f"retrieve results for {database}\n")
        while True:
            response = requests_until_success(
                "GET", f"https://rest.uniprot.org/idmapping/status/{jobId}"
            )
            # NEW, RUNNING, PROCESSING, UNFINISHED, FINISHED, ERROR, ABORTED
            status = json.loads(response.text)["jobStatus"]
            if status == "FINISHED":
                response = requests_until_success(
                    "GET", f"https://rest.uniprot.org/idmapping/stream/{jobId}"
                )
                results = json.loads(response.text)["results"]
                if len(results) > 0:
                    with open(
                        f"convert_any_accession_to_uniprotKB_accession/{database}.txt",
                        "w",
                    ) as fd:
                        for result in results:
                            fd.write(f"""{result["from"]}\t{result["to"]}\n""")
                sys.stdout.write(f"retrieve results for {database} {status}\n")
                break
            elif status in ["UNFINISHED", "ERROR", "ABORTED"]:
                new_todo_database_list.append(database)
                sys.stdout.write(f"retrieve results for {database} {status}\n")
                break
            time.sleep(3)

    todo_database_list = new_todo_database_list

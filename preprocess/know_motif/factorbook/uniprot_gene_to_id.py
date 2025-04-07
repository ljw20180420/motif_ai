#!/usr/bin/env python

import sys
import requests
import json
import time

# # list all database
# response = requests.get("https://rest.uniprot.org/configure/idmapping/fields")
# database_dict = json.loads(response.text)

# for group in database_dict["groups"]:
#     for item in group["items"]:
#         sys.stderr.write(item["name"] + "\n")

genes = [gene.strip() for gene in sys.stdin]
job_done = False
while not job_done:
    response = requests.post(
        "https://rest.uniprot.org/idmapping/run",
        data={
            "ids": ",".join(genes),
            "from": "Gene_Name",
            "to": "UniProtKB",
            "taxId": "10090",
        },
        allow_redirects=False,
    )
    if response.text.find("parameters is invalid") != -1:
        sys.stdout.write(f"request job is invalid\n")
        exit(1)
    jobId = json.loads(response.text)["jobId"]
    while True:
        response = requests.get(
            f"https://rest.uniprot.org/idmapping/status/{jobId}",
            allow_redirects=False,
        )
        # NEW, RUNNING, PROCESSING, UNFINISHED, FINISHED, ERROR, ABORTED
        status = json.loads(response.text)["jobStatus"]
        if status == "FINISHED":
            response = requests.get(
                f"https://rest.uniprot.org/idmapping/stream/{jobId}",
                allow_redirects=False,
            )
            results = json.loads(response.text)["results"]
            if len(results) > 0:
                with open(3, "w") as fd:
                    for result in results:
                        fd.write(f"""{result["from"]}\t{result["to"]}\n""")
            job_done = True
            sys.stderr.write(f"retrieve results {status}\n")
            break
        elif status in ["UNFINISHED", "ERROR", "ABORTED"]:
            sys.stderr.write(f"retrieve results {status}\n")
            break
        time.sleep(3)

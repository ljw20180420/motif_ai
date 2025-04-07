#!/usr/bin/env python

import sys
import requests

genes = [gene.strip() for gene in sys.stdin]

# requests.post("https://rest.uniprot.org/idmapping/run", data={ids=",".join(genes)})

#!/usr/bin/env python

import xml.etree.ElementTree as ET
from signal import signal, SIGPIPE, SIG_DFL
import sys

signal(SIGPIPE, SIG_DFL)
root = ET.parse(sys.stdin).getroot()

feature_table = next(root[0].iterfind("INSDSeq_feature-table"))
seq = next(root[0].iterfind("INSDSeq_sequence")).text.upper()

sys.stdout.write(f"{seq}\n")

for feature in feature_table:
    if next(feature.iterfind("INSDFeature_key")).text != "Region":
        continue

    intervals = next(feature.iterfind("INSDFeature_intervals"))
    starts, ends, accessions = [], [], []
    for interval in intervals:
        starts.append(
            int(next(interval.iterfind("INSDInterval_from")).text) - 1
        )  # from 1-based close interval to 0-based open interval
        ends.append(int(next(interval.iterfind("INSDInterval_to")).text))
        accessions.append(next(interval.iterfind("INSDInterval_accession")).text)

    name, note, db_xref = "", "", ""
    quals = next(feature.iterfind("INSDFeature_quals"))
    for qual in quals:
        qual_name = next(qual.iterfind("INSDQualifier_name")).text
        qual_value = (
            next(qual.iterfind("INSDQualifier_value"))
            .text.replace(" ", "_")
            .replace("\t", "_")
        )
        if qual_name == "region_name":
            name = qual_value
        elif qual_name == "note":
            note = qual_value
        elif qual_name == "db_xref":
            db_xref = qual_value

    for start, end, accession in zip(starts, ends, accessions):
        sys.stdout.write(f"{name}\t{note}\t{db_xref}\t{start}\t{end}\t{accession}\n")

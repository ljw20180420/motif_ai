#!/usr/bin/env python
import re
import sys
import json


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def parse_ft(file):
    with open(file) as fd:
        lines = []
        for line in fd:
            line = line.rstrip()
            if len(line) == 0:
                continue
            if line.startswith(">Feature ") and len(lines) > 0:
                yield lines
                lines = []
            lines.append(line)
        yield lines


@static_vars(feature_re=re.compile(r"^>Feature (.+)\|(.+)\|(.*)"))
def parse_record(protein):
    def put_feature():
        if feature[0].endswith("Region"):
            regions.append(feature)
        elif feature[0].endswith("Site"):
            sites.append(feature)
        elif feature[0].endswith("Protein"):
            proteins.append(feature)
        elif feature[0].endswith("CDS"):
            CDSs.append(feature)
        elif feature[0].endswith("gene"):
            genes.append(feature)
        elif feature[0].endswith("Bond"):
            bonds.append(feature)
        elif feature[0].endswith("SecStr"):
            SecStrs.append(feature)
        elif feature[0].endswith("Het"):
            Hets.append(feature)
        else:
            sys.stderr.write(f"unknown feature {feature[0].split()[-1]}\n")

    mat = parse_record.feature_re.search(protein[0])
    source, accession, name = mat.group(1), mat.group(2), mat.group(3)
    regions, sites, proteins, CDSs, genes, bonds, SecStrs, Hets, feature, has_indent = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        False,
    )
    for line in protein[1:]:
        if len(feature) > 0 and has_indent and re.match(r"\S", line):
            put_feature()
            feature = []
            has_indent = False
        feature.append(line)
        if re.match(r"\s", line):
            has_indent = True
    put_feature()
    return (
        source,
        accession,
        name,
        regions,
        sites,
        proteins,
        CDSs,
        genes,
        bonds,
        SecStrs,
        Hets,
    )


@static_vars(
    zinc_finger_re=re.compile(r"c2h2|h2c2|zn[ -]finger|zinc[ -]finger", re.IGNORECASE)
)
def parse_region(region):
    (start, end, _) = region[0].split()
    # transform 1-based closed interval to 0-based open interval
    start, end, name, structural_motif, db_xref, zinc_finger = (
        int(start.lstrip("<")) - 1,
        int(end.lstrip(">")),
        "",
        False,
        "",
        False,
    )
    for line in region[1:]:
        line = line.lstrip()
        (key, value) = line.split("\t")
        if value.find("[structural motif]") != -1:
            structural_motif = True
        if parse_region.zinc_finger_re.search(value):
            zinc_finger = True
        if key == "region":
            name = value
        elif key == "db_xref":
            db_xref = value
    return {
        "name": name,
        "start": start,
        "end": end,
        "structural_motif": structural_motif,
        "db_xref": db_xref,
        "zinc_finger": zinc_finger,
    }


for record in parse_ft("zf.seq"):
    (
        source,
        accession,
        name,
        regions,
        sites,
        proteins,
        CDSs,
        genes,
        bonds,
        SecStrs,
        Hets,
    ) = parse_record(record)

    for region in regions:
        breakpoint()
        domain = parse_region(region)

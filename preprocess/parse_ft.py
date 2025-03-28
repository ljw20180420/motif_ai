#!/usr/bin/env python

import os

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import re
import sys
import pandas as pd
import bioframe as bf


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def parse_ft(file_ft, file_seq):
    with open(file_ft) as ft_fd, open(file_seq) as seq_fd:
        lines = []
        for line in ft_fd:
            line = line.rstrip()
            if len(line) == 0:
                continue
            if line.startswith(">Feature ") and len(lines) > 0:
                yield lines, seq_fd.readline().strip()
                lines = []
            lines.append(line)
        yield lines, seq_fd.readline().strip()


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
    zinc_finger_re=re.compile(
        r"c2h2|h2c2|zn[ -_]finger|zinc[ -_]finger", re.IGNORECASE
    ),
    coiled_coil_re=re.compile(r"coiled[ -_]coil", re.IGNORECASE),
)
def parse_region(region, accession):
    (start, end, _) = region[0].split()
    # transform 1-based closed interval to 0-based open interval
    (
        start,
        end,
        name,
        structural_motif,
        db_xref,
        zinc_finger,
        coiled_coil,
        disordered,
        basic,
        acidic,
        polar,
        pro,
        p_rich,
        g_rich,
    ) = (
        int(start.lstrip("<")) - 1,
        int(end.lstrip(">")),
        "",
        False,
        "",
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    is_region_domain = False
    is_compositionally_biased_region = False
    for line in region[1:]:
        line = line.lstrip()
        (key, value) = line.split("\t")
        if value.lower().find("[structural motif]") != -1:
            structural_motif = True
        if parse_region.zinc_finger_re.search(value):
            zinc_finger = True
        if parse_region.coiled_coil_re.search(value):
            coiled_coil = True
        if value.lower().find("disordered.") != -1:
            disordered = True
        if key == "region":
            if value != "Domain":
                name = value.replace(" ", "_").replace("\t", "_")
                if value == "Compositionally biased region":
                    is_compositionally_biased_region = True
                elif value == "P-rich sequence":
                    p_rich = True
                elif value == "G-rich sequence":
                    g_rich = True
            else:
                is_region_domain = True
        elif key == "note":
            if is_region_domain:
                name = (
                    value.split(". ")[0]
                    .rstrip(".")
                    .replace(" ", "_")
                    .replace("\t", "_")
                )
                is_region_domain = False
            if is_compositionally_biased_region:
                if value.lower().find("acidic") != -1:
                    acidic = True
                if value.lower().find("basic") != -1:
                    basic = True
                if value.lower().find("polar") != -1:
                    polar = True
                if value.lower().find("pro") != -1:
                    pro = True
                is_compositionally_biased_region = False
        elif key == "db_xref":
            db_xref = value
    return {
        "accession": accession,
        "start": start,
        "end": end,
        "name": name,
        "structural_motif": structural_motif,
        "db_xref": db_xref,
        "zinc_finger": zinc_finger,
        "coiled_coil": coiled_coil,
        "disordered": disordered,
        "basic": basic,
        "acidic": acidic,
        "polar": polar,
        "pro": pro,
        "p_rich": p_rich,
        "g_rich": g_rich,
    }


def format_data_frame(data_dict_list):
    df = pd.DataFrame(
        data_dict_list,
        columns=[
            "accession",
            "start",
            "end",
            "name",
            "structural_motif",
            "db_xref",
            "zinc_finger",
            "coiled_coil",
            "disordered",
            "basic",
            "acidic",
            "polar",
            "pro",
            "p_rich",
            "g_rich",
        ],
    )
    df["accession"] = df["accession"].astype("string")
    df["start"] = df["start"].astype("int")
    df["end"] = df["end"].astype("int")
    df["name"] = df["name"].astype("string")
    df["structural_motif"] = df["structural_motif"].astype("boolean")
    df["db_xref"] = df["db_xref"].astype("string")
    df["zinc_finger"] = df["zinc_finger"].astype("boolean")
    df["coiled_coil"] = df["coiled_coil"].astype("boolean")
    df["disordered"] = df["disordered"].astype("boolean")
    df["basic"] = df["basic"].astype("boolean")
    df["acidic"] = df["acidic"].astype("boolean")
    df["polar"] = df["polar"].astype("boolean")
    df["pro"] = df["pro"].astype("boolean")
    df["p_rich"] = df["p_rich"].astype("boolean")
    df["g_rich"] = df["g_rich"].astype("boolean")

    return df


def remove_pfam_db_xref(df):
    # For the mouse C2H2 zinc finger protein, all pfam records seem to be included in NCBI records.
    return df.loc[~df["db_xref"].str.startswith("CDD:pfam")]


def remove_useless_regions(df):
    black_list = [
        "Region_of_interest_in_the_sequence",
        "Conflict",
        "Mature_chain",
        "Splicing_variant",
        "Short_sequence_motif_of_biological_interest",
        "Repetitive_region",
        "Variant",
        "Domain_1",
        "Domain_2",
        "Domain_3",
        "Domain_4",
    ]
    return df.loc[
        ~df["name"].isin(black_list)
        | df["zinc_finger"]
        | df["coiled_coil"]
        | df["disordered"]
        | df["basic"]
        | df["acidic"]
        | df["polar"]
        | df["pro"]
        | df["p_rich"]
        | df["g_rich"]
    ].reset_index(drop=True)


# TODO: basic, acidic, polar, pro can overlap other regions
def preference_based_overlap_removing(df):
    def cluster_first(df):
        if df.size == 0:
            return df
        cols = ["accession", "start", "end"]
        return (
            bf.cluster(df, cols=cols)
            .groupby("cluster")
            .first()
            .reset_index(drop=True)
            .drop(columns=["cluster_start", "cluster_end"])
        )

    cols = ["accession", "start", "end"]

    def split_df_by_preferences(df, preferences):
        if len(preferences) > 0:
            return split_df_by_preferences(
                df.loc[preferences[0], :], preferences[1:]
            ) + split_df_by_preferences(df.loc[~preferences[0], :], preferences[1:])
        return [df]

    df_preference_list = split_df_by_preferences(
        df,
        preferences=[
            df["structural_motif"],
            df["db_xref"] != "",
            df["zinc_finger"],
            df["coiled_coil"],
            df["disordered"],
        ],
    )

    df_non_overlap = cluster_first(df_preference_list[0])
    for df_block in df_preference_list[1:]:
        df_non_overlap = pd.concat(
            [
                df_non_overlap,
                cluster_first(
                    bf.setdiff(df_block, df_non_overlap, cols1=cols, cols2=cols)
                ),
            ]
        )

    return bf.sort_bedframe(df_non_overlap, reset_index=True, cols=cols)


def mutually_exclude(df):
    cols = ["accession", "start", "end"]
    counts = bf.count_overlaps(
        df,
        df,
        cols1=cols,
        cols2=cols,
        return_input=False,
    )
    return (counts == 1).all().item().item()


dfs = []
for record, seq in parse_ft("zf.ft", "zf.seq"):
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

    df = format_data_frame([parse_region(region, accession) for region in regions])
    df = remove_pfam_db_xref(df)
    df = remove_useless_regions(df)
    df = preference_based_overlap_removing(df)

    if not mutually_exclude(df):
        sys.stderr.write(df)
        sys.stderr.write("\n")

    dfs.append(df)

    white_list = [
        "Helical_region",
        "Beta-strand_region",
        "Hydrogen_bonded_turn",
        "UIM",
        "Propeptide",
        "bipartite_nuclear_targeting_sequence",
        "RNA-binding_motif_RGG",
        "alanine_repeat",
    ]
    white_mask = (
        ~df["structural_motif"]
        & (df["db_xref"] == "")
        & ~df["zinc_finger"]
        & ~df["coiled_coil"]
        & ~df["disordered"]
        & ~df["basic"]
        & ~df["acidic"]
        & ~df["polar"]
        & ~df["pro"]
        & ~df["p_rich"]
        & ~df["g_rich"]
    )
    stranges = set(df["name"].loc[white_mask]).difference(white_list)

    if len(stranges) > 0:
        print(stranges)

pd.concat(dfs).to_csv("protein.tsv", sep="\t", index=False)

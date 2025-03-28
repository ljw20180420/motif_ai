#!/usr/bin/env python

import os

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import re
import sys
import pandas as pd
import bioframe as bf
import subprocess


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


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


with subprocess.Popen(
    "cat zf.xml | xtract -insd complete Region INSDInterval_from INSDInterval_to region_name note db_xref",
    shell=True,
    stdout=subprocess.PIPE,
) as ft:
    df = pd.read_table(
        ft.stdout,
        names=[
            "accession",
            "start",
            "end",
            "point",
            "name",
            "note",
            "db_xref",
        ],
        sep="\t",
    )

point_mask = df["start"] == "-"
df["start"].loc[point_mask] = df["end"].loc[point_mask] = df["point"].loc[point_mask]

df["start"].astype("int")
breakpoint()
# 1-based闭区间到0-based开区间
df["start"] = df["start"] - 1

# 去掉accession后的版本号
strip_version = re.compile(r"\.\d+$")
df["accession"] = df["accession"].apply(
    lambda accession: strip_version.sub("", accession)
)

zinc_finger_re = re.compile(r"c2h2|h2c2|zn[ -_]finger|zinc[ -_]finger", re.IGNORECASE)
df["zinc_finger"] = df["name"].apply(
    lambda name: bool(zinc_finger_re.search(name))
) & df["note"].apply(lambda note: bool(zinc_finger_re.search(note)))
df["sec_struct"] = False
df["disordered"] = df["name"].apply(
    lambda name: name.lower().find("disordered.") != -1
) & df["note"].apply(lambda note: note.lower().find("disordered.") != -1)
df["structural_motif"] = df["name"].apply(
    lambda name: name.lower().find("[structural motif]") != -1
) & df["note"].apply(lambda note: note.lower().find("[structural motif]") != -1)


def check_consistent():
    strip_version = re.compile(r"\.\d+$")
    df_sec = pd.read_table("secondary_structure.tsv", sep="\t", header=0).rename(
        columns={"accession": "uniprotKBacc"}
    )
    with open("zf.acc") as rd_acc, open("zf.seq") as rd_seq:
        df_seq = pd.DataFrame(
            {
                "accession": [strip_version.sub("", acc.strip()) for acc in rd_acc],
                "sequence": [seq.strip() for seq in rd_seq],
            },
        )

    df_map = pd.concat(
        [
            pd.read_table(
                f"convert_any_accession_to_uniprotKB_accession/{map_file}",
                sep="\t",
                names=["accession", "uniprotKBacc"],
            )
            for map_file in os.listdir("convert_any_accession_to_uniprotKB_accession")
        ]
    )
    df_map["accession"] = df_map["accession"].apply(
        lambda accession: strip_version.sub("", accession)
    )
    df_map = df_map.merge(df_seq, on="accession", how="left")

    df_sec = df_sec.merge(df_map, on="uniprotKBacc", how="left")

    from Bio import Align

    uniprot_start, ncbi_start, mismatch, gap, the_same = [], [], [], [], []
    aligner = Align.PairwiseAligner(scoring="blastp", mode="local")
    for i in range(len(df_sec)):
        if df_sec.loc[i]["sequence_x"] != df_sec.loc[i]["sequence_y"]:
            alignments = aligner.align(
                df_sec.iloc[i]["sequence_x"], df_sec.iloc[i]["sequence_y"]
            )
            breakpoint()
            print(alignments[0])

    breakpoint()
    # .apply(
    #     lambda row: row["sequence_x"] != row["sequence_y"], axis=1
    # )


check_consistent()

# dfs = []
# for record, seq in parse_ft("zf.ft", "zf.seq"):
#     (
#         source,
#         accession,
#         name,
#         regions,
#         sites,
#         proteins,
#         CDSs,
#         genes,
#         bonds,
#         SecStrs,
#         Hets,
#     ) = parse_record(record)

#     df = format_data_frame([parse_region(region, accession) for region in regions])
#     df = remove_pfam_db_xref(df)
#     df = remove_useless_regions(df)
#     df = preference_based_overlap_removing(df)

#     if not mutually_exclude(df):
#         sys.stderr.write(df)
#         sys.stderr.write("\n")

#     dfs.append(df)

#     white_list = [
#         "Helical_region",
#         "Beta-strand_region",
#         "Hydrogen_bonded_turn",
#         "UIM",
#         "Propeptide",
#         "bipartite_nuclear_targeting_sequence",
#         "RNA-binding_motif_RGG",
#         "alanine_repeat",
#     ]
#     white_mask = (
#         ~df["structural_motif"]
#         & (df["db_xref"] == "")
#         & ~df["zinc_finger"]
#         & ~df["coiled_coil"]
#         & ~df["disordered"]
#         & ~df["basic"]
#         & ~df["acidic"]
#         & ~df["polar"]
#         & ~df["pro"]
#         & ~df["p_rich"]
#         & ~df["g_rich"]
#     )
#     stranges = set(df["name"].loc[white_mask]).difference(white_list)

#     if len(stranges) > 0:
#         print(stranges)

# pd.concat(dfs).to_csv("protein.tsv", sep="\t", index=False)

#!/usr/bin/env python

import os
import pathlib

import bioframe as bf
import pandas as pd

os.chdir(pathlib.Path(__file__).resolve().parent.parent)


def get_pcdh_exons() -> pd.DataFrame:
    df = (
        pd
        .read_csv(
            "AI/dataset/genome/mm9.refGene.gtf.gz",
            sep="\t",
            names=[
                "seqid",
                "source",
                "type",
                "start",
                "end",
                "score",
                "strand",
                "phase",
                "attributes",
            ],
        )
        .query(
            "seqid == 'chr18' and type == 'exon' and attributes.str.contains(pat=r'pcdh\D', case=False) and attributes.str.contains(pat='exon_number \"1\"')"
        )
        .reset_index(drop=True)
    )

    breakpoint()

    df = (
        df
        .assign(
            pcdh=lambda df: df["attributes"].str.extract(
                r"gene_id \"(Pcdh\w{1,2}\d{1,2})\""
            )[0]
        )[["seqid", "start", "end", "pcdh"]]
        .rename(columns={"seqid": "chrom"})
        .assign(start=lambda df: df["start"] - 1)
    )

    return df


df_exon = get_pcdh_exons()

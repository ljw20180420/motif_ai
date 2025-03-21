#!/bin/bash

# 获取脚本路径
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

nextflow -log $SCRIPT_DIR/logs/.nextflow.log run $SCRIPT_DIR/../../../pipelines/chipseq/dev \
    -work-dir $SCRIPT_DIR/.nextflow \
    -profile singularity \
    -resume \
    -params-file $SCRIPT_DIR/params.yaml \
    --input $SCRIPT_DIR/samplesheet.csv \
    --outdir $SCRIPT_DIR/results \
    --fasta $SRA_CACHE/ncbi_dataset/data/GCF_000001635.27/GCF_000001635.27_GRCm39_genomic.fna \
    --gtf $SRA_CACHE/ncbi_dataset/data/GCF_000001635.27/genomic.gtf

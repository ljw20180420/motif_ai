#!/bin/bash

# 获取脚本路径
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

nextflow \
    -log $SCRIPT_DIR/logs/.nextflow.log \
    -config $SCRIPT_DIR/nextflow.config \
    run $SCRIPT_DIR/../../../pipelines/chipseq/dev \
        -output-dir $SCRIPT_DIR/.nextflow \
        -work-dir $SCRIPT_DIR/work \
        -profile singularity \
        -resume \
        -with-trace \
        -params-file $SCRIPT_DIR/params.yaml \
            --input $SCRIPT_DIR/samplesheet.csv \
            --outdir $SCRIPT_DIR/results \
            --fasta $SRA_CACHE/ncbi_dataset/data/GCF_000001635.27/GCF_000001635.27_GRCm39_genomic.fna \
            --gtf $SRA_CACHE/ncbi_dataset/data/GCF_000001635.27/genomic.gtf

if ! test -d $SCRIPT_DIR/results/bowtie2/merged_library/macs3
then
    for bam in $(find $SCRIPT_DIR/results/bowtie2/merged_library -name "*.sorted.bam")
    do
        macs3 callpeak --gsize 2495461690 --format BAMPE --outdir $SCRIPT_DIR/results/bowtie2/merged_library/macs3/narrowPeak/$(basename $bam) --name $(basename $bam) --treatment $bam
    done
fi
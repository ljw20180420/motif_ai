#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

nextflow \
    -log logs/.nextflow.log \
    run ../../../pipelines/chipseq/dev \
        -profile singularity \
        -resume \
        -with-trace \
        -params-file params.yaml \
            --fasta $SRA_CACHE/ncbi_dataset/data/GCF_000001635.27/GCF_000001635.27_GRCm39_genomic.fna \
            --gtf $SRA_CACHE/ncbi_dataset/data/GCF_000001635.27/genomic.gtf \
            --bowtie2_index $SRA_CACHE/ncbi_dataset/data/GCF_000001635.27/index/bowtie2

if ! test -d results/bowtie2/merged_library/macs3
then
    for bam in $(find results/bowtie2/merged_library -name "*.sorted.bam")
    do
        macs3 callpeak --gsize 2495461690 --format BAMPE --outdir results/bowtie2/merged_library/macs3/narrowPeak/$(basename $bam) --name $(basename $bam) --treatment $bam
    done
fi
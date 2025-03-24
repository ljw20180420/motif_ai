#!/bin/bash

# 获取脚本路径
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 下载小鼠最新基因组GRCm39
# $SCRIPT_DIR/../database/datasets.sh

# 下载sra原始数据
# $SCRIPT_DIR/../database/sra-tools.sh $SCRIPT_DIR/SraAccList.txt

# 提取fastq文件
# $SCRIPT_DIR/../fasterq_dump.sh $SCRIPT_DIR/SraAccList.txt

# 压缩fastq文件
# for fastq in $(find $SRA_CACHE/sra -name "*.fastq")
# do
#     gzip $fastq
# done

# 运行nextflow的chipseq pipeline
# TODO: 利用--skip选项略过不必要的步骤
# nextflow/runs/chipseq/test/run.sh

# 获取函数
. $SCRIPT_DIR/../AI_models/bind_transformer/preprocess.sh

# # 合并所有narrowPeak
# narrowPeak_files=$(find $SCRIPT_DIR/../nextflow/runs/chipseq/test/results/bowtie2/merged_library/macs3/narrowPeak -name "*.narrowPeak")
# cat $narrowPeak_files |
# bedtools sort |
# bedtools merge \
#     > $SCRIPT_DIR/peaks.bed    

# # 提取peak的序列
# # The flag -U/--update-faidx is recommended to ensure the .fai file matches the FASTA file.
# seqkit subseq \
#     < $SRA_CACHE/ncbi_dataset/data/GCF_000001635.27/GCF_000001635.27_GRCm39_genomic.fna \
#     --update-faidx \
#     --bed $SCRIPT_DIR/peaks.bed \
#     --up-stream 50 \
#     --down-stream 50 \
#     > $SCRIPT_DIR/peaks.fa

# # 预测motif
# streme \
#     --text \
#     --thres 0.05 \
#     --nmotifs 1 \
#     --minw 10 \
#     --maxw 20 \
#     --p $SCRIPT_DIR/peaks.fa \
#     > $SCRIPT_DIR/meme.txt
# meme2images -png $SCRIPT_DIR/meme.txt $SCRIPT_DIR/images


# # 搜索motif，并根据搜索结果选择最好的motif
# # 最多搜索nmotifs个motif
# fimo \
#     --best-site \
#     --thresh 1 \
#     --no-qvalue \
#     --max-strand \
#     --max-stored-scores 999999999 \
#     $SCRIPT_DIR/meme.txt \
#     $SCRIPT_DIR/peaks.fa \
#     > $SCRIPT_DIR/motifs.bed

# 提取结合位点序列
sed -r 's/^([^\t]+)_([0-9]+)-([0-9]+)/\1\t\2\t\3/' \
    < $SCRIPT_DIR/motifs.bed |
awk -F '\t' -v OFS='\t' '{print $1, $2 + $4, $2 + $5, "*", $9, $8}' >$SCRIPT_DIR/motifs.global.bed
seq_len=32
motif_len=$(awk '{print $3 - $2; exit}' $SCRIPT_DIR/motifs.global.bed)
if [ $seq_len -lt $motif_len ]
then
    echo "ERROR: motif length is larger than seq length"
    exit 1
fi
TOTAL_EXT=$((seq_len - motif_len))
UP_EXT=$((TOTAL_EXT / 2))
DOWN_EXT=$((TOTAL_EXT - UP_EXT))
seqkit subseq \
    < $SRA_CACHE/ncbi_dataset/data/GCF_000001635.27/GCF_000001635.27_GRCm39_genomic.fna \
    --update-faidx \
    --bed $SCRIPT_DIR/motifs.global.bed \
    --up-stream $UP_EXT \
    --down-stream $DOWN_EXT |
seqkit fx2tab | cut -f2 | dd conv=ucase \
    > $SCRIPT_DIR/positive.txt
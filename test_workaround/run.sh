#!/bin/bash

# # 提取peak的序列
# # The flag -U/--update-faidx is recommended to ensure the .fai file matches the FASTA file.
# for narrowPeak in $(ls $DATA_DIR/*.final.narrowPeak)
# do
#     accession=$(basename ${narrowPeak%%.*})
#     seqkit subseq \
#         < $GENOME \
#         --update-faidx \
#         --bed $narrowPeak \
#         --up-stream 50 \
#         --down-stream 50 \
#         > $DATA_DIR/$accession.fasta
# done

# 预测motif
for fasta in $(ls $DATA_DIR/*.fasta)
do
    accession=$(basename ${fasta%.*})
    streme \
        --text \
        --thres 0.05 \
        --nmotifs 1 \
        --minw 10 \
        --maxw 30 \
        --p $DATA_DIR/$accession.fasta \
        > $DATA_DIR/$accession.meme
    meme2images -png $DATA_DIR/$accession.meme $DATA_DIR/$accession.png
done

# # 搜索motif，并根据搜索结果选择最好的motif
# for meme in $(ls $DATA_DIR/*.meme)
# do
#     accession=$(basename ${meme%.*})
#     fimo \
#         --best-site \
#         --thresh 1 \
#         --no-qvalue \
#         --max-strand \
#         --max-stored-scores 999999999 \
#         $DATA_DIR/$accession.meme \
#         $DATA_DIR/$accession.fasta \
#         > $DATA_DIR/$accession.bed
# done

# # 提取结合位点序列
# for bed in $(ls $DATA_DIR/*.bed)
# do
#     accession=$(basename ${bed%.*})
#     sed -r 's/^([^\t]+)_([0-9]+)-([0-9]+)/\1\t\2\t\3/' \
#     < $DATA_DIR/$accession.bed |
#     awk -F '\t' -v OFS='\t' '{print $1, $2 + $4, $2 + $5, "*", $9, $8}' > $DATA_DIR/$accession.global.bed
#     seq_len=100
#     motif_len=$(awk '{print $3 - $2; exit}' $DATA_DIR/$accession.global.bed)
#     if [ $seq_len -lt $motif_len ]
#     then
#         echo "ERROR: motif length is larger than seq length"
#         exit 1
#     fi
#     TOTAL_EXT=$((seq_len - motif_len))
#     UP_EXT=$((TOTAL_EXT / 2))
#     DOWN_EXT=$((TOTAL_EXT - UP_EXT))
#     seqkit subseq \
#         < $GENOME \
#         --update-faidx \
#         --bed $DATA_DIR/$accession.global.bed \
#         --up-stream $UP_EXT \
#         --down-stream $DOWN_EXT |
#     seqkit fx2tab | cut -f2 | dd conv=ucase \
#         > $DATA_DIR/$accession.positive
# done


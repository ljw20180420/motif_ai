#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 使用严格模式
set -euo pipefail

# 收集所有accession
accessions=()
for narrowPeak in $(ls $DATA_DIR/sorted/*.sorted.narrowPeak)
do
    accession=$(basename ${narrowPeak%%.*})
    accessions+=($accession)
done

# 准备DeepZF输入文件
mkdir -p DeepZF/zinc_finger_protein_motifs
printf "label,seq,res_12,groups\n" > DeepZF/zinc_finger_protein_motifs/DeepZF_input.csv
ext=40
for accession in "${accessions[@]}"
do
    awk -F "\t" -v accession="$accession" -v ext=$ext '
        $1 == accession {
            seq = $4
            while (1) {
                match(seq, /..C(..|.{4})C.{12}H.{3,5}H/)
                if (RLENGTH == -1) {
                    break
                }
                zinc = substr(seq, RSTART, RLENGTH)
                printf("0.0,%s,%s,%s\n", substr(seq, RSTART - ext, RLENGTH + 2 * ext), gensub(/..C(..|.{4})C(.{12})H.{3,5}H/, "\\2", "g", zinc), accession)
                seq = substr(seq, RSTART + RLENGTH)
            }
        }
    ' \
        ../preprocess/protein.tsv \
        >> DeepZF/zinc_finger_protein_motifs/DeepZF_input.csv
done

# 计算锌指结合概率
DeepZF/.conda/bin/python DeepZF/BindZF_predictor/code/main_bindzfpredictor_predict.py -in DeepZF/zinc_finger_protein_motifs/DeepZF_input.csv -out DeepZF/zinc_finger_protein_motifs/BindZF_output.txt -m DeepZF/BindZF_predictor/code/model.p -e DeepZF/BindZF_predictor/code/encoder.p -r 1

# 计算锌指PWM
DeepZF/.conda/bin/python DeepZF/PWMpredictor/code/main_PWMpredictor.py -in DeepZF/zinc_finger_protein_motifs/DeepZF_input.csv -out DeepZF/zinc_finger_protein_motifs/PWM_output.txt -m DeepZF/PWMpredictor/code/transfer_model100.h5

# 合并结果
paste \
    <(
        tail -n+2 DeepZF/zinc_finger_protein_motifs/DeepZF_input.csv |
        cut -d',' -f4
    ) \
    DeepZF/zinc_finger_protein_motifs/BindZF_output.txt \
    <(
        sed -nr 'N;N;N;N;N;N;N;N;N;N;N;s/\n/\t/g;p' \
            DeepZF/zinc_finger_protein_motifs/PWM_output.txt
    ) \
    > DeepZF/zinc_finger_protein_motifs/merge_output.tsv

# 产生meme基序
mkdir -p DeepZF/zinc_finger_protein_motifs/motifs
thres=0.5
for accession in "${accessions[@]}"
do
    awk -v accession=$accession -v thres=$thres '
        $1 == accession && $2 > thres {
            for (i=0; i<3; ++i) {
                printf("%s\t%s\t%s\t%s\n", $(i * 4 + 3), $(i * 4 + 4), $(i * 4 + 5), $(i * 4 + 6))
            }
        }
    ' DeepZF/zinc_finger_protein_motifs/merge_output.tsv |
    matrix2meme -dna |
    sed -r "s/^MOTIF .+ .+$/MOTIF $accession/" \
        > DeepZF/zinc_finger_protein_motifs/motifs/$accession.meme
    if [ "$(wc -l < DeepZF/zinc_finger_protein_motifs/motifs/$accession.meme)" -eq 0 ]
    then
        awk -v accession=$accession '
            $1 == accession && $2 > max_bind {
                max_bind = $2
                max_line = $0
            }
            END {
                split(max_line, arr)
                for (i=0; i<3; ++i) {
                    printf("%s\t%s\t%s\t%s\n", arr[i * 4 + 3], arr[i * 4 + 4], arr[i * 4 + 5], arr[i * 4 + 6])
                }
            }
        ' DeepZF/zinc_finger_protein_motifs/merge_output.tsv |
        matrix2meme -dna |
        sed -r "s/^MOTIF .+ .+$/MOTIF $accession/" \
            > DeepZF/zinc_finger_protein_motifs/motifs/$accession.meme
    fi
done

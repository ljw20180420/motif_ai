#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 收集所有accession
accessions=()
for narrowPeak in $(ls $DATA_DIR/sorted/*.sorted.narrowPeak)
do
    accession=$(basename ${narrowPeak%%.*})
    accessions+=($accession)
done

# 准备DeepZF输入文件
mkdir -p $DATA_DIR/DeepZF
printf "label,seq,res_12,groups\n" > $DATA_DIR/DeepZF/DeepZF_input.csv
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
        >> $DATA_DIR/DeepZF/DeepZF_input.csv
done

# 计算锌指结合概率
DeepZF/.conda/bin/python DeepZF/BindZF_predictor/code/main_bindzfpredictor_predict.py -in $DATA_DIR/DeepZF/DeepZF_input.csv -out $DATA_DIR/DeepZF/BindZF_output.txt -m DeepZF/BindZF_predictor/code/model.p -e DeepZF/BindZF_predictor/code/encoder.p -r 1

# 计算锌指PWM
DeepZF/.conda/bin/python DeepZF/PWMpredictor/code/main_PWMpredictor.py -in $DATA_DIR/DeepZF/DeepZF_input.csv -out $DATA_DIR/DeepZF/PWM_output.txt -m DeepZF/PWMpredictor/code/transfer_model100.h5

# 合并结果
paste \
    <(
        tail -n+2 $DATA_DIR/DeepZF/DeepZF_input.csv |
        cut -d',' -f4
    ) \
    $DATA_DIR/DeepZF/BindZF_output.txt \
    <(
        sed -nr 'N;N;N;N;N;N;N;N;N;N;N;s/\n/\t/g;p' \
            $DATA_DIR/DeepZF/PWM_output.txt
    ) \
    > $DATA_DIR/DeepZF/merge_output.tsv

# 产生meme基序
mkdir -p $DATA_DIR/DeepZF/motifs
thres=0.5
for accession in "${accessions[@]}"
do
    awk -v accession=$accession -v thres=$thres '
        $1 == accession && $2 > thres {
            for (i=0; i<3; ++i) {
                printf("%s\t%s\t%s\t%s\n", $(i * 4 + 3), $(i * 4 + 4), $(i * 4 + 5), $(i * 4 + 6))
            }
        }
    ' $DATA_DIR/DeepZF/merge_output.tsv |
    matrix2meme -dna |
    sed -r "s/^MOTIF .+ .+$/MOTIF $accession/" \
        > $DATA_DIR/DeepZF/motifs/$accession.meme
    if [ "$(wc -l < $DATA_DIR/DeepZF/motifs/$accession.meme)" -eq 0 ]
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
        ' $DATA_DIR/DeepZF/merge_output.tsv |
        matrix2meme -dna |
        sed -r "s/^MOTIF .+ .+$/MOTIF $accession/" \
            > $DATA_DIR/DeepZF/motifs/$accession.meme
    fi
done

# 搜索motif
mkdir -p $DATA_DIR/DeepZF/sites
shear_DNA_len=100
for accession in "${accessions[@]}"
do
    cut -d, -f2 \
        $BIND_TRANSFORMER_DATA_DIR/DNA_data/$accession.csv |
    awk -v shear_DNA_len=$shear_DNA_len '
        {
            print substr($0, int((length($0) - shear_DNA_len) / 2) + 1, shear_DNA_len)
        }
    ' |
    sed '=' |
    sed "1~2s/^/>$accession/" \
        > $DATA_DIR/DeepZF/tempfile
    fimo \
        --best-site \
        --thresh 1 \
        --no-qvalue \
        --max-strand \
        --max-stored-scores 99999999 \
        $DATA_DIR/DeepZF/motifs/$accession.meme \
        $DATA_DIR/DeepZF/tempfile |
    sort -k1,1V \
        > $DATA_DIR/DeepZF/sites/$accession.site
done
rm $DATA_DIR/DeepZF/tempfile

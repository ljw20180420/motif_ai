#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 使用严格模式
# set -euo pipefail
set -eu

title() {
    printf "\n----------\n%s\n----------\n" "$1" >&2
}

download_mm9() {
    title "download mm9"
    mkdir -p genome
    pushd genome
    wget https://github.com/Boyle-Lab/Blacklist/raw/refs/heads/master/lists/Blacklist_v1/mm9-blacklist.bed.gz
    gzip -d mm9-blacklist.bed.gz
    wget https://hgdownload.cse.ucsc.edu/goldenpath/mm9/bigZips/mm9.chrom.sizes
    wget https://hgdownload.cse.ucsc.edu/goldenpath/mm9/bigZips/mm9.2bit
    twoBitToFa mm9.2bit mm9.fa
    wget https://hgdownload.gi.ucsc.edu/goldenPath/mm9/bigZips/genes/mm9.refGene.gtf.gz
    popd
}

download_uniprot_C2H2_protein_table() {
    title "download uniprot C2H2 protein table"
    ./scripts/download_uniprot_C2H2_protein_table.py \
        'ft_zn_fing:C2H2' \
        'organism_name:"Mus musculus"'
}

download_alphafoldDB_mmcif() {
    title "download alphafoldDB mmcif"
    ./scripts/download_alphafoldDB_mmcif.py
}

infer_secondary_structure() {
    title "infer secondary structure"
    printf "accession,sequence,secondary_structure\n" >secondary_structure.csv
    local mmcif
    local stem
    for mmcif in $(find alphafoldDB_mmcif/ -name "*.mmcif")
    do
        stem=${mmcif##*/}
        stem=${stem%.mmcif}
        printf "%s," ${stem} >> secondary_structure.csv
        mkdssp --output-format dssp $mmcif | sed '1,/^  #/d' | cut  -c14 | tr -d '\n' >> secondary_structure.csv
        printf "," >> secondary_structure.csv
        mkdssp --output-format dssp $mmcif | sed '1,/^  #/d' | cut  -c17 | tr -d '\n' | tr ' ' '-' >> secondary_structure.csv
        printf "\n" >> secondary_structure.csv
    done
}

parse_protein_feature() {
    title "parse protein feature"
    ./scripts/parse_protein_feature.py
}

collect_accession() {
    local -n ref_accessions="$1"
    local narrowPeak
    local accession
    for narrowPeak in $(ls $DATA_DIR/sorted/*.sorted.narrowPeak)
    do
        accession=$(basename ${narrowPeak%%.*})
        ref_accessions+=($accession)
    done
}

clean_sorted_peak() {
    title "clean sorted peak"
    local accessions=()
    collect_accession accessions
    local accession
    for accession in "${accessions[@]}"
    do
        printf "clean sorted narrowPeak for %s\n" $accession
        awk '
            NF == 10 {
                print
            }
        ' $DATA_DIR/sorted/$accession.sorted.narrowPeak \
        > $DATA_DIR/sorted/$accession.sorted.narrowPeak2
        mv $DATA_DIR/sorted/$accession.sorted.narrowPeak2 $DATA_DIR/sorted/$accession.sorted.narrowPeak
    done
}

split_by_SRR() {
    title "split by SRR"
    local accessions=()
    collect_accession accessions
    local accession
    for accession in "${accessions[@]}"
    do 
        printf "split by SRR for %s\n" $accession
        mkdir -p ${DATA_DIR}/splited/${accession}
        gawk -v accession=${accession} -f scripts/split_by_srr.awk -- ${DATA_DIR}/sorted/${accession}.sorted.narrowPeak
    done
}

assess_SRR() {
    title "assess SRR"
    scripts/assess_SRR.py
}

select_SRR() {
    title "select SRR"
    scripts/select_SRR.py
}

assess_peak_width() {
    title "assess peak width"
    scripts/assess_peak_width.py
}

filter_peak_by_width() {
    title "filter peak by width"
    local accessions=()
    collect_accession accessions
    mkdir -p $DATA_DIR/filtered
    local width_upper_bound=$1
    local accession
    for accession in "${accessions[@]}"
    do
        if [ -f "$DATA_DIR/filtered/$accession.sorted.narrowPeak" ]
        then
            continue
        fi
        printf "filtered peak for %s\n" $accession
        awk -v width_upper_bound=${width_upper_bound} '
            $3 - $2 <= width_upper_bound {print}
        ' $DATA_DIR/single/$accession.sorted.narrowPeak \
            > $DATA_DIR/filtered/$accession.sorted.narrowPeak
    done
}

remove_black_peak() {
    title "remove black peak"
    local accessions=()
    collect_accession accessions
    mkdir -p $DATA_DIR/white
    local accession
    for accession in "${accessions[@]}"
    do
        if [ -f "$DATA_DIR/white/$accession.sorted.narrowPeak" ]
        then
            continue
        fi
        printf "remove black peak for %s\n" $accession
        bedtools intersect -sorted -v \
            -a $DATA_DIR/filtered/$accession.sorted.narrowPeak \
            -b <(
                bedtools sort -i genome/mm9-blacklist.bed
            ) \
            > $DATA_DIR/white/$accession.sorted.narrowPeak
    done
}

resize_peak_and_sort_by_summit() {
    title "resize peak and sort by summit"
    local accessions=()
    collect_accession accessions
    mkdir -p $DATA_DIR/sized
    local seq_len=256
    local accession
    for accession in "${accessions[@]}"
    do
        if [ -f "$DATA_DIR/sized/$accession.sized.narrowPeak" ]
        then
            continue
        fi
        printf "resize peak for %s\n" $accession
        bedClip \
            <(
                awk -v seq_len=$seq_len '
                    {
                        start = $2
                        end = $3
                        summit = $10
                        new_start = start + summit - int(seq_len / (end - start) * summit)
                        new_end = new_start + seq_len
                        new_summit = start + summit
                        printf("%s\t%d\t%d\t%d\n", $1, new_start, new_end, new_summit)
                    }
                ' $DATA_DIR/white/$accession.sorted.narrowPeak |
                sort -k1,1 -k4,4n
            ) \
            genome/mm9.chrom.sizes \
            $DATA_DIR/sized/$accession.sized.narrowPeak
    done
}

extract_peak_site_sequence() {
    title "extract peak site sequence"
    local accessions=()
    collect_accession accessions
    mkdir -p $DATA_DIR/positive
    local accession
    for accession in "${accessions[@]}"
    do
        if [ -f "$DATA_DIR/positive/$accession.positive" ]
        then
            continue
        fi
        printf "extract peak sequence for %s\n" $accession
        # --line-width 0 防止fasta换行
        paste \
            $DATA_DIR/sized/$accession.sized.narrowPeak \
            <(
                seqkit subseq \
                    < genome/mm9.fa \
                    --update-faidx \
                    --line-width 0 \
                    --bed $DATA_DIR/sized/$accession.sized.narrowPeak |
                sed -nr '2~2{y/acgtn/ACGTN/; p}'
            ) |
        grep -vE "\sN|[ACGT]N" \
            > $DATA_DIR/positive/$accession.positive
    done
}

get_protein_pairwise_closest_peak_distance() {
    title "get protein pairwise closest peak distance"
    local accessions=()
    collect_accession accessions
    mkdir -p $DATA_DIR/train_data
    local dis_files
    local accession
    local accession2
    for accession in "${accessions[@]}"
    do
        if [ -f "$DATA_DIR/train_data/${accession}.csv" ]
        then
            continue
        fi
        printf "calculate the closest peak from other proteins for %s\n" ${accession}
        dis_files=()
        for accession2 in "${accessions[@]}"
        do
            if [ "${accession2}" != "${accession}" ]
            then
                bedtools closest -sorted -d -t first \
                    -a <(
                        awk '
                            {
                                printf("%s\t%s\t%s\n", $1, $4, $4 + 1)
                            }
                        ' $DATA_DIR/positive/${accession}.positive
                    ) \
                    -b <(
                        awk '
                            {
                                printf("%s\t%s\t%s\n", $1, $4, $4 + 1)
                            }
                        ' $DATA_DIR/positive/${accession2}.positive
                    ) |
                awk -F $'\t' -v accession=${accession2} '
                    BEGIN{print accession}
                    {print $7}
                ' > $DATA_DIR/train_data/${accession}_${accession2}
            else
                awk -v accession=${accession2} '
                    BEGIN{print accession}
                    {print 0}
                ' $DATA_DIR/positive/${accession}.positive \
                    > $DATA_DIR/train_data/${accession}_${accession2}
            fi
            dis_files+=($DATA_DIR/train_data/${accession}_${accession2})
        done
        paste -d, \
            <(
                awk -v accession=${accession} '
                    BEGIN{
                        printf("protein,DNA\n")
                    }
                    {
                        printf("%s,%s\n", accession, $5)
                    }
                ' "$DATA_DIR/positive/${accession}.positive"
            ) "${dis_files[@]}" \
            > "$DATA_DIR/train_data/${accession}.csv"
        rm "${dis_files[@]}"
    done
}

balance_data() {
    title "balance data"
    local minimal_unbind_summit_distance=300
    local seed=63036
    scripts/balance_data.py ${minimal_unbind_summit_distance} ${seed}
}

split_data() {
    title "split data"
    local validation_ratio=0.05
    local test_ratio=0.05
    local seed=63036
    scripts/split_data.py ${validation_ratio} ${test_ratio} ${seed}
}

random_DNA()
{
    local length=$1
    local chars="ACGT"
    local str=""
    for ((i = 0; i < ${length}; ++i)); do
        str+=${chars:RANDOM%${#chars}:1}
    done
    printf $str
}

generate_inference_data() {
    local number=100
    local seq_len=256
    printf "DNA,protein\n" > inference_data.csv
    paste -d, \
        <(
            for (( i=0; i<${number}; ++i ))
            do
                echo $(random_DNA ${seq_len})
            done
        ) \
        <(
            tail -n+2 \
                protein_feature.csv |
            cut -d, -f1 | shuf -n ${number} -r
        ) \
        >> inference_data.csv
}

generate_unittest_data() {
    scripts/generate_unittest_data.py
}

# download_mm9

# download_uniprot_C2H2_protein_table

# download_alphafoldDB_mmcif

# infer_secondary_structure

# parse_protein_feature

# clean_sorted_peak

# split_by_SRR

# assess_SRR

# select_SRR

# assess_peak_width

# filter_peak_by_width 1300

# remove_black_peak

# resize_peak_and_sort_by_summit

# extract_peak_site_sequence

# get_protein_pairwise_closest_peak_distance

balance_data

# split_data

# generate_inference_data

# generate_unittest_data

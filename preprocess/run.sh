#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

title() {
    printf "\n----------\n%s\n----------\n" $1 >&2
}

# title "下载蛋白文件"
# ./uniprot_download.py \
#     'ft_zn_fing:C2H2' \
#     'organism_name:"Mus musculus"' \
#     3> uniprot_mouse_C2H2_protein.tsv

# title "下载蛋白结构"
# ./get_mmcif_from_alphafoldDB.py \
#     < uniprot_mouse_C2H2_protein.tsv

# title "计算蛋白二级结构"
# # dssp需要libcifpp: sudo apt install libcifpp-dev
# # dssp需要更新libcifpp的资源文件: https://github.com/PDB-REDO/dssp/issues/3
# # curl -o /var/cache/libcifpp/components.cif https://files.wwpdb.org/pub/pdb/data/monomers/components.cif
# # curl -o /var/cache/libcifpp/mmcif_pdbx.dic https://mmcif.wwpdb.org/dictionaries/ascii/mmcif_pdbx_v50.dic
# # curl -o /var/cache/libcifpp/mmcif_ma.dic https://github.com/ihmwg/ModelCIF/raw/master/dist/mmcif_ma.dic
# ./dssp.py \
#     3> secondary_structure.tsv

# title "去掉没有结构的蛋白"
# # 去掉uniprot和alphafoldDB蛋白长度不相同的蛋白
# # 在原来9种二级结构（包括没有结构）的基础上，标注KRAB和锌指蛋白结构
# ./parse_ft.py \
#     < uniprot_mouse_C2H2_protein.tsv \
#     4< secondary_structure.tsv \
#     3> protein.tsv

title "收集所有accession"
accessions=()
for narrowPeak in $(ls $DATA_DIR/sorted/*.sorted.narrowPeak)
do
    accession=$(basename ${narrowPeak%%.*})
    # 如果accession蛋白没有二级结构
    if ! grep -F $accession protein.tsv > /dev/null
    then
        printf "%s没有二级结构, 不处理\n" $accession >&2
        continue
    fi
    accessions+=($accession)
done

title "把黑名单的peak去掉|把peak聚类"
mkdir -p $DATA_DIR/clustered
cluster_max_distance="-50"
for accession in "${accessions[@]}"
do
    bedtools intersect -v \
        -a $DATA_DIR/sorted/$accession.sorted.narrowPeak \
        -b $GENOME_BLACK |
    bedtools cluster \
        -d $cluster_max_distance \
        > $DATA_DIR/clustered/$accession.clustered.narrowPeak
done

title "每个聚类选择好的分位数|不选择最大值|防止异常值"
mkdir -p $DATA_DIR/selected
cluster_quantile=0.9
for accession in "${accessions[@]}"
do
    ./peak_cluster_select.py \
        < $DATA_DIR/clustered/$accession.clustered.narrowPeak \
        $cluster_quantile \
        > $DATA_DIR/selected/$accession.selected.narrowPeak
done

title "去掉太宽太窄的峰|去掉太显著太不显著的峰"
mkdir -p $DATA_DIR/filtered
for accession in "${accessions[@]}"
do
    wlb=$(
        awk '{print $3 - $2}' \
            < $DATA_DIR/selected/$accession.selected.narrowPeak |
        sort -n |
        perl -e '$d=0.1;@l=<>;print $l[int($d*$#l)]'
    )
    wub=$(
        awk '{print $3 - $2}' \
            < $DATA_DIR/selected/$accession.selected.narrowPeak |
        sort -n |
        perl -e '$d=0.9;@l=<>;print $l[int($d*$#l)]'
    )
    plb=$(
        awk '{print $8}' \
            < $DATA_DIR/selected/$accession.selected.narrowPeak |
        sort -g |
        perl -e '$d=0.1;@l=<>;print $l[int($d*$#l)]'
    )
    pub=$(
        awk '{print $8}' \
            < $DATA_DIR/selected/$accession.selected.narrowPeak |
        sort -g |
        perl -e '$d=0.9;@l=<>;print $l[int($d*$#l)]'
    )
    awk -v wlb=$wlb -v wub=$wub -v plb=$plb -v pub=$pub '$3 - $2 <= wub && $3 - $2 >= wlb && $8 >= plb && $8 <= pub {print}' \
        < $DATA_DIR/selected/$accession.selected.narrowPeak \
        > $DATA_DIR/filtered/$accession.filtered.narrowPeak
done

title "调整peak大小"
mkdir -p $DATA_DIR/sized
seq_len=256
for accession in "${accessions[@]}"
do
    bedClip \
        <(
            awk -v seq_len=$seq_len '
                {
                    start = $2
                    end = $3
                    summit = $10
                    new_start = start + summit - int(seq_len / (end - start) * summit)
                    new_end = new_start + seq_len
                    printf("%s\t%d\t%d\n", $1, new_start, new_end)
                }
            ' $DATA_DIR/filtered/$accession.filtered.narrowPeak
        ) \
        $GENOME_SIZE \
        $DATA_DIR/sized/$accession.sized.narrowPeak
done

title "提取结合位点序列"
mkdir -p $DATA_DIR/positive
for accession in "${accessions[@]}"
do
    # --line-width 0 防止fasta换行
    seqkit subseq \
        < $GENOME \
        --update-faidx \
        --line-width 0 \
        --bed $DATA_DIR/sized/$accession.sized.narrowPeak |
    sed '2~2y/acgtn/ACGTN/' |
    sed -nr 'N;s/\n/\t/;p' |
    grep -vE "\sN|[ACGT]N" |
    sed -r 's/\t/\n/'\
        > $DATA_DIR/positive/$accession.positive
done

title "产生不结合位点序列"
mkdir -p $DATA_DIR/negative
for accession in "${accessions[@]}"
do
    # -line 999999999 防止fasta换行
    fasta-shuffle-letters \
        -kmer 1 \
        -dna \
        -line 999999999 \
        -seed 63036 \
        $DATA_DIR/positive/$accession.positive \
        $DATA_DIR/negative/$accession.negative
done

title "生成训练数据集的蛋白"
mkdir -p $DATA_DIR/train_data
>$DATA_DIR/train_data/protein_data.csv
for ((i=0;i<${#accessions[@]};++i))
do
    grep -F "${accessions[$i]}" \
        protein.tsv |
    cut -f1,4,5 |
    tr '\n\t' ',' \
        >> $DATA_DIR/train_data/protein_data.csv

    # 得到每个蛋白的锌指数量
    zinc_num=$(
        grep -F "${accessions[$i]}" \
            < uniprot_mouse_C2H2_protein.tsv |
            sed -r 's/^.+note="C2H2-type ([0-9]+)".+$/\1/' |
            sed -r '/^[^1-9]/s/^.+$/1/'
    )
    printf "%d\n" $zinc_num >> $DATA_DIR/train_data/protein_data.csv
done

title "生成训练数据集的DNA"
mkdir -p $DATA_DIR/train_data/DNA_data
for ((i=0;i<${#accessions[@]};++i))
do
    (
        sed -nr '2~2{s/$/,1.0/;p}' \
            "$DATA_DIR/positive/${accessions[$i]}.positive"
        sed -nr '2~2{s/$/,0.0/;p}' \
            "$DATA_DIR/negative/${accessions[$i]}.negative"
    ) |
    sed -r "s/^/$i,/" \
        > "$DATA_DIR/train_data/DNA_data/${accessions[$i]}.csv"
done

title "生成小训练数据集"
get_seeded_random()
{
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

mkdir -p $DATA_DIR/small_train_data/DNA_data
cp $DATA_DIR/train_data/protein_data.csv $DATA_DIR/small_train_data/protein_data.csv
small_line_num=3000
for accession in "${accessions[@]}"
do
    shuf -n $small_line_num \
        --random-source=<(get_seeded_random 63036) \
        $DATA_DIR/train_data/DNA_data/${accession}.csv \
        > $DATA_DIR/small_train_data/DNA_data/${accession}.csv
done

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

# 把基因名字映射到uniprot accession
grep -E "\sC2H2 ZF\s" TF_Information.txt |
cut -f7 |
../uniprot_gene_to_id.py \
    "Gene_Name" \
    "UniProtKB" \
    10090 \
    > ids.map

# 把下了数据的蛋白的motif提取出来
mkdir -p c2h2_motifs
while read gene_name accession
do
    read filename < <(
        grep -E "\s$gene_name\s" TF_Information.txt |
        head -n1 |
        cut -f4
    )
    if [ "$filename" = "." ]
    then
        continue
    fi
    cut -f2-5 pwms_all_motifs/$filename.txt |
    tail -n+2 |
    matrix2meme |
    sed -r "s/^MOTIF 1 .+$/MOTIF $accession $gene_name/" \
        > c2h2_motifs/CIS-BP_$accession.meme
    if [ ! -s "c2h2_motifs/CIS-BP_$accession.meme" ]
    then
        rm "c2h2_motifs/CIS-BP_$accession.meme"
    fi
done < <(
    grep \
        -f <(printf "%s\n" "${accessions[@]}") \
        ids.map
)

# 把下了数据的蛋白的motif合并在一起
meme2meme $(ls c2h2_motifs/CIS-BP_*.meme) \
    > CIS-BP_motifs.dbs

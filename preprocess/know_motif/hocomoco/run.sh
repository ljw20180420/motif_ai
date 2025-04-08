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

# 把下了数据的蛋白的motif提取出来
mkdir -p c2h2_motifs
while read uniprot_ac uniprot_id gene_symbol gene_synonyms
do
    gene_names=(${uniprot_id%_*} $gene_symbol $gene_synonyms)
    for gene_name in "${gene_names[@]}"
    do
        id=$(grep -i "^MOTIF $gene_name.H13CORE.0." H13CORE_meme_format.meme | sed 's/^MOTIF //')
        if [ -n "$id" ]
        then
            meme-get-motif -id $id H13CORE_meme_format.meme |
            sed -r "s/^MOTIF (.+)\.H13CORE\.0\.(.+)$/MOTIF $uniprot_ac \1/" \
                > c2h2_motifs/hocomoco_$uniprot_ac.meme
            break
        fi
    done
done < <(
    grep \
        -f <(printf "%s\n" "${accessions[@]}") \
        tf_masterlist.tsv |
    cut -d $'\t' -f 6,8,10,11
)

# 把下了数据的蛋白的motif合并在一起
meme2meme $(ls c2h2_motifs/hocomoco_*.meme) \
    > hocomoco_motifs.dbs
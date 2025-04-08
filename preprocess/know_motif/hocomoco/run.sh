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

# 把uniprot accession映射到基因名字
gene_names=()
while read accession gene_name
do
    gene_names+=($gene_name)
done < <(
    printf "%s\n" "${accessions[@]}" |
    ../uniprot_gene_to_id.py \
        "UniProtKB_AC-ID" \
        "Gene_Name"
)

# 把下了数据的蛋白的motif提取出来
mkdir -p c2h2_motifs
# 先找对应的基因名
for ((i=0; i<"${#accessions[@]}"; i++))
do
    while read uniprot_ac uniprot_id gene_symbol gene_synonyms
    do
        names=(${uniprot_id%_*} $gene_symbol $gene_synonyms)
        name_match="false"
        for name in "${names[@]}"
        do
            if [ "${gene_names[$i]^^}" = "${name^^}" ]
            then
                name_match="true"
                break
            fi
        done
        if [ "$name_match" = "true" ]
        then
            for name in "${names[@]}"
            do
                id=$(grep -i "^MOTIF $name.H13CORE.0." H13CORE_meme_format.meme | sed 's/^MOTIF //')
                if [ -n "$id" ]
                then
                    meme-get-motif -id $id H13CORE_meme_format.meme |
                    sed -r "s/^MOTIF .+$/MOTIF ${accessions[$i]} ${gene_names[$i]}/" \
                        > "c2h2_motifs/hocomoco_${accessions[$i]}.meme"
                    break
                fi
            done
            break
        fi
    done < <(
        cut -d $'\t' -f 6,8,10,11 tf_masterlist.tsv
    )
done
# 再找对应的uniprot accession来覆盖基因名的结果
for ((i=0; i<"${#accessions[@]}"; i++))
do
    while read uniprot_ac uniprot_id gene_symbol gene_synonyms
    do
        names=(${uniprot_id%_*} $gene_symbol $gene_synonyms)
        if [ "${accessions[$i]}" = "$uniprot_ac" ]
        then
            for name in "${names[@]}"
            do
                id=$(grep -i "^MOTIF $name.H13CORE.0." H13CORE_meme_format.meme | sed 's/^MOTIF //')
                if [ -n "$id" ]
                then
                    meme-get-motif -id $id H13CORE_meme_format.meme |
                    sed -r "s/^MOTIF .+$/MOTIF ${accessions[$i]} ${gene_names[$i]}/" \
                        > "c2h2_motifs/hocomoco_${accessions[$i]}.meme"
                        break
                fi
            done
            break
        fi
    done < <(
        cut -d $'\t' -f 6,8,10,11 tf_masterlist.tsv
    )
done

# 把下了数据的蛋白的motif合并在一起
meme2meme $(ls c2h2_motifs/hocomoco_*.meme) \
    > hocomoco_motifs.dbs
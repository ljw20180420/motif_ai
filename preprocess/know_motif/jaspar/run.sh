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

# 收集jaspar的motif id，uniprot accessions，基因名
ids=()
uniprot_idss=()
names=()
while IFS="|" read id uniprot_ids name
do
    ids+=("$id")
    uniprot_idss+=("$uniprot_ids")
    names+=("$name")
done < <(
    gawk '
        $0 ~ /^AC / {
            id = $0;
            sub(/^AC /, "", id);
        }
        $0 ~ /^CC uniprot_ids:/ {
            uniprot_ids = $0;
            sub(/^CC uniprot_ids:/, "", uniprot_ids)
            sub(/ /, "", uniprot_ids)
        }
        $0 ~ /^ID / {
            name = $0;
            sub(/^ID /, "", name)
        }
        $0 ~ /^\/\/$/ {
            printf("%s|%s|%s\n", id, uniprot_ids, name)
        }
    ' \
        < JASPAR2024_CORE_non-redundant_pfms_transfac.txt
)

# 把下了数据的蛋白的motif提取出来
mkdir -p c2h2_motifs
# 先找对应的基因名
for ((i=0; i<"${#accessions[@]}"; i++))
do
    for ((j=0; j<"${#ids[@]}"; ++j))
    do
        name_match="false"
        while IFS="::" read single_name
        do
            if [ "${gene_names[$i]^^}" = "${single_name^^}" ]
            then
                name_match="true"
                break
            fi
        done <<< "${names[$j]}"
        if [ "$name_match" = "true" ]
        then
            meme-get-motif -id "${ids[$j]}" JASPAR2024_CORE_non-redundant_pfms_meme.txt |
            sed -r "s/^MOTIF .+$/MOTIF ${accessions[$i]} ${gene_names[$i]}/" \
                > "c2h2_motifs/jaspar_${accessions[$i]}.meme"
            break
        fi
    done
done
# 再找对应的uniprot accession来覆盖基因名的结果
for ((i=0; i<"${#accessions[@]}"; i++))
do
    for ((j=0; j<"${#ids[@]}"; ++j))
    do
        if [[ "${uniprot_idss[$j]}" =~ "${accessions[$i]}" ]]
        then
            meme-get-motif -id "${ids[$j]}" JASPAR2024_CORE_non-redundant_pfms_meme.txt |
            sed -r "s/^MOTIF .+$/MOTIF ${accessions[$i]} ${gene_names[$i]}/" \
                > "c2h2_motifs/jaspar_${accessions[$i]}.meme"
            break
        fi
    done
done

# 把下了数据的蛋白的motif合并在一起
meme2meme $(ls c2h2_motifs/jaspar_*.meme) \
    > jaspar_motifs.dbs

#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 收集所以accession
accessions=()
for narrowPeak in $(ls $DATA_DIR/sorted/*.sorted.narrowPeak)
do
    accession=$(basename ${narrowPeak%%.*})
    accessions+=($accession)
done

# 把下了数据的蛋白的motif提取出来
mkdir -p c2h2_motifs
while read pwm_id uniprot_ids
do
    for accession in "${accessions[@]}"
    do
        if [[ $uniprot_ids =~ $accession ]]
        then
            meme-get-motif -id $pwm_id JASPAR2024_CORE_non-redundant_pfms_meme.txt |
            sed -r "s/^MOTIF (.+) (.+)$/MOTIF $accession \2/" \
                > c2h2_motifs/jaspar_$accession.meme
            break
        fi
    done
done < <(
    gawk '
        $0 ~ /^AC / {
            pwm_id = $0;
            sub(/^AC /, "", pwm_id);
        }
        $0 ~ /^CC uniprot_ids:/ {
            uniprot_ids = $0;
            sub(/^CC uniprot_ids:/, "", uniprot_ids)
            sub(/ /, "", uniprot_ids)
        }
        $0 ~ /^\/\/$/ {
            printf("%s\t%s\n", pwm_id, uniprot_ids)
        }
    ' \
        < JASPAR2024_CORE_non-redundant_pfms_transfac.txt
)

# 把下了数据的蛋白的motif合并在一起
meme2meme $(ls c2h2_motifs/jaspar_*.meme) \
    > jaspar_motifs.dbs

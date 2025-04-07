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

tail -n+2 factorbook_chipseq_meme_motifs.tsv |
cut -f3 |
sort |
uniq |
./uniprot_gene_to_id.py \
    3> ids.map

# 把下了数据的蛋白的motif提取出来
mkdir -p c2h2_motifs
while read gene_name accession
do
    read gene_name ENC < <(
        grep -E "\s$gene_name\s" factorbook_chipseq_meme_motifs.tsv |
        head -n1 |
        cut -f3,4
    )
    if grep "^MOTIF ${ENC}_" complete-factorbook-catalog.meme.corrected
    then
        read MOTIF id < <(
            grep "^MOTIF ${ENC}_" complete-factorbook-catalog.meme.corrected | head -n1
        )
        meme-get-motif -id $id complete-factorbook-catalog.meme.corrected |
        sed -r "s/^MOTIF .+$/MOTIF $accession $gene_name/" \
            > c2h2_motifs/factorbook_$accession.meme
    fi
done < <(
    grep \
        -f <(printf "%s\n" "${accessions[@]}") \
        ids.map
)

# 把下了数据的蛋白的motif合并在一起
meme2meme $(ls c2h2_motifs/factorbook_*.meme) \
    > factorbook_motifs.dbs

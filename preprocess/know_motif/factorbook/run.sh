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
./uniprot_gene_to_id.py
#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 下载蛋白文件
./uniprot_download.py \
    'ft_zn_fing:C2H2' \
    'organism_name:"Mus musculus"' \
    3> uniprot_mouse_C2H2_protein.tsv

# 下载蛋白结构
./get_mmcif_from_alphafoldDB.py \
    < uniprot_mouse_C2H2_protein.tsv

# 计算蛋白二级结构
# dssp需要libcifpp: sudo apt install libcifpp-dev
# dssp需要更新libcifpp的资源文件: https://github.com/PDB-REDO/dssp/issues/3
# curl -o /var/cache/libcifpp/components.cif https://files.wwpdb.org/pub/pdb/data/monomers/components.cif
# curl -o /var/cache/libcifpp/mmcif_pdbx.dic https://mmcif.wwpdb.org/dictionaries/ascii/mmcif_pdbx_v50.dic
# curl -o /var/cache/libcifpp/mmcif_ma.dic https://github.com/ihmwg/ModelCIF/raw/master/dist/mmcif_ma.dic
./dssp.py \
    3> secondary_structure.tsv

# 去掉没有结构的蛋白
# 去掉uniprot和alphafoldDB蛋白长度不相同的蛋白
# 在原来9种二级结构（包括没有结构）的基础上，标注KRAB和锌指蛋白结构
./parse_ft.py \
    3> protein.tsv

# 提取peak的序列
# The flag -U/--update-faidx is recommended to ensure the .fai file matches the FASTA file.
for narrowPeak in $(ls $DATA_DIR/*.final.narrowPeak)
do
    accession=$(basename ${narrowPeak%%.*})
    seqkit subseq \
        < $GENOME \
        --update-faidx \
        --bed $narrowPeak \
        --up-stream 50 \
        --down-stream 50 \
        > $DATA_DIR/$accession.fasta
done

# 预测motif
for fasta in $(ls $DATA_DIR/*.fasta)
do
    accession=$(basename ${fasta%.*})
    streme \
        --text \
        --thres 0.05 \
        --nmotifs 1 \
        --minw 10 \
        --maxw 30 \
        --p $fasta \
        > $DATA_DIR/$accession.meme
    meme2images -png $DATA_DIR/$accession.meme $DATA_DIR/$accession.png
done

# 搜索motif，并根据搜索结果选择最好的motif
for meme in $(ls $DATA_DIR/*.meme)
do
    accession=$(basename ${meme%.*})
    fimo \
        --best-site \
        --thresh 1 \
        --no-qvalue \
        --max-strand \
        --max-stored-scores 999999999 \
        $meme \
        $DATA_DIR/$accession.fasta \
        > $DATA_DIR/$accession.bed
done

# 提取结合位点序列
for bed in $(ls $DATA_DIR/*.bed)
do
    accession=$(basename ${bed%.*})
    sed -r 's/^([^\t]+)_([0-9]+)-([0-9]+)/\1\t\2\t\3/' \
        < $DATA_DIR/$bed |
    awk -F '\t' -v OFS='\t' '{print $1, $2 + $4, $2 + $5, "*", $9, $8}' > $DATA_DIR/$accession.global.bed
    seq_len=100
    motif_len=$(awk '{print $3 - $2; exit}' $DATA_DIR/$accession.global.bed)
    if [ $seq_len -lt $motif_len ]
    then
        echo "ERROR: motif length is larger than seq length"
        exit 1
    fi
    TOTAL_EXT=$((seq_len - motif_len))
    UP_EXT=$((TOTAL_EXT / 2))
    DOWN_EXT=$((TOTAL_EXT - UP_EXT))
    seqkit subseq \
        < $GENOME \
        --update-faidx \
        --bed $DATA_DIR/$accession.global.bed \
        --up-stream $UP_EXT \
        --down-stream $DOWN_EXT |
    sed '2~2y/acgt/ACGT/' \
        > $DATA_DIR/$accession.positive.fasta
done

# 产生不结合位点序列
for positive in $(ls $DATA_DIR/*.positive.fasta)
do
    accession=$(basename ${positive%%.*})
    seq_length_p1=$(
        sed -n '2{p;q}' \
            < $positive |
        wc -c
    )
    
    fasta-shuffle-letters \
        -kmer 1 \
        -dna \
        -line $((seq_length_p1 - 1)) \
        -seed 63036
        $positive \
        $DATA_DIR/$accession.negative.fasta
done

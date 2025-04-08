#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# mask基因组
bedtools maskfasta -fi $GENOME -bed $GENOME_BLACK -fo $GENOME_MASK

# # 下载蛋白文件
# ./uniprot_download.py \
#     'ft_zn_fing:C2H2' \
#     'organism_name:"Mus musculus"' \
#     3> uniprot_mouse_C2H2_protein.tsv

# # 下载蛋白结构
# ./get_mmcif_from_alphafoldDB.py \
#     < uniprot_mouse_C2H2_protein.tsv

# # 计算蛋白二级结构
# # dssp需要libcifpp: sudo apt install libcifpp-dev
# # dssp需要更新libcifpp的资源文件: https://github.com/PDB-REDO/dssp/issues/3
# # curl -o /var/cache/libcifpp/components.cif https://files.wwpdb.org/pub/pdb/data/monomers/components.cif
# # curl -o /var/cache/libcifpp/mmcif_pdbx.dic https://mmcif.wwpdb.org/dictionaries/ascii/mmcif_pdbx_v50.dic
# # curl -o /var/cache/libcifpp/mmcif_ma.dic https://github.com/ihmwg/ModelCIF/raw/master/dist/mmcif_ma.dic
# ./dssp.py \
#     3> secondary_structure.tsv

# # 去掉没有结构的蛋白
# # 去掉uniprot和alphafoldDB蛋白长度不相同的蛋白
# # 在原来9种二级结构（包括没有结构）的基础上，标注KRAB和锌指蛋白结构
# ./parse_ft.py \
#     3> protein.tsv

# 收集所有accession
# 收集所有accession
accessions=()
for narrowPeak in $(ls $DATA_DIR/sorted/*.sorted.narrowPeak)
do
    accession=$(basename ${narrowPeak%%.*})
    accessions+=($accession)
done

# accessions=(Q61164)

# 选择最好的peak
select_peak_num=30000
for accession in "${accessions[@]}"
do
    sort -k7,7nr \
        < $DATA_DIR/$accession.sorted.narrowPeak |
    head -n $select_peak_num \
        > $DATA_DIR/$accession.best.narrowPeak
done

# 提取peak的序列
# The flag -U/--update-faidx is recommended to ensure the .fai file matches the FASTA file.
for accession in "${accessions[@]}"
do
    seqkit subseq \
        < $GENOME_MASK \
        --update-faidx \
        --bed $DATA_DIR/$accession.best.narrowPeak \
        --up-stream 0 \
        --down-stream 0 \
        > $DATA_DIR/$accession.best.fasta
done

# 预测motif
for accession in "${accessions[@]}"
do
    streme \
        --text \
        --thres 0.05 \
        --nmotifs 3 \
        --minw 10 \
        --maxw 30 \
        --p $DATA_DIR/$accession.best.fasta \
        > $DATA_DIR/$accession.meme
    meme2images -png $DATA_DIR/$accession.meme $DATA_DIR/$accession
    # png=$(ls $DATA_DIR/$accession/*)
    # mv $png $DATA_DIR/$accession.png
    # rm -r $DATA_DIR/$accession
done

# 



# # 搜索motif，并根据搜索结果选择最好的motif
# for accession in "${accessions[@]}"
# do
#     fimo \
#         --best-site \
#         --thresh 1 \
#         --no-qvalue \
#         --max-strand \
#         --max-stored-scores 999999999 \
#         $DATA_DIR/$accession.meme \
#         $DATA_DIR/$accession.fasta \
#         > $DATA_DIR/$accession.bed
# done

# # 提取结合位点序列
# for accession in "${accessions[@]}"
# do
#     sed -r 's/^([^\t]+)_([0-9]+)-([0-9]+)/\1\t\2\t\3/' \
#         < $DATA_DIR/$accession.bed |
#     awk -F '\t' -v OFS='\t' '{print $1, $2 + $4, $2 + $5, "*", $9, $8}' > $DATA_DIR/$accession.global.bed
#     seq_len=100
#     motif_len=$(awk '{print $3 - $2; exit}' $DATA_DIR/$accession.global.bed)
#     if [ $seq_len -lt $motif_len ]
#     then
#         echo "ERROR: motif length is larger than seq length"
#         exit 1
#     fi
#     TOTAL_EXT=$((seq_len - motif_len))
#     UP_EXT=$((TOTAL_EXT / 2))
#     DOWN_EXT=$((TOTAL_EXT - UP_EXT))
#     seqkit subseq \
#         < $GENOME \
#         --update-faidx \
#         --bed $DATA_DIR/$accession.global.bed \
#         --up-stream $UP_EXT \
#         --down-stream $DOWN_EXT |
#     sed '2~2y/acgt/ACGT/' \
#         > $DATA_DIR/$accession.positive.fasta
# done

# # 聚类motif
# # 重命名motif
# for accession in "${accessions[@]}"
# do
#     sed -i "s/^MOTIF.*$/MOTIF $accession/" $DATA_DIR/$accession.meme
# done
# # 合并motif
# meme2meme $(ls $DATA_DIR/*.meme) > $DATA_DIR/motif.dbs
# # 计算motif相似度
# tomtom \
#     -motif-pseudo 0.1 \
# 	-dist kullback \
#     -min-overlap 1 \
# 	-text \
# 	$DATA_DIR/motif.dbs $DATA_DIR/motif.dbs \
# > $DATA_DIR/motif.tomtom

# # 产生不结合位点序列
# for accession in "${accessions[@]}"
# do
#     seq_length_p1=$(
#         sed -n '2{p;q}' \
#             < $DATA_DIR/$accession.positive.fasta |
#         wc -c
#     )
    
#     fasta-shuffle-letters \
#         -kmer 1 \
#         -dna \
#         -line $((seq_length_p1 - 1)) \
#         -seed 63036
#         $DATA_DIR/$accession.positive.fasta \
#         $DATA_DIR/$accession.negative.fasta
# done

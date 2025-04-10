#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

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
accessions=()
for narrowPeak in $(ls $DATA_DIR/sorted/*.sorted.narrowPeak)
do
    accession=$(basename ${narrowPeak%%.*})
    accessions+=($accession)
done

# # 把黑名单的peak去掉
# # 把peak聚类
# mkdir -p $DATA_DIR/clustered
# cluster_max_distance="-50"
# for accession in "${accessions[@]}"
# do
#     bedtools intersect -v \
#         -a $DATA_DIR/sorted/$accession.sorted.narrowPeak \
#         -b $GENOME_BLACK |
#     bedtools cluster \
#         -d $cluster_max_distance \
#         > $DATA_DIR/clustered/$accession.clustered.narrowPeak
# done

# # 每个聚类选择好的分位数，不选择最大值，防止异常值
# mkdir -p $DATA_DIR/selected
# cluster_quantile=0.9
# for accession in "${accessions[@]}"
# do
#     ./peak_cluster_select.py \
#         < $DATA_DIR/clustered/$accession.clustered.narrowPeak \
#         $cluster_quantile \
#         > $DATA_DIR/selected/$accession.selected.narrowPeak
# done

# # 从选出的不重复的peak中选择p值最好的peak
# mkdir -p $DATA_DIR/best
# select_peak_num=30000
# for accession in "${accessions[@]}"
# do
#     sort -k8,8gr \
#         < $DATA_DIR/selected/$accession.selected.narrowPeak |
#     head -n $select_peak_num \
#         > $DATA_DIR/best/$accession.best.narrowPeak
# done

# # 从最好的peak的summit周围截取固定长度
# mkdir -p $DATA_DIR/same
# centrimo_length=500
# for accession in "${accessions[@]}"
# do
#     gawk -v centrimo_length=$centrimo_length '
#         {
#             start=$2
#             end=$3
#             summit=$10
#             if (summit - start <= int(centrimo_length / 2)) {
#                 new_start = start
#                 new_end = new_start + centrimo_length
#                 new_end = new_end <= end ? new_end : end
#             } else if (end - summit <= int(centrimo_length / 2)) {
#                 new_end = end
#                 new_start = new_end - centrimo_length
#                 new_start = new_start >= start ? new_start : start
#             } else {
#                 new_start = summit - int(centrimo_length / 2)
#                 new_end = new_start + centrimo_length
#             }
#             printf("%s\t%d\t%d\n", $1, new_start, new_end)
#         }
#     ' \
#         < $DATA_DIR/best/$accession.best.narrowPeak \
#         > $DATA_DIR/same/$accession.same.bed
# done

# # 提取peak的序列
# for accession in "${accessions[@]}"
# do
#     bed2fasta \
#         -o $DATA_DIR/same/$accession.same.fasta \
#         $DATA_DIR/same/$accession.same.bed \
#         $GENOME
# done

# # 预测motif
# mkdir -p $DATA_DIR/meme-chip
# for accession in "${accessions[@]}"
# do
#     meme-chip \
#         -oc $DATA_DIR/meme-chip/$accession \
#         -seed 63036 \
#         -db know_motif/jaspar/jaspar_motifs.dbs \
#         -db know_motif/hocomoco/hocomoco_motifs.dbs \
#         -dna \
#         -filter-thresh 0.05 \
#         -minw 10 \
#         -maxw 30 \
#         -ccut 100 \
#         -meme-nmotifs 3 \
#         -meme-norand \
#         -meme-mod oops \
#         -streme-nmotifs 3 \
#         -spamo-skip \
#         -fimo-skip \
#         $DATA_DIR/same/$accession.same.fasta
# done

# # 选择和已知motif最像的motif
# # 否则如何不是wiz，选择centrimo报告的motif
# # 否则如果有hocomoco motif，选hocomoco motif
# # 否则如果有jaspar motif，选jaspar motif
# # 否则选择combine.meme的第一个motif
# mkdir -p $DATA_DIR/motifs
# situations=()
# for accession in "${accessions[@]}"
# do
#     best_line=$(
#         cat \
#             <(
#                 head -n-4 \
#                     < $DATA_DIR/meme-chip/$accession/meme_tomtom_out/tomtom.tsv
#             ) \
#             <(
#                 head -n-4 \
#                     < $DATA_DIR/meme-chip/$accession/streme_tomtom_out/tomtom.tsv
#             ) |
#         grep -F $accession |
#         sort -k5,5g |
#         head -n1
#     )
#     id=$(
#         sed -r 's/^([^\t]+)\t.+$/\1/' \
#             <<<$best_line
#     )
#     if [ -n "$id" ] # 找到了和已知motif像的motif
#     then
#         is_streme='^[0-9]+-'
#         if [[ $id =~ $is_streme ]]
#         then
#             discovery_program="streme"
#         else
#             discovery_program="meme"
#         fi
#         strand=$(
#             cut -f10 \
#                 <<<$best_line
#         )
#         if [ "$strand" = "+" ]
#         then
#             revcomp=""
#         else
#             revcomp="-rc"
#         fi
#         db_source=$(
#             cut -f2 \
#                 <<<$best_line |
#             grep -oE "jaspar|hocomoco|factorbook|CIS-BP"
#         )
#         meme-get-motif $revcomp -id $id \
#             $DATA_DIR/meme-chip/$accession/${discovery_program}_out/${discovery_program}.txt |
#         sed -r "s/^MOTIF .+$/MOTIF $accession/" \
#             > $DATA_DIR/motifs/$accession.meme
#         situations+=("${discovery_program}-$db_source")
#     else # 找不到和已知motif像的motif
#         found_motif="false"
#         if [ "$accession" != "O88286" ] # 如果不是wiz
#         then
#             db_id=$(
#                 head -n-4 \
#                     $DATA_DIR/meme-chip/$accession/centrimo_out/centrimo.tsv |
#                 grep -F $accession |
#                 sort -k5,5g |
#                 head -n1 |
#                 cut -f2
#             )
#             if [ -n "$db_id" ] # 如果centrimo找到了数据库中的已知motif
#             then
#                 db_source=$(
#                     grep -oE "jaspar|hocomoco|factorbook|CIS-BP" \
#                         <<<$db_id
#                 )
#                 meme-get-motif -id $db_id \
#                     know_motif/$db_source/${db_source}_motifs.dbs |
#                 sed -r "s/^MOTIF .+$/MOTIF $accession/" \
#                     > $DATA_DIR/motifs/$accession.meme
#                 situations+=("centrimo-$db_source")
#                 found_motif="true"
#             else
#                 for db_source in hocomoco jaspar
#                 do
#                     if grep -F "${db_source}_${accession}" know_motif/${db_source}/${db_source}_motifs.dbs > /dev/null # db_source有motif
#                     then
#                         meme-get-motif -id "${db_source}_${accession}" \
#                             know_motif/${db_source}/${db_source}_motifs.dbs |
#                         sed -r "s/^MOTIF .+$/MOTIF $accession/" \
#                             > $DATA_DIR/motifs/$accession.meme
#                         situations+=("$db_source")
#                         found_motif="true"
#                         break
#                     fi
#                 done
#             fi
#         fi
#         if [ "${found_motif}" = "false" ] # 之前的方法都没找到motif
#         then
#             meme-get-motif -id 1 \
#                 $DATA_DIR/meme-chip/$accession/combined.meme |
#             sed -r "s/^MOTIF .+$/MOTIF $accession/" \
#                 > $DATA_DIR/motifs/$accession.meme
#             situations+=("combine1")
#         fi
#     fi    
# done

# # 产生motif背靠背比较
# mkdir -p $DATA_DIR/images
# > $DATA_DIR/images/side_by_side.md
# for ((i=0;i<${#accessions[@]};++i))
# do
#     accession="${accessions[$i]}"
#     meme2images -png \
#         $DATA_DIR/motifs/$accession.meme \
#         $DATA_DIR/images/$accession
#     if [ -s "know_motif/jaspar/c2h2_motifs/jaspar_$accession.meme" ]
#     then
#         meme2images -png \
#             know_motif/jaspar/c2h2_motifs/jaspar_$accession.meme \
#             $DATA_DIR/images/$accession
#     fi
#     if [ -s "know_motif/hocomoco/c2h2_motifs/hocomoco_$accession.meme" ]
#     then
#         meme2images -png \
#             know_motif/hocomoco/c2h2_motifs/hocomoco_$accession.meme \
#             $DATA_DIR/images/$accession
#     fi
#     printf \
#         "<p float=\"left\">\n  <img src=\"%s/logo%s.png\" title=\"%s\" alt=\"%s\" width=\"200\" />\n  <img src=\"%s/logojaspar_%s.png\" title=\"jaspar_%s\" alt=\"jaspar_%s\" width=\"200\" />\n  <img src=\"%s/logohocomoco_%s.png\" title=\"hocomoco_%s\" alt=\"hocomoco_%s\" width=\"200\" />\n  <span>\"%s\"</span>\n</p>\n" \
#         $accession $accession $accession $accession $accession $accession $accession $accession $accession $accession $accession $accession "${situations[$i]}" \
#         >> $DATA_DIR/images/side_by_side.md
# done

# # 搜索motif
# mkdir -p $DATA_DIR/sites
# for accession in "${accessions[@]}"
# do
#     bed2fasta \
#         -o $DATA_DIR/selected/$accession.selected.fasta \
#         $DATA_DIR/selected/$accession.selected.narrowPeak \
#         $GENOME
#     fimo \
#         --best-site \
#         --thresh 1 \
#         --no-qvalue \
#         --max-strand \
#         $DATA_DIR/motifs/$accession.meme \
#         $DATA_DIR/selected/$accession.selected.fasta \
#         > $DATA_DIR/sites/$accession.site
# done

# # 提取结合位点序列
# # 得到每个蛋白的锌指数量
# mkdir -p $DATA_DIR/positive
# for accession in "${accessions[@]}"
# do
#     zinc_num=$(
#         grep -F "$accession" \
#             < uniprot_mouse_C2H2_protein.tsv |
#             sed -r 's/^.+note="C2H2-type ([0-9]+)".+$/\1/' |
#             sed -r '/^[^1-9]/s/^.+$/1/'
#     )
#     seq_len=$((10 + zinc_num * 3))
#     motif_len=$(awk '{print $3 - $2; exit}' $DATA_DIR/sites/$accession.site)
#     if [ $seq_len -lt $motif_len ]
#     then
#         UP_EXT=0
#         DOWN_EXT=0
#     else
#         TOTAL_EXT=$((seq_len - motif_len))
#         UP_EXT=$((TOTAL_EXT / 2))
#         DOWN_EXT=$((TOTAL_EXT - UP_EXT))
#     fi
#     # --line-width 0 防止fasta换行
#     seqkit subseq \
#         < $GENOME \
#         --update-faidx \
#         --line-width 0 \
#         --bed $DATA_DIR/sites/$accession.site \
#         --up-stream $UP_EXT \
#         --down-stream $DOWN_EXT |
#     sed '2~2y/acgt/ACGT/' \
#         > $DATA_DIR/positive/$accession.positive
# done

# 产生不结合位点序列
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

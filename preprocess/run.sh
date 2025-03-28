#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source lib.sh

# 下载小鼠的C2H2锌指蛋白的feature table
esearch -db protein -query "C2H2 AND Mus musculus [ORGN]" |
efetch -format ft \
    >zf.ft

# 下载小鼠的C2H2锌指蛋白的INSDSet XML格式
esearch -db protein -query "C2H2 AND Mus musculus [ORGN]" |
efetch -format gpc -mode xml \
    >zf.xml

# 下载小鼠的C2H2锌指蛋白的accession
esearch -db protein -query "C2H2 AND Mus musculus [ORGN]" |
efetch -format acc \
    >zf.acc

# 下载小鼠的C2H2锌指蛋白的序列
esearch -db protein -query "C2H2 AND Mus musculus [ORGN]" |
efetch -format fasta |
seqkit fx2tab |
cut -f2 \
    >zf.seq

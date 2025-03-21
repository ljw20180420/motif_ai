#!/bin/bash

# 使用方法：path/to/fasterq_dump.sh SRA_ACCESSION_LIST_FILE.txt

# 提取fastq文件
SraAccList=$1
while read accsession
do
    fasterq-dump $accsession -O $SRA_CACHE/sra -p 
done < $SraAccList
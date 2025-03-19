#!/bin/bash

# 提取fastq文件
while read accsession
do
    fasterq-dump $accsession -O $SRA_CACHE/sra -p 
done < SraAccList.txt

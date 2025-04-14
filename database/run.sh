#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 下载小鼠最新基因组GRCm39
# ./datasets.sh

# 下载sra原始数据
# ./sra-tools.sh SraAccList.txt

# 提取fastq文件
# ./fasterq_dump.sh SraAccList.txt

# 压缩fastq文件
# for fastq in $(find $SRA_CACHE/sra -name "*.fastq")
# do
#     gzip $fastq
# done

# 运行nextflow的chipseq pipeline
# ../nextflow/runs/chipseq/test/run.sh
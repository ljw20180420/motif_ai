#!/bin/bash

# 获取脚本路径
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

java -jar $SCRIPT_DIR/bin/ena-file-downloader.jar --accessions=ERR164407 --format=READS_FASTQ --location=$SRA_CACHE --protocol=ASPERA --asperaLocation=$SCRIPT_DIR
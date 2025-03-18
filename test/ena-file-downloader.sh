#!/bin/bash

# ena-file-downloader.jar总是在--asperaLocation/bin中寻找ascp。但是ascp在~/.aspera/sdk/ascp，所以没办法用--asperaLocation指定正确路径。通过创建软链接绕过这个问题。
ln -s ~/.aspera/sdk/ascp ~/bin/ascp

# 获取脚本路径
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

java -jar $SCRIPT_DIR/ena-file-downloader.jar --accessions=ERR164407 --format=READS_FASTQ --location=$HOME/sdc1/SRA_cache --protocol=ASPERA --asperaLocation=$HOME
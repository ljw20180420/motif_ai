#!/bin/bash

# 使用方法：path/to/sra-tools.sh SRA_ACCESSION_LIST_FILE.txt

download_sra_until_success() {
    # 第一个参数是SRR的accession编号列表文件
    accessionListFile=$1

    # 一直下到成功
    until prefetch -p --option-file=$accessionListFile
    do
        pass
    done
}

validate_sra_until_success() {
    # 第一个参数是SRR的accession编号
    accessionListFile=$1
    
    # 一直下到成功
    download_sra_until_success $accessionListFile
    # 一直检查到成功
    until vdb-validate --option-file=$accessionListFile
    do
        # 失败了还得重新下载到成功
        download_sra_until_success $accessionListFile
    done
}

# 允许缓存
vdb-config -s "cache-enabled=true"

# 下载到user repository
vdb-config  --prefetch-to-user-repo

# 设置user repository路径
vdb-config -s "/repository/user/main/public/root=$SRA_CACHE"

# 一直检查到成功
SraAccList=$1
validate_sra_until_success $SraAccList

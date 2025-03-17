#!/bin/bash

download_sra_until_success() {
    # 第一个参数是SRR的accession编号
    accession=$1

    # 一直下到成功
    until prefetch -p $accession
    do
        echo "failed to prefetch $accession, retry"
    done
}

validate_sra_until_success() {
    # 第一个参数是SRR的accession编号
    accession=$1
    
    # 一直下到成功
    download_sra_until_success $accession
    # 一直检查到成功
    # /home/ljw/sdc1/SRA_cache/sra/是vdb-config -i设置的user-repository
    until vdb-validate /home/ljw/sdc1/SRA_cache/sra/$accession.sra
    do
        echo "failed to vdb-validate $accession, retry"
        # 失败了还得重新下载到成功
        download_sra_until_success $accession
    done
}

for accession in SRR32612225 SRR32612226
do
    # 一直检查到成功
    validate_sra_until_success $accession
done

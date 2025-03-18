#!/bin/bash

# ascp私钥就是ascli.sh的--ssh-keys（https://github.com/IBM/aspera-cli?tab=readme-ov-file#authentication-on-server-with-ssh-session）
$HOME/.aspera/sdk/ascp -T -l 300m -P 33001 -i $HOME/.aspera/sdk/aspera_bypass_rsa.pem era-fasp@fasp.sra.ebi.ac.uk:vol1/fastq/ERR164/ERR164407/ERR164407.fastq.gz $HOME/sdc1/SRA_cache

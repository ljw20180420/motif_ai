#!/bin/bash

preset_server() {
    # 配置服务器
    
    # ENA文档的方法https://ena-docs.readthedocs.io/en/latest/retrieval/file-download.html#using-aspera-ascli有问题
    # ascli conf preset update era --url=ssh://fasp.sra.ebi.ac.uk:33001 --username=era-fasp --ssh-keys=@ruby:Fasp::Installation.instance.bypass_keys.first --ts=@json:'{"target_rate_kbps":300000}'

    # ascli文档https://github.com/IBM/aspera-cli?tab=readme-ov-file#authentication-on-server-with-ssh-session
    # Note: If you need to use the Aspera public keys, then specify an empty token: --ts=@json:'{"token":""}' : Aspera public SSH keys will be used, but the protocol will ignore the empty token.
    # 可以设置空token使用Aspera的ssh公钥
    # ascli conf preset update era --url=ssh://fasp.sra.ebi.ac.uk:33001 --username=era-fasp --ts=@json:'{"token":"","target_rate_kbps":300000}'

    # 也可以使用：--ssh-keys=~/.aspera/sdk/aspera_bypass_rsa.pem
    ascli conf preset update era --url=ssh://fasp.sra.ebi.ac.uk:33001 --username=era-fasp --ssh-keys=$HOME/.aspera/sdk/aspera_bypass_rsa.pem --ts=@json:'{"target_rate_kbps":300000}'
}

preset_server

ascli -Pera server download vol1/fastq/ERR164/ERR164407/ERR164407.fastq.gz --to-folder=$HOME/sdc1/SRA_cache
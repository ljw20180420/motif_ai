#!/bin/bash

# 使用严格模式
set -euo pipefail

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 指定目标超参取值
declare -A choices
choices[depth]="6 5 4"
choices[dim_emb]="128 64 32"
choices[dim_heads]="64 32 16"
choices[num_heads]="4 3 2"
choices[dim_ffn]="256 128 64"

for target in depth dim_emb dim_heads num_heads dim_ffn
do
    for value in ${choices[$target]}
    do
        # 替换自定义设置设置
        sed -r "
            /^xxx =$/s/^xxx =$/${target} = ${value}/
            /^hp_study_name =$/s/$/ ${target}/
        " bind_transformer/config_custom.ini.template \
        > bind_transformer/config_custom.ini
        
        ./run_bind_transformer.py --command train
    done
done
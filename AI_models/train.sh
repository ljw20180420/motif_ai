#!/bin/bash

# 使用严格模式
set -euo pipefail

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 将自定义设置调为训练
cp bind_transformer/config_custom.ini.train bind_transformer/config_custom.ini
./run_bind_transformer.py --command train

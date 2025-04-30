#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 使用严格模式
set -euo pipefail

# # # 初始化数据库集群
# pg_ctl init -D optuna_postgresql

# # 启动数据库集群
# pg_ctl start -D optuna_postgresql -l optuna_postgresql.log -o "-h localhost -p 5432"

# 数据库集群状态
pg_ctl status -D optuna_postgresql

# # 停止数据库集群
# pg_ctl stop -D optuna_postgresql

# # 创建数据库
# createdb -h localhost -p 5432 hpo
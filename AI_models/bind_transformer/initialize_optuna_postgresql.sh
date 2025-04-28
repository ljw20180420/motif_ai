#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# # 初始化数据库集群
# pg_ctl init -D optuna_postgresql

# 启动数据库
pg_ctl start -D optuna_postgresql -l optuna_postgresql.log

# # 停止数据库
# pg_ctl stop -D optuna_postgresql
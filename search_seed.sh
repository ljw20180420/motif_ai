#!/bin/bash

preprocess="COP"
model_cls="COP"
mkdir -p zlog
logfile="zlog/${preprocess}:${model_cls}.log"
> ${logfile}
for (( i=63036; i<63037; ++i )) {
    SEED=$i PRE_MODEL="${preprocess}:${model_cls}" LOGFILE=${logfile} scripts/formal/train_copy.sh
    PRE_MODEL="${preprocess}:${model_cls}" LOGFILE=${logfile} paper/infer_all_models_copy.sh

    let total_bind_num=0
    for file in \
        "paper/infer_all_models/output/${model_cls}.csv" \
        "paper/infer_all_models/output/${model_cls}_train.csv"
    do
        bind_num=$(
            awk -F ',' '
                BEGIN{
                    sum=0
                }
                NR > 1 {
                    if ($2 > 0.5)
                        sum += 1
                }
                END{
                    print FILENAME, sum
                }
            ' $file |
            cut -d' ' -f2
        )
        echo $file bind_num $bind_num >> ${logfile}
        let 'total_bind_num = total_bind_num + bind_num' >> ${logfile}
    done
    echo total_bind_num $total_bind_num >> ${logfile}
    if [ "$total_bind_num" -gt 80 ]
    then
        break
    fi
}

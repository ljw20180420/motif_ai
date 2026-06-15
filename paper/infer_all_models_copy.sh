#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ..

title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} $1 ${sharps}
}

infer_config=AI/infer.yaml
output_dir=${OUTPUT_DIR:-"${HOME}/COP_results"}
run_type="formal"
run_name="default"
trial_name="default"
data_name=mouse_C2H2
pre_model=${PRE_MODEL}
logfile=${LOGFILE}

title ${pre_model} >> ${logfile}

IFS=":" read preprocess model_cls <<<${pre_model}
checkpoints_path=${output_dir}/${run_type}/${run_name}/checkpoints/${preprocess}/${model_cls}/${data_name}/${trial_name}
logs_path=${output_dir}/${run_type}/${run_name}/logs/${preprocess}/${model_cls}/${data_name}/${trial_name}

title Infer >> ${logfile}

for suffix in \
    "" \
    "_train"
do
    input_file="paper/infer_all_models/wiz${suffix}.csv"
    output_file="paper/infer_all_models/output/${model_cls}${suffix}.csv"
    ./run.py infer \
        --config ${infer_config} \
        --input ${input_file} \
        --output ${output_file} \
        --test.checkpoints_path ${checkpoints_path} \
        --test.logs_path ${logs_path}
done

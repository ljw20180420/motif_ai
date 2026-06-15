#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} "$1" ${sharps}
}

train_config=AI/train.yaml
output_dir=${OUTPUT_DIR:-"${HOME}/COP_results"}
run_type="formal"
run_name="default"
trial_name="default"
seed=${SEED:-"63036"}
pre_model=${PRE_MODEL}
logfile=${LOGFILE}

title "seed ${seed}" >> ${logfile}

title ${pre_model} >> ${logfile}

IFS=":" read preprocess model_cls <<<${pre_model}
model_config=AI/preprocess/${preprocess}/${model_cls}.yaml

title Train >> ${logfile}
case ${model_cls} in
    COP)
        ./run.py train \
            --config ${train_config} \
            --train.output_dir ${output_dir}/${run_type}/${run_name} \
            --train.trial_name ${trial_name} \
            --train.num_epochs 33 \
            --train.last_epoch 3 \
            --train.evaluation_only false \
            --model ${model_config} \
            --dataset.data_dir AI/dataset/formal_data \
            --generator.seed ${seed}
    ;;
    LightGBM|XGBoost|RandomForest|DecisionTree)
        ./run.py train \
            --config ${train_config} \
            --train.output_dir ${output_dir}/${run_type}/${run_name} \
            --train.trial_name ${trial_name} \
            --train.batch_size 1000000 \
            --train.num_epochs 1 \
            --train.evaluation_only false \
            --train.device cpu \
            --model ${model_config} \
            --dataset.data_dir AI/dataset/formal_data \
            --generator.seed ${seed}
    ;;
    DeepZF)
        ./run.py train \
            --config ${train_config} \
            --train.output_dir ${output_dir}/${run_type}/${run_name} \
            --train.trial_name ${trial_name} \
            --train.batch_size 1000000 \
            --train.num_epochs 1 \
            --train.evaluation_only false \
            --model ${model_config} \
            --dataset.data_dir AI/dataset/formal_data \
            --generator.seed ${seed}
    ;;
    PassiveAggressiveClassifier|Perceptron|SGDClassifier|SupportVectorMachine)
        ./run.py train \
            --config ${train_config} \
            --train.output_dir ${output_dir}/${run_type}/${run_name} \
            --train.trial_name ${trial_name} \
            --train.batch_size 100000 \
            --train.evaluation_only false \
            --model ${model_config} \
            --dataset.data_dir AI/dataset/formal_data \
            --generator.seed ${seed}
    ;;
    CategoricalNB)
        ./run.py train \
            --config ${train_config} \
            --train.output_dir ${output_dir}/${run_type}/${run_name} \
            --train.trial_name ${trial_name} \
            --train.batch_size 100000 \
            --train.num_epochs 1 \
            --train.evaluation_only false \
            --model ${model_config} \
            --dataset.data_dir AI/dataset/formal_data
    ;;
    *)
        ./run.py train \
            --config ${train_config} \
            --train.output_dir ${output_dir}/${run_type}/${run_name} \
            --train.trial_name ${trial_name} \
            --train.evaluation_only false \
            --model ${model_config} \
            --dataset.data_dir AI/dataset/formal_data \
            --generator.seed ${seed}
    ;;
esac

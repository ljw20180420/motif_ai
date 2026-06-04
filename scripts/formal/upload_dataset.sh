#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} "$1" ${sharps}
}

upload_dataset_config=AI/upload_dataset.yaml
data_dir=AI/dataset/formal_data

title "Upload dataset"
./run.py upload_dataset \
    --config ${upload_dataset_config} \
    --dataset.data_dir ${data_dir}

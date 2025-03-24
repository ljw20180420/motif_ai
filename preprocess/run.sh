#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source lib.sh

get_protein_info CTCF | ./zinc_finger_protein_xml_parser.py

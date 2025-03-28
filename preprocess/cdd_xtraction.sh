#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cat zf.xml | xtract -insd complete Region region_name note db_xref

# < zf.xml xtract --pattern INSDSeq \
#     -ACCESSION INSDSeq_primary-accession \
#     -block INSDFeature \
#         -def "" \
#         -if INSDFeature_key -equals Region \
#             -element &ACCESSION INSDInterval_from INSDInterval_to \
#             -subset INSDQualifier \
#                 -if INSDQualifier_name -quals

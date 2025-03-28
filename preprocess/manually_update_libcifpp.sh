#!/bin/bash

# dssp需要libcifpp: sudo apt install libcifpp-dev
# dssp需要更新libcifpp的资源文件: https://github.com/PDB-REDO/dssp/issues/3

curl -o /var/cache/libcifpp/components.cif https://files.wwpdb.org/pub/pdb/data/monomers/components.cif
curl -o /var/cache/libcifpp/mmcif_pdbx.dic https://mmcif.wwpdb.org/dictionaries/ascii/mmcif_pdbx_v50.dic
curl -o /var/cache/libcifpp/mmcif_ma.dic https://github.com/ihmwg/ModelCIF/raw/master/dist/mmcif_ma.dic
#!/usr/bin/env python

import os

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import re
import sys


def eight_to_three(sec):
    # The DSSP codes for secondary structure used here are:
    # =====     ====
    # Code      Structure
    # =====     ====
    # H         α-helix
    # B         residue in isolated β-bridge
    # E         extended strand, participates in β ladder
    # G         310-helix
    # I         π-helix
    # P         κ-helix (poly-proline II helix)
    # T         hydrogen-bonded turn
    # S         bend
    # -         None
    # =====     ====

    sec = re.sub(r"[-ITS]", "-", sec)
    sec = re.sub(r"G", "H", sec)
    sec = re.sub(r"B", "E", sec)
    return sec


with open(3, "w") as fd:
    fd.write("accession\tsequence\tsecondary_structure\n")
    for file in enumerate(os.listdir("get_mmcif_from_alphafoldDB")):
        file = file[1]
        if not file.endswith(".pdb"):
            continue
        p = PDBParser()
        id = os.path.splitext(file)[0]
        structure = p.get_structure(id, f"get_mmcif_from_alphafoldDB/{file}")
        # use only the first model
        model = structure[0]
        # DBREF is not in the standard format, delete it temperarily
        with open(f"get_mmcif_from_alphafoldDB/tmp.{file}", "w") as wd, open(
            f"get_mmcif_from_alphafoldDB/{file}", "r"
        ) as rd:
            for line in rd:
                if not line.startswith("DBREF"):
                    wd.write(line)
        # calculate DSSP
        dssp = DSSP(model, f"get_mmcif_from_alphafoldDB/tmp.{file}")
        os.remove(f"get_mmcif_from_alphafoldDB/tmp.{file}")
        seq = "".join([dssp[key][1] for key in dssp.keys()])
        sec = "".join([dssp[key][2] for key in dssp.keys()])
        # if desired, convert DSSP's 8-state assignments into 3-state [C - coil, E - extended (beta-strand), H - helix]
        # sec = eight_to_three(sec)

        fd.write(f"{id}\t{seq}\t{sec}\n")
        fd.flush()

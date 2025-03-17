#!/usr/bin/env python

from utils.random_seq_methods import generate_random_DNA
import numpy as np
import pandas as pd
from pymemesuite.common import MotifFile


# Generate a random instance from motif.
def generate_random_motif_instances(motif, rng):
    return "".join(
        [
            rng.choice(
                ["A", "C", "G", "T"], p=np.array(frequency) / frequency.sum()
            ).item()
            for frequency in motif.frequencies
        ]
    )


# Put a random instance in a random position of a random sequence.
def put_instances_in_seqs(instance, rng, seq_length):
    instance_length = len(instance)
    start = rng.integers(seq_length - instance_length + 1).item()
    return (
        generate_random_DNA(start, rng)
        + instance
        + generate_random_DNA(seq_length - start - instance_length, rng)
    )


def generate_random_seqs(rng, seq_length=32, seq_num=10000, positive_ratio=0.2):
    with MotifFile("MA0139.1.meme") as motif_file:
        motif = motif_file.read()
        seqs, binds = [], []
        for idx in range(seq_num):
            if rng.random() < positive_ratio:
                instance = generate_random_motif_instances(motif, rng)
                seq = put_instances_in_seqs(instance, rng, seq_length)
                seqs.append(seq)
                binds.append(1.0)
            else:
                seq = generate_random_DNA(seq_length, rng)
                seqs.append(seq)
                binds.append(0.0)
    return seqs, binds


rng = np.random.default_rng(63036)
seqs, binds = generate_random_seqs(
    rng, seq_length=32, seq_num=10000, positive_ratio=0.2
)
pd.DataFrame({"seq": seqs, "bind": binds}).to_csv("data.csv", index=False)

seqs, binds = generate_random_seqs(
    rng, seq_length=32, seq_num=10000, positive_ratio=0.2
)
pd.DataFrame({"seq": seqs}).to_csv("inference.csv", index=False)
pd.DataFrame({"bind": binds}).to_csv("answer.csv", index=False)

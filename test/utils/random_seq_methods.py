#!/usr/bin/env python

import numpy as np

def generate_random_DNA(length, rng):
    # usage: generate_random_DNA of length
    return "".join(rng.choice(["A","C","G","T"], length))

def SNP_DNA(DNA, probability, rng):
    DNA = list(DNA)
    for i in range(len(DNA)):
        if rng.random() < probability:
            DNA[i] = rng.choice(["A","C","G","T"])
    return "".join(DNA)

def indel_DNA(DNA, probability, rng):
    inslens = rng.negative_binomial(1, 1 - probability, len(DNA) + 1)
    insDNAs = generate_random_DNA(np.sum(inslens), rng)
    DNAnew, start = [], 0
    for i in range(len(DNA)):
        end = start + inslens[i]
        DNAnew.append(insDNAs[start:end])
        if rng.random() > probability:
            DNAnew.append(DNA[i])
        start = end
    DNAnew.append(insDNAs[start:])
    return "".join(DNAnew)

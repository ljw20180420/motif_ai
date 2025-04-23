import numpy as np
import torch
from typing import List


class DNA_Tokenizer:
    """
    DNA: mcACGTN->0123456
    m: mask token
    c: [CLS]token
    ACGT: 碱基token
    N: 位置碱基
    从输入的DNA中截取DNA_length长度的DNA序列, 并在开头加入分类token c
    如果DNA_length <= 0, 使用10 + 3 * zinc_num作为DNA_length
    """

    def __init__(self, DNA_length: int) -> None:
        self.DNA_length = DNA_length

        self.DNA_tokenmap = torch.zeros(110, dtype=torch.int64)
        self.DNA_tokenmap[
            np.frombuffer("mcACGTN".encode(), dtype=np.int8).astype(np.int64)
        ] = torch.arange(7)

    def __call__(self, DNAs: List[str], zinc_nums: List[int]) -> torch.Tensor:
        extracted_DNAs, max_length = [], 0
        for DNA, zinc_num in zip(DNAs, zinc_nums):
            DNA_length = self.DNA_length if self.DNA_length > 0 else 10 + 3 * zinc_num
            if len(DNA) < DNA_length:
                raise ValueError("DNA太短")
            start = (len(DNA) - DNA_length) // 2
            extracted_DNA = "c" + DNA[start : start + DNA_length]
            max_length = (
                max_length if max_length >= len(extracted_DNA) else len(extracted_DNA)
            )
            extracted_DNAs.append(extracted_DNA)

        return torch.stack(
            [
                self.DNA_tokenmap[
                    np.frombuffer(
                        (
                            extracted_DNA + "m" * (max_length - len(extracted_DNA))
                        ).encode(),
                        dtype=np.int8,
                    ).astype(np.int64)
                ]
                for extracted_DNA in extracted_DNAs
            ]
        )


class Second_Tokenizer:
    """
    second: mHBEGIPTS-KZ->0-11
    m: mask token
    HBEGIPTS-: 二级结构
    K: KRAB
    Z: 锌指
    """

    def __init__(self) -> None:
        self.second_tokenmap = torch.zeros(110, dtype=torch.int64)
        self.second_tokenmap[
            np.frombuffer("mHBEGIPTS-KZ".encode(), dtype=np.int8).astype(np.int64)
        ] = torch.arange(12)

    def __call__(self, seconds: List[str]) -> torch.Tensor:
        max_length = max([len(second) for second in seconds])
        return torch.stack(
            [
                self.second_tokenmap[
                    np.frombuffer(
                        (second + "m" * (max_length - len(second))).encode(),
                        dtype=np.int8,
                    ).astype(np.int64)
                ]
                for second in seconds
            ]
        )


class Protein_Tokenizer:
    """
    protein: mACDEFGHIKLMNPQRSTVWY->0-20
    m: mask token
    ACDEFGHIKLMNPQRSTVWY: 氨基酸
    """

    def __init__(self) -> None:
        self.protein_tokenmap = torch.zeros(110, dtype=torch.int64)
        self.protein_tokenmap[
            np.frombuffer("mACDEFGHIKLMNPQRSTVWY".encode(), dtype=np.int8).astype(
                np.int64
            )
        ] = torch.arange(21)

    def __call__(self, proteins: List[str]) -> torch.Tensor:
        max_length = max([len(protein) for protein in proteins])
        return torch.stack(
            [
                self.protein_tokenmap[
                    np.frombuffer(
                        (protein + "m" * (max_length - len(protein))).encode(),
                        dtype=np.int8,
                    ).astype(np.int64)
                ]
                for protein in proteins
            ]
        )


class Protein_Bert_Tokenizer:
    """
    protein: ACDEFGHIKLMNPQRSTUVWXYosep->0-25
    ACDEFGHIKLMNPQRSTVWY: 氨基酸
    U: 硒半胱氨酸
    X: 未定义氨基酸
    o: 其它氨基酸
    s: 蛋白序列起始位置
    e: 蛋白序列终止位置
    p: pad
    max_length: protein bert不使用mask, 而是用pad. 因此需要固定max_length.
    """

    def __init__(self, max_length: int) -> None:
        self.max_length = max_length
        self.protein_tokenmap = torch.zeros(116, dtype=torch.int64)
        self.protein_tokenmap[
            np.frombuffer("ACDEFGHIKLMNPQRSTUVWXYosep".encode(), dtype=np.int8).astype(
                np.int64
            )
        ] = torch.arange(26)

    def __call__(self, proteins: List[str]) -> torch.Tensor:
        return torch.stack(
            [
                self.protein_tokenmap[
                    np.frombuffer(
                        (
                            "s" + protein + "e" + "p" * (self.max_length - len(protein))
                        ).encode(),
                        dtype=np.int8,
                    ).astype(np.int64)
                ]
                for protein in proteins
            ]
        )

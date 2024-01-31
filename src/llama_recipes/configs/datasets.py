# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str = "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class medcqa_dataset:
    dataset: str = "medcqa_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"


# alpaca_medcqa_dataset


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"


@dataclass
class alpaca_medcqa_dataset:
    dataset: str = "alpaca_medcqa_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = "src/llama_recipes/datasets/alpca_med_cqa_modi_prompt_in_modify_ou_T_option_en.json"


@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
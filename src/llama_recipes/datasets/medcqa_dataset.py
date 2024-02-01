# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets


def get_preprocessed_medmcqa(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("daze-unlv/medmcqa", split=split)
    # This is a real-world medical entrance exam question, please give the true answer based on the question and options:
    prompt = (
        f"This is a real-world medical entrance exam question, please give the true answer based on the question and options: \n{{question}}\n{{opa}}, {{opb}}, {{opc}}, {{opd}}\nanswer:\n"
    )

    def apply_prompt_template(sample):
        True_answer=(sample['opa'] if sample['cop'] == 0 else
                     sample['opb'] if sample['cop'] == 1 else
                     sample['opc'] if sample['cop'] == 2 else
                     sample['opd'] if sample['cop'] == 3 else "No valid answer found")
        return {
            "prompt": prompt.format(question=sample['question'], opa=sample['opa'], opb=sample['opb'], opc=sample['opc'], opd=sample['opd']),
            "answer": True_answer,
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(sample["answer"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + answer,
            "attention_mask" : [1] * (len(prompt) + len(answer)),
            "labels": [-100] * len(prompt) + answer,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset

if __name__ == "__main__":
    from transformers import LlamaForCausalLM, LlamaTokenizer
    model_id = "meta-llama/Llama-2-7b"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
    from llama_recipes.configs.datasets import medcqa_dataset

    train_dataset = get_preprocessed_dataset(tokenizer=tokenizer, dataset_config=medcqa_dataset, split='train')
    print(datasets)
    # print(train_dataset['prompt'])
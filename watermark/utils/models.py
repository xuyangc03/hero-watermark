from typing import Any

import torch


MODEL_MAP = {
    "Llama2_7B": "meta-llama/Llama-2-7b-hf",
}


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    model_path = MODEL_MAP.get(model_name, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: Any = "auto",
):
    from transformers import AutoModelForCausalLM

    model_path = MODEL_MAP.get(model_name, model_name)
    tokenizer = load_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    ).eval()
    return model, tokenizer

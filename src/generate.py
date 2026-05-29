import argparse
import copy
import json
import logging
import os
import sys
import time

import torch
from tqdm import tqdm
from transformers import LogitsProcessorList, set_seed
from transformers.generation.logits_process import (
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    MinPLogitsWarper,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)

try:
    from transformers.generation.logits_process import TopHLogitsWarper
except ImportError:
    TopHLogitsWarper = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watermark.generation import (
    GumbelLogitsProcessor,
    HeRoLogitsProcessor,
)
from watermark.utils import load_data, load_model
from watermark.utils.data import count_existing_samples, load_config

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to generation_config.json"
    )
    return parser.parse_args()


def prepare_logits_processor_list(
    exp_config: dict,
    model_gen_config,
    watermark_processor,
    model_device,
):
    generation_config = copy.deepcopy(model_gen_config)
    relevant_keys = [
        "num_beams",
        "temperature",
        "top_h",
        "top_k",
        "top_p",
        "min_p",
        "typical_p",
        "epsilon_cutoff",
        "eta_cutoff",
    ]
    for key in relevant_keys:
        if key in exp_config:
            setattr(generation_config, key, exp_config[key])
            logger.info("Set %s to %s", key, exp_config[key])

    processors = LogitsProcessorList()

    # Adapt from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
    # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
    # better score (i.e. keep len(list(generation_config._eos_token_tensor)) + 1)
    if generation_config.num_beams is not None and generation_config.num_beams > 1:
        if isinstance(generation_config._eos_token_tensor, list):
            min_tokens_to_keep = len(generation_config._eos_token_tensor) + 1
        elif isinstance(generation_config._eos_token_tensor, torch.Tensor):
            min_tokens_to_keep = generation_config._eos_token_tensor.shape[0] + 1
        else:
            min_tokens_to_keep = 2
    else:
        min_tokens_to_keep = 1

    # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
    # all samplers can be found in `generation_utils_samplers.py`
    if (
        generation_config.temperature is not None
        and generation_config.temperature != 1.0
    ):
        processors.append(TemperatureLogitsWarper(generation_config.temperature))
    if hasattr(generation_config, "top_h") and generation_config.top_h is not None:
        processors.append(TopHLogitsWarper(top_h=generation_config.top_h))
    if generation_config.top_k is not None and generation_config.top_k != 0:
        processors.append(
            TopKLogitsWarper(
                top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep
            )
        )
    if generation_config.top_p is not None and generation_config.top_p < 1.0:
        processors.append(
            TopPLogitsWarper(
                top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep
            )
        )
    if generation_config.min_p is not None:
        # Applied after temperature scaling (see https://github.com/ggerganov/llama.cpp/pull/3841#issuecomment-2073826084)
        processors.append(
            MinPLogitsWarper(
                min_p=generation_config.min_p, min_tokens_to_keep=min_tokens_to_keep
            )
        )
    if generation_config.typical_p is not None and generation_config.typical_p < 1.0:
        processors.append(
            TypicalLogitsWarper(
                mass=generation_config.typical_p, min_tokens_to_keep=min_tokens_to_keep
            )
        )
    if (
        generation_config.epsilon_cutoff is not None
        and 0.0 < generation_config.epsilon_cutoff < 1.0
    ):
        processors.append(
            EpsilonLogitsWarper(
                epsilon=generation_config.epsilon_cutoff,
                min_tokens_to_keep=min_tokens_to_keep,
            )
        )
    if (
        generation_config.eta_cutoff is not None
        and 0.0 < generation_config.eta_cutoff < 1.0
    ):
        processors.append(
            EtaLogitsWarper(
                epsilon=generation_config.eta_cutoff,
                min_tokens_to_keep=min_tokens_to_keep,
                device=model_device,
            )
        )
    if watermark_processor is not None:
        processors.append(watermark_processor)
    return processors


def create_watermark_processor(config: dict, vocab_size: int):
    base_wm = config["base_watermark"]
    multibit_wm = config["multibit_watermark"]
    context_width = config["context_width"]
    salt_key = config["salt_key"]
    seeding = config["seeding"]

    if base_wm == "Gumbel":
        base_processor = GumbelLogitsProcessor(
            vocab_size=vocab_size,
            context_width=context_width,
            seed=config["seed"],
            salt_key=salt_key,
            seeding=seeding,
        )
    else:
        raise ValueError(f"Unknown base_watermark: {base_wm}")

    if multibit_wm == "HeRo":
        processor = HeRoLogitsProcessor(
            base_processor=base_processor,
            bits=config["bits"],
            n_layers=len(config["layer_base"]),
            layer_base=config["layer_base"],
            split_schedule=config["split_schedule"],
            layer_salts=config["layer_salts"],
            position_salt=config["position_salt"],
        )
    processor.set_messages(config["messages"])
    return processor


def truncate_prompts(prompts, tokenizer, max_prompt_len):
    results = []
    for prompt in prompts:
        tokens = tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_len,
        )
        results.append(tokenizer.decode(tokens))
    return results


def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    args = get_args()
    exp_config = load_config(args.config_path)
    logger.info("Config loaded: %s", exp_config)

    set_seed(exp_config["seed"])

    # Load data
    input_path = exp_config["input_path"]
    prompts = load_data(input_path, exp_config["nsamples"], exp_config["input_key"])
    logger.info(f"Loaded {len(prompts)} prompts from {input_path}")

    # Load model
    model, tokenizer = load_model(exp_config["model_name"])
    # logger.info("Model default config: %s", model.generation_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompts = truncate_prompts(prompts, tokenizer, exp_config["max_prompt_len"])
    watermark_processor = create_watermark_processor(
        exp_config, model.config.vocab_size
    )
    logits_processor_list = prepare_logits_processor_list(
        exp_config,
        model.generation_config,
        watermark_processor,
        model.device,
    )

    output_path = exp_config["output_path"]
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Resume check
    start_point = count_existing_samples(output_path)
    if start_point > 0:
        logger.info("Resuming from sample %s", start_point)

    batch_size = exp_config["batch_size"]
    max_new_tokens = exp_config["max_new_tokens"]

    with open(output_path, "a", encoding="utf-8") as f:
        for ii in tqdm(range(start_point, len(prompts), batch_size)):
            chunk_prompts = prompts[ii : ii + batch_size]
            inputs = tokenizer(
                chunk_prompts,
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            prompt_len = inputs.input_ids.shape[-1]
            if watermark_processor:
                watermark_processor.set_prompt_len(prompt_len)

            start_time = time.time()

            # Default from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    num_beams=1,
                    temperature=1.0,
                    # top_h=None,
                    top_k=0,
                    top_p=1.0,
                    min_p=None,
                    typical_p=1.0,
                    epsilon_cutoff=0.0,
                    eta_cutoff=0.0,
                    logits_processor=logits_processor_list,
                    pad_token_id=tokenizer.pad_token_id,
                )
            duration = time.time() - start_time

            generated_tokens = outputs[:, inputs.input_ids.shape[1] :]
            decoded_texts = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

            # Save results
            for j, text in enumerate(decoded_texts):
                result = {
                    "index": ii + j,
                    "prompt": chunk_prompts[j],
                    "result": text,
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

            logger.info(
                "Batch %s: Speed %.2f samples/s",
                ii // batch_size,
                len(chunk_prompts) / duration,
            )

    # Log peak memory usage
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
        logger.info(f"Peak GPU memory usage: {peak_memory:.2f} GB")


if __name__ == "__main__":
    main()
